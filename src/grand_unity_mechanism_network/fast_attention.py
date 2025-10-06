import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from flash_attn.flash_attn_interface import flash_attn_varlen_func




class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) with NTK-aware scaling
    Based on Qwen/LLaMA-style implementation
    """

    def __init__(self, dim: int, max_position: int = 4096, base: int = 10000, ntk_factor: float = 1.0):
        """
        Args:
            dim: dimension of head (must be even)
            max_position: maximum sequence length (training)
            base: frequency base (default: 10000, like Vaswani)
            ntk_factor: scaling factor for long-context extension
        """
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"

        # Apply NTK scaling to extend context
        base = base * (ntk_factor ** (dim / (dim - 2)))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, dim/2]

        # Precompute cos/sin
        self.register_buffer("cos_cached", freqs.cos()[None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        """
        Args:
            x: [batch, seq_len, n_heads, head_dim]
            seq_len: actual sequence length
        Returns:
            rotated x
        """
        if seq_len is None:
            seq_len = x.shape[1]

        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]

        # Split last dim into 2 parts for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]  # [B, L, H, D/2]

        # Apply rotation in complex plane
        x_rot = torch.cat([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], dim=-1)

        return x_rot

class RotaryEmbeddingWithNTKFlashAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, seq_len: int = 4096,
                 use_flash: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.use_flash = use_flash

        # Projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_embedding = RotaryEmbedding(self.d_head, max_position=seq_len)  # <-- FIXED

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)

        if self.use_flash:
            return self._flash_attention(qkv)
        else:
            return self._pytorch_attention(qkv)

    def _flash_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        B, N, _, H, D = qkv.shape
        q, k, v = qkv.unbind(2)  # [B, N, H, D]

        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)

        out = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=True
        )
        out = out.to(qkv.dtype).reshape(B, N, -1)
        return self.out_proj(out)

    def _pytorch_attention(self, qkv: torch.Tensor) -> torch.Tensor:
        B, N, _, H, D = qkv.shape
        q, k, v = qkv.unbind(2)  # [B, N, H, D]

        q = self.rotary_embedding(q)
        k = self.rotary_embedding(k)

        q = q.transpose(1, 2)  # -> [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(N, N, device=q.device), diagonal=1).bool()
        scores.masked_fill_(causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class FlashAttention(nn.Module):
    """
    Attention using Flash Attention
    """

    def __init__(self, d_model: int, n_heads: int,
                 use_flash: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.use_flash = use_flash

        # Projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)


    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, N, C = x.shape

        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)

        if self.use_flash:
            return self._flash_attention(qkv, causal)
        else:
            return self._pytorch_attention(qkv, causal)

    def _flash_attention(self, qkv: torch.Tensor, causal: bool) -> torch.Tensor:
        """Use Flash Attention with block-sparse pattern"""
        B, N, _, H, D = qkv.shape

        # Flash attention expects [B, N, H, D] for q, k, v
        q, k, v = qkv.unbind(2)

        # Flash attention with causal mask
        out = flash_attn_func(
            q, k, v,
            dropout_p=0.0,
            softmax_scale=self.scale,
            causal=causal
        )

        # Convert back to original dtype and project
        out = out.to(qkv.dtype).reshape(B, N, -1)
        return self.out_proj(out)

    def _pytorch_attention(self, qkv: torch.Tensor, causal: bool) -> torch.Tensor:
        """Fallback to PyTorch implementation (less efficient)"""
        B, N, _, H, D = qkv.shape

        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask only if causal=True
        if causal:
            causal_mask = torch.triu(torch.ones(N, N, device=q.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)