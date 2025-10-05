import dataclasses
import json
import math
import os
from pathlib import Path

from typing import Any, Union

import torch

from typing import Optional
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryPositionEmbedding
from .yarn import YaRNRotaryEmbedding


class LearnableActivation(nn.Module):
    """
    Learnable activation function that maps scalars through a small MLP.
    Each input value gets its own learned transformation.
    """

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Initialize to approximate identity function
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, 1)

        hidden = torch.tanh(self.fc1(x_flat))
        out = self.fc2(hidden)

        # Add residual connection to maintain gradient flow
        out = out + x_flat

        return out.reshape(shape)


class MultiQueryFlashAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads_q: int,
            n_heads_kv: int,
            dropout_p: float = 0.0,
            use_yarn=True,
            max_seq_len: int = 256,
            use_flash: bool = True
    ):
        super().__init__()
        assert n_heads_q % n_heads_kv == 0

        self.d_model = d_model
        self.n_heads_q = n_heads_q
        self.n_heads_kv = n_heads_kv
        self.d_head = d_model // n_heads_q
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.dropout_p = dropout_p
        self.use_flash = use_flash
        self.use_yarn = use_yarn

        if self.use_flash:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            self.flash_attn_func = flash_attn_func
            self.flash_attn_varlen_func = flash_attn_varlen_func

        if use_yarn:
            # ✅ Use d_head, not d_model
            self.yarn_rope = YaRNRotaryEmbedding(
                d_model=self.d_head,  # Changed!
                max_len=max_seq_len,
                original_max_len=max_seq_len
            )

        self.q_proj = nn.Linear(d_model, n_heads_q * self.d_head, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * n_heads_kv * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            causal: bool = True,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads_q, self.d_head)
        kv = self.kv_proj(x).reshape(B, N, 2, self.n_heads_kv, self.d_head)
        k, v = kv.unbind(2)

        if self.use_yarn:
            q, k = self.yarn_rope(q, k)

        if self.use_flash and attention_mask is None:
            q, k, v = q.half(), k.half(), v.half()
            out = self.flash_attn_func(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=causal
            )
            out = out.to(dtype=x.dtype).reshape(B, N, -1)

        elif self.use_flash and attention_mask is not None:
            seqlens = attention_mask.sum(dim=1, dtype=torch.int32)
            cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0)).to(dtype=torch.int32, device=x.device)
            max_seqlen = int(seqlens.max())

            valid_mask = attention_mask.bool().flatten()

            q = q.reshape(B * N, self.n_heads_q, self.d_head)[valid_mask]
            k = k.reshape(B * N, self.n_heads_kv, self.d_head)[valid_mask]
            v = v.reshape(B * N, self.n_heads_kv, self.d_head)[valid_mask]

            q, k, v = q.half(), k.half(), v.half()

            out = self.flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=causal
            )

            out_padded = torch.zeros(B * N, self.n_heads_q, self.d_head,
                                     dtype=out.dtype, device=out.device)

            out_padded[valid_mask] = out
            out = out_padded.reshape(B, N, -1)
            out = out.to(dtype=self.out_proj.weight.dtype)
        else:
            # Fallback manual MHA
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            repeat_factor = self.n_heads_q // self.n_heads_kv
            if repeat_factor > 1:
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if causal:
                causal_mask = torch.triu(
                    torch.ones(N, N, dtype=torch.bool, device=scores.device),
                    diagonal=1
                )
                scores.masked_fill_(causal_mask, float("-inf"))

            if attention_mask is not None:
                mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
                scores = scores.masked_fill(mask == 0, float("-inf"))

            attn = F.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0:
                attn = F.dropout(attn, p=self.dropout_p)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B, N, -1)

        return self.out_proj(out)


class SwiGLU(nn.Module):
    """
    SwiGLU activation from PaLM/LLaMA
    Outperforms standard FFN in most benchmarks
    """
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        # Need 3 projections for gated activation
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: x * SiLU(gate)
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Used in: LLaMA, T5
    Benefits: Simpler, faster than LayerNorm, often better performance
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm


@dataclasses.dataclass
class TransformerParallelBlockConfig:
    d_model: int = 256
    num_heads_q: int = 8
    num_heads_kv: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256
    use_yarn: bool = True
    use_flash: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformerParallelBlockConfig":
        return cls(**data)

class TransformerParallelBlock(nn.Module):
    """
    Parallel attention and FFN from GPT-J/PaLM
    Reduces latency by computing attention and FFN in parallel
    """
    def __init__(self, d_model, num_heads_q, num_heads_kv, d_ff, max_seq_len=256, dropout=0.0, use_yarn=True, use_flash=True):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.attn = MultiQueryFlashAttention(d_model, n_heads_q=num_heads_q, n_heads_kv=num_heads_kv, dropout_p=dropout, max_seq_len=max_seq_len, use_yarn=use_yarn, use_flash=use_flash)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True, attention_mask=None):
        # Single normalization, then parallel paths
        normalized = self.norm(x)
        attn_out = self.attn(normalized, causal, attention_mask)
        ffn_out = self.ffn(normalized)

        # Combine and add residual
        return x + self.dropout(attn_out + ffn_out)


class ModernTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, n_heads_q, n_heads_k, d_ff, dropout=0.0, max_seq_len=512,
                 use_flash=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = RotaryPositionEmbedding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            TransformerParallelBlock(d_model=d_model, n_heads=n_heads_q,
                          n_heads_kv=n_heads_k, d_ff=d_ff, max_seq_len=max_seq_len,
                          dropout=dropout, use_yarn=True, use_flash=use_flash)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model)for _ in range(num_layers)])

    def forward(self, x, causal=True, attention_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for layer, norm in zip(self.layers, self.layer_norms):
            x_out = layer(x, causal=causal, attention_mask=attention_mask)
            x = x + self.dropout(x_out)
            x = norm(x)

        x = self.out_proj(x)
        return x
