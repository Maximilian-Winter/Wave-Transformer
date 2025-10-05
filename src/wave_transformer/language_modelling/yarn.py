import torch
import torch.nn as nn
import math


class YaRNRotaryEmbedding(nn.Module):
    def __init__(
            self,
            d_model,
            max_len=2048,
            base=10000,
            scale=1.0,
            original_max_len=2048,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0
    ):
        super().__init__()
        self.d_model = d_model

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))

        # YaRN scaling
        if scale > 1.0:
            wavelengths = 2 * math.pi / inv_freq
            smooth_factor = (wavelengths - beta_fast) / (beta_slow - beta_fast)
            smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)

            low_freq_factor = original_max_len / max_len
            scale_factors = (1 - smooth_factor) * 1.0 + smooth_factor * low_freq_factor
            inv_freq = inv_freq / scale_factors

            if mscale != 1.0:
                self.mscale = mscale * math.sqrt(1 + math.log(scale) / math.log(original_max_len))
            else:
                self.mscale = 1.0
        else:
            self.mscale = 1.0

        self.register_buffer('inv_freq', inv_freq)

        # Precompute for max length
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)

        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, q, k):
        """
        Apply RoPE to Q and K tensors.
        Expected shape: (B, N, n_heads, d_head)
        """
        seq_len = q.shape[1]

        cos = self.cos[:seq_len]  # (seq_len, d_head//2)
        sin = self.sin[:seq_len]  # (seq_len, d_head//2)

        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)

        return q_rot, k_rot

    def _apply_rotary(self, x, cos, sin):
        """
        x: (B, seq_len, n_heads, d_head)
        cos, sin: (seq_len, d_head//2)
        """
        # Split into even and odd features
        x_even = x[..., 0::2]  # (B, seq_len, n_heads, d_head//2)
        x_odd = x[..., 1::2]  # (B, seq_len, n_heads, d_head//2)

        # Reshape cos/sin for broadcasting: (1, seq_len, 1, d_head//2)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Apply rotation
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos

        return out * self.mscale
