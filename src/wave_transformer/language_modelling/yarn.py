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
        self.max_len = max_len
        self.base = base
        self.scale = scale
        self.original_max_len = original_max_len

        # Calculate dimensions for different scaling regions
        dim = d_model // 2

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))

        # YaRN: Apply NTK-aware interpolation with wavelength-dependent scaling
        if scale > 1.0:
            # Calculate the scaling factors for each dimension
            low_freq_factor = original_max_len / max_len
            high_freq_factor = 1.0

            # Wavelength for each frequency
            wavelengths = 2 * math.pi / inv_freq

            # Smooth interpolation between low and high frequency factors
            smooth_factor = (wavelengths - beta_fast) / (beta_slow - beta_fast)
            smooth_factor = torch.clamp(smooth_factor, 0.0, 1.0)

            # Interpolate scaling factors
            scale_factors = (1 - smooth_factor) * high_freq_factor + smooth_factor * low_freq_factor

            # Apply NTK-aware scaling
            inv_freq = inv_freq / scale_factors

            # Compute mscale for attention entropy preservation
            if mscale != 1.0:
                self.mscale = mscale * math.sqrt(1 + math.log(scale) / math.log(original_max_len))
            else:
                self.mscale = 1.0
        else:
            self.mscale = 1.0

        self.register_buffer('inv_freq', inv_freq)

        # Precompute embeddings
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)

        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x):
        seq_len = x.shape[1]

        cos = self.cos[:seq_len, :].unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0)

        # Interleave even and odd dimensions
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # Apply rotation
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)

        # Apply mscale
        return rotated * self.mscale


# Example usage
if __name__ == "__main__":
    # Standard configuration (no extension)
    rope_standard = YaRNRotaryEmbedding(
        d_model=128,
        max_len=2048,
        original_max_len=2048
    )

    # Extended context with YaRN (4x extension)
    rope_yarn = YaRNRotaryEmbedding(
        d_model=128,
        max_len=8192,
        original_max_len=2048,
        scale=4.0,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0  # Set to sqrt(scale) for attention preservation
    )

    x = torch.randn(2, 4096, 128)
    output = rope_yarn(x)
    print(f"Output shape: {output.shape}")