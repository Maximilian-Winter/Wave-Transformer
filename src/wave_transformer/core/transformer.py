import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple


@dataclass
class Wave:
    frequencies: torch.Tensor
    amplitudes: torch.Tensor
    phases: torch.Tensor

    def to_representation(self) -> torch.Tensor:
        return torch.cat([
            self.frequencies,
            self.amplitudes,
            self.phases
        ], dim=-1)

    @classmethod
    def from_representation(cls, x: torch.Tensor):
        chunks = x.chunk(3, dim=-1)
        return cls(
            frequencies=chunks[0],
            amplitudes=chunks[1],
            phases=chunks[2]
        )

    def synthesize(self, t: torch.Tensor) -> torch.Tensor:
        """Generate wave signal at time points t."""
        t = t.unsqueeze(-1)
        return (self.amplitudes * torch.sin(
            2 * np.pi * self.frequencies * t + self.phases
        )).sum(dim=-1)

    def plot_waveform(self, duration: float = 1.0, sample_rate: int = 1000,
                      ax: Optional[plt.Axes] = None, **kwargs):
        """Plot the time-domain waveform."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        t = torch.linspace(0, duration, int(duration * sample_rate))
        signal = self.synthesize(t).detach().numpy()

        ax.plot(t.numpy(), signal, **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Waveform')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_spectrum(self, ax: Optional[plt.Axes] = None, **kwargs):
        """Plot the frequency spectrum (amplitude vs frequency)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        freqs = self.frequencies.detach().numpy().flatten()
        amps = self.amplitudes.detach().numpy().flatten()

        ax.stem(freqs, amps, **kwargs)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Frequency Spectrum')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_phase_spectrum(self, ax: Optional[plt.Axes] = None, **kwargs):
        """Plot phase vs frequency."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        freqs = self.frequencies.detach().numpy().flatten()
        phases = self.phases.detach().numpy().flatten()

        ax.stem(freqs, phases, **kwargs)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase (radians)')
        ax.set_title('Phase Spectrum')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_spectrogram(self, duration: float = 1.0, sample_rate: int = 1000,
                         ax: Optional[plt.Axes] = None, **kwargs):
        """Plot spectrogram using STFT."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        t = torch.linspace(0, duration, int(duration * sample_rate))
        signal = self.synthesize(t).detach().numpy()

        ax.specgram(signal, Fs=sample_rate, **kwargs)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Spectrogram')

        return ax

    def plot_components(self, duration: float = 1.0, sample_rate: int = 1000,
                        ax: Optional[plt.Axes] = None):
        """Plot individual frequency components."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        t = torch.linspace(0, duration, int(duration * sample_rate))
        t_np = t.numpy()

        for i in range(self.frequencies.shape[-1]):
            component = (self.amplitudes[..., i] * torch.sin(
                2 * np.pi * self.frequencies[..., i] * t.unsqueeze(-1) +
                self.phases[..., i]
            )).detach().numpy()

            freq_val = self.frequencies[..., i].item()
            ax.plot(t_np, component.flatten(),
                    label=f'{freq_val:.1f} Hz', alpha=0.7)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Individual Wave Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_summary(self, duration: float = 1.0, sample_rate: int = 1000):
        """Create a comprehensive visualization with multiple subplots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        self.plot_waveform(duration, sample_rate, ax=axes[0, 0])
        self.plot_spectrum(ax=axes[0, 1])
        self.plot_phase_spectrum(ax=axes[1, 0])
        self.plot_components(duration, sample_rate, ax=axes[1, 1])

        plt.tight_layout()
        return fig, axes


def plot_wave_series(waves: List[Wave], duration: float = 1.0,
                     sample_rate: int = 1000, labels: Optional[List[str]] = None):
    """Plot multiple waves for comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Waveforms
    t = torch.linspace(0, duration, int(duration * sample_rate))
    for i, wave in enumerate(waves):
        label = labels[i] if labels else f'Wave {i + 1}'
        signal = wave.synthesize(t).detach().numpy()
        axes[0, 0].plot(t.numpy(), signal, label=label, alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Waveforms Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Spectra
    for i, wave in enumerate(waves):
        label = labels[i] if labels else f'Wave {i + 1}'
        freqs = wave.frequencies.detach().numpy().flatten()
        amps = wave.amplitudes.detach().numpy().flatten()
        axes[0, 1].stem(freqs, amps, label=label,
                        linefmt=f'C{i}-', markerfmt=f'C{i}o')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Frequency Spectra')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Phase spectra
    for i, wave in enumerate(waves):
        label = labels[i] if labels else f'Wave {i + 1}'
        freqs = wave.frequencies.detach().numpy().flatten()
        phases = wave.phases.detach().numpy().flatten()
        axes[1, 0].scatter(freqs, phases, label=label, s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Phase (radians)')
    axes[1, 0].set_title('Phase Spectra')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Overlaid components
    for i, wave in enumerate(waves):
        label = labels[i] if labels else f'Wave {i + 1}'
        for j in range(wave.frequencies.shape[-1]):
            component = (wave.amplitudes[..., j] * torch.sin(
                2 * np.pi * wave.frequencies[..., j] * t.unsqueeze(-1) +
                wave.phases[..., j]
            )).detach().numpy()
            axes[1, 1].plot(t.numpy(), component.flatten(),
                            alpha=0.3, color=f'C{i}')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('All Components')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, axes


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
            q.half(), k.half(), v.half(),
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


class ParallelBlock(nn.Module):
    """
    Parallel attention and FFN from GPT-J/PaLM
    Reduces latency by computing attention and FFN in parallel
    """
    def __init__(self, d_model, n_heads, n_heads_kv, d_ff, dropout=0.0, use_flash=True):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.attn = FlashAttention(d_model, n_heads, use_flash)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True, attention_mask=None):
        # Single normalization, then parallel paths
        normalized = self.norm(x)
        attn_out = self.attn(normalized, causal)
        ffn_out = self.ffn(normalized)

        # Combine and add residual
        return x + self.dropout(attn_out + ffn_out)

class DeepNormParallelBlock(nn.Module):
    """
    Parallel attention + FFN block with DeepNorm residual scaling.
    """
    def __init__(self, d_model, n_heads, n_heads_kv, d_ff, dropout=0.1, use_flash=True, num_layers=1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.attn = FlashAttention(d_model, n_heads, use_flash)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

        # DeepNorm residual scaling factor
        self.residual_scale = 1.0 / math.sqrt(2 * num_layers)

    def forward(self, x, attention_mask=None, causal=True):
        normed = self.norm(x)
        attn_out = self.attn(normed, causal=causal)
        ffn_out = self.ffn(normed)

        return x + self.dropout((attn_out + ffn_out) * self.residual_scale)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

class NonCausalParallelBlock(nn.Module):
    """
    Parallel attention and FFN from GPT-J/PaLM
    Reduces latency by computing attention and FFN in parallel
    """
    def __init__(self, d_model, n_heads, n_heads_kv, d_ff, dropout=0.1, use_flash=True):
        super().__init__()

        self.norm = RMSNorm(d_model)
        self.attn = FlashAttention(d_model, n_heads, use_flash)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        # Single normalization, then parallel paths
        normalized = self.norm(x)
        attn_out = self.attn(normalized, False)
        ffn_out = self.ffn(normalized)

        # Combine and add residual
        return x + self.dropout(attn_out + ffn_out)

class WaveTransformer(nn.Module):
    def __init__(self, wave_encoder, wave_decoder, num_harmonics=64, transformer_num_layers=6, transformer_num_heads=8, transformer_heads_kv=4,  transformer_d_ff_multi=4, dropout=0.1, use_flash=True):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.input_dim = num_harmonics * 3

        self.wave_encoder = wave_encoder

        self.layers = nn.ModuleList([
            ParallelBlock(d_model=self.input_dim, n_heads=transformer_num_heads, n_heads_kv=transformer_heads_kv, d_ff=self.input_dim * transformer_d_ff_multi,  dropout=dropout, use_flash=use_flash) for _ in range(transformer_num_layers)
        ])
        self.norm_f = RMSNorm(self.input_dim)

        self.wave_decoder = wave_decoder

    def forward(self, encoder_input: dict[str, Any], causal=True, return_encoder_outputs=False, attention_mask=None,
                plot_waves=False):
        wave = self.wave_encoder(attention_mask=attention_mask, **encoder_input)
        x = wave.to_representation()

        if plot_waves:
            wave.plot_summary()
            plt.savefig(f'wave_layer_0_input.png')
            plt.close()

        for i, block in enumerate(self.layers):
            x = block(x, causal=causal, attention_mask=attention_mask)

            if plot_waves:
                # Create temporary wave for visualization
                layer_wave = Wave.from_representation(x)
                layer_wave.plot_summary()
                plt.savefig(f'wave_layer_{i + 1}.png')
                plt.close()

        x = self.norm_f(x)
        output = self.wave_decoder(x, attention_mask=attention_mask)

        if return_encoder_outputs:
            return output, wave  # Returns original encoder wave
        return output

