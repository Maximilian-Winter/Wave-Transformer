import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from wave_transformer.core.transformer import FlashAttention, RMSNorm, Wave, ParallelBlock, \
    NonCausalParallelBlock
from wave_transformer.language_modelling.embeddings import RotaryPositionEmbedding, HashEmbedding, \
    SinusoidalPositionEmbedding


# --- Slim Encoder ---
class TokenToWaveEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_harmonics: int = 64,
        d_model: int = 256,
        hidden_mult: float = 2.0,
        num_heads: int = 4,
        num_heads_kv: int = 4,
        num_layers: int = 2,
        shared_projector: bool = False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.num_harmonics = num_harmonics
        hidden_dim = int(d_model * hidden_mult)

        # Attention + projection
        self.self_attention = FlashAttention(d_model, num_heads)
        self.input_projection = nn.Linear(d_model, hidden_dim)

        # Semantic encoder layers
        self.semantic_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                RMSNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ) for _ in range(num_layers)
        ])

        # Wave projectors
        if shared_projector:
            self.projector = nn.Linear(hidden_dim, num_harmonics * 4)
        else:
            self.freq_projector = nn.Linear(hidden_dim, num_harmonics)
            self.amp_projector = nn.Linear(hidden_dim, num_harmonics)
            self.phase_projector = nn.Linear(hidden_dim, num_harmonics)
            self.phase_projector = nn.Linear(hidden_dim, num_harmonics)

        self.shared_projector = shared_projector

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)

        # Attention
        attn_out = self.self_attention(x, False, attention_mask)
        x = x + attn_out

        # Hidden expansion
        x = self.input_projection(x)

        # Semantic depth
        for layer in self.semantic_layers:
            x = x + layer(x)

        # Generate harmonics
        if self.shared_projector:
            proj = self.projector(x)
            f, a, p = proj.chunk(4, dim=-1)
        else:
            f = self.freq_projector(x)
            a = self.amp_projector(x)
            p = self.phase_projector(x)

        frequencies = torch.sigmoid(f) * 20.0 + 0.1     # >0
        amplitudes = F.softplus(a)                      # >0, unnormalized
        phases = torch.tanh(p) * np.pi                  # [-π, π]

        return Wave(frequencies, amplitudes, phases)


class WaveAwarePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Frequency path: Use standard sinusoidal (naturally periodic)
        self.freq_pe = SinusoidalPositionEmbedding(d_model, max_len)

        # Amplitude path: Use learnable positions (more flexible for magnitude)
        self.amp_pe = nn.Embedding(max_len, d_model)

        # Phase path: Use rotary embeddings (natural for phase information)
        self.phase_pe = RotaryPositionEmbedding(d_model, max_len)

    def forward(self, x, component='freq'):
        if component == 'freq':
            return self.freq_pe(x)
        elif component == 'amp':
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device)
            return x + self.amp_pe(positions).unsqueeze(0)
        elif component == 'phase':
            return self.phase_pe(x)
        return x


class FrequencyAwareEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, num_freq_bands=8):
        super().__init__()
        self.base_embedding = nn.Embedding(vocab_size, d_model // 2)

        # Fourier features for tokens
        self.freq_bands = nn.Parameter(torch.randn(num_freq_bands, d_model // 2))
        self.token_to_freq = nn.Embedding(vocab_size, num_freq_bands)

    def forward(self, token_ids):
        base_emb = self.base_embedding(token_ids)

        # Get frequency weights for each token
        freq_weights = torch.softmax(self.token_to_freq(token_ids), dim=-1)

        # Combine with frequency bands
        freq_emb = torch.einsum('bsf,fd->bsd', freq_weights, self.freq_bands)

        return torch.cat([base_emb, freq_emb], dim=-1)


class TokenToWaveEncoderImproved(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 4,
                 d_ff: int = 1024, num_harmonics: int = 64, dropout: float = 0.1):
        super().__init__()

        # Use frequency-aware embeddings
        self.embedding = FrequencyAwareEmbedding(vocab_size, d_model)

        # Component-specific positional encodings
        self.wave_pos_encoding = WaveAwarePositionalEncoding(d_model)

        # Component-specific encoders with different configurations
        self.freq_generator = WaveEncoderBlock(
            d_model, num_harmonics, num_harmonics, d_ff, dropout, num_harmonics, num_layers
        )
        self.amp_generator = WaveEncoderBlock(
            d_model, num_harmonics // 2, num_harmonics // 2, d_ff, dropout, num_harmonics, num_layers
        )
        self.phase_generator = WaveEncoderBlock(
            d_model, num_harmonics, num_harmonics, d_ff, dropout, num_harmonics, num_layers
        )

        # Normalization layers for stability
        self.freq_norm = nn.LayerNorm(d_model)
        self.amp_norm = nn.LayerNorm(d_model)
        self.phase_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, attention_mask=None):
        # Initial embedding
        x = self.embedding(token_ids)

        # Generate wave components
        f = self.freq_generator(self.wave_pos_encoding(x, 'freq'), attention_mask)
        a = self.amp_generator(self.wave_pos_encoding(x, 'amp'), attention_mask)
        p = self.phase_generator(self.wave_pos_encoding(x, 'phase'), attention_mask)

        # Improved activation functions for wave properties
        frequencies = torch.exp(torch.clamp(f, -3, 3)) * 2.0  # Exponential for frequency
        amplitudes = F.softplus(a) + 1e-6  # Ensure non-zero
        phases = torch.atan2(torch.sin(p), torch.cos(p))  # Proper phase wrapping

        return Wave(frequencies, amplitudes, phases)

class WaveEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_heads_kv, d_ff, dropout, num_harmonics, num_layers: int = 2, use_flash=False):
        super().__init__()
        self.layers = nn.ModuleList([NonCausalParallelBlock(d_model, num_heads, num_heads_kv, d_ff, dropout, use_flash)
                                     for _ in range(num_layers)])
        self.proj = nn.Linear(d_model, num_harmonics)

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x_out = layer(x, attention_mask=attention_mask)
            x = x + x_out

        return self.proj(x)


class TokenToWaveEncoderSimple(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 4,
                 d_ff: int = 1024, num_harmonics: int = 64, dropout: float = 0.1, use_flash=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = SinusoidalPositionEmbedding(d_model, 512)

        self.freq_generator = WaveEncoderBlock(d_model, num_harmonics, num_harmonics, d_ff, dropout,  num_harmonics, num_layers, use_flash)

        self.amp_generator = WaveEncoderBlock(d_model, num_harmonics, num_harmonics, d_ff, dropout, num_harmonics, num_layers, use_flash)

        self.phase_generator = WaveEncoderBlock(d_model, num_harmonics, num_harmonics, d_ff, dropout, num_harmonics, num_layers, use_flash)

    def forward(self, token_ids: torch.Tensor, attention_mask=None):
        x = self.embedding(token_ids)
        x = self.position_embedding(x)

        f = self.freq_generator(x=x, attention_mask=attention_mask)
        a = self.amp_generator(x=x, attention_mask=attention_mask)
        p = self.phase_generator(x=x, attention_mask=attention_mask)

        frequencies = torch.sigmoid(f) * 20.0 + 0.1
        amplitudes = F.softplus(a)
        phases = torch.tanh(p) * np.pi

        return Wave(frequencies, amplitudes, phases)