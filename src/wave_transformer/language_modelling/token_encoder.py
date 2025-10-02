import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from wave_transformer.core.transformer import Wave, NonCausalParallelBlock
from wave_transformer.language_modelling.embeddings import SinusoidalPositionEmbedding, RotaryPositionEmbedding

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


class TokenToWaveEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 4,
                 d_ff: int = 1024, num_harmonics: int = 64, dropout: float = 0.1, max_seq_len=512, use_flash=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = RotaryPositionEmbedding(d_model, max_seq_len)

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