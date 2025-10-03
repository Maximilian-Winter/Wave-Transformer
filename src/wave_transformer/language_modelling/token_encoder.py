from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from wave_transformer.core.transformer import Wave, NonCausalParallelBlock, ParallelBlock
from wave_transformer.language_modelling.embeddings import SinusoidalPositionEmbedding, RotaryPositionEmbedding

class WaveEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_heads_kv, d_ff, dropout, num_harmonics, num_layers: int = 2, use_flash=False):
        super().__init__()
        self.layers = nn.ModuleList([ParallelBlock(d_model, num_heads, num_heads_kv, d_ff, dropout, use_flash)
                                     for _ in range(num_layers)])
        self.proj = nn.Linear(d_model, num_harmonics)

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x_out = layer(x, attention_mask=attention_mask)
            x = x + x_out

        return self.proj(x)


# --- TokenToWaveEncoder ---
class TokenToWaveEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 4,
                 d_ff: int = 1024, num_harmonics: int = 64, dropout: float = 0.1,
                 max_seq_len=512, use_flash=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.num_harmonics = num_harmonics
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_flash = use_flash

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = RotaryPositionEmbedding(d_model, max_seq_len)

        self.freq_generator = WaveEncoderBlock(d_model, num_harmonics, num_harmonics, d_ff,
                                               dropout, num_harmonics, num_layers, use_flash)
        self.amp_generator = WaveEncoderBlock(d_model, num_harmonics, num_harmonics, d_ff,
                                              dropout, num_harmonics, num_layers, use_flash)
        self.phase_generator = WaveEncoderBlock(d_model, num_harmonics, num_harmonics, d_ff,
                                                dropout, num_harmonics, num_layers, use_flash)

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

    def save(self, path: Union[str, Path]):
        """Save encoder state and configuration."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'num_layers': self.num_layers,
                'd_ff': self.d_ff,
                'num_harmonics': self.num_harmonics,
                'dropout': self.dropout,
                'max_seq_len': self.max_seq_len,
                'use_flash': self.use_flash,
            }
        }

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Union[str, Path], map_location=None):
        """Load encoder from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location)

        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])

        return model