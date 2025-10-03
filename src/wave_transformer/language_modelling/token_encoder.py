import json
import os.path
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from wave_transformer.core.transformer import Wave, ParallelBlock
from wave_transformer.language_modelling.embeddings import RotaryPositionEmbedding

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
                 max_seq_len=4096, use_flash=True):
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

        self.freq_generator = WaveEncoderBlock(d_model, 8, 4, d_ff,
                                               dropout, num_harmonics, num_layers, use_flash)
        self.amp_generator = WaveEncoderBlock(d_model, 8, 4, d_ff,
                                              dropout, num_harmonics, num_layers, use_flash)
        self.phase_generator = WaveEncoderBlock(d_model, 8, 4, d_ff,
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

    def save(self, model_dir: Union[str, Path]):
        """Save encoder state and configuration."""
        path = Path(model_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_path = path / 'encoder_config.json'
        checkpoint_path = path / 'encoder_state_dict.pt'
        config = {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'num_layers': self.num_layers,
                'd_ff': self.d_ff,
                'num_harmonics': self.num_harmonics,
                'dropout': self.dropout,
                'max_seq_len': self.max_seq_len,
                'use_flash': self.use_flash,

        }
        checkpoint = {
            'encoder_state_dict': self.state_dict(),
        }
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config, f)
        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load(cls, model_dir: Union[str, Path], map_location=None):
        """Load encoder from model directory."""
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            path = Path(model_dir)

            config_path = path / 'encoder_config.json'
            checkpoint_path = path / 'encoder_state_dict.pt'

            with open(config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)
            model = cls(**config)

            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint['encoder_state_dict'])

            return model
        else:
            raise FileNotFoundError