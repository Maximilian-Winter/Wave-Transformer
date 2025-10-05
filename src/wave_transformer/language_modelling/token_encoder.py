import dataclasses
import json
import math
import os.path
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from wave_transformer.core.wave import Wave
from wave_transformer.core.transformer import ParallelBlock, MultiQueryFlashAttention, RMSNorm



class WaveEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_heads_kv, d_ff, dropout, num_harmonics, num_layers: int = 2, max_seq_len=256, use_flash=False):
        super().__init__()
        self.layers = nn.ModuleList([ParallelBlock(d_model=d_model, n_heads=num_heads, n_heads_kv=num_heads_kv, d_ff=d_ff, max_seq_len=max_seq_len,dropout=dropout, use_yarn=True, use_flash=use_flash)
                                     for _ in range(num_layers)])
        self.norm_input = nn.LayerNorm(d_model)
        self.norm_f = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, num_harmonics)

    def forward(self, x, attention_mask=None):
        x = self.norm_input(x)
        for layer in self.layers:
            x_out = layer(x, attention_mask=attention_mask)
            x = x + x_out

        return self.proj(self.norm_f(x))

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


# --- TokenToWaveEncoder ---
class TokenToWaveEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, num_layers: int = 4, num_heads: int = 8, num_heads_kv: int = 8,
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
        self.scale = math.sqrt(d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)


        self.freq_generator = WaveEncoderBlock(d_model, num_heads, num_heads_kv, d_ff,
                                               dropout, num_harmonics, num_layers, max_seq_len, use_flash)
        self.amp_generator = WaveEncoderBlock(d_model, num_heads, num_heads_kv, d_ff,
                                              dropout, num_harmonics, num_layers, max_seq_len, use_flash)
        self.phase_generator = WaveEncoderBlock(d_model, num_heads, num_heads_kv, d_ff,
                                                dropout, num_harmonics, num_layers, max_seq_len, use_flash)

    def forward(self, token_ids: torch.Tensor, attention_mask=None):
        x = self.embedding(token_ids) * self.scale

        f = self.freq_generator(x, attention_mask)
        a = self.amp_generator(x, attention_mask)
        p = self.phase_generator(x, attention_mask)

        frequencies = F.sigmoid(f) * 20.0 + 0.1
        amplitudes = F.softplus(a)
        phases = F.tanh(p) * np.pi

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
            json.dump(config, f, indent=4)
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


# --- Slim Encoder ---
class TokenToWaveEncoderSlim(nn.Module):
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

        self.vocab_size = vocab_size
        self.num_harmonics = num_harmonics
        self.d_model = d_model
        self.hidden_mult = hidden_mult
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.num_layers = num_layers
        self.shared_projector = shared_projector

        self.embedding = nn.Embedding(vocab_size, d_model)
        hidden_dim = int(d_model * hidden_mult)

        # Attention + projection
        self.self_attention = MultiQueryFlashAttention(d_model, num_heads, num_heads_kv)
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
            self.projector = nn.Linear(hidden_dim, num_harmonics * 3)
        else:
            self.freq_projector = nn.Linear(hidden_dim, num_harmonics)
            self.amp_projector = nn.Linear(hidden_dim, num_harmonics)
            self.phase_projector = nn.Linear(hidden_dim, num_harmonics)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)

        # Attention
        attn_out = self.self_attention(x, True, attention_mask)
        x = x + attn_out

        # Hidden expansion
        x = self.input_projection(x)

        # Semantic depth
        for layer in self.semantic_layers:
            x = x + layer(x)

        # Generate harmonics
        if self.shared_projector:
            proj = self.projector(x)
            f, a, p = proj.chunk(3, dim=-1)
        else:
            f = self.freq_projector(x)
            a = self.amp_projector(x)
            p = self.phase_projector(x)

        frequencies = torch.sigmoid(f) * 20.0 + 0.1     # >0
        amplitudes = F.softplus(a)                      # >0, unnormalized
        phases = torch.tanh(p) * np.pi                  # [-π, π]

        return Wave(frequencies, amplitudes, phases)

    def save(self, model_dir: Union[str, Path]):
        """Save encoder state and configuration."""
        path = Path(model_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_path = path / 'encoder_slim_config.json'
        checkpoint_path = path / 'encoder_slim_state_dict.pt'
        config = {
            'vocab_size': self.vocab_size,
            'num_harmonics': self.num_harmonics,
            'd_model': self.d_model,
            'hidden_mult': self.hidden_mult,
            'num_heads': self.num_heads,
            'num_heads_kv': self.num_heads_kv,
            'num_layers': self.num_layers,
            'shared_projector': self.shared_projector,
        }
        checkpoint = {
            'encoder_state_dict': self.state_dict(),
        }
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load(cls, model_dir: Union[str, Path], map_location=None):
        """Load encoder from model directory."""
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            path = Path(model_dir)

            config_path = path / 'encoder_slim_config.json'
            checkpoint_path = path / 'encoder_slim_state_dict.pt'

            with open(config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)
            model = cls(**config)

            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint['encoder_state_dict'])

            return model
        else:
            raise FileNotFoundError