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


from .signal_core import SignalConfig
from .transformer import TransformerParallelBlock, MultiQueryFlashAttention, RMSNorm, TransformerParallelBlockConfig


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, encoder_layer_config: TransformerParallelBlockConfig):
        super().__init__()
        self.norm_input = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([TransformerParallelBlock(**encoder_layer_config.to_dict())
                                     for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x, causal=True, attention_mask=None):
        x = self.norm_input(x)
        for layer in self.layers:
            x_out = layer(x, causal=causal, attention_mask=attention_mask)
            x = x + x_out

        return self.norm_f(x)

class SignalEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, output_signals: list[SignalConfig], num_layers: int, layer_config: TransformerParallelBlockConfig, share_encoder_layers: bool):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.encoder_layer_config = layer_config
        self.scale = math.sqrt(d_model)
        self.share_encoder_layers= share_encoder_layers

        self.embedding = nn.Embedding(vocab_size, d_model)

        for output_signal in output_signals:



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