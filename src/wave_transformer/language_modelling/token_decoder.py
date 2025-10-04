import json
import os.path
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from wave_transformer.core.transformer import MultiQueryFlashAttention, RMSNorm


# --- WaveToTokenDecoder ---
class WaveToTokenDecoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            num_harmonics: int = 64,
            d_model: int = 256,
            hidden_mult: float = 2.0,
            num_heads: int = 4,
            num_heads_kv: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
            low_rank_output: int = None,
            use_flash=False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_harmonics = num_harmonics
        self.d_model = d_model
        self.hidden_mult = hidden_mult
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.num_layers = num_layers
        self.dropout = dropout
        self.low_rank_output = low_rank_output
        self.use_flash = use_flash

        input_dim = num_harmonics * 3
        hidden_dim = int(d_model * hidden_mult)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.self_attention = MultiQueryFlashAttention(d_model, num_heads, num_heads_kv, self.dropout, use_yarn=True, use_flash=use_flash)
        self.hidden_projection = nn.Linear(d_model, hidden_dim)

        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                RMSNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ) for _ in range(num_layers)
        ])

        if low_rank_output is not None:
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim, low_rank_output),
                nn.GELU(),
                nn.Linear(low_rank_output, vocab_size),
            )
        else:
            self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, representation: torch.Tensor, attention_mask=None) -> torch.Tensor:
        x = self.input_projection(representation)
        attn_out = self.self_attention(x, True)
        x = x + attn_out
        x = self.hidden_projection(x)

        for layer in self.decoder_layers:
            x = x + layer(x)

        logits = self.output_projection(x)
        return logits

    def save(self, model_dir: Union[str, Path]):
        """Save decoder state and configuration."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        config_path = path / 'decoder_config.json'
        checkpoint_path = path / 'decoder_state_dict.pt'

        config = {
            'vocab_size': self.vocab_size,
            'num_harmonics': self.num_harmonics,
            'd_model': self.d_model,
            'hidden_mult': self.hidden_mult,
            'num_heads': self.num_heads,
            'num_heads_kv': self.num_heads_kv,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'low_rank_output': self.low_rank_output,
            'use_flash': self.use_flash,
        }

        checkpoint = {
            'decoder_state_dict': self.state_dict(),
        }

        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load(cls, model_dir: Union[str, Path], map_location=None):
        """Load decoder from model directory."""
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            path = Path(model_dir)

            config_path = path / 'decoder_config.json'
            checkpoint_path = path / 'decoder_state_dict.pt'

            with open(config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)
            model = cls(**config)

            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint['decoder_state_dict'])

            return model
        else:
            raise FileNotFoundError