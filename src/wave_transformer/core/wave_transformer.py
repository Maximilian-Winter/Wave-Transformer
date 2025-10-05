import dataclasses
import json
import math
import os
from pathlib import Path

from typing import Any, Union

import torch

from typing import Optional
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import RotaryPositionEmbedding
from .transformer import TransformerParallelBlock, RMSNorm
from .yarn import YaRNRotaryEmbedding


class WaveTransformer(nn.Module):
    def __init__(self, wave_encoder, wave_decoder, num_harmonics=64, transformer_num_layers=6,
                 transformer_num_heads=8, transformer_heads_kv=4, transformer_d_ff_multi=4, max_seq_len=512,
                 dropout=0.1, use_flash=True):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.input_dim = num_harmonics * 3
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.transformer_heads_kv = transformer_heads_kv
        self.transformer_d_ff_multi = transformer_d_ff_multi
        self.dropout_p = dropout
        self.use_flash = use_flash
        self.wave_encoder = wave_encoder

        self.layers = nn.ModuleList([
            TransformerParallelBlock(d_model=self.input_dim, num_heads_q=transformer_num_heads,
                          num_heads_kv=transformer_heads_kv, d_ff=self.input_dim * transformer_d_ff_multi, max_seq_len=max_seq_len,
                          dropout=dropout, use_yarn=True,use_flash=use_flash)
            for _ in range(transformer_num_layers)
        ])
        self.norm_f = RMSNorm(self.input_dim)

        self.wave_decoder = wave_decoder

    def forward(self, encoder_input: dict[str, Any], causal=True, return_encoder_outputs=False,
                attention_mask=None):
        wave = self.wave_encoder(attention_mask=attention_mask, **encoder_input)
        x = wave.to_representation()


        for i, block in enumerate(self.layers):
            x = block(x, causal=causal, attention_mask=attention_mask)

        x = self.norm_f(x)
        output = self.wave_decoder(x, attention_mask=attention_mask)

        if return_encoder_outputs:
            return output, wave
        return output

    def save(self, model_dir: Union[str, Path]):
        """Save model state and configuration, including encoder and decoder."""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        config_path = path / 'transformer_config.json'
        checkpoint_path = path / 'transformer_state_dict.pt'

        config = {
            'num_harmonics': self.num_harmonics,
            'transformer_num_layers': self.transformer_num_layers,
            'transformer_num_heads': self.transformer_num_heads,
            'transformer_heads_kv': self.transformer_heads_kv,
            'transformer_d_ff_multi': self.transformer_d_ff_multi,
            'dropout': self.dropout_p,
            'use_flash': self.use_flash,
        }

        checkpoint = {
            'transformer_state_dict': self.state_dict(),
        }

        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        torch.save(checkpoint, checkpoint_path)

        # Save encoder and decoder
        self.wave_encoder.save(model_dir)
        self.wave_decoder.save(model_dir)

    @classmethod
    def load(cls, model_dir: Union[str, Path], encoder_cls, decoder_cls, map_location=None):
        """Load model from directory, including encoder and decoder."""
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            path = Path(model_dir)

            config_path = path / 'transformer_config.json'
            checkpoint_path = path / 'transformer_state_dict.pt'

            # Load encoder and decoder
            wave_encoder = encoder_cls.load(path, map_location=map_location)
            wave_decoder = decoder_cls.load(path, map_location=map_location)

            # Load transformer config and create model
            with open(config_path, 'r', encoding="utf-8") as f:
                config = json.load(f)

            model = cls(
                wave_encoder=wave_encoder,
                wave_decoder=wave_decoder,
                **config
            )

            # Load transformer state dict
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.load_state_dict(checkpoint['transformer_state_dict'])

            return model
        else:
            raise FileNotFoundError

