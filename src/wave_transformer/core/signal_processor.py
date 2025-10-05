import dataclasses
import json
import math
import os.path
from pathlib import Path
from typing import Union, Any

import torch
import torch.nn as nn


from .signal_core import SignalConfig, MultiSignal
from .transformer import TransformerParallelBlock, TransformerParallelBlockConfig, MultiQueryFlashAttention, RMSNorm


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, encoder_layer_config: TransformerParallelBlockConfig, d_out: int = None):
        super().__init__()
        self.norm_input = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([TransformerParallelBlock(**encoder_layer_config.to_dict())
                                     for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.d_out = d_out
        if self.d_out is not None:
            self.projection = nn.Linear(d_model, d_out)

    def forward(self, x, causal=True, attention_mask=None):
        x = self.norm_input(x)
        for layer in self.layers:
            x_out = layer(x, causal=causal, attention_mask=attention_mask)
            x = x + x_out
        if self.d_out:
            return self.projection(self.norm_f(x))
        return self.norm_f(x)

class SignalEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, output_signals: list[SignalConfig], num_layers: int, layer_config: TransformerParallelBlockConfig, share_encoder_layers: bool):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.encoder_layer_config = layer_config
        self.scale = math.sqrt(d_model)
        self.share_encoder_layers = share_encoder_layers
        self.signal_configs = output_signals

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.signal_output_encoders = nn.ModuleDict({})
        if share_encoder_layers:
            self.shared_encoder = Encoder(d_model, num_layers, layer_config)

        for output_signal in self.signal_configs:
            if share_encoder_layers:
                self.signal_output_encoders[output_signal.signal_name] = nn.Linear(d_model, output_signal.num_dimensions)
            else:
                self.signal_output_encoders[output_signal.signal_name] = Encoder(d_model, num_layers, layer_config, output_signal.num_dimensions)

    def forward(self, token_ids: torch.Tensor, causal=True, attention_mask=None):
        x = self.embedding(token_ids) * self.scale

        signal_list = []
        if self.share_encoder_layers:
            x = self.shared_encoder(x, causal=causal, attention_mask=attention_mask)

        for output_signal in self.output_signals:
            if self.share_encoder_layers:
                x = self.signal_output_encoders[output_signal.signal_name](x)
            else:
                x = self.signal_output_encoders[output_signal.signal_name](x, causal=causal, attention_mask=attention_mask)
            x = output_signal.normalization.apply(x)
            signal_list.append(x)

        return MultiSignal.from_signals(signal_list)

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


class SignalDecoder(nn.Module):
    def __init__(
            self, vocab_size: int, d_model: int, output_signals: list[SignalConfig], num_layers: int, num_heads, num_heads_kv, max_seq_len, use_flash: bool,
            low_rank_output: int = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.output_signals = output_signals
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.max_seq_len = max_seq_len
        self.use_flash = use_flash
        self.low_rank_output = low_rank_output


        self.input_projection = nn.Linear(self.d_model, d_model)
        self.self_attention = MultiQueryFlashAttention(d_model, num_heads, num_heads_kv, self.dropout, max_seq_len=max_seq_len, use_flash=use_flash)
        self.hidden_projection = nn.Linear(d_model, self.d_model)

        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                RMSNorm(self.d_model),
                nn.GELU(),
                nn.Dropout(0.1),
            ) for _ in range(num_layers)
        ])

        if low_rank_output is not None:
            self.output_projection = nn.Sequential(
                nn.Linear(self.d_model, low_rank_output),
                nn.GELU(),
                nn.Linear(low_rank_output, vocab_size),
            )
        else:
            self.output_projection = nn.Linear(self.d_model, vocab_size)

    def forward(self, representation: torch.Tensor, causal=True, attention_mask=None) -> torch.Tensor:
        x = self.input_projection(representation)
        attn_out = self.self_attention(x, causal, attention_mask)
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




class SignalTransformer(nn.Module):
    def __init__(self, vocab_size, signals: list[SignalConfig], encoder_d_model: int = 256, encoder_num_layers: int = 3,
                 encoder_layer_config: TransformerParallelBlockConfig = TransformerParallelBlockConfig(),
                 share_encoder_layer: bool = False, decoder_d_model: int = 256, decoder_num_layers: int = 3, decoder_num_heads_q: int = 8, decoder_num_heads_kv: int = 8, decoder_low_rank_dim: int = None,
                 transformer_num_layers: int = 6, transformer_layer_config: TransformerParallelBlockConfig = TransformerParallelBlockConfig(), max_seq_len: int = 256, use_flash: bool = False,):
        super().__init__()
        self.vocab_size = vocab_size
        self.signals = signals
        self.encoder_d_model = encoder_d_model
        self.encoder_num_layers = encoder_num_layers
        self.encoder_layer_config = encoder_layer_config
        self.share_encoder_layer = share_encoder_layer
        self.decoder_d_model = decoder_d_model
        self.decoder_num_layers = decoder_num_layers
        self.decoder_num_heads_q = decoder_num_heads_q
        self.decoder_num_heads_kv = decoder_num_heads_kv
        self.transformer_num_layers = transformer_num_layers
        self.transformer_layer_config = transformer_layer_config

        self.signal_encoder = SignalEncoder(vocab_size, encoder_d_model, signals, encoder_num_layers, encoder_layer_config,
                                            share_encoder_layer)

        self.layers = nn.ModuleList([
            TransformerParallelBlock(**transformer_layer_config.to_dict())
            for _ in range(transformer_num_layers)
        ])
        self.norm_f = RMSNorm(self.input_dim)
        self.signal_decoder = SignalDecoder(vocab_size, decoder_d_model, signals, decoder_num_layers, decoder_num_heads_q,decoder_num_heads_kv, max_seq_len=max_seq_len, use_flash=use_flash, low_rank_output=decoder_low_rank_dim)


    def forward(self, input_ids, causal=True, return_encoder_outputs=False,
                attention_mask=None):
        signal = self.signal_encoder(input_ids, causal=causal, attention_mask=attention_mask)
        x = signal.to_flat()


        for i, block in enumerate(self.layers):
            x = block(x, causal=causal, attention_mask=attention_mask)

        x = self.norm_f(x)
        output = self.signal_decoder(x, causal=causal, attention_mask=attention_mask)

        if return_encoder_outputs:
            return output, signal
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
