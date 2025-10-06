import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np

from wave_transformer.core.normalization import linear_norm
from wave_transformer.core.signal_core import SignalConfig, MultiSignal
from wave_transformer.core.transformer import TransformerParallelBlockConfig


def create_hierarchical_text_signals() -> List[SignalConfig]:
    """
    Create hierarchical signal configurations for text processing.
    Each level captures different linguistic granularities.
    """

    # Character/subword level - rapid local patterns
    character_signal = SignalConfig(
        signal_name="character_patterns",
        torch_activation_function=torch.sigmoid,
        normalization=linear_norm(scale=1.0, offset=0.0),
        num_dimensions=32
    )

    # Word/token level - lexical semantics
    word_signal = SignalConfig(
        signal_name="word_semantics",
        torch_activation_function=torch.tanh,
        normalization=linear_norm(scale=1.0, offset=0.0),
        num_dimensions=64
    )

    # Phrase level - compositional meanings
    phrase_signal = SignalConfig(
        signal_name="phrase_composition",
        torch_activation_function=F.gelu,
        normalization=linear_norm(scale=1.0, offset=0.0),
        num_dimensions=48
    )

    # Sentence level - complete thoughts
    sentence_signal = SignalConfig(
        signal_name="sentence_structure",
        torch_activation_function=lambda x: F.softplus(x) - 0.5,  # Shifted softplus
        normalization=linear_norm(scale=2.0, offset=0.0),
        num_dimensions=64
    )

    # Discourse level - long-range dependencies
    discourse_signal = SignalConfig(
        signal_name="discourse_flow",
        torch_activation_function=torch.tanh,
        normalization=linear_norm(scale=1.0, offset=0.0),
        num_dimensions=32
    )

    return [
        character_signal,
        word_signal,
        phrase_signal,
        sentence_signal,
        discourse_signal
    ]


class HierarchicalSignalEncoder(nn.Module):
    """
    Enhanced encoder that processes signals at different temporal resolutions.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            signal_configs: List[SignalConfig],
            num_layers: int = 3,
            layer_config: TransformerParallelBlockConfig = None
    ):
        super().__init__()

        self.signal_configs = signal_configs
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Different attention spans for different signals
        self.temporal_processors = nn.ModuleDict({
            "character_patterns": self._make_temporal_block(d_model, kernel_size=3),
            "word_semantics": self._make_temporal_block(d_model, kernel_size=7),
            "phrase_composition": self._make_temporal_block(d_model, kernel_size=15),
            "sentence_structure": self._make_temporal_block(d_model, kernel_size=31),
            "discourse_flow": self._make_temporal_block(d_model, kernel_size=63),
        })

        # Signal-specific encoders
        self.signal_encoders = nn.ModuleDict({
            config.signal_name: nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, config.num_dimensions)
            )
            for config in signal_configs
        })

    def _make_temporal_block(self, d_model: int, kernel_size: int) -> nn.Module:
        """Create a conv block with specific receptive field."""
        return nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model),
            nn.GroupNorm(8, d_model),
            nn.SiLU()
        )

    def forward(self, token_ids: torch.Tensor) -> MultiSignal:
        x = self.embedding(token_ids)  # (batch, seq, d_model)
        x_conv = x.transpose(1, 2)  # (batch, d_model, seq) for conv1d

        signals = []
        for config in self.signal_configs:
            # Apply temporal processing for this signal's scale
            x_temporal = self.temporal_processors[config.signal_name](x_conv)
            x_temporal = x_temporal.transpose(1, 2)  # Back to (batch, seq, d_model)

            # Combine with original for residual
            x_combined = x + x_temporal

            # Encode to signal dimensions
            signal = self.signal_encoders[config.signal_name](x_combined)

            # Apply activation and normalization
            signal = config.torch_activation_function(signal)
            signal = config.normalization.apply(signal)

            signals.append(signal)

        return MultiSignal.from_signals(signals)


class CrossScaleAttention(nn.Module):
    """
    Allow signals at different scales to interact.
    """

    def __init__(self, signal_dims: List[int]):
        super().__init__()

        self.cross_attentions = nn.ModuleList()
        for i, dim_i in enumerate(signal_dims):
            for j, dim_j in enumerate(signal_dims):
                if i != j:  # Cross-attention between different scales
                    self.cross_attentions.append(
                        nn.MultiheadAttention(
                            dim_i,
                            num_heads=max(1, dim_i // 8),
                            kdim=dim_j,
                            vdim=dim_j,
                            batch_first=True
                        )
                    )

    def forward(self, signals: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply cross-scale attention between all signal pairs.
        """
        updated_signals = []
        attn_idx = 0

        for i, signal_i in enumerate(signals):
            updates = [signal_i]  # Start with original

            for j, signal_j in enumerate(signals):
                if i != j:
                    # Attend from signal_i to signal_j
                    attended, _ = self.cross_attentions[attn_idx](
                        signal_i, signal_j, signal_j
                    )
                    updates.append(attended * 0.1)  # Small weight for cross-attention
                    attn_idx += 1

            # Combine all updates
            updated = sum(updates)
            updated_signals.append(updated)

        return updated_signals