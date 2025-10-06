from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from wave_transformer.core.signal_core import MultiSignal
from wave_transformer.core.signal_processor import SignalTransformer
from hierachial import create_hierarchical_text_signals, HierarchicalSignalEncoder, CrossScaleAttention
from wave_transformer.core.transformer import TransformerParallelBlockConfig

# Complete Hierarchical Signal Transformer System

class HierarchicalSignalTransformer(nn.Module):
    """
    Complete hierarchical text processing system with SignalTransformer.
    """

    def __init__(
            self,
            vocab_size: int,
            d_model: int = 512,
            encoder_num_layers: int = 3,
            transformer_num_layers: int = 12,
            decoder_num_layers: int = 3,
            max_seq_len: int = 512,
            use_cross_scale: bool = True
    ):
        super().__init__()

        # Create hierarchical signal configurations
        self.signal_configs = create_hierarchical_text_signals()

        # Enhanced hierarchical encoder
        self.hierarchical_encoder = HierarchicalSignalEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            signal_configs=self.signal_configs,
            num_layers=encoder_num_layers
        )

        # Cross-scale attention (optional)
        self.use_cross_scale = use_cross_scale
        if use_cross_scale:
            signal_dims = [config.num_dimensions for config in self.signal_configs]
            self.cross_scale_attention = CrossScaleAttention(signal_dims)

        # Calculate total dimensions
        input_dim = sum(config.num_dimensions for config in self.signal_configs)

        # Main SignalTransformer
        self.signal_transformer = SignalTransformer(
            vocab_size=vocab_size,
            signals=self.signal_configs,
            encoder_d_model=d_model,
            encoder_num_layers=encoder_num_layers,
            transformer_num_layers=transformer_num_layers,
            transformer_layer_config=TransformerParallelBlockConfig(
                d_model=input_dim,
                num_heads_q=8,
                num_heads_kv=4,  # MQA for efficiency
                d_ff=input_dim * 4,
                max_seq_len=max_seq_len
            ),
            decoder_num_layers=decoder_num_layers,
            decoder_d_model=d_model,
            max_seq_len=max_seq_len,
            share_encoder_layer=False  # Independent encoders for each signal
        )

        # Signal-specific predictors for auxiliary losses
        self.auxiliary_heads = nn.ModuleDict({
            "character_patterns": nn.Linear(32, vocab_size),  # Next character
            "word_semantics": nn.Linear(64, vocab_size),  # Word prediction
            "phrase_composition": nn.Linear(48, 2),  # Phrase boundary
            "sentence_structure": nn.Linear(64, 5),  # Sentence type
            "discourse_flow": nn.Linear(32, 3)  # Discourse relation
        })

    def forward(
            self,
            input_ids: torch.Tensor,
            return_auxiliary: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional auxiliary outputs.
        """
        # Get hierarchical signals
        signals = self.hierarchical_encoder(input_ids)

        # Apply cross-scale attention if enabled
        if self.use_cross_scale:
            signal_list = signals.get_all_signals()
            signal_list = self.cross_scale_attention(signal_list)
            signals = MultiSignal.from_signals(signal_list)

        # Main transformer processing
        # Use the signal_transformer's forward directly
        logits = self.signal_transformer.signal_encoder(input_ids)
        x = logits.to_flat()

        # Process through transformer layers
        for block in self.signal_transformer.layers:
            x = block(x)
        x = self.signal_transformer.norm_f(x)

        # Decode to vocabulary
        main_logits = self.signal_transformer.signal_decoder(x)

        outputs = {"logits": main_logits}

        # Compute auxiliary predictions if requested
        if return_auxiliary:
            aux_outputs = {}
            for i, config in enumerate(self.signal_configs):
                signal_data = signals.get_signal_data(i)
                aux_logits = self.auxiliary_heads[config.signal_name](signal_data)
                aux_outputs[config.signal_name] = aux_logits
            outputs["auxiliary"] = aux_outputs
            outputs["signals"] = signals

        return outputs


class HierarchicalLoss(nn.Module):
    """
    Multi-scale loss function for hierarchical signals.
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            labels: torch.Tensor,
            auxiliary_labels: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute main and auxiliary losses.
        """
        losses = {}

        # Main language modeling loss
        logits = outputs["logits"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        main_loss = self.ce_loss(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1)
        )
        losses["main"] = main_loss

        # Auxiliary losses (if provided)
        if "auxiliary" in outputs and auxiliary_labels is not None:
            aux_outputs = outputs["auxiliary"]

            # Character-level loss (next character prediction)
            if "character_patterns" in aux_outputs:
                char_loss = self.ce_loss(
                    aux_outputs["character_patterns"].view(-1, self.vocab_size),
                    shift_labels.view(-1)
                )
                losses["character"] = char_loss * 0.1

            # Add other auxiliary losses as needed
            # These would require additional labels in your dataset

        # Signal diversity loss (encourage different signals to be different)
        if "signals" in outputs:
            signals = outputs["signals"].get_all_signals()
            diversity_loss = 0
            for i in range(len(signals)):
                for j in range(i + 1, len(signals)):
                    # Cosine similarity between signals
                    sim = F.cosine_similarity(
                        signals[i].view(signals[i].size(0), -1),
                        signals[j].view(signals[j].size(0), -1),
                        dim=-1
                    ).mean()
                    # Penalize high similarity
                    diversity_loss += torch.abs(sim)

            losses["diversity"] = diversity_loss * 0.01

        # Total loss
        total_loss = sum(losses.values())
        losses["total"] = total_loss

        return losses
