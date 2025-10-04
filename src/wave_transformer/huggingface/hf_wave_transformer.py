"""
Hugging Face compatible WaveTransformer implementation
"""

import torch.nn as nn

from typing import Optional

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.generation import GenerationMixin

from wave_transformer.core.transformer import ParallelBlock, RMSNorm
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoderSlim
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder


class WaveTransformerConfig(PretrainedConfig):
    """Configuration class for WaveTransformer"""

    model_type = "wave_transformer"

    def __init__(
            self,
            vocab_size=50257,
            num_layers=3,
            num_heads=8,
            dropout=0.1,
            max_seq_len=5000,
            num_harmonics=64,
            encoder_d_model=256,
            encoder_hidden_mult=2.0,
            encoder_num_heads=4,
            encoder_num_layers=2,
            decoder_d_model=256,
            decoder_hidden_mult=1.5,
            decoder_num_heads=4,
            decoder_num_layers=2,
            decoder_low_rank_output=None,
            use_flash=False,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.num_hidden_layers = num_layers
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.num_harmonics = num_harmonics

        # Encoder config
        self.encoder_d_model = encoder_d_model
        self.encoder_hidden_mult = encoder_hidden_mult
        self.encoder_num_heads = encoder_num_heads
        self.encoder_num_layers = encoder_num_layers

        # Decoder config
        self.decoder_d_model = decoder_d_model
        self.decoder_hidden_mult = decoder_hidden_mult
        self.decoder_num_heads = decoder_num_heads
        self.decoder_num_layers = decoder_num_layers
        self.decoder_low_rank_output = decoder_low_rank_output
        self.use_flash = use_flash


class WaveTransformerForCausalLM(PreTrainedModel, GenerationMixin):
    """
    WaveTransformer model for causal language modeling, compatible with HF Trainer
    """

    config_class = WaveTransformerConfig
    base_model_prefix = "wave_transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ParallelBlock", "TokenToWaveEncoderSimple", "WaveToTokenDecoder"]

    def __init__(self, config: WaveTransformerConfig):
        super().__init__(config)
        self.config = config

        # Initialize wave encoder
        self.wave_encoder = TokenToWaveEncoderSlim(
            config.vocab_size,
            config.num_harmonics,
            config.encoder_d_model,
            config.encoder_hidden_mult,
            config.encoder_num_heads,
            config.encoder_num_heads,
            config.encoder_num_layers,
            False
        )
        # self.wave_encoder = TokenToWaveEncoder(
        #    config.vocab_size,
        #    config.encoder_d_model,
        #    config.encoder_num_layers,
        #    int(config.encoder_hidden_mult * config.encoder_d_model),
        #    config.num_harmonics,
        #    use_flash=config.use_flash,
        # )

        # Transformer layers
        input_dim = config.num_harmonics * 3
        self.layers = nn.ModuleList([
            ParallelBlock(
                input_dim,
                config.num_heads,
                config.num_heads,
                input_dim * 4,
                dropout=config.dropout,
                use_flash=config.use_flash,
            ) for _ in range(config.num_layers)
        ])

        self.norm_f = RMSNorm(input_dim)

        # Wave decoder
        self.wave_decoder = WaveToTokenDecoder(
            vocab_size=config.vocab_size,
            num_harmonics=config.num_harmonics,
            d_model=config.decoder_d_model,
            hidden_mult=config.decoder_hidden_mult,
            num_heads=config.decoder_num_heads,
            num_heads_kv=config.decoder_num_heads,
            num_layers=config.decoder_num_layers,
            low_rank_output=config.decoder_low_rank_output,
            use_flash=config.use_flash,
        )

        # Initialize weights
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode
        wave = self.wave_encoder(input_ids, attention_mask=attention_mask)
        x = wave.to_representation()

        all_hidden_states = () if output_hidden_states else None
        for block in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
            x = block(x, attention_mask=attention_mask)

        x = self.norm_f(x)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        # Decode
        logits = self.wave_decoder(x, attention_mask=attention_mask)

        # Loss (labels already masked at dataset level)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size),
                            shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + (all_hidden_states,) if output_hidden_states else (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs
    ):
        """Prepare inputs for generation (text generation support)"""

        # Only use input_ids
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorder cache for beam search (not implemented yet)"""
        return past_key_values

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        """Resize token embeddings if vocab size changes"""
        print("Resizing token embeddings not implemented yet")
        pass


# Register the model with transformers
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("wave_transformer", WaveTransformerConfig)
AutoModelForCausalLM.register(WaveTransformerConfig, WaveTransformerForCausalLM)
