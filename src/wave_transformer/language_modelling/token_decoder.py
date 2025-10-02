
import torch
import torch.nn as nn

from wave_transformer.core.transformer import FlashAttention, RMSNorm


# --- Slim Decoder ---
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
        low_rank_output: int = None,
        use_flash=False,# e.g. 256 for low-rank
    ):
        super().__init__()

        input_dim = num_harmonics * 3
        hidden_dim = int(d_model * hidden_mult)

        # Project raw wave representation into d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Attention + expansion
        self.self_attention = FlashAttention(d_model, num_heads, use_flash=use_flash)
        self.hidden_projection = nn.Linear(d_model, hidden_dim)

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                RMSNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ) for _ in range(num_layers)
        ])

        # Output projection (normal or low-rank)
        if low_rank_output is not None:
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim, low_rank_output),
                nn.GELU(),
                nn.Linear(low_rank_output, vocab_size),
            )
        else:
            self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, representation: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # Project to model space
        x = self.input_projection(representation)

        # Attention
        attn_out = self.self_attention(x, True)
        x = x + attn_out

        # Expand hidden dimension
        x = self.hidden_projection(x)

        # Depth
        for layer in self.decoder_layers:
            x = x + layer(x)


        logits = self.output_projection(x)
        return logits
