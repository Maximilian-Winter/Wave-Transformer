import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import math
from typing import List, Dict, Tuple
from tqdm import tqdm

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


def train_hierarchical_model(
        model: HierarchicalSignalTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 5,
        learning_rate: float = 5e-4,
        device: str = "cuda"
):
    """
    Training loop for hierarchical signal transformer.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader)
    )

    loss_fn = HierarchicalLoss(model.signal_transformer.vocab_size)

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {"total": 0, "main": 0, "diversity": 0}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)

            # Forward pass with auxiliary outputs
            outputs = model(input_ids, return_auxiliary=True)

            # Compute losses
            losses = loss_fn(outputs, input_ids)

            # Backward pass
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update metrics
            for key, value in losses.items():
                if key in epoch_losses:
                    epoch_losses[key] += value.item()

            # Update progress bar
            if batch_idx % 10 == 0:
                avg_losses = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                progress_bar.set_postfix(avg_losses)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                outputs = model(input_ids, return_auxiliary=False)

                logits = outputs["logits"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, model.signal_transformer.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, Val Perplexity: {math.exp(avg_val_loss):.2f}")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": avg_val_loss
        }
        torch.save(checkpoint, f"hierarchical_checkpoint_epoch_{epoch + 1}.pt")


# Visualization function to inspect learned signals
@torch.no_grad()
def visualize_hierarchical_signals(
        model: HierarchicalSignalTransformer,
        text: str,
        tokenizer,
        device: str = "cuda"
):
    """
    Visualize the hierarchical signals for a given text.
    """
    model.eval()

    # Tokenize
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokens["input_ids"].to(device)

    # Get signals
    outputs = model(input_ids, return_auxiliary=True)
    signals = outputs["signals"]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(model.signal_configs), 1, figsize=(15, 10))

    for i, (config, ax) in enumerate(zip(model.signal_configs, axes)):
        signal_data = signals.get_signal_data(i).squeeze(0).cpu().numpy()

        im = ax.imshow(signal_data.T, aspect='auto', cmap='coolwarm')
        ax.set_title(f"{config.signal_name} ({config.num_dimensions} dims)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Dimension")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("hierarchical_signals.png")
    plt.show()

    return signals


# Example usage
if __name__ == "__main__":
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = HierarchicalSignalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        encoder_num_layers=3,
        transformer_num_layers=12,
        decoder_num_layers=3,
        use_cross_scale=True
    ).to(device="cuda")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    test_input = torch.randint(0, tokenizer.vocab_size, (2, 128)).to(device="cuda")
    outputs = model(test_input, return_auxiliary=True)

    print("Output shapes:")
    print(f"  Main logits: {outputs['logits'].shape}")
    print(f"  Signals: {outputs['signals'].shape}")

    # Visualize for sample text
    sample_text = "The quick brown fox jumps over the lazy dog."
    visualize_hierarchical_signals(model, sample_text, tokenizer, device)