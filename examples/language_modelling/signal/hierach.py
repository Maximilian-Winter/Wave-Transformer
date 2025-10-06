import copy
import json
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from tokenizers import processors, Tokenizer
from matplotlib import pyplot as plt

import wandb

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from wave_transformer.core.normalization import linear_norm
from wave_transformer.core.signal_core import SignalConfig, MultiSignal
from wave_transformer.core.signal_processor import SignalTransformer
from wave_transformer.core.transformer import TransformerParallelBlockConfig
from wave_transformer.language_modelling.text_datasets import TextDatasetPaddedSimple
from wave_transformer.language_modelling.train_utils import (
    prepare_autoregressive_batch,
    compute_language_modeling_loss,
    cosine_schedule_with_warmup,
    camel_to_snake,
    extract_architecture_details,
    test_generation,
    diversity_report,
    save_training_chronicle,
    get_logits_tensor
)


# Import hierarchical components (assuming they're in your package)
# from wave_transformer.core.hierarchical import (
#     HierarchicalSignalEncoder,
#     CrossScaleAttention,
#     create_hierarchical_text_signals
# )


def create_hierarchical_text_signals():
    """Create hierarchical signal configurations for text processing."""
    return [
        SignalConfig(
            signal_name="character_patterns",
            torch_activation_function=torch.sigmoid,
            normalization=linear_norm(scale=1.0, offset=0.0),
            num_dimensions=32
        ),
        SignalConfig(
            signal_name="word_semantics",
            torch_activation_function=torch.tanh,
            normalization=linear_norm(scale=1.0, offset=0.0),
            num_dimensions=64
        ),
        SignalConfig(
            signal_name="phrase_composition",
            torch_activation_function=torch.nn.functional.gelu,
            normalization=linear_norm(scale=1.0, offset=0.0),
            num_dimensions=48
        ),
        SignalConfig(
            signal_name="sentence_structure",
            torch_activation_function=lambda x: torch.nn.functional.softplus(x) - 0.5,
            normalization=linear_norm(scale=2.0, offset=0.0),
            num_dimensions=64
        ),
        SignalConfig(
            signal_name="discourse_flow",
            torch_activation_function=torch.tanh,
            normalization=linear_norm(scale=1.0, offset=0.0),
            num_dimensions=32
        )
    ]


def train_hierarchical_epoch(
        result_dir, epoch, model, dataloader, optimizer, scheduler, pad_token_id,
        device, accumulation_steps=1, global_step=[0], use_wandb=True,
        use_diversity_loss=True, diversity_weight=0.01
):
    """Training epoch for hierarchical model with signal diversity loss."""
    model.train()
    epoch_losses = []
    epoch_diversity_losses = []

    progress = tqdm(dataloader, desc='Training')

    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0
    current_lr = 0

    for batch_idx, raw_batch in enumerate(progress):
        batch = {
            'input_ids': raw_batch['input_ids'].to(device, non_blocking=True),
            'attention_mask': raw_batch['attention_mask'].to(device, non_blocking=True)
        }

        inputs, targets, input_mask = prepare_autoregressive_batch(batch, pad_token_id)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Forward pass with signal tracking
            logits, signals = model(inputs, attention_mask=input_mask, return_encoder_outputs=True)
            logits = get_logits_tensor(logits)

            # Main language modeling loss
            lm_loss = compute_language_modeling_loss(logits, targets, pad_token_id)

            # Signal diversity loss (optional)
            diversity_loss = torch.tensor(0.0, device=device)
            if use_diversity_loss and signals is not None:
                signal_list = signals.get_all_signals()
                for i in range(len(signal_list)):
                    for j in range(i + 1, len(signal_list)):
                        # Cosine similarity between signals
                        sim = torch.nn.functional.cosine_similarity(
                            signal_list[i].flatten(start_dim=1),
                            signal_list[j].flatten(start_dim=1),
                            dim=-1
                        ).mean()
                        # Penalize high similarity
                        diversity_loss += torch.abs(sim)

                diversity_loss = diversity_loss * diversity_weight

            # Total loss
            loss = lm_loss + diversity_loss

        if not torch.isfinite(loss):
            print(f"Loss is NaN/Inf at batch {batch_idx}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        normalized_loss = loss / accumulation_steps
        normalized_loss.backward()

        loss_value = loss.detach().item()
        lm_loss_value = lm_loss.detach().item()
        diversity_loss_value = diversity_loss.detach().item() if use_diversity_loss else 0.0

        epoch_losses.append(lm_loss_value)
        epoch_diversity_losses.append(diversity_loss_value)
        accumulated_loss += loss_value

        if ((batch_idx + 1) % accumulation_steps) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            accumulated_loss = 0
            global_step[0] += 1
            current_lr = optimizer.param_groups[0]['lr']

        # Save checkpoint periodically
        if (batch_idx + 1) % 5000 == 0:
            model.save(f"./{result_dir}/hierarchical_epoch_{epoch + 1}_batch_{batch_idx + 1}")

        # Log to wandb
        if ((batch_idx + 1) % 50) == 0 and use_wandb and wandb.run is not None:
            wandb.log({
                'train/lm_loss': lm_loss_value,
                'train/diversity_loss': diversity_loss_value,
                'train/total_loss': loss_value,
                'train/perplexity': math.exp(lm_loss_value),
                'train/learning_rate': current_lr,
                'train/global_step': global_step[0],
                'train/epoch': epoch,
            }, step=global_step[0])

        # Update progress bar
        progress.set_postfix({
            'lm_loss': f"{lm_loss_value:.4f}",
            'div_loss': f"{diversity_loss_value:.4f}",
            'ppl': f"{math.exp(lm_loss_value):.2f}",
            'lr': f"{current_lr:.2e}",
            'step': global_step[0]
        })

    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
    avg_diversity_loss = sum(epoch_diversity_losses) / len(epoch_diversity_losses) if epoch_diversity_losses else 0.0

    # Plot losses
    if len(epoch_losses) > 0:
        plot_hierarchical_losses(
            epoch_losses,
            epoch_diversity_losses,
            f"hierarchical_losses_epoch_{epoch}.png"
        )

    return avg_loss, avg_diversity_loss


def plot_hierarchical_losses(lm_losses, diversity_losses, save_path=None, window_size=100):
    """Visualize both LM and diversity losses."""
    if not lm_losses:
        print("No losses to plot")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Language Modeling Loss
    ax1 = axes[0]
    ax1.plot(lm_losses, alpha=0.7, color='blue', linewidth=0.5)
    ax1.set_ylabel('LM Loss')
    ax1.set_title('Language Modeling Loss')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Diversity Loss
    ax2 = axes[1]
    ax2.plot(diversity_losses, alpha=0.7, color='green', linewidth=0.5)
    ax2.set_ylabel('Diversity Loss')
    ax2.set_title('Signal Diversity Loss')
    ax2.grid(True, alpha=0.3)

    # Plot 3: LM Loss with smoothing
    ax3 = axes[2]
    ax3.plot(lm_losses, alpha=0.3, color='gray', linewidth=0.5, label='Raw')

    if len(lm_losses) > window_size:
        moving_avg = np.convolve(lm_losses,
                                 np.ones(window_size) / window_size,
                                 mode='valid')
        x_smooth = np.arange(len(moving_avg)) + (window_size - 1) // 2
        ax3.plot(x_smooth, moving_avg, color='red', linewidth=2,
                 label=f'Moving Avg (window={window_size})')

    ax3.set_xlabel('Batch')
    ax3.set_ylabel('LM Loss')
    ax3.set_title('Language Modeling Loss with Smoothing')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    plt.close()


def visualize_signal_evolution(model, tokenizer, text, device, save_path=None):
    """Visualize how hierarchical signals evolve during training."""
    model.eval()

    tokens = tokenizer(text, truncation=True, padding=True)
    input_ids = torch.tensor(tokens['ids']).unsqueeze(0).to(device)

    with torch.no_grad():
        _, signals = model(input_ids, return_encoder_outputs=True)

    signal_list = signals.get_all_signals()
    signal_configs = model.signals

    fig, axes = plt.subplots(len(signal_configs), 1, figsize=(15, 2 * len(signal_configs)))

    for i, (signal_data, config, ax) in enumerate(zip(signal_list, signal_configs, axes)):
        signal_np = signal_data.squeeze(0).cpu().numpy()

        im = ax.imshow(signal_np.T, aspect='auto', cmap='coolwarm')
        ax.set_title(f"{config.signal_name} ({config.num_dimensions} dims)")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Dimension")
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"Hierarchical Signals: '{text[:50]}...'")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    plt.close()


def train_hierarchical_language_model():
    """Main training function for hierarchical signal transformer."""
    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        print(f"WARNING: CUDA not available, using CPU")

    result_dir = "./results_hierarchical"
    os.makedirs(result_dir, exist_ok=True)
    print(f"Result directory: {result_dir}")
    print(f"Using device: {device}")

    # Model Parameters
    seq_len = 256
    d_model = 512
    num_layers = 24  # Reduced from 32 for hierarchical model
    num_heads = 8
    dropout = 0.1

    # Hierarchical-specific parameters
    use_diversity_loss = True
    diversity_weight = 0.01

    # Hyperparameters
    epochs = 5
    batch_size = 16 if torch.cuda.is_available() else 4
    accumulation_steps = 1
    base_lr = 3e-4
    final_lr = 3e-5
    warmup_pct = 0.1

    # Initialize wandb
    use_wandb = True
    if use_wandb:
        wandb.init(
            project="hierarchical-signal-transformer",
            name=f"hierarchical_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "seq_len": seq_len,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "dropout": dropout,
                "epochs": epochs,
                "batch_size": batch_size,
                "accumulation_steps": accumulation_steps,
                "base_lr": base_lr,
                "final_lr": final_lr,
                "warmup_pct": warmup_pct,
                "use_diversity_loss": use_diversity_loss,
                "diversity_weight": diversity_weight,
            }
        )

    # Load tokenizer
    model_name = "SmolLM2-135M-Instruct-Tokenizer.json"
    train_tokenizer = Tokenizer.from_file(model_name)

    train_tokenizer.add_special_tokens(["<|bos|>", "<|eos|>", "<|pad|>"])

    bos_token_id = train_tokenizer.token_to_id("<|bos|>")
    eos_token_id = train_tokenizer.token_to_id("<|eos|>")
    pad_token_id = train_tokenizer.token_to_id("<|pad|>")

    tokenizer = copy.deepcopy(train_tokenizer)
    train_tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        pair="<|bos|> $A <|eos|> <|bos|> $B <|eos|>",
        special_tokens=[
            ("<|bos|>", bos_token_id),
            ("<|eos|>", eos_token_id),
        ],
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A",
        pair="<|bos|> $A <|bos|> $B",
        special_tokens=[
            ("<|bos|>", bos_token_id)
        ],
    )
    train_tokenizer.enable_padding(pad_id=pad_token_id, pad_token="<|pad|>", length=seq_len)
    train_tokenizer.enable_truncation(max_length=seq_len - 2)

    vocab_size = train_tokenizer.get_vocab_size()

    def load_dao_teachings():
        with open("dao_de_jing.json", "r", encoding="utf-8") as file:
            chapters = json.load(file)
        random.shuffle(chapters)
        texts = [chapter["text"] for chapter in chapters]
        train_corpus = texts * 50
        random.shuffle(train_corpus)
        eval_corpus = None
        return train_corpus, eval_corpus

    train_corpus, eval_corpus = load_dao_teachings()
    train_dataset = TextDatasetPaddedSimple(train_corpus, train_tokenizer, pad_token_id, seq_len)

    prompts = [
        "The tao that can be told",
        "Success is as dangerous as failure.",
        "The wise man"
    ]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    print("Dataloaders created...")
    print("Creating hierarchical model...")

    # Create hierarchical signal configuration
    hierarchical_signals = create_hierarchical_text_signals()
    input_dim = sum([signal.num_dimensions for signal in hierarchical_signals])

    # Create model
    hierarchical_model = SignalTransformer(
        vocab_size=vocab_size,
        signals=hierarchical_signals,
        encoder_d_model=d_model,
        decoder_d_model=d_model,
        encoder_num_layers=4,
        decoder_num_layers=4,
        transformer_num_layers=num_layers,
        transformer_layer_config=TransformerParallelBlockConfig(
            d_model=input_dim,
            num_heads_q=num_heads,
            num_heads_kv=num_heads,
            max_seq_len=seq_len,
            d_ff=input_dim * 4
        ),
        encoder_layer_config=TransformerParallelBlockConfig(
            num_heads_q=num_heads,
            num_heads_kv=num_heads,
            max_seq_len=seq_len
        ),
        max_seq_len=seq_len,
        share_encoder_layer=False,  # Independent encoders for hierarchical signals
    ).to(device)

    print("Hierarchical Model:\n", hierarchical_model)
    total_params = sum(p.numel() for p in hierarchical_model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Log signal configuration
    for signal in hierarchical_signals:
        print(f"  {signal.signal_name}: {signal.num_dimensions} dimensions")

    # Log model info to wandb
    if use_wandb and wandb.run is not None:
        wandb.config.update({
            "total_parameters": total_params,
            "signal_configs": [s.signal_name for s in hierarchical_signals],
            "total_signal_dims": input_dim
        })

    # Create optimizer
    optimizer = optim.AdamW(
        hierarchical_model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1
    )

    # Create scheduler
    steps_per_epoch = len(train_loader) // accumulation_steps
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_pct * total_steps))

    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=base_lr,
        final_lr=final_lr
    )

    print(f"Training with base LR: {base_lr} | final LR: {final_lr} | "
          f"warmup steps: {warmup_steps} | total steps: {total_steps}")

    # Log training setup to wandb
    if use_wandb and wandb.run is not None:
        wandb.config.update({
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch
        })

    # Create chronicle
    timestamp = datetime.now().isoformat()
    chronicle = {
        'experiment_name': "hierarchical_signal_transformer_experiment",
        'timestamp': timestamp,
        'architecture': extract_architecture_details(hierarchical_model),
        'signal_hierarchy': [s.to_dict() for s in hierarchical_signals],
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'accumulation_steps': accumulation_steps,
            'base_lr': base_lr,
            'final_lr': final_lr,
            'total_steps': total_steps,
            'warmup_steps': warmup_steps,
            'warmup_pct': warmup_pct,
            'seq_len': seq_len,
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'use_diversity_loss': use_diversity_loss,
            'diversity_weight': diversity_weight
        },
        'epoch_records': [],
        'generation_samples': []
    }

    print("\n🚀 Hierarchical training initiated...")

    last_train_loss = None
    last_train_ppl = None
    global_step = [0]

    # Training loop
    for epoch in range(epochs):
        train_loss, diversity_loss = train_hierarchical_epoch(
            result_dir, epoch, hierarchical_model, train_loader,
            optimizer, scheduler, pad_token_id, device, accumulation_steps,
            global_step, use_wandb, use_diversity_loss, diversity_weight
        )

        train_ppl = math.exp(train_loss)

        # Log epoch metrics to wandb
        if use_wandb and wandb.run is not None:
            wandb.log({
                'epoch/train_lm_loss': train_loss,
                'epoch/train_diversity_loss': diversity_loss,
                'epoch/train_perplexity': train_ppl,
                'epoch/epoch': epoch + 1
            }, step=global_step[0])

        # Visualize signals for sample text
        if (epoch + 1) % 2 == 0:  # Every 2 epochs
            visualize_signal_evolution(
                hierarchical_model,
                tokenizer,
                prompts[0],
                device,
                f"{result_dir}/signals_epoch_{epoch + 1}.png"
            )

            if use_wandb and wandb.run is not None:
                wandb.log({
                    "signal_visualization": wandb.Image(f"{result_dir}/signals_epoch_{epoch + 1}.png")
                }, step=global_step[0])

        # Generation and reporting
        generations = test_generation(
            hierarchical_model, tokenizer, 50,
            device,
            prompts=prompts,
            temperature=0.65,
            top_p=0.9,
            min_p=0.025,
            repetition_penalty=1.10,
            max_seq_length=seq_len
        )
        diversity = diversity_report(generations)

        print(f"\nEpoch {epoch + 1}/{epochs}:")
        print(f"  Train LM Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}")
        print(f"  Signal Diversity Loss: {diversity_loss:.6f}")

        if last_train_loss is not None:
            print(f"  Train Improvement: {train_loss - last_train_loss:.4f}, "
                  f"Perplexity: {train_ppl - last_train_ppl:.2f}")

        last_train_loss = train_loss
        last_train_ppl = train_ppl

        print(f"  Generation Diversity: {diversity}")

        # Log diversity metrics to wandb
        if use_wandb and wandb.run is not None:
            wandb.log({
                'generation/diversity_score': diversity.get('score', 0),
                'epoch/epoch': epoch + 1
            }, step=global_step[0])

            # Log sample generations as a table
            if generations:
                generation_table = wandb.Table(
                    columns=["Epoch", "Prompt", "Generation"],
                    data=[[epoch + 1, prompt, generation]
                          for prompt, generation in zip(prompts[:len(generations)], generations)]
                )
                wandb.log({"generations": generation_table}, step=global_step[0])

        # Save model
        hierarchical_model.save(f"./{result_dir}/hierarchical_epoch_{epoch + 1}")

        # Save checkpoint
        chronicle['generation_samples'].append({
            'epoch': epoch + 1,
            'samples': generations,
            'diversity': diversity,
        })

        # Record epoch
        epoch_data = {
            'epoch': epoch + 1,
            'train_lm_loss': train_loss,
            'train_diversity_loss': diversity_loss,
            'train_perplexity': train_ppl,
            'diversity_metrics': diversity
        }
        chronicle['epoch_records'].append(epoch_data)

        # Save chronicle
        chronicle_path = save_training_chronicle(
            chronicle, result_dir,
            "hierarchical_signal_transformer_experiment",
            timestamp
        )
        print(f"  Session saved: {chronicle_path}")

    # Finish wandb run
    if use_wandb and wandb.run is not None:
        wandb.finish()

    return hierarchical_model, chronicle


def main():
    """Main entry point for hierarchical training."""
    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    print("Running Hierarchical Signal Transformer Training")
    train_hierarchical_language_model()


if __name__ == "__main__":
    main()