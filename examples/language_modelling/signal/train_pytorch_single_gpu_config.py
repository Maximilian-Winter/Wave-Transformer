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
from wave_transformer.core.signal_core import SignalConfig
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


def train_epoch(result_dir, epoch, model, dataloader, optimizer, scheduler, pad_token_id, device,
                accumulation_steps=1, global_step=[0], use_wandb=True):
    """Training epoch with detailed logging."""
    model.train()
    epoch_losses = []

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
            logits = model(inputs, attention_mask=input_mask)
            logits = get_logits_tensor(logits)
            loss = compute_language_modeling_loss(logits, targets, pad_token_id)
           
        if not torch.isfinite(loss):
            print(f"Loss is NaN/Inf at batch {batch_idx}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        normalized_loss = loss / accumulation_steps
        normalized_loss.backward()

        loss_value = loss.detach().item()
        epoch_losses.append(loss_value)
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

        # Log to wandb
        if ((batch_idx + 1) % 50) == 0 and use_wandb and wandb.run is not None:
            wandb.log({
                'train/loss': loss_value,
                'train/perplexity': math.exp(loss_value),
                'train/learning_rate': current_lr,
                'train/global_step': global_step[0],
                'train/epoch': epoch,
            }, step=global_step[0])



        # Update progress bar
        progress.set_postfix({
            'loss': f"{loss_value:.4f}",
            'ppl': f"{math.exp(loss_value):.2f}",
            'lr': f"{current_lr:.2e}",
            'step': global_step[0]
        })

    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

    # Plot losses
    if len(epoch_losses) > 0:
        model_name = model.__class__.__name__
        plot_epoch_losses(epoch_losses, f"{camel_to_snake(model_name)}_epoch_{epoch}.png")

    return avg_loss


def plot_epoch_losses(epoch_losses, save_path=None, window_size=100):
    """Visualize training loss development over an epoch."""
    if not epoch_losses:
        print("No losses to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Raw losses
    ax1.plot(epoch_losses, alpha=0.7, color='blue', linewidth=0.5)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss per Batch')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Raw + Smoothed losses
    ax2.plot(epoch_losses, alpha=0.3, color='gray', linewidth=0.5, label='Raw')

    if len(epoch_losses) > window_size:
        moving_avg = np.convolve(epoch_losses,
                                 np.ones(window_size) / window_size,
                                 mode='valid')
        x_smooth = np.arange(len(moving_avg)) + (window_size - 1) // 2
        ax2.plot(x_smooth, moving_avg, color='red', linewidth=2,
                 label=f'Moving Avg (window={window_size})')

    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss with Smoothing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    plt.close()


@torch.no_grad()
def evaluate_epoch(model, dataloader, pad_token_id, device, use_wandb=True,
                   epoch=None, global_step=None):
    """Evaluation epoch with wandb logging."""
    model.eval()
    epoch_losses = []

    progress = tqdm(dataloader, desc='Evaluating')

    for idx, raw_batch in enumerate(progress):
        batch = {
            'input_ids': raw_batch['input_ids'].to(device, non_blocking=True),
            'attention_mask': raw_batch['attention_mask'].to(device, non_blocking=True)
        }

        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs, targets, input_mask = prepare_autoregressive_batch(batch, pad_token_id)
            logits = model({"token_ids": inputs}, attention_mask=input_mask)

        loss = compute_language_modeling_loss(logits, targets, pad_token_id)
        loss_value = float(loss)
        epoch_losses.append(loss_value)

        progress.set_postfix({
            'loss': f"{loss_value:.4f}",
            'ppl': f"{math.exp(loss_value):.2f}"
        })

    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

    # Log validation metrics to wandb
    if use_wandb and wandb.run is not None and global_step is not None:
        wandb.log({
            'val/loss': avg_loss,
            'val/perplexity': math.exp(avg_loss),
            'val/epoch': epoch if epoch is not None else 0
        }, step=global_step)

    return avg_loss


def train_language_model():
    """Main training function with wandb support."""
    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        print(f"WARNING: CUDA not available, using CPU")

    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    print(f"Result directory: {result_dir}")
    print(f"Using device: {device}")

    # Model Parameters
    seq_len = 256
    d_model = 512
    num_layers = 32
    num_heads = 8
    dropout = 0.1

    # Hyperparameters
    epochs = 5
    batch_size = 16 if torch.cuda.is_available() else 4
    eval_batch_size = 1
    accumulation_steps = 1
    base_lr = 3e-4
    final_lr = 3e-5
    warmup_pct = 0.1

    # Initialize wandb
    use_wandb = True
    if use_wandb:
        wandb.init(
            project="signal-transformer-training",
            name=f"signal_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
        random.shuffle(chapters)
        texts = [chapter["text"] for chapter in chapters]
        train_corpus = texts * 5
        random.shuffle(train_corpus)
        random.shuffle(train_corpus)
        random.shuffle(train_corpus)
        random.shuffle(train_corpus)
        eval_corpus = None
        return train_corpus, eval_corpus

    train_corpus, eval_corpus = load_dao_teachings()
    train_dataset = TextDatasetPaddedSimple(train_corpus, train_tokenizer, pad_token_id, seq_len)

    prompts = [
        "The tao that can be told",
        "Success is as dangerous as failure."
    ]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    print("Dataloaders created...")
    print("Creating model...")

    dtype = torch.bfloat16
    signal_configs = [
        SignalConfig(
            signal_name="frequency",
            torch_activation_function=torch.sigmoid,
            normalization=linear_norm(scale=20.0, offset=0.1),
            num_dimensions=128
        ),
        SignalConfig(
            signal_name="amplitude",
            torch_activation_function=torch.nn.functional.softplus,
            normalization=linear_norm(scale=1.0, offset=0.0),
            num_dimensions=64
        ),
        SignalConfig(
            signal_name="phase",
            torch_activation_function=torch.tanh,
            normalization=linear_norm(scale=np.pi, offset=0.0),
            num_dimensions=32
        ),
    ]
    input_dim = sum([signal.num_dimensions for signal in signal_configs])
    signal_transformer_model = SignalTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        signals=signal_configs,
        encoder_d_model=d_model,
        decoder_d_model=d_model,
        transformer_num_layers=num_layers,
        transformer_layer_config=TransformerParallelBlockConfig(num_heads_q=num_heads, num_heads_kv=num_heads,
                                                                max_seq_len=seq_len, d_ff=input_dim * 4),
        # 3 Signals with each 32 dimensions
        encoder_layer_config=TransformerParallelBlockConfig(num_heads_q=num_heads, num_heads_kv=num_heads,
                                                            max_seq_len=seq_len),
        max_seq_len=seq_len,
        share_encoder_layer=True,
    ).to(device)

    print("Model:\n", signal_transformer_model)
    total_params = sum(p.numel() for p in signal_transformer_model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Log model info to wandb
    if use_wandb and wandb.run is not None:
        wandb.config.update({"total_parameters": total_params})

    # Create optimizer
    optimizer = optim.AdamW(
        signal_transformer_model.parameters(),
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
        'experiment_name': f"{camel_to_snake(signal_transformer_model.__class__.__name__)}_experiment",
        'timestamp': timestamp,
        'architecture': extract_architecture_details(signal_transformer_model),
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
            'num_heads': num_heads
        },
        'epoch_records': [],
        'generation_samples': []
    }

    print("\n🚀 Training initiated...")

    last_train_loss = None
    last_train_ppl = None
    global_step = [0]

    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(
            result_dir, epoch, signal_transformer_model, train_loader,
            optimizer, scheduler, pad_token_id, device, accumulation_steps,
            global_step, use_wandb
        )

        train_ppl = math.exp(train_loss)

        # Log epoch metrics to wandb
        if use_wandb and wandb.run is not None:
            wandb.log({
                'epoch/train_loss': train_loss,
                'epoch/train_perplexity': train_ppl,
                'epoch/epoch': epoch + 1
            }, step=global_step[0])

        # Generation and reporting
        generations = test_generation(
            signal_transformer_model, tokenizer, 50,
            device,
            prompts=prompts,
            temperature=0.65,
            top_p=0.9,
            #min_p=0.025,  # typical sampling floor
            repetition_penalty=1.10,
            #max_seq_length=256
        )
        diversity = diversity_report(generations)

        print(f"\nEpoch {epoch + 1}/{epochs}:")
        print(f"  Train: Loss={train_loss:.4f}, Perplexity={train_ppl:.2f}")

        if last_train_loss is not None:
            print(f"  Train Improvement: {train_loss - last_train_loss:.4f}, "
                  f"Perplexity: {train_ppl - last_train_ppl:.2f}")

        last_train_loss = train_loss
        last_train_ppl = train_ppl

        print(f"  Diversity: {diversity}")

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
                          for prompt, generation in zip(prompts, generations)]
                )
                wandb.log({"generations": generation_table}, step=global_step[0])
        signal_transformer_model.save_pretrained(
            f"./{result_dir}/epoch_{epoch + 1}",
            save_optimizer=True,
            save_scheduler=True,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=-1,

        )
        # Save checkpoint
        chronicle['generation_samples'].append({
            'epoch': epoch + 1,
            'samples': generations,
            'diversity': diversity,
        })

        # Record epoch
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_perplexity': train_ppl,
            'diversity_metrics': diversity
        }
        chronicle['epoch_records'].append(epoch_data)

        # Save chronicle
        chronicle_path = save_training_chronicle(
            chronicle, result_dir,
            f"{camel_to_snake(signal_transformer_model.__class__.__name__)}_experiment",
            timestamp
        )
        print(f"  Session saved: {chronicle_path}")

    # Finish wandb run
    if use_wandb and wandb.run is not None:
        wandb.finish()

    return signal_transformer_model, chronicle


def main():
    """Main entry point for single GPU training."""
    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    print("Running in single GPU/CPU mode")
    train_language_model()


if __name__ == "__main__":
    main()
