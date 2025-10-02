import json
import math
import random
from time import sleep
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch import optim, nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb

from wave_transformer.core.transformer import WaveTransformer

from wave_transformer.language_modelling.text_datasets import MultiBoundedStreamingDatasetDef, BoundedStreamingDataset, TextDatasetPadded
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoderSimple

from wave_transformer.language_modelling.train_utils import prepare_autoregressive_batch, compute_language_modeling_loss, \
    cosine_schedule_with_warmup, camel_to_snake, extract_architecture_details, test_generation, diversity_report, \
    save_training_chronicle, compute_distillation_loss


def train_epoch(epoch, model, dataloader, optimizer, scheduler, pad_token_id, accumulation_steps=1):
    model.train()
    epoch_losses = []

    progress = tqdm(dataloader, desc='Training')
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0

    for batch_idx, raw_batch in enumerate(progress):
        batch = {
            'input_ids': raw_batch['input_ids'],
            'attention_mask': raw_batch['attention_mask']
        }

        inputs, targets, input_mask = prepare_autoregressive_batch(batch, pad_token_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model({"token_ids": inputs}, attention_mask=input_mask)
            loss = compute_language_modeling_loss(logits, targets, pad_token_id)

        if not torch.isfinite(loss):
            print("Loss is NaN or Inf; skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        normalized_loss = loss / accumulation_steps
        normalized_loss.backward()

        loss_value = loss.detach().item()
        epoch_losses.append(loss_value)
        accumulated_loss += loss_value
        
        # Log batch metrics to wandb
        global_step = epoch * len(dataloader) + batch_idx
        wandb.log({
            "train/batch_loss": loss_value,
            "train/batch_perplexity": math.exp(loss_value),
            "train/learning_rate": optimizer.param_groups[0]['lr'] if scheduler else base_lr,
        }, step=global_step)
        
        if (batch_idx + 1) % 10000 == 0:
            checkpoint_path = f"wave_transformer_batch_{batch_idx + 1}.pt"
            torch.save({
                'epoch': batch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            wandb.save(checkpoint_path)

        # step only after we have accumulated grads
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            accumulated_loss = 0

        # (optional) show current LR
        if scheduler is not None and len(optimizer.param_groups) > 0:
            cur_lr = optimizer.param_groups[0]['lr']
            progress.set_postfix({'loss': f"{loss_value:.4f}", 'ppl': f"{math.exp(loss_value):.2f}", 'lr': f"{cur_lr:.2e}"})
        else:
            progress.set_postfix({'loss': f"{loss_value:.4f}", 'ppl': f"{math.exp(loss_value):.2f}"})

    plot_path = f"{camel_to_snake(model.__class__.__name__)}_epoch_{epoch}.png"
    plot_epoch_losses(epoch_losses, plot_path)
    wandb.log({"train/loss_plot": wandb.Image(plot_path)}, step=epoch)
    
    return sum(epoch_losses) / max(1, len(epoch_losses))


def plot_epoch_losses(epoch_losses, save_path=None, window_size=100):
    """
    Visualize training loss development over an epoch.

    Args:
        epoch_losses: List of loss values from training
        save_path: Optional path to save the plot
        window_size: Window size for moving average smoothing
    """
    if not epoch_losses:
        print("No losses to plot")
        return

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Raw losses
    ax1.plot(epoch_losses, alpha=0.7, color='blue', linewidth=0.5)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss per Batch')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Raw + Smoothed losses
    ax2.plot(epoch_losses, alpha=0.3, color='gray', linewidth=0.5, label='Raw')

    # Add moving average if enough data points
    if len(epoch_losses) > window_size:
        moving_avg = np.convolve(epoch_losses,
                                 np.ones(window_size) / window_size,
                                 mode='valid')
        # Create x coordinates for the smoothed line (centered)
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
def evaluate_epoch(model, dataloader, pad_token_id):
    model.eval()
    epoch_losses = []

    progress = tqdm(dataloader, desc='Evaluating')
    for idx, raw_batch in enumerate(progress):
        batch = {
            'input_ids': raw_batch['input_ids'],
            'attention_mask': raw_batch['attention_mask']
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs, targets, input_mask = prepare_autoregressive_batch(batch, pad_token_id)
            logits = model({"token_ids": inputs}, attention_mask=input_mask)
        loss = compute_language_modeling_loss(logits, targets, pad_token_id)

        loss_value = float(loss)
        epoch_losses.append(loss_value)
        progress.set_postfix({'loss': f"{loss_value:.4f}", 'ppl': f"{math.exp(loss_value):.2f}"})

    return sum(epoch_losses) / max(1, len(epoch_losses))


def train_language_model(big_training: bool = False, wandb_project: str = "wave-transformer", wandb_name: str = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model Parameters
    seq_len = 512
    d_model = 512
    num_layers = 42
    num_heads = 8
    dropout = 0.1
    num_harmonics = 64

    # Hyperparameters
    epochs = 3
    batch_size = 16
    eval_batch_size = 1
    accumulation_steps = 2
    base_lr = 3e-4
    final_lr = 5e-5
    warmup_pct = 0.025

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Load tokenizer ...
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0

    print("Pad Token ID:", pad_token_id)
    print("Pad Token:", tokenizer.decode([pad_token_id], False))

    vocab_size = tokenizer.get_vocab_size()
    torch.set_float32_matmul_precision('high')
    
    if big_training:
        dataset_specs = [
            {"name": "wikimedia/wikipedia", "subset": "20231101.en", "skip": 0, "max_entries": 1_500_000, "weight": 0.2},
            {"name": "roneneldan/TinyStories", "skip": 0, "max_entries": 500_000, "weight": 0.3},
            {"name": "HuggingFaceFW/fineweb", "skip": 1000, "max_entries": 1_500_000, "weight": 0.5},
        ]
        train_dataset = MultiBoundedStreamingDatasetDef(dataset_specs, tokenizer, pad_token_id, seq_len, device=device)
        eval_dataset = BoundedStreamingDataset("HuggingFaceFW/fineweb", tokenizer, pad_token_id, seq_len,
                                               max_entries=10000, skip_first=2_500_000, device=device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, drop_last=False)
    else:
        def load_dao_teachings():
            with open("dao_de_jing.json", "r", encoding="utf-8") as file:
                chapters = json.load(file)
            random.shuffle(chapters)
            random.shuffle(chapters)
            texts = [chapter["text"] for chapter in chapters]
            factor = int(len(texts) * 0.95)
            train_corpus = texts * 50
            eval_corpus = texts[factor:]

            return train_corpus, eval_corpus

        train_corpus, eval_corpus = load_dao_teachings()
        train_dataset = TextDatasetPadded(train_corpus, tokenizer, pad_token_id, seq_len, device=device)
        eval_dataset = TextDatasetPadded(eval_corpus, tokenizer, pad_token_id, seq_len, device=device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, drop_last=False)

    wave_encoder = TokenToWaveEncoderSimple(vocab_size=vocab_size, num_harmonics=num_harmonics, d_model=d_model)

    wave_decoder = WaveToTokenDecoder(vocab_size=vocab_size, num_harmonics=num_harmonics, d_model=d_model, hidden_mult=1.5, num_heads=8, num_layers=3,
                                 low_rank_output=256)
    # Model
    model = WaveTransformer(
        wave_encoder=wave_encoder,
        wave_decoder=wave_decoder,
        num_harmonics=num_harmonics,
        transformer_num_heads=num_heads,
        transformer_heads_kv=num_heads,
        transformer_num_layers=num_layers,
        transformer_d_ff_multi= 4,
        dropout=dropout
    ).to(device, dtype=torch.bfloat16)

    print("Model:\n", model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1
    )

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

    print(f"Training with base LR: {base_lr} | final LR: {final_lr} | warmup steps: {warmup_steps} | total steps: {total_steps}")

    # Initialize wandb
    config = {
        'model': 'WaveTransformer',
        'epochs': epochs,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'accumulation_steps': accumulation_steps,
        'base_lr': base_lr,
        'final_lr': final_lr,
        'warmup_pct': warmup_pct,
        'warmup_steps': warmup_steps,
        'total_steps': total_steps,
        'seq_len': seq_len,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dropout': dropout,
        'num_harmonics': num_harmonics,
        'total_params': total_params,
        'big_training': big_training,
    }
    
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config=config
    )
    
    # Watch model gradients and parameters
    wandb.watch(model, log='all', log_freq=100)

    # Create chronicle
    timestamp = datetime.now().isoformat()
    chronicle = {
        'experiment_name': f"{camel_to_snake(model.__class__.__name__)}_experiment",
        'timestamp': timestamp,
        'architecture': extract_architecture_details(model),
        'hyperparameters': config,
        'epoch_records': [],
        'generation_samples': []
    }

    print("\n🚀 Training initiated...")
    last_train_loss = None
    last_eval_loss = None
    last_train_ppl = None
    last_eval_ppl = None
    
    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(
            epoch, model, train_loader, optimizer, scheduler,
            pad_token_id, accumulation_steps
        )
        eval_loss = evaluate_epoch(model, eval_loader, pad_token_id)

        train_ppl = math.exp(train_loss)
        eval_ppl = math.exp(eval_loss)

        generations = test_generation(model, tokenizer, 50, device, prompts=[
            "The tao that can be told",
            "Success is as dangerous as failure."]
        )
        diversity = diversity_report(generations)
        
        # Log epoch metrics to wandb
        epoch_metrics = {
            "epoch": epoch + 1,
            "train/epoch_loss": train_loss,
            "train/epoch_perplexity": train_ppl,
            "eval/loss": eval_loss,
            "eval/perplexity": eval_ppl,
        }
        
        # Add diversity metrics
        if diversity:
            for key, value in diversity.items():
                epoch_metrics[f"diversity/{key}"] = value
        
        # Add improvement metrics
        if last_train_loss is not None:
            epoch_metrics["train/loss_improvement"] = last_train_loss - train_loss
            epoch_metrics["train/ppl_improvement"] = last_train_ppl - train_ppl
            epoch_metrics["eval/loss_improvement"] = last_eval_loss - eval_loss
            epoch_metrics["eval/ppl_improvement"] = last_eval_ppl - eval_ppl
        
        wandb.log(epoch_metrics, step=epoch)
        
        # Log generation samples as table
        if generations:
            samples_table = wandb.Table(columns=["prompt", "generation"])
            for gen in generations:
                samples_table.add_data(gen.get('prompt', ''), gen.get('generation', ''))
            wandb.log({f"generations/epoch_{epoch + 1}": samples_table}, step=epoch)
        
        print(f"\nEpoch {epoch + 1}/{epochs}:")
        print(f"  Train: Loss={train_loss:.4f}, Perplexity={train_ppl:.2f}")
        print(f"  Eval: Loss={eval_loss:.4f}, Perplexity={eval_ppl:.2f}")
        
        if last_train_loss is not None:
            print(f"  Train Improvement: {train_loss - last_train_loss:.4f}, Perplexity: {train_ppl - last_train_ppl:.2f}")
            print(f"  Eval Improvement: {eval_loss - last_eval_loss:.4f}, Perplexity: {eval_ppl - last_eval_ppl:.2f}")
        
        last_train_loss = train_loss
        last_eval_loss = eval_loss
        last_train_ppl = train_ppl
        last_eval_ppl = eval_ppl
        
        print(f"  Diversity: {diversity}")
        
        # Save checkpoint
        checkpoint_path = f"wave_transformer_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_path)
        wandb.save(checkpoint_path)
        print(f"  Checkpoint: {checkpoint_path}")
        
        chronicle['generation_samples'].append({
            'epoch': epoch + 1,
            'samples': generations,
            'diversity': diversity,
        })

        # Record epoch
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'train_perplexity': train_ppl,
            'eval_perplexity': eval_ppl,
        }
        chronicle['epoch_records'].append(epoch_data)

        # Save chronicle
        chronicle_path = save_training_chronicle(chronicle, f"{camel_to_snake(model.__class__.__name__)}_experiment", timestamp)
        wandb.save(chronicle_path)
        print(f"  Session saved: {chronicle_path}")

        sleep(0.025)

    wandb.finish()
    return model, chronicle


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, chronicle = train_language_model()
