import json
import math
import os
import random
from time import sleep
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt

from torch import optim, nn
from torch.utils.data import DataLoader, get_worker_info, IterableDataset

from tqdm import tqdm

from wave_transformer.core.transformer import WaveTransformer
from wave_transformer.language_modelling.text_datasets import MultiBoundedStreamingDataset, BoundedStreamingDataset
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoder
from wave_transformer.language_modelling.train_utils import (
    prepare_autoregressive_batch,
    compute_language_modeling_loss,
    cosine_schedule_with_warmup,
    camel_to_snake,
    extract_architecture_details,
    test_generation,
    diversity_report,
    save_training_chronicle,
    save_model_bundle
)


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


class DistributedIterableWrapper(torch.utils.data.IterableDataset):
    """
    Wrapper to make IterableDataset work with distributed training.
    Each GPU will get different batches from the stream.
    """

    def __init__(self, dataset, rank, world_size):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        # Each GPU skips samples based on its rank
        for i, sample in enumerate(self.dataset):
            if i % self.world_size == self.rank:
                yield sample


def train_epoch(result_dir, epoch, model, dataloader, optimizer, scheduler, pad_token_id, rank, accumulation_steps=1):
    """Training epoch with distributed support."""
    model.train()
    epoch_losses = []

    # Only show progress bar on main process
    if rank == 0:
        progress = tqdm(dataloader, desc='Training')
    else:
        progress = dataloader

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
            if rank == 0:
                print("Loss is NaN or Inf; skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        normalized_loss = loss / accumulation_steps
        normalized_loss.backward()

        loss_value = loss.detach().item()
        epoch_losses.append(loss_value)
        accumulated_loss += loss_value

        # Save checkpoint from main process only
        if rank == 0 and (batch_idx + 1) % 10000 == 0:
            save_model_bundle(
                model.module,  # Access underlying model from DDP wrapper
                result_dir,
                f"wave_transformer_batch_{batch_idx + 1}",
                epoch=epoch + 1,
                optimizer=optimizer,
                scheduler=scheduler
            )

        # Step after accumulation
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            accumulated_loss = 0

        # Update progress bar on main process
        if rank == 0:
            if scheduler is not None and len(optimizer.param_groups) > 0:
                cur_lr = optimizer.param_groups[0]['lr']
                progress.set_postfix({
                    'loss': f"{loss_value:.4f}",
                    'ppl': f"{math.exp(loss_value):.2f}",
                    'lr': f"{cur_lr:.2e}"
                })
            else:
                progress.set_postfix({
                    'loss': f"{loss_value:.4f}",
                    'ppl': f"{math.exp(loss_value):.2f}"
                })

    # Gather losses from all processes
    if len(epoch_losses) > 0:
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_loss_tensor = torch.tensor(avg_loss).cuda(rank)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / dist.get_world_size()
    else:
        avg_loss = 0.0

    # Plot losses only on main process
    if rank == 0:
        plot_epoch_losses(epoch_losses, f"{camel_to_snake(model.module.__class__.__name__)}_epoch_{epoch}.png")

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
def evaluate_epoch(model, dataloader, pad_token_id, rank):
    """Evaluation epoch with distributed support."""
    model.eval()
    epoch_losses = []

    # Only show progress bar on main process
    if rank == 0:
        progress = tqdm(dataloader, desc='Evaluating')
    else:
        progress = dataloader

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

        if rank == 0:
            progress.set_postfix({
                'loss': f"{loss_value:.4f}",
                'ppl': f"{math.exp(loss_value):.2f}"
            })

    # Gather losses from all processes
    if len(epoch_losses) > 0:
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_loss_tensor = torch.tensor(avg_loss).cuda(rank)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / dist.get_world_size()
    else:
        avg_loss = 0.0

    return avg_loss


def train_language_model_distributed(rank, world_size):
    """Main training function for distributed training."""
    # Setup distributed training
    setup(rank, world_size)

    if rank == 0:
        print(f"Training on {world_size} GPUs")

    result_dir = "./results"
    if rank == 0:
        os.makedirs(result_dir, exist_ok=True)
        print(f"Result directory: {result_dir}")

    # Model Parameters
    seq_len = 512
    d_model = 512
    num_layers = 48
    num_heads = 8
    dropout = 0.1
    num_harmonics = 64

    # Hyperparameters - adjust batch size per GPU
    epochs = 5
    batch_size = 32  # Per GPU batch size (16 total across 2 GPUs)
    eval_batch_size = 1
    accumulation_steps = 1
    base_lr = 3e-4
    final_lr = 5e-5
    warmup_pct = 0.025

    model_name = "SmolLM2-135M-Instruct-Tokenizer.json"

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(model_name)
    pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0

    if rank == 0:
        print("Pad Token ID:", pad_token_id)
        print("Pad Token:", tokenizer.decode([pad_token_id], False))

    vocab_size = tokenizer.get_vocab_size()
    torch.set_float32_matmul_precision('high')

    # Load datasets
    dataset_specs = [
        {"name": "wikimedia/wikipedia", "subset": "20231101.en", "skip": 0, "max_entries": 500_000, "weight": 0.5},
        {"name": "HuggingFaceFW/fineweb", "skip": 0, "max_entries": 500_000, "weight": 0.5},
    ]

    # Create datasets with different seeds for each rank to ensure different data
    train_dataset = MultiBoundedStreamingDataset(
        dataset_specs, tokenizer, pad_token_id, seq_len,
        device=torch.device(f'cuda:{rank}'),
        seed=42 + rank,  # Different seed per GPU for different data
        preloaded_data=None
    )

    eval_dataset = BoundedStreamingDataset(
        "HuggingFaceFW/fineweb", tokenizer, pad_token_id, seq_len,
        max_entries=5000, skip_first=500_000 + rank * 5000,  # Different skip per GPU
        device=torch.device(f'cuda:{rank}'),
        preloaded_data=None
    )

    if rank == 0:
        print("Datasets created...")

    # Wrap datasets for distributed training
    distributed_train_dataset = DistributedIterableWrapper(train_dataset, rank, world_size)
    distributed_eval_dataset = DistributedIterableWrapper(eval_dataset, rank, world_size)

    # Create dataloaders WITHOUT samplers (they don't work with IterableDataset)
    train_loader = DataLoader(
        distributed_train_dataset,
        batch_size=batch_size,
        drop_last=True,

    )

    eval_loader = DataLoader(
        distributed_eval_dataset,
        batch_size=eval_batch_size,
        drop_last=False,

    )

    if rank == 0:
        print("Dataloaders created...")

    # Create model components
    wave_encoder = TokenToWaveEncoder(
        vocab_size=vocab_size,
        num_harmonics=num_harmonics,
        num_layers=4,
        d_model=d_model,
        dropout=dropout,
        max_seq_len=seq_len
    )

    wave_decoder = WaveToTokenDecoder(
        vocab_size=vocab_size,
        num_harmonics=num_harmonics,
        d_model=d_model,
        hidden_mult=2.0,
        num_heads=8,
        num_layers=3,
        low_rank_output=512
    )

    if rank == 0:
        print("Creating model...")

    # Create model and move to GPU
    wave_transformer_model = WaveTransformer(
        wave_encoder=wave_encoder,
        wave_decoder=wave_decoder,
        num_harmonics=num_harmonics,
        transformer_num_heads=num_heads,
        transformer_heads_kv=num_heads,
        transformer_num_layers=num_layers,
        transformer_d_ff_multi=4,
        dropout=dropout
    ).to(rank, dtype=torch.bfloat16)

    # Wrap model with DDP
    wave_transformer_model = DDP(
        wave_transformer_model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    if rank == 0:
        print("Model:\n", wave_transformer_model.module)
        print(f"Model parameters: {sum(p.numel() for p in wave_transformer_model.parameters()):,}")

    # Create optimizer
    optimizer = optim.AdamW(
        wave_transformer_model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1
    )

    # Create scheduler - note: steps should account for distributed training
    # Each GPU sees fewer batches, so adjust accordingly
    steps_per_epoch = (1000000 // world_size) // (batch_size * accumulation_steps)  # Approximate
    total_steps = epochs * steps_per_epoch
    warmup_steps = max(1, int(warmup_pct * total_steps))

    scheduler = cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=base_lr,
        final_lr=final_lr
    )

    if rank == 0:
        print(f"Training with base LR: {base_lr} | final LR: {final_lr} | "
              f"warmup steps: {warmup_steps} | total steps: {total_steps}")

    # Create chronicle (only on main process)
    if rank == 0:
        timestamp = datetime.now().isoformat()
        chronicle = {
            'experiment_name': f"{camel_to_snake(wave_transformer_model.module.__class__.__name__)}_experiment",
            'timestamp': timestamp,
            'architecture': extract_architecture_details(wave_transformer_model.module),
            'hyperparameters': {
                'epochs': epochs,
                'batch_size': batch_size * world_size,  # Total batch size
                'batch_size_per_gpu': batch_size,
                'num_gpus': world_size,
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
    last_eval_loss = None
    last_train_ppl = None
    last_eval_ppl = None

    # Training loop
    for epoch in range(epochs):
        # For streaming datasets, we don't need to set epoch

        # Train
        train_loss = train_epoch(
            result_dir, epoch, wave_transformer_model, train_loader,
            optimizer, scheduler, pad_token_id, rank, accumulation_steps
        )

        # Evaluate
        eval_loss = evaluate_epoch(
            wave_transformer_model, eval_loader, pad_token_id, rank
        )

        train_ppl = math.exp(train_loss)
        eval_ppl = math.exp(eval_loss)

        # Generation and reporting (only on main process)
        if rank == 0:
            generations = test_generation(
                wave_transformer_model.module, tokenizer, 100,
                torch.device(f'cuda:{rank}'),
                prompts=[
                    "The tao that can be told",
                    "Success is as dangerous as failure."
                ]
            )
            diversity = diversity_report(generations)

            print(f"\nEpoch {epoch + 1}/{epochs}:")
            print(f"  Train: Loss={train_loss:.4f}, Perplexity={train_ppl:.2f}")
            print(f"  Eval: Loss={eval_loss:.4f}, Perplexity={eval_ppl:.2f}")

            if last_train_loss is not None:
                print(f"  Train Improvement: {train_loss - last_train_loss:.4f}, "
                      f"Perplexity: {train_ppl - last_train_ppl:.2f}")
                print(f"  Eval Improvement: {eval_loss - last_eval_loss:.4f}, "
                      f"Perplexity: {eval_ppl - last_eval_ppl:.2f}")

            last_train_loss = train_loss
            last_eval_loss = eval_loss
            last_train_ppl = train_ppl
            last_eval_ppl = eval_ppl

            print(f"  Diversity: {diversity}")

            # Save checkpoint
            chronicle['generation_samples'].append({
                'epoch': epoch + 1,
                'samples': generations,
                'diversity': diversity,
            })

            save_model_bundle(
                wave_transformer_model.module,
                result_dir,
                "wave_transformer",
                epoch=epoch + 1,
                optimizer=optimizer,
                scheduler=scheduler
            )

            # Record epoch
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'train_perplexity': train_ppl,
                'eval_perplexity': eval_ppl,
                'diversity_metrics': diversity
            }
            chronicle['epoch_records'].append(epoch_data)

            # Save chronicle
            chronicle_path = save_training_chronicle(
                chronicle, result_dir,
                f"{camel_to_snake(wave_transformer_model.module.__class__.__name__)}_experiment",
                timestamp
            )
            print(f"  Session saved: {chronicle_path}")

        # Synchronize between processes
        dist.barrier()
        sleep(0.025)

    cleanup()
    return wave_transformer_model, chronicle if rank == 0 else (wave_transformer_model, None)


def main():
    """Main entry point for multi-GPU training."""
    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Number of GPUs to use
    world_size = 2

    if torch.cuda.device_count() < world_size:
        print(f"Warning: Requested {world_size} GPUs but only {torch.cuda.device_count()} available")
        world_size = min(world_size, torch.cuda.device_count())

    # Spawn processes for distributed training
    mp.spawn(
        train_language_model_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()