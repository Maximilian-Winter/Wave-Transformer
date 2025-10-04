import copy
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
from tokenizers import processors, Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt
from wave_transformer.language_modelling.text_datasets import TextDatasetPaddedSimple

import wandb

from torch import optim, nn
from torch.utils.data import DataLoader, get_worker_info, IterableDataset

from tqdm import tqdm

from wave_transformer.core.transformer import WaveTransformer
from wave_transformer.language_modelling.streaming_dataset import MultiBoundedStreamingDataset, BoundedStreamingDataset
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
    save_model_bundle, lm_total_loss
)


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Detect available backend
    if torch.cuda.is_available() and torch.distributed.is_nccl_available():
        backend = "nccl"
        # Set device for this process
        torch.cuda.set_device(rank)
    elif torch.distributed.is_gloo_available():
        backend = "gloo"
        # Gloo works with both CPU and GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
    else:
        raise RuntimeError("No distributed backend available. Install NCCL or ensure Gloo is available.")

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if rank == 0:
        print(f"Using distributed backend: {backend}")


def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedIterableWrapper(torch.utils.data.IterableDataset):
    """
    Wrapper to make IterableDataset work with distributed training.
    Each GPU will get different batches from the stream.
    """

    def __init__(self, dataset, rank, world_size, max_entries):
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.max_entries = max_entries

    def __len__(self):
        return self.max_entries
    def __iter__(self):
        # Each GPU skips samples based on its rank
        for i, sample in enumerate(self.dataset):
            if i % self.world_size == self.rank:
                yield sample


def train_epoch(result_dir, epoch, model, dataloader, optimizer, scheduler, pad_token_id, rank, device,
                accumulation_steps=1, use_ddp=True, global_step=[0], use_wandb=True):
    """Training epoch with distributed support and detailed debugging."""
    model.train()
    epoch_losses = []

    # Initial synchronization with detailed logging
    if use_ddp:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Starting epoch {epoch}")
        try:
            dist.barrier()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Passed initial barrier")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: ERROR at initial barrier: {e}")
            raise

    # Check if dataloader has length
    try:
        dataloader_len = len(dataloader)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Dataloader has {dataloader_len} batches")
    except TypeError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Dataloader is iterable (length unknown)")
        dataloader_len = None

    # Only show progress bar on main process
    if rank == 0:
        progress = tqdm(dataloader, desc='Training')
    else:
        progress = dataloader

    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0
    batch_count = 0
    last_log_time = datetime.now()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Starting iteration over dataloader")

    for batch_idx, raw_batch in enumerate(progress):
        batch_count += 1

        # Log first batch reception
        if batch_idx == 0:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Got first batch (shape: {raw_batch['input_ids'].shape if 'input_ids' in raw_batch else 'unknown'})")

        batch = {
            'input_ids': raw_batch['input_ids'].to(device, non_blocking=True),
            'attention_mask': raw_batch['attention_mask'].to(device, non_blocking=True)
        }

        inputs, targets, input_mask = prepare_autoregressive_batch(batch, pad_token_id)

        # Forward pass with error handling
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model({"token_ids": inputs}, attention_mask=input_mask)
            loss = compute_language_modeling_loss(logits, targets, pad_token_id)
           
        if not torch.isfinite(loss):
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Loss is NaN/Inf at batch {batch_idx}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        normalized_loss = loss / accumulation_steps

        # Backward pass with error handling
        normalized_loss.backward()

        loss_value = loss.detach().item()
        epoch_losses.append(loss_value)
        accumulated_loss += loss_value

        # Step after accumulation - FIXED CONDITION
        should_step = ((batch_idx + 1) % accumulation_steps == 0)

        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Step the scheduler AFTER optimizer.step()
            if scheduler is not None:
                scheduler.step()

            accumulated_loss = 0

            # Increment global step
            global_step[0] += 1

            # Get current learning rate AFTER scheduler step
            current_lr = optimizer.param_groups[0]['lr']

            # Debug learning rate
            if batch_idx < 5 and rank == 0:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Step {global_step[0]}, LR = {current_lr:.6f}")

            # Log to wandb (fixed modulo condition)
            if rank == 0 and global_step[0] % 50 == 0 and use_wandb and wandb.run is not None:
                wandb.log({
                    'train/loss': loss_value,
                    'train/perplexity': math.exp(loss_value),
                    'train/learning_rate': current_lr,
                    'train/global_step': global_step[0],
                    'train/epoch': epoch,
                    
                }, step=global_step[0])

            # Save checkpoint
            if rank == 0 and global_step[0] % 5000 == 0:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Saving checkpoint at step {global_step[0]}")
                model_to_save = model.module if use_ddp else model
                save_model_bundle(
                    model_to_save,
                    f"{result_dir}/epoch_{epoch}_batch_{batch_idx}",
                    epoch,
                    global_step=global_step[0]
                )
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Checkpoint saved")

            # Update progress bar with current learning rate
            if rank == 0:
                progress.set_postfix({
                    'loss': f"{loss_value:.4f}",
                    
                    'ppl': f"{math.exp(loss_value):.2f}",
                    'lr': f"{current_lr:.2e}",
                    'step': global_step[0]
                })
        else:
            # Update progress bar without LR when not stepping
            if rank == 0:
                progress.set_postfix({
                    'loss': f"{loss_value:.4f}",
                   
                    'ppl': f"{math.exp(loss_value):.2f}",
                    'step': global_step[0]
                })



    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Dataloader exhausted after {batch_count} batches")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Completed {batch_count} batches total")

    # Gather losses from all processes
    if len(epoch_losses) > 0:
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Local avg loss = {avg_loss:.4f}")

        if use_ddp and dist.is_initialized():
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Starting all_reduce for losses")
            avg_loss_tensor = torch.tensor(avg_loss).cuda(rank) if torch.cuda.is_available() else torch.tensor(avg_loss)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / dist.get_world_size()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Global avg loss = {avg_loss:.4f}")
    else:
        avg_loss = 0.0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: WARNING - No losses recorded")

    # Final synchronization before plotting
    if use_ddp:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Final epoch barrier")
        dist.barrier()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Passed final barrier")

    # Plot losses only on main process
    if rank == 0 and len(epoch_losses) > 0:
        model_name = model.module.__class__.__name__ if use_ddp else model.__class__.__name__
        plot_epoch_losses(epoch_losses, f"{camel_to_snake(model_name)}_epoch_{epoch}.png")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Saved loss plot")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Rank {rank}: Epoch complete, returning avg_loss={avg_loss:.4f}")
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
def evaluate_epoch(model, dataloader, pad_token_id, rank, device, use_ddp=True, use_wandb=True,
                   epoch=None, global_step=None):
    """Evaluation epoch with distributed support and wandb logging."""
    model.eval()
    epoch_losses = []

    # Only show progress bar on main process
    if rank == 0:
        progress = tqdm(dataloader, desc='Evaluating')
    else:
        progress = dataloader

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

        if rank == 0:
            progress.set_postfix({
                'loss': f"{loss_value:.4f}",
                'ppl': f"{math.exp(loss_value):.2f}"
            })

    # Gather losses from all processes if using distributed training
    if len(epoch_losses) > 0:
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if use_ddp and dist.is_initialized():
            avg_loss_tensor = torch.tensor(avg_loss).cuda(rank) if torch.cuda.is_available() else torch.tensor(avg_loss)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss_tensor.item() / dist.get_world_size()
    else:
        avg_loss = 0.0

    # Log validation metrics to wandb
    if rank == 0 and use_wandb and wandb.run is not None and global_step is not None:
        wandb.log({
            'val/loss': avg_loss,
            'val/perplexity': math.exp(avg_loss),
            'val/epoch': epoch if epoch is not None else 0
        }, step=global_step)

    return avg_loss


def train_language_model_distributed(rank, world_size):
    """Main training function for distributed training with wandb support."""
    # Check if we're doing distributed training
    use_ddp = world_size > 1

    if use_ddp:
        # Setup distributed training
        setup(rank, world_size)

    if rank == 0:
        if use_ddp:
            print(f"Training on {world_size} GPUs")
        else:
            print("Training on single GPU/CPU")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
    else:
        device = torch.device('cpu')
        print(f"WARNING: CUDA not available, using CPU")

    result_dir = "./results"
    if rank == 0:
        os.makedirs(result_dir, exist_ok=True)
        print(f"Result directory: {result_dir}")
        print(f"Using device: {device}")

    # Model Parameters
    seq_len = 1024
    d_model = 512
    num_layers = 32
    num_heads = 8
    dropout = 0.1
    num_harmonics = 64

    # Hyperparameters - adjust batch size per GPU
    epochs = 2
    batch_size = 8 if torch.cuda.is_available() else 4
    eval_batch_size = 1
    accumulation_steps = 1
    base_lr = 3e-4
    final_lr = 5e-5
    warmup_pct = 0.1

    # Initialize wandb (only on rank 0)
    use_wandb = True
    if rank == 0 and use_wandb:
        wandb.init(
            project="wave-transformer-training",
            name=f"wave_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "seq_len": seq_len,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "dropout": dropout,
                "num_harmonics": num_harmonics,
                "epochs": epochs,
                "batch_size": batch_size * world_size,
                "batch_size_per_gpu": batch_size,
                "num_gpus": world_size,
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
        with open("corpus.json", "r", encoding="utf-8") as file:
            chapters = json.load(file)
        random.shuffle(chapters)
        random.shuffle(chapters)
        texts = [chapter["text"] for chapter in chapters]
        factor = int(len(texts) * 0.97)
        train_corpus = texts
        random.shuffle(train_corpus)
        random.shuffle(train_corpus)
        random.shuffle(train_corpus)
        random.shuffle(train_corpus)
        eval_corpus = None
        return train_corpus, eval_corpus

    train_corpus, eval_corpus = load_dao_teachings()
    avg_train_length = 0
    max_train_length = -1
    min_train_length = 10000
    entries_per_dataset = len(train_corpus)
    cleaned_corpus = []
    for train_text in train_corpus:
        train_length = len(tokenizer.encode(train_text).ids)
        if train_length >= 6:
            cleaned_corpus.append(train_text)
            avg_train_length += train_length
            max_train_length = max(max_train_length, train_length)
            min_train_length = min(min_train_length, train_length)
    entries_per_dataset = len(cleaned_corpus)
    avg_train_length = avg_train_length / len(train_corpus)
    print(
        f"Dataset Entries: {entries_per_dataset}, Average length: {avg_train_length}, Max length: {max_train_length}, Min length: {min_train_length}")
    train_dataset = TextDatasetPaddedSimple(train_corpus, train_tokenizer, pad_token_id, seq_len)

    prompts = [
        "The tao that can be told",
        "Success is as dangerous as failure."
    ]
    # Use wrapper for distribution
    if use_ddp:
        train_dataset_wrapped = DistributedIterableWrapper(
            train_dataset, rank, world_size, entries_per_dataset
        )
        train_loader = DataLoader(
            train_dataset_wrapped,
            batch_size=batch_size,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )

    if rank == 0:
        print("Dataloaders created...")

    # Create model components
    wave_encoder = TokenToWaveEncoder(
        vocab_size=vocab_size,
        num_harmonics=num_harmonics,
        num_layers=3,
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
        num_heads_kv=8,
        num_layers=3,
        low_rank_output=512
    )

    if rank == 0:
        print("Creating model...")
    dtype = torch.float32

    wave_transformer_model = WaveTransformer(
        wave_encoder=wave_encoder,
        wave_decoder=wave_decoder,
        num_harmonics=num_harmonics,
        transformer_num_heads=num_heads,
        transformer_heads_kv=num_heads,
        transformer_num_layers=num_layers,
        transformer_d_ff_multi=4,
        dropout=dropout
    ).to(device, dtype=dtype)

    # Wrap model with DDP if using multiple GPUs
    if use_ddp:
        wave_transformer_model = DDP(
            wave_transformer_model,
            device_ids=[rank] if torch.cuda.is_available() else None,
            output_device=rank if torch.cuda.is_available() else None,
            find_unused_parameters=False
        )

    if rank == 0:
        model_to_print = wave_transformer_model.module if use_ddp else wave_transformer_model
        print("Model:\n", model_to_print)
        total_params = sum(p.numel() for p in wave_transformer_model.parameters())
        print(f"Model parameters: {total_params:,}")

        # Log model info to wandb
        if use_wandb and wandb.run is not None:
            wandb.config.update({"total_parameters": total_params})

    # Create optimizer
    optimizer = optim.AdamW(
        wave_transformer_model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1
    )

    # Create scheduler
    entries_per_rank = entries_per_dataset // world_size
    steps_per_epoch = entries_per_rank // (batch_size * accumulation_steps)
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

        # Log training setup to wandb
        if use_wandb and wandb.run is not None:
            wandb.config.update({
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "steps_per_epoch": steps_per_epoch
            })

    # Create chronicle (only on main process)
    if rank == 0:
        timestamp = datetime.now().isoformat()
        model_to_save = wave_transformer_model.module if use_ddp else wave_transformer_model
        chronicle = {
            'experiment_name': f"{camel_to_snake(model_to_save.__class__.__name__)}_experiment",
            'timestamp': timestamp,
            'architecture': extract_architecture_details(model_to_save),
            'hyperparameters': {
                'epochs': epochs,
                'batch_size': batch_size * world_size,
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
    last_train_ppl = None

    # Global step counter that persists across epochs
    global_step = [0]

    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            result_dir, epoch, wave_transformer_model, train_loader,
            optimizer, scheduler, pad_token_id, rank, device, accumulation_steps,
            use_ddp, global_step, use_wandb
        )

        train_ppl = math.exp(train_loss)

        # Log epoch metrics to wandb
        if rank == 0 and use_wandb and wandb.run is not None:
            wandb.log({
                'epoch/train_loss': train_loss,
                'epoch/train_perplexity': train_ppl,
                'epoch/epoch': epoch + 1
            }, step=global_step[0])

        # Generation and reporting (only on main process)
        if rank == 0:
            model_for_gen = wave_transformer_model.module if use_ddp else wave_transformer_model
            generations = test_generation(
                model_for_gen, tokenizer, 50,
                device,
                prompts=prompts
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
                              for prompt, generation in zip(prompts, generations)]  # Log first 5 samples
                    )
                    wandb.log({"generations": generation_table}, step=global_step[0])

            # Save checkpoint
            chronicle['generation_samples'].append({
                'epoch': epoch + 1,
                'samples': generations,
                'diversity': diversity,
            })

            save_model_bundle(
                model_for_gen,
                f"{result_dir}/epoch_{epoch}_final",
                (entries_per_dataset // batch_size * (epoch + 1)),
                optimizer=optimizer,
                scheduler=scheduler
            )

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
                f"{camel_to_snake(model_for_gen.__class__.__name__)}_experiment",
                timestamp
            )
            print(f"  Session saved: {chronicle_path}")

        # Synchronize between processes if using distributed training
        if use_ddp:
            dist.barrier()
        sleep(0.025)

    # Finish wandb run
    if rank == 0 and use_wandb and wandb.run is not None:
        wandb.finish()

    if use_ddp:
        cleanup()

    return wave_transformer_model, chronicle if rank == 0 else (wave_transformer_model, None)


def main():
    """Main entry point for multi-GPU training."""
    # Set random seeds
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Number of GPUs/processes to use
    world_size = 1  # Set to 1 for single GPU/CPU, 2+ for multi-GPU

    if world_size == 1:
        # Single GPU/CPU mode - run directly without spawning
        print("Running in single GPU/CPU mode")
        train_language_model_distributed(0, 1)
    else:
        # Multi-GPU mode
        if not torch.cuda.is_available():
            print("WARNING: Multi-GPU training requested but CUDA not available. Falling back to single CPU.")
            train_language_model_distributed(0, 1)
        else:
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