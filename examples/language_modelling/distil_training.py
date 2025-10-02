import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from wave_transformer.language_modelling.train_utils import prepare_autoregressive_batch, compute_distillation_loss, \
    camel_to_snake


def train_epoch_distil(epoch, model, teacher_model, dataloader, optimizer, scheduler, pad_token_id, accumulation_steps=1):
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
            student_logits = model(inputs)

            with torch.no_grad():
                teacher_logits = teacher_model(inputs).logits

            loss, lm_loss_val, kl_loss_val = compute_distillation_loss(
                student_logits, teacher_logits, targets, pad_token_id,
                alpha=0.5, temperature=2.0
            )

        if not torch.isfinite(loss):
            print("Loss is NaN or Inf; skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue

        normalized_loss = loss / accumulation_steps
        normalized_loss.backward()

        loss_value = loss.detach().item()
        epoch_losses.append(loss_value)
        accumulated_loss += loss_value
        if (batch_idx + 1) % 10000 == 0:
            torch.save({
                'epoch': batch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"wave_transformer_batch_{batch_idx + 1}.pt")

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

    plot_epoch_losses(epoch_losses, f"{camel_to_snake(model.__class__.__name__)}_epoch_{epoch}.png")
    return sum(epoch_losses) / max(1, len(epoch_losses))


@torch.no_grad()
def evaluate_epoch_distil(model, teacher_model, dataloader, pad_token_id, alpha=0.5, temperature=2.0):
    model.eval()
    teacher_model.eval()

    epoch_losses = []
    epoch_lm_losses = []
    epoch_kl_losses = []

    progress = tqdm(dataloader, desc='Evaluating (distil)')
    for idx, raw_batch in enumerate(progress):
        batch = {
            'input_ids': raw_batch['input_ids'],
            'attention_mask': raw_batch['attention_mask']
        }

        inputs, targets, input_mask = prepare_autoregressive_batch(batch, pad_token_id)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            student_logits = model(inputs)

            with torch.no_grad():
                teacher_logits = teacher_model(inputs).logits

            loss, lm_loss_val, kl_loss_val = compute_distillation_loss(
                student_logits, teacher_logits, targets, pad_token_id,
                alpha=alpha, temperature=temperature
            )

        loss_value = float(loss)
        epoch_losses.append(loss_value)
        epoch_lm_losses.append(lm_loss_val)
        epoch_kl_losses.append(kl_loss_val)

        progress.set_postfix({
            'loss': f"{loss_value:.4f}",
            'lm': f"{lm_loss_val:.4f}",
            'kl': f"{kl_loss_val:.4f}",
            'ppl': f"{math.exp(loss_value):.2f}"
        })

    return (
        sum(epoch_losses) / max(1, len(epoch_losses)),
        sum(epoch_lm_losses) / max(1, len(epoch_lm_losses)),
        sum(epoch_kl_losses) / max(1, len(epoch_kl_losses)),
    )


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

    plt.show()
