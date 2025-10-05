"""
Example: How to integrate signal visualization into train_dao_wandb_signal.py

Add these modifications to your training script to unleash visual wisdom!
"""

# ============================================================================
# STEP 1: Add imports at the top of train_dao_wandb_signal.py
# ============================================================================

# Add these imports after your existing imports:
from signal_training_monitor import SignalTrainingMonitor, add_signal_visualization_to_training


# ============================================================================
# STEP 2: Initialize monitor in train_language_model_distributed()
# ============================================================================

# Add this after creating your result_dir (around line 358):

    if rank == 0:
        # Create signal visualization monitor
        signal_viz_dir = f"{result_dir}/signal_visualizations"
        signal_monitor = SignalTrainingMonitor(
            save_dir=signal_viz_dir,
            signal_names=['frequency', 'amplitude', 'phase'],
            log_interval=100,        # Log stats every 100 steps
            visualize_interval=500   # Create plots every 500 steps
        )
        print(f"Signal visualization directory: {signal_viz_dir}")


# ============================================================================
# STEP 3: Modify train_epoch() to capture signals
# ============================================================================

# In the train_epoch function, add this INSIDE the training loop
# (around line 175, right after you compute loss):

        # Visualize signals periodically (only on rank 0)
        if rank == 0 and ((batch_idx + 1) % 500) == 0:
            add_signal_visualization_to_training(
                model.module if use_ddp else model,
                batch['input_ids'],
                tokenizer,  # You'll need to pass tokenizer to train_epoch
                global_step[0],
                signal_monitor,
                max_tokens_display=15
            )
            
            # Get current signal stats for logging
            current_stats = signal_monitor.get_current_stats()
            
            # Optionally log to wandb
            if use_wandb and wandb.run is not None:
                wandb.log({
                    'signals/frequency_mean': current_stats.get('frequency_mean', 0),
                    'signals/frequency_std': current_stats.get('frequency_std', 0),
                    'signals/amplitude_mean': current_stats.get('amplitude_mean', 0),
                    'signals/amplitude_std': current_stats.get('amplitude_std', 0),
                    'signals/phase_mean': current_stats.get('phase_mean', 0),
                    'signals/phase_std': current_stats.get('phase_std', 0),
                }, step=global_step[0])


# ============================================================================
# STEP 4: Update train_epoch function signature
# ============================================================================

# Change the function signature to include tokenizer and signal_monitor:

def train_epoch(result_dir, epoch, model, dataloader, optimizer, scheduler, pad_token_id, rank, device,
                accumulation_steps=1, use_ddp=True, global_step=[0], use_wandb=True,
                tokenizer=None, signal_monitor=None):  # ADD THESE TWO PARAMS


# ============================================================================
# STEP 5: Update the train_epoch calls in main training loop
# ============================================================================

# Around line 451, update the call to train_epoch:

        train_loss = train_epoch(
            result_dir, epoch, signal_transformer_model, train_loader,
            optimizer, scheduler, pad_token_id, rank, device, accumulation_steps,
            use_ddp, global_step, use_wandb,
            tokenizer=tokenizer if rank == 0 else None,  # ADD THIS
            signal_monitor=signal_monitor if rank == 0 else None  # ADD THIS
        )


# ============================================================================
# STEP 6: Plot training evolution at end of training
# ============================================================================

# After the training loop completes (around line 505), add:

    # Plot signal evolution over entire training
    if rank == 0:
        signal_monitor.plot_training_history()
        print("✨ Signal training evolution saved!")


# ============================================================================
# COMPLETE MINIMAL EXAMPLE
# ============================================================================

"""
Here's a minimal working example you can test:

import torch
from wave_transformer.core.signal_core import SignalConfig, MultiSignal
from wave_transformer.core.normalization import linear_norm
from signal_visualizer import SignalVisualizer
import numpy as np

# Create some test signals
signal_configs = [
    SignalConfig("frequency", torch.sigmoid, linear_norm(20.0, 0.1), 32),
    SignalConfig("amplitude", torch.nn.functional.softplus, linear_norm(1.0, 0.0), 32),
    SignalConfig("phase", torch.tanh, linear_norm(np.pi, 0.0), 32),
]

# Generate random signal data
batch_size, seq_len = 4, 50
freq_data = torch.sigmoid(torch.randn(batch_size, seq_len, 32)) * 20 + 0.1
amp_data = torch.nn.functional.softplus(torch.randn(batch_size, seq_len, 32))
phase_data = torch.tanh(torch.randn(batch_size, seq_len, 32)) * np.pi

multi_signal = MultiSignal.from_signals([freq_data, amp_data, phase_data])

# Create visualizations
viz = SignalVisualizer(['frequency', 'amplitude', 'phase'])

# Dashboard view
viz.plot_comprehensive_dashboard(
    multi_signal,
    tokens=['The', 'tao', 'that', 'can', 'be'],
    save_path='test_signals.png'
)

print("✓ Test visualization created: test_signals.png")
"""
