"""
Training Integration for Signal Visualization

Hooks into the training loop to capture and visualize signals at key moments.
"""

import torch
from pathlib import Path
from typing import Optional, List
from signal_visualizer import SignalVisualizer, extract_signal_stats


class SignalTrainingMonitor:
    """Monitor and visualize signals during training."""
    
    def __init__(
        self, 
        save_dir: str,
        signal_names: Optional[List[str]] = None,
        log_interval: int = 500,
        visualize_interval: int = 1000
    ):
        """
        Initialize training monitor.
        
        Args:
            save_dir: Directory to save visualizations
            signal_names: Names of signals being tracked
            log_interval: How often to log stats
            visualize_interval: How often to create visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer = SignalVisualizer(signal_names)
        self.log_interval = log_interval
        self.visualize_interval = visualize_interval
        
        self.signal_history = []
        
    def log_batch_signals(
        self, 
        multi_signal,
        global_step: int,
        tokens: Optional[List[str]] = None
    ):
        """
        Log signal statistics and create visualizations if needed.
        
        Args:
            multi_signal: MultiSignal from encoder
            global_step: Current training step
            tokens: Optional decoded tokens for labeling
        """
        # Extract stats
        stats = extract_signal_stats(multi_signal)
        stats['step'] = global_step
        self.signal_history.append(stats)
        
        # Create visualizations at intervals
        if global_step % self.visualize_interval == 0:
            self._create_checkpoint_viz(multi_signal, global_step, tokens)
    
    def _create_checkpoint_viz(
        self,
        multi_signal,
        global_step: int,
        tokens: Optional[List[str]] = None
    ):
        """Create comprehensive visualization at checkpoint."""
        save_path = self.save_dir / f"signals_step_{global_step}.png"
        
        self.visualizer.plot_comprehensive_dashboard(
            multi_signal,
            tokens=tokens,
            save_path=str(save_path),
            title=f"Signal Analysis - Step {global_step}"
        )
        
        print(f"📊 Saved signal visualization: {save_path}")
    
    def plot_training_history(self):
        """Plot how signals evolved during training."""
        if not self.signal_history:
            print("No signal history to plot")
            return
        
        save_path = self.save_dir / "signal_training_evolution.png"
        self.visualizer.plot_training_evolution(
            self.signal_history,
            save_path=str(save_path)
        )
        
        print(f"📈 Saved training evolution: {save_path}")
    
    def get_current_stats(self) -> dict:
        """Get most recent signal statistics."""
        if self.signal_history:
            return self.signal_history[-1]
        return {}


def add_signal_visualization_to_training(
    model,
    batch_input_ids: torch.Tensor,
    tokenizer,
    global_step: int,
    monitor: SignalTrainingMonitor,
    max_tokens_display: int = 10
):
    """
    Helper function to add to your training loop.
    
    Example usage in train_epoch():
        if batch_idx % 500 == 0:
            add_signal_visualization_to_training(
                signal_transformer_model,
                batch['input_ids'],
                tokenizer,
                global_step[0],
                signal_monitor
            )
    
    Args:
        model: SignalTransformer model
        batch_input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: Tokenizer for decoding tokens
        global_step: Current training step
        monitor: SignalTrainingMonitor instance
        max_tokens_display: How many tokens to decode for labels
    """
    model.eval()
    
    with torch.no_grad():
        # Get signals from encoder
        signal = model.signal_encoder(
            batch_input_ids[:1],  # Just first sample
            causal=True
        )
        
        # Decode tokens for labeling
        tokens = None
        try:
            token_ids = batch_input_ids[0, :max_tokens_display].cpu().tolist()
            tokens = [tokenizer.decode([tid]) for tid in token_ids]
        except:
            pass  # Skip token decoding if it fails
        
        # Log to monitor
        monitor.log_batch_signals(signal, global_step, tokens)
    
    model.train()
