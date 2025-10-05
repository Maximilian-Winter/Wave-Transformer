"""
Signal Visualization for Signal Transformer Architecture

Illuminates the inner workings of frequency, amplitude, and phase signals
during training and generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from typing import List, Optional, Dict
from pathlib import Path


class SignalVisualizer:
    """Visualize signal transformer internals with fierce clarity."""
    
    def __init__(self, signal_names: List[str] = None):
        """
        Initialize visualizer.
        
        Args:
            signal_names: Names of signals (default: ['frequency', 'amplitude', 'phase'])
        """
        self.signal_names = signal_names or ['frequency', 'amplitude', 'phase']
        self.num_signals = len(self.signal_names)
        
    def plot_signal_evolution(
        self, 
        multi_signal, 
        seq_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        title: str = "Signal Evolution"
    ):
        """
        Plot how signals evolve across sequence positions.
        
        Args:
            multi_signal: MultiSignal object from encoder
            seq_indices: Which sequence positions to plot (default: first 5)
            save_path: Where to save plot
            title: Plot title
        """
        if seq_indices is None:
            seq_len = multi_signal.shape[1]
            seq_indices = list(range(min(5, seq_len)))
        
        signals = multi_signal.get_all_signals()
        
        fig, axes = plt.subplots(self.num_signals, 1, figsize=(12, 3 * self.num_signals))
        if self.num_signals == 1:
            axes = [axes]
        
        for idx, (signal_data, signal_name, ax) in enumerate(zip(signals, self.signal_names, axes)):
            # Get first sample from batch
            signal_np = signal_data[0].detach().cpu().numpy()  # [seq_len, dims]
            
            # Plot selected positions
            for seq_idx in seq_indices:
                ax.plot(signal_np[seq_idx], alpha=0.7, label=f'pos_{seq_idx}')
            
            ax.set_title(f'{signal_name.capitalize()} Signal Evolution')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_signal_heatmaps(
        self,
        multi_signal,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: str = "Signal Heatmaps"
    ):
        """
        Create heatmaps showing signal values across sequence and dimensions.
        
        Args:
            multi_signal: MultiSignal object
            tokens: Optional token strings for y-axis labels
            save_path: Where to save
            title: Plot title
        """
        signals = multi_signal.get_all_signals()
        
        fig, axes = plt.subplots(1, self.num_signals, figsize=(5 * self.num_signals, 8))
        if self.num_signals == 1:
            axes = [axes]
        
        for signal_data, signal_name, ax in zip(signals, self.signal_names, axes):
            # Get first sample
            signal_np = signal_data[0].detach().cpu().numpy()  # [seq_len, dims]
            
            im = ax.imshow(signal_np, aspect='auto', cmap='viridis')
            ax.set_title(f'{signal_name.capitalize()}')
            ax.set_xlabel('Dimension')
            ax.set_ylabel('Sequence Position')
            
            if tokens:
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens, fontsize=8)
            
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_signal_distribution(
        self,
        multi_signal,
        save_path: Optional[str] = None,
        title: str = "Signal Value Distributions"
    ):
        """
        Plot distribution of values for each signal type.
        
        Args:
            multi_signal: MultiSignal object
            save_path: Where to save
            title: Plot title
        """
        signals = multi_signal.get_all_signals()
        
        fig, axes = plt.subplots(self.num_signals, 1, figsize=(10, 3 * self.num_signals))
        if self.num_signals == 1:
            axes = [axes]
        
        for signal_data, signal_name, ax in zip(signals, self.signal_names, axes):
            signal_np = signal_data[0].detach().cpu().numpy().flatten()
            
            ax.hist(signal_np, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{signal_name.capitalize()} Distribution')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add stats
            mean_val = np.mean(signal_np)
            std_val = np.std(signal_np)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'μ={mean_val:.3f}, σ={std_val:.3f}')
            ax.legend()
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_comprehensive_dashboard(
        self,
        multi_signal,
        tokens: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        title: str = "Signal Analysis Dashboard"
    ):
        """
        Create comprehensive dashboard with multiple views.
        
        Args:
            multi_signal: MultiSignal object
            tokens: Optional token strings
            save_path: Where to save
            title: Dashboard title
        """
        signals = multi_signal.get_all_signals()
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(self.num_signals, 3, figure=fig)
        
        for idx, (signal_data, signal_name) in enumerate(zip(signals, self.signal_names)):
            signal_np = signal_data[0].detach().cpu().numpy()
            
            # Heatmap
            ax_heat = fig.add_subplot(gs[idx, 0])
            im = ax_heat.imshow(signal_np, aspect='auto', cmap='viridis')
            ax_heat.set_title(f'{signal_name.capitalize()} Heatmap')
            ax_heat.set_xlabel('Dimension')
            ax_heat.set_ylabel('Sequence')
            if tokens and idx == 0:
                ax_heat.set_yticks(range(min(len(tokens), signal_np.shape[0])))
                ax_heat.set_yticklabels(tokens[:signal_np.shape[0]], fontsize=6)
            plt.colorbar(im, ax=ax_heat)
            
            # Evolution plot
            ax_evol = fig.add_subplot(gs[idx, 1])
            for pos in range(min(5, signal_np.shape[0])):
                ax_evol.plot(signal_np[pos], alpha=0.6, label=f'pos_{pos}')
            ax_evol.set_title(f'{signal_name.capitalize()} Evolution')
            ax_evol.set_xlabel('Dimension')
            ax_evol.set_ylabel('Value')
            ax_evol.legend(fontsize=8)
            ax_evol.grid(True, alpha=0.3)
            
            # Distribution
            ax_dist = fig.add_subplot(gs[idx, 2])
            flat_signal = signal_np.flatten()
            ax_dist.hist(flat_signal, bins=40, alpha=0.7, edgecolor='black')
            ax_dist.set_title(f'{signal_name.capitalize()} Distribution')
            ax_dist.set_xlabel('Value')
            ax_dist.set_ylabel('Count')
            mean_val = np.mean(flat_signal)
            std_val = np.std(flat_signal)
            ax_dist.axvline(mean_val, color='red', linestyle='--',
                          label=f'μ={mean_val:.3f}\nσ={std_val:.3f}')
            ax_dist.legend(fontsize=8)
            ax_dist.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_evolution(
        self,
        signal_history: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Plot how signal statistics evolve during training.
        
        Args:
            signal_history: List of dicts with keys like 'frequency_mean', 'amplitude_std', etc.
            save_path: Where to save
        """
        if not signal_history:
            print("No signal history to plot")
            return
        
        fig, axes = plt.subplots(self.num_signals, 2, figsize=(14, 3 * self.num_signals))
        
        steps = list(range(len(signal_history)))
        
        for idx, signal_name in enumerate(self.signal_names):
            # Mean evolution
            ax_mean = axes[idx, 0] if self.num_signals > 1 else axes[0]
            means = [h.get(f'{signal_name}_mean', 0) for h in signal_history]
            ax_mean.plot(steps, means, linewidth=2)
            ax_mean.set_title(f'{signal_name.capitalize()} Mean Evolution')
            ax_mean.set_xlabel('Step')
            ax_mean.set_ylabel('Mean Value')
            ax_mean.grid(True, alpha=0.3)
            
            # Std evolution
            ax_std = axes[idx, 1] if self.num_signals > 1 else axes[1]
            stds = [h.get(f'{signal_name}_std', 0) for h in signal_history]
            ax_std.plot(steps, stds, linewidth=2, color='orange')
            ax_std.set_title(f'{signal_name.capitalize()} Std Evolution')
            ax_std.set_xlabel('Step')
            ax_std.set_ylabel('Std Value')
            ax_std.grid(True, alpha=0.3)
        
        plt.suptitle('Signal Statistics During Training', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def extract_signal_stats(multi_signal) -> Dict[str, float]:
    """
    Extract statistics from MultiSignal for tracking during training.
    
    Returns:
        Dict with keys like 'frequency_mean', 'amplitude_std', etc.
    """
    signals = multi_signal.get_all_signals()
    signal_names = ['frequency', 'amplitude', 'phase']
    
    stats = {}
    for signal_data, name in zip(signals, signal_names):
        signal_np = signal_data.detach().cpu().numpy()
        stats[f'{name}_mean'] = float(np.mean(signal_np))
        stats[f'{name}_std'] = float(np.std(signal_np))
        stats[f'{name}_min'] = float(np.min(signal_np))
        stats[f'{name}_max'] = float(np.max(signal_np))
    
    return stats
