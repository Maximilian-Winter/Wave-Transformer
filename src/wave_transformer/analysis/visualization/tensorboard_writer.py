"""
TensorBoard Integration for Wave Transformer Analysis

Provides utilities for logging wave statistics, visualizations, and metrics
to TensorBoard for real-time monitoring and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Union, Any
from pathlib import Path
import warnings

# Optional tensorboard import with graceful fallback
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from wave_transformer.core.wave import Wave


class WaveTensorBoardWriter:
    """
    TensorBoard writer with specialized support for Wave representations.

    Provides methods to log:
    - Wave statistics as scalars
    - Wave component heatmaps as images
    - Spectral visualizations as matplotlib figures
    - Layer-wise analysis and comparisons
    - Gradient flow visualizations

    Args:
        log_dir: Directory for TensorBoard logs
        comment: Optional comment to append to run name
        flush_secs: How often to flush pending events (default: 120)
        max_queue: Maximum number of events to queue (default: 10)

    Example:
        >>> writer = WaveTensorBoardWriter(log_dir='runs/wave_analysis')
        >>> writer.add_wave_statistics(wave, tag='encoder/wave', step=100)
        >>> writer.add_wave_heatmaps(wave, tag='layer_0/wave', step=100)
        >>> writer.close()
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        comment: str = '',
        flush_secs: int = 120,
        max_queue: int = 10
    ):
        if not TENSORBOARD_AVAILABLE:
            warnings.warn(
                "tensorboard is not installed. Install with 'pip install tensorboard'. "
                "WaveTensorBoardWriter will not log anything.",
                ImportWarning
            )
            self.enabled = False
            self.writer = None
            return

        self.enabled = True
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            flush_secs=flush_secs,
            max_queue=max_queue
        )

    def add_wave_statistics(
        self,
        wave: Wave,
        tag: str,
        step: int,
        batch_idx: Optional[int] = None,
        component: str = 'all'
    ):
        """
        Log wave statistics as scalar metrics.

        Computes and logs mean, std, min, max, median for wave components.

        Args:
            wave: Wave object to analyze [B, S, H]
            tag: Base tag for TensorBoard (e.g., 'encoder/wave')
            step: Global step/iteration number
            batch_idx: If specified, analyze only this batch element
            component: Which components to log ('all', 'frequencies', 'amplitudes', 'phases')
        """
        if not self.enabled:
            return

        from wave_transformer.analysis.core.wave_statistics import WaveStatistics

        # Compute basic statistics
        stats = WaveStatistics.compute_basic_stats(wave, component=component, dim=None)

        # Log each component's statistics
        for comp_name, wave_stats in stats.items():
            if hasattr(wave_stats, 'to_dict'):
                # WaveStats object
                stats_dict = wave_stats.to_dict()
                for stat_name, value in stats_dict.items():
                    self.writer.add_scalar(
                        f'{tag}/{comp_name}/{stat_name}',
                        value,
                        step
                    )

        # Compute additional spectral metrics
        spectral_centroid = WaveStatistics.compute_spectral_centroid(
            wave, batch_idx=batch_idx
        )
        self.writer.add_scalar(
            f'{tag}/spectral_centroid_mean',
            spectral_centroid.mean().item(),
            step
        )

        total_energy = WaveStatistics.compute_total_energy(
            wave, batch_idx=batch_idx, per_position=False
        )
        self.writer.add_scalar(
            f'{tag}/total_energy',
            total_energy.item() if torch.is_tensor(total_energy) else total_energy,
            step
        )

        # Harmonic entropy
        entropy = WaveStatistics.compute_harmonic_entropy(wave, batch_idx=batch_idx)
        self.writer.add_scalar(
            f'{tag}/harmonic_entropy_mean',
            entropy.mean().item(),
            step
        )

    def add_wave_heatmaps(
        self,
        wave: Wave,
        tag: str,
        step: int,
        batch_idx: int = 0,
        max_seq_len: Optional[int] = None
    ):
        """
        Log wave components as heatmap images.

        Creates heatmaps for frequencies, amplitudes, and phases.

        Args:
            wave: Wave object [B, S, H]
            tag: Base tag for TensorBoard
            step: Global step number
            batch_idx: Which batch element to visualize
            max_seq_len: Maximum sequence length to visualize (for truncation)
        """
        if not self.enabled:
            return
        # Extract single batch element
        freqs = wave.frequencies[batch_idx].detach().cpu().numpy()  # [S, H]
        amps = wave.amplitudes[batch_idx].detach().cpu().numpy()
        phases = wave.phases[batch_idx].detach().cpu().numpy()

        if max_seq_len is not None:
            freqs = freqs[:max_seq_len]
            amps = amps[:max_seq_len]
            phases = phases[:max_seq_len]

        # Normalize for visualization
        def normalize_for_display(arr):
            """Normalize to [0, 1] for heatmap display"""
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                return (arr - arr_min) / (arr_max - arr_min)
            return arr

        # Add heatmaps as images (TensorBoard expects [C, H, W])
        # We'll use colormaps to create RGB images
        import matplotlib.cm as cm

        # Frequencies heatmap
        freqs_norm = normalize_for_display(freqs)
        freqs_colored = cm.viridis(freqs_norm)[:, :, :3]  # [S, H, 3]
        freqs_colored = np.transpose(freqs_colored, (2, 0, 1))  # [3, S, H]
        self.writer.add_image(f'{tag}/frequencies_heatmap', freqs_colored, step)

        # Amplitudes heatmap
        amps_norm = normalize_for_display(amps)
        amps_colored = cm.hot(amps_norm)[:, :, :3]
        amps_colored = np.transpose(amps_colored, (2, 0, 1))
        self.writer.add_image(f'{tag}/amplitudes_heatmap', amps_colored, step)

        # Phases heatmap (phases are in [-π, π])
        phases_norm = (phases + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        phases_colored = cm.twilight(phases_norm)[:, :, :3]
        phases_colored = np.transpose(phases_colored, (2, 0, 1))
        self.writer.add_image(f'{tag}/phases_heatmap', phases_colored, step)

    def add_wave_spectrum(
        self,
        wave: Wave,
        tag: str,
        step: int,
        batch_idx: int = 0,
        seq_position: int = 0,
        figsize: tuple = (10, 6)
    ):
        """
        Log spectrum visualization as matplotlib figure.

        Creates a multi-panel figure showing frequency spectrum, amplitude distribution,
        and phase information.

        Args:
            wave: Wave object [B, S, H]
            tag: Tag for TensorBoard
            step: Global step number
            batch_idx: Which batch element
            seq_position: Which sequence position to visualize
            figsize: Figure size
        """
        if not self.enabled:
            return
        # Extract data for specific position
        freqs = wave.frequencies[batch_idx, seq_position].detach().cpu().numpy()
        amps = wave.amplitudes[batch_idx, seq_position].detach().cpu().numpy()
        phases = wave.phases[batch_idx, seq_position].detach().cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: Amplitude spectrum
        ax = axes[0]
        sorted_indices = np.argsort(freqs)
        ax.stem(freqs[sorted_indices], amps[sorted_indices], basefmt=' ')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Amplitude Spectrum')
        ax.grid(True, alpha=0.3)

        # Panel 2: Amplitude distribution (histogram)
        ax = axes[1]
        ax.hist(amps, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Count')
        ax.set_title('Amplitude Distribution')
        ax.grid(True, alpha=0.3)

        # Panel 3: Phase distribution
        ax = axes[2]
        ax.scatter(freqs, phases, c=amps, cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase (rad)')
        ax.set_title('Phase vs Frequency')
        ax.set_ylim(-np.pi, np.pi)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Amplitude')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Add to TensorBoard
        self.writer.add_figure(f'{tag}/spectrum', fig, step)
        plt.close(fig)

    def add_layer_comparison(
        self,
        layer_snapshots: List[Dict[str, Any]],
        tag: str,
        step: int,
        metrics: List[str] = ['spectral_centroid', 'total_energy', 'bandwidth'],
        figsize: tuple = (12, 8)
    ):
        """
        Log layer-wise comparison of wave metrics.

        Args:
            layer_snapshots: List of dicts with 'layer_name', 'wave', and optional metrics
            tag: Tag for TensorBoard
            step: Global step number
            metrics: Which metrics to plot
            figsize: Figure size
        """
        if not self.enabled:
            return
        from wave_transformer.analysis.core.wave_statistics import WaveStatistics

        layer_names = [snap['layer_name'] for snap in layer_snapshots]
        num_layers = len(layer_names)

        # Compute metrics for each layer
        metric_values = {metric: [] for metric in metrics}

        for snap in layer_snapshots:
            wave = snap['wave']

            if 'spectral_centroid' in metrics:
                centroid = WaveStatistics.compute_spectral_centroid(wave)
                metric_values['spectral_centroid'].append(centroid.mean().item())

            if 'total_energy' in metrics:
                energy = WaveStatistics.compute_total_energy(wave, per_position=False)
                metric_values['total_energy'].append(
                    energy.item() if torch.is_tensor(energy) else energy
                )

            if 'bandwidth' in metrics:
                bandwidth = WaveStatistics.compute_frequency_bandwidth(wave)
                metric_values['bandwidth'].append(bandwidth.mean().item())

            if 'harmonic_entropy' in metrics:
                entropy = WaveStatistics.compute_harmonic_entropy(wave)
                metric_values['harmonic_entropy'].append(entropy.mean().item())

        # Create comparison plot
        num_metrics = len(metrics)
        num_cols = min(2, num_metrics)
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = metric_values[metric]

            ax.plot(range(num_layers), values, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Layer Index')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
            ax.set_xticks(range(num_layers))
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        self.writer.add_figure(f'{tag}/layer_comparison', fig, step)
        plt.close(fig)

    def add_gradient_flow(
        self,
        named_parameters: List[tuple],
        tag: str,
        step: int,
        figsize: tuple = (14, 6)
    ):
        """
        Log gradient flow visualization.

        Shows gradient magnitudes across layers to diagnose vanishing/exploding gradients.

        Args:
            named_parameters: List of (name, parameter) tuples from model.named_parameters()
            tag: Tag for TensorBoard
            step: Global step number
            figsize: Figure size
        """
        if not self.enabled:
            return
        # Collect gradient statistics
        ave_grads = []
        max_grads = []
        layers = []

        for name, param in named_parameters:
            if param.grad is not None and param.requires_grad:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())

        if not layers:
            return  # No gradients to plot

        # Create gradient flow plot
        fig, ax = plt.subplots(figsize=figsize)

        x_pos = np.arange(len(layers))
        ax.bar(x_pos - 0.2, ave_grads, 0.4, alpha=0.7, label='Mean', color='blue')
        ax.bar(x_pos + 0.2, max_grads, 0.4, alpha=0.7, label='Max', color='red')

        ax.set_xlabel('Layers')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title('Gradient Flow Through Network')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(layers, rotation=90, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')

        plt.tight_layout()
        self.writer.add_figure(f'{tag}/gradient_flow', fig, step)
        plt.close(fig)

    def add_harmonic_importance(
        self,
        importance_scores: np.ndarray,
        tag: str,
        step: int,
        top_k: int = 16,
        figsize: tuple = (10, 6)
    ):
        """
        Log harmonic importance visualization.

        Args:
            importance_scores: Array of importance scores [num_harmonics]
            tag: Tag for TensorBoard
            step: Global step number
            top_k: Number of top harmonics to highlight
            figsize: Figure size
        """
        if not self.enabled:
            return
        num_harmonics = len(importance_scores)

        # Normalize scores
        scores_norm = importance_scores / (importance_scores.max() + 1e-8)

        # Get top-k indices
        top_indices = np.argsort(importance_scores)[::-1][:top_k]

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: Bar chart
        ax = axes[0]
        colors = ['red' if i in top_indices else 'steelblue'
                  for i in range(num_harmonics)]
        ax.bar(range(num_harmonics), scores_norm, color=colors, alpha=0.7)
        ax.set_xlabel('Harmonic Index')
        ax.set_ylabel('Normalized Importance')
        ax.set_title(f'Harmonic Importance (Top-{top_k} in red)')
        ax.grid(True, alpha=0.3)

        # Panel 2: Cumulative importance
        ax = axes[1]
        sorted_scores = np.sort(importance_scores)[::-1]
        cumulative = np.cumsum(sorted_scores) / sorted_scores.sum()
        ax.plot(range(num_harmonics), cumulative, linewidth=2, color='green')
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%')
        ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
        ax.set_xlabel('Number of Harmonics')
        ax.set_ylabel('Cumulative Importance')
        ax.set_title('Cumulative Harmonic Importance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.writer.add_figure(f'{tag}/harmonic_importance', fig, step)
        plt.close(fig)

    def add_scalar(self, tag: str, value: float, step: int):
        """Add a scalar value (passthrough to underlying writer)."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Add multiple scalar values (passthrough to underlying writer)."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def flush(self):
        """Flush pending events to disk."""
        if self.enabled:
            self.writer.flush()

    def close(self):
        """Close the TensorBoard writer and flush all pending events."""
        if self.enabled:
            self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
