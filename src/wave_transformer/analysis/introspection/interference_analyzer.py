"""
Wave Interference Analysis

Analyzes wave interference patterns between layers to understand
how waves interact and transform through the network.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from wave_transformer.core.wave import Wave


@dataclass
class InterferenceMetrics:
    """Container for interference analysis results"""
    mode: str
    phase_alignment: float  # Cosine similarity of phases
    frequency_coupling: float  # Correlation of frequencies
    amplitude_correlation: float  # Correlation of amplitudes
    energy_transfer: float  # Ratio of output to input energy
    spectral_overlap: float  # Frequency domain overlap


class WaveInterferenceAnalyzer:
    """
    Analyzes wave interference patterns between consecutive layers.

    Uses the Wave.interfere_with() method to study how waves interact
    and provides metrics for understanding wave transformations.

    Args:
        model: WaveTransformer model to analyze
        device: Device for computation

    Example:
        >>> analyzer = WaveInterferenceAnalyzer(model)
        >>> pattern = analyzer.compute_interference_pattern(wave1, wave2, mode='constructive')
        >>> layer_analysis = analyzer.analyze_layer_interference(snapshots)
        >>> analyzer.plot_interference_patterns(layer_analysis)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device

    @torch.no_grad()
    def compute_interference_pattern(
        self,
        wave1: Wave,
        wave2: Wave,
        mode: str = 'constructive',
        batch_idx: int = 0
    ) -> Tuple[Wave, InterferenceMetrics]:
        """
        Compute wave interference and analyze the resulting pattern.

        Args:
            wave1: First Wave object
            wave2: Second Wave object
            mode: Interference mode ('constructive', 'destructive', 'modulate')
            batch_idx: Which batch element to analyze

        Returns:
            Tuple of (interfered_wave, metrics)
        """
        # Compute interference using Wave's built-in method
        interfered = wave1.interfere_with(wave2, mode=mode)

        # Extract tensors for metric computation
        f1 = wave1.frequencies[batch_idx].detach().cpu().numpy()
        f2 = wave2.frequencies[batch_idx].detach().cpu().numpy()
        a1 = wave1.amplitudes[batch_idx].detach().cpu().numpy()
        a2 = wave2.amplitudes[batch_idx].detach().cpu().numpy()
        p1 = wave1.phases[batch_idx].detach().cpu().numpy()
        p2 = wave2.phases[batch_idx].detach().cpu().numpy()

        f_out = interfered.frequencies[batch_idx].detach().cpu().numpy()
        a_out = interfered.amplitudes[batch_idx].detach().cpu().numpy()
        p_out = interfered.phases[batch_idx].detach().cpu().numpy()

        # Compute metrics
        metrics = self._compute_metrics(
            f1, f2, f_out,
            a1, a2, a_out,
            p1, p2, p_out,
            mode
        )

        return interfered, metrics

    def _compute_metrics(
        self,
        f1: np.ndarray, f2: np.ndarray, f_out: np.ndarray,
        a1: np.ndarray, a2: np.ndarray, a_out: np.ndarray,
        p1: np.ndarray, p2: np.ndarray, p_out: np.ndarray,
        mode: str
    ) -> InterferenceMetrics:
        """
        Compute interference metrics from wave components.

        Args:
            f1, f2, f_out: Frequency arrays [S, H]
            a1, a2, a_out: Amplitude arrays [S, H]
            p1, p2, p_out: Phase arrays [S, H]
            mode: Interference mode

        Returns:
            InterferenceMetrics object
        """
        # Phase alignment: cosine similarity of phase vectors
        # cos(θ) = (p1 · p2) / (|p1| |p2|)
        phase_cos = np.cos(p1 - p2)  # Pairwise phase difference
        phase_alignment = float(phase_cos.mean())

        # Frequency coupling: correlation between input frequencies
        f1_flat = f1.flatten()
        f2_flat = f2.flatten()
        if f1_flat.std() > 1e-8 and f2_flat.std() > 1e-8:
            frequency_coupling = float(np.corrcoef(f1_flat, f2_flat)[0, 1])
        else:
            frequency_coupling = 0.0

        # Amplitude correlation
        a1_flat = a1.flatten()
        a2_flat = a2.flatten()
        if a1_flat.std() > 1e-8 and a2_flat.std() > 1e-8:
            amplitude_correlation = float(np.corrcoef(a1_flat, a2_flat)[0, 1])
        else:
            amplitude_correlation = 0.0

        # Energy transfer: ratio of output to input energy
        energy_in = (a1 ** 2).sum() + (a2 ** 2).sum()
        energy_out = (a_out ** 2).sum()
        energy_transfer = float(energy_out / (energy_in + 1e-8))

        # Spectral overlap: how much do frequency distributions overlap?
        # Use histogram overlap (Bhattacharyya coefficient)
        f_min = min(f1.min(), f2.min())
        f_max = max(f1.max(), f2.max())
        bins = np.linspace(f_min, f_max, 50)

        hist1, _ = np.histogram(f1.flatten(), bins=bins, weights=a1.flatten(), density=True)
        hist2, _ = np.histogram(f2.flatten(), bins=bins, weights=a2.flatten(), density=True)

        # Bhattacharyya coefficient
        spectral_overlap = float(np.sum(np.sqrt(hist1 * hist2)))

        return InterferenceMetrics(
            mode=mode,
            phase_alignment=phase_alignment,
            frequency_coupling=frequency_coupling,
            amplitude_correlation=amplitude_correlation,
            energy_transfer=energy_transfer,
            spectral_overlap=spectral_overlap
        )

    def analyze_layer_interference(
        self,
        snapshots: List,  # List[LayerWaveSnapshot] from LayerWaveAnalyzer
        modes: List[str] = ['constructive', 'destructive', 'modulate'],
        batch_idx: int = 0
    ) -> Dict[str, List[InterferenceMetrics]]:
        """
        Analyze interference between consecutive layers.

        Args:
            snapshots: List of LayerWaveSnapshot from LayerWaveAnalyzer.analyze_input()
            modes: Interference modes to test
            batch_idx: Which batch element to analyze

        Returns:
            Dictionary mapping mode -> list of InterferenceMetrics for each layer pair
        """
        results = {mode: [] for mode in modes}

        # Analyze interference between consecutive layers
        for i in range(len(snapshots) - 1):
            wave1 = snapshots[i].wave
            wave2 = snapshots[i + 1].wave

            for mode in modes:
                _, metrics = self.compute_interference_pattern(
                    wave1, wave2, mode=mode, batch_idx=batch_idx
                )
                results[mode].append(metrics)

        return results

    def plot_interference_patterns(
        self,
        interference_results: Dict[str, List[InterferenceMetrics]],
        layer_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize interference patterns across layers.

        Args:
            interference_results: Output from analyze_layer_interference()
            layer_names: Names of layer transitions (or None for auto-generate)
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Figure and axes
        """
        modes = list(interference_results.keys())
        num_transitions = len(interference_results[modes[0]])

        if layer_names is None:
            layer_names = [f'L{i}→L{i+1}' for i in range(num_transitions)]

        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # Metric names to plot
        metric_names = [
            'phase_alignment',
            'frequency_coupling',
            'amplitude_correlation',
            'energy_transfer',
            'spectral_overlap'
        ]

        # Plot each metric across modes
        for idx, metric_name in enumerate(metric_names):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            for mode in modes:
                metrics_list = interference_results[mode]
                values = [getattr(m, metric_name) for m in metrics_list]

                ax.plot(range(num_transitions), values, marker='o',
                       label=mode, alpha=0.7, linewidth=2)

            ax.set_xlabel('Layer Transition')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Across Layers')
            ax.set_xticks(range(num_transitions))
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Use the last subplot for a summary heatmap
        ax = axes[2, 1]

        # Create heatmap of all metrics for constructive mode
        mode_to_plot = 'constructive' if 'constructive' in modes else modes[0]
        metrics_list = interference_results[mode_to_plot]

        heatmap_data = np.array([
            [getattr(m, metric_name) for m in metrics_list]
            for metric_name in metric_names
        ])

        im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_yticks(range(len(metric_names)))
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metric_names])
        ax.set_xticks(range(num_transitions))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_title(f'Metrics Heatmap ({mode_to_plot})')
        plt.colorbar(im, ax=ax, label='Metric Value')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    @torch.no_grad()
    def visualize_interference_components(
        self,
        wave1: Wave,
        wave2: Wave,
        batch_idx: int = 0,
        seq_position: int = 0,
        modes: List[str] = ['constructive', 'destructive', 'modulate'],
        figsize: Tuple[int, int] = (18, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Detailed visualization of how two waves interfere for a specific position.

        Shows:
        - Original wave spectra
        - Interfered spectra for each mode
        - Phase relationships
        - Amplitude changes

        Args:
            wave1: First wave
            wave2: Second wave
            batch_idx: Batch element index
            seq_position: Sequence position to visualize
            modes: Interference modes to show
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Figure and axes
        """
        num_modes = len(modes)
        fig, axes = plt.subplots(3, num_modes + 1, figsize=figsize)

        # Extract data for specific position
        f1 = wave1.frequencies[batch_idx, seq_position].detach().cpu().numpy()
        a1 = wave1.amplitudes[batch_idx, seq_position].detach().cpu().numpy()
        p1 = wave1.phases[batch_idx, seq_position].detach().cpu().numpy()

        f2 = wave2.frequencies[batch_idx, seq_position].detach().cpu().numpy()
        a2 = wave2.amplitudes[batch_idx, seq_position].detach().cpu().numpy()
        p2 = wave2.phases[batch_idx, seq_position].detach().cpu().numpy()

        num_harmonics = len(f1)

        # Column 0: Original waves
        # Row 0: Frequency spectrum
        axes[0, 0].stem(f1, a1, linefmt='C0-', markerfmt='C0o', basefmt='C0-',
                       label='Wave 1')
        axes[0, 0].stem(f2, a2, linefmt='C1-', markerfmt='C1o', basefmt='C1-',
                       label='Wave 2', alpha=0.6)
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Original Spectra')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Row 1: Phase comparison
        axes[1, 0].scatter(range(num_harmonics), p1, c='C0', label='Wave 1', alpha=0.7)
        axes[1, 0].scatter(range(num_harmonics), p2, c='C1', label='Wave 2', alpha=0.7)
        axes[1, 0].set_ylabel('Phase (rad)')
        axes[1, 0].set_title('Original Phases')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Row 2: Amplitude comparison
        axes[2, 0].bar(range(num_harmonics), a1, alpha=0.6, label='Wave 1', color='C0')
        axes[2, 0].bar(range(num_harmonics), a2, alpha=0.6, label='Wave 2', color='C1')
        axes[2, 0].set_xlabel('Harmonic Index')
        axes[2, 0].set_ylabel('Amplitude')
        axes[2, 0].set_title('Amplitude Distribution')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Columns 1+: Each interference mode
        for col_idx, mode in enumerate(modes, start=1):
            interfered, metrics = self.compute_interference_pattern(
                wave1, wave2, mode=mode, batch_idx=batch_idx
            )

            f_out = interfered.frequencies[batch_idx, seq_position].detach().cpu().numpy()
            a_out = interfered.amplitudes[batch_idx, seq_position].detach().cpu().numpy()
            p_out = interfered.phases[batch_idx, seq_position].detach().cpu().numpy()

            # Row 0: Interfered spectrum
            axes[0, col_idx].stem(f_out, a_out, linefmt='C2-', markerfmt='C2o',
                                 basefmt='C2-')
            axes[0, col_idx].set_title(f'{mode.title()} Mode\nEnergy: {metrics.energy_transfer:.2f}')
            axes[0, col_idx].grid(True, alpha=0.3)

            # Row 1: Phase changes
            phase_diff = p_out - (p1 + p2) / 2  # Compare to average input phase
            axes[1, col_idx].scatter(range(num_harmonics), p_out, c='C2', alpha=0.7)
            axes[1, col_idx].set_title(f'Phase Alignment: {metrics.phase_alignment:.2f}')
            axes[1, col_idx].grid(True, alpha=0.3)

            # Row 2: Amplitude changes
            axes[2, col_idx].bar(range(num_harmonics), a_out, alpha=0.7, color='C2')
            axes[2, col_idx].set_xlabel('Harmonic Index')
            axes[2, col_idx].set_title(f'Spectral Overlap: {metrics.spectral_overlap:.2f}')
            axes[2, col_idx].grid(True, alpha=0.3)

        fig.suptitle(f'Wave Interference Analysis [Batch {batch_idx}, Pos {seq_position}]',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def compute_pairwise_interference_matrix(
        self,
        snapshots: List,  # List[LayerWaveSnapshot]
        mode: str = 'constructive',
        batch_idx: int = 0,
        metric: str = 'energy_transfer'
    ) -> np.ndarray:
        """
        Compute pairwise interference metrics between all layers.

        Creates a matrix where entry [i, j] is the interference metric
        when interfering layer i with layer j.

        Args:
            snapshots: Layer snapshots from LayerWaveAnalyzer
            mode: Interference mode
            batch_idx: Batch element to analyze
            metric: Which metric to extract ('energy_transfer', 'phase_alignment', etc.)

        Returns:
            Symmetric matrix [num_layers, num_layers] of interference metrics
        """
        num_layers = len(snapshots)
        matrix = np.zeros((num_layers, num_layers))

        for i in range(num_layers):
            for j in range(num_layers):
                if i == j:
                    matrix[i, j] = 1.0  # Self-interference is identity
                else:
                    _, metrics = self.compute_interference_pattern(
                        snapshots[i].wave,
                        snapshots[j].wave,
                        mode=mode,
                        batch_idx=batch_idx
                    )
                    matrix[i, j] = getattr(metrics, metric)

        return matrix

    def plot_interference_matrix(
        self,
        matrix: np.ndarray,
        layer_names: Optional[List[str]] = None,
        metric_name: str = 'Metric',
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize pairwise interference matrix as a heatmap.

        Args:
            matrix: Output from compute_pairwise_interference_matrix()
            layer_names: Layer labels (or None for auto-generate)
            metric_name: Name of metric being displayed
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Figure and axis
        """
        num_layers = matrix.shape[0]
        if layer_names is None:
            layer_names = [f'L{i}' for i in range(num_layers)]

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')

        ax.set_xticks(range(num_layers))
        ax.set_yticks(range(num_layers))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_yticklabels(layer_names)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Layer')
        ax.set_title(f'Pairwise Layer Interference: {metric_name}')

        plt.colorbar(im, ax=ax, label=metric_name)

        # Add text annotations
        for i in range(num_layers):
            for j in range(num_layers):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha='center', va='center', color='white' if matrix[i, j] < 0.5 else 'black',
                             fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax
