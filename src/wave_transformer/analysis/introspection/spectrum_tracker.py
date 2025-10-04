"""
Spectrum Evolution Tracking

Tracks how the frequency spectrum evolves through transformer layers,
providing insights into spectral transformations and energy redistribution.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from wave_transformer.core.wave import Wave


@dataclass
class SpectrumSnapshot:
    """Container for spectrum data at a specific layer"""
    layer_idx: int
    layer_name: str
    frequencies: np.ndarray  # [num_harmonics]
    amplitudes: np.ndarray   # [num_harmonics]
    spectral_centroid: float
    bandwidth: float
    peak_frequency: float
    total_energy: float


@dataclass
class SpectralShift:
    """Metrics describing spectral changes between layers"""
    centroid_shift: float  # Change in spectral centroid
    bandwidth_change: float  # Change in spectral bandwidth
    energy_redistribution: float  # How much energy moved between harmonics
    frequency_drift: float  # Average frequency change
    amplitude_ratio: float  # Ratio of output to input total amplitude


class SpectrumEvolutionTracker:
    """
    Tracks spectrum evolution through transformer layers.

    Provides 3D visualizations and metrics for understanding how
    the frequency spectrum changes as waves propagate through the network.

    Args:
        model: WaveTransformer model to analyze
        device: Device for computation

    Example:
        >>> tracker = SpectrumEvolutionTracker(model)
        >>> evolution = tracker.extract_spectrum_evolution(snapshots)
        >>> shifts = tracker.compute_spectral_shift(evolution)
        >>> tracker.plot_spectrum_evolution(evolution)
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device

    def extract_spectrum_evolution(
        self,
        snapshots: List,  # List[LayerWaveSnapshot] from LayerWaveAnalyzer
        batch_idx: int = 0,
        seq_position: int = 0
    ) -> List[SpectrumSnapshot]:
        """
        Extract frequency spectrum at each layer for a specific sequence position.

        Args:
            snapshots: Layer snapshots from LayerWaveAnalyzer.analyze_input()
            batch_idx: Which batch element to analyze
            seq_position: Which sequence position to analyze

        Returns:
            List of SpectrumSnapshot objects, one per layer
        """
        spectrum_snapshots = []

        for snapshot in snapshots:
            wave = snapshot.wave

            # Extract data for specific batch and position
            freqs = wave.frequencies[batch_idx, seq_position].detach().cpu().numpy()
            amps = wave.amplitudes[batch_idx, seq_position].detach().cpu().numpy()

            # Compute spectral metrics
            total_energy = (amps ** 2).sum()

            # Spectral centroid: amplitude-weighted mean frequency
            if amps.sum() > 1e-8:
                spectral_centroid = (freqs * amps).sum() / amps.sum()
            else:
                spectral_centroid = freqs.mean()

            # Bandwidth: amplitude-weighted standard deviation of frequency
            if amps.sum() > 1e-8:
                freq_variance = ((freqs - spectral_centroid) ** 2 * amps).sum() / amps.sum()
                bandwidth = np.sqrt(freq_variance)
            else:
                bandwidth = freqs.std()

            # Peak frequency: frequency with maximum amplitude
            peak_idx = amps.argmax()
            peak_frequency = freqs[peak_idx]

            spectrum_snapshots.append(SpectrumSnapshot(
                layer_idx=snapshot.layer_idx,
                layer_name=snapshot.layer_name,
                frequencies=freqs,
                amplitudes=amps,
                spectral_centroid=spectral_centroid,
                bandwidth=bandwidth,
                peak_frequency=peak_frequency,
                total_energy=total_energy
            ))

        return spectrum_snapshots

    def compute_spectral_shift(
        self,
        spectrum_evolution: List[SpectrumSnapshot]
    ) -> List[SpectralShift]:
        """
        Compute spectral shift metrics between consecutive layers.

        Args:
            spectrum_evolution: Output from extract_spectrum_evolution()

        Returns:
            List of SpectralShift objects for each layer transition
        """
        shifts = []

        for i in range(len(spectrum_evolution) - 1):
            spec1 = spectrum_evolution[i]
            spec2 = spectrum_evolution[i + 1]

            # Centroid shift
            centroid_shift = spec2.spectral_centroid - spec1.spectral_centroid

            # Bandwidth change
            bandwidth_change = spec2.bandwidth - spec1.bandwidth

            # Energy redistribution: measure how much energy moved between harmonics
            # Using Earth Mover's Distance approximation
            energy1 = spec1.amplitudes ** 2
            energy2 = spec2.amplitudes ** 2

            # Normalize to probability distributions
            energy1_norm = energy1 / (energy1.sum() + 1e-8)
            energy2_norm = energy2 / (energy2.sum() + 1e-8)

            # Simple L1 distance as redistribution metric
            energy_redistribution = np.abs(energy1_norm - energy2_norm).sum()

            # Frequency drift: average change in frequency values
            frequency_drift = np.abs(spec2.frequencies - spec1.frequencies).mean()

            # Amplitude ratio
            total_amp1 = spec1.amplitudes.sum()
            total_amp2 = spec2.amplitudes.sum()
            amplitude_ratio = total_amp2 / (total_amp1 + 1e-8)

            shifts.append(SpectralShift(
                centroid_shift=centroid_shift,
                bandwidth_change=bandwidth_change,
                energy_redistribution=energy_redistribution,
                frequency_drift=frequency_drift,
                amplitude_ratio=amplitude_ratio
            ))

        return shifts

    def plot_spectrum_evolution(
        self,
        spectrum_evolution: List[SpectrumSnapshot],
        mode: str = '3d',
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize spectrum evolution through layers.

        Args:
            spectrum_evolution: Output from extract_spectrum_evolution()
            mode: Visualization mode:
                - '3d': 3D surface plot (layer, harmonic, amplitude)
                - '2d_stacked': Stacked 2D spectra
                - 'waterfall': Waterfall plot
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Figure and axes
        """
        if mode == '3d':
            return self._plot_3d_spectrum(spectrum_evolution, figsize, save_path)
        elif mode == '2d_stacked':
            return self._plot_2d_stacked(spectrum_evolution, figsize, save_path)
        elif mode == 'waterfall':
            return self._plot_waterfall(spectrum_evolution, figsize, save_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _plot_3d_spectrum(
        self,
        spectrum_evolution: List[SpectrumSnapshot],
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create 3D surface plot of spectrum evolution"""
        num_layers = len(spectrum_evolution)
        num_harmonics = len(spectrum_evolution[0].frequencies)

        # Prepare data for 3D plot
        layers = np.arange(num_layers)
        harmonics = np.arange(num_harmonics)
        L, H = np.meshgrid(layers, harmonics)

        # Build amplitude surface
        amplitudes = np.array([spec.amplitudes for spec in spectrum_evolution]).T  # [H, L]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Surface plot
        surf = ax.plot_surface(L, H, amplitudes, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.8)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Harmonic Index')
        ax.set_zlabel('Amplitude')
        ax.set_title('Spectrum Evolution Through Layers (3D)', fontsize=14, fontweight='bold')

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Amplitude')

        # Add layer names as x-tick labels
        ax.set_xticks(layers)
        ax.set_xticklabels([spec.layer_name[:10] for spec in spectrum_evolution], rotation=45, ha='right')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def _plot_2d_stacked(
        self,
        spectrum_evolution: List[SpectrumSnapshot],
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Create stacked 2D spectrum plots"""
        num_layers = len(spectrum_evolution)

        fig, axes = plt.subplots(num_layers, 1, figsize=figsize, sharex=True)
        if num_layers == 1:
            axes = [axes]

        for idx, spec in enumerate(spectrum_evolution):
            ax = axes[idx]

            # Sort by frequency for cleaner visualization
            sorted_indices = np.argsort(spec.frequencies)
            freqs_sorted = spec.frequencies[sorted_indices]
            amps_sorted = spec.amplitudes[sorted_indices]

            # Spectrum plot
            ax.fill_between(freqs_sorted, 0, amps_sorted, alpha=0.6, color=f'C{idx % 10}')
            ax.plot(freqs_sorted, amps_sorted, linewidth=2, color=f'C{idx % 10}')

            # Mark spectral centroid
            ax.axvline(spec.spectral_centroid, color='red', linestyle='--',
                      alpha=0.7, linewidth=1.5, label='Centroid')

            ax.set_ylabel('Amplitude')
            ax.set_title(f'{spec.layer_name} (E={spec.total_energy:.2f})', fontsize=10)
            ax.grid(True, alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=8)

        axes[-1].set_xlabel('Frequency (Hz)')

        fig.suptitle('Spectrum Evolution (Stacked)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def _plot_waterfall(
        self,
        spectrum_evolution: List[SpectrumSnapshot],
        figsize: Tuple[int, int],
        save_path: Optional[str]
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create waterfall plot"""
        fig, ax = plt.subplots(figsize=figsize)

        num_layers = len(spectrum_evolution)
        offset_step = 0.5  # Vertical offset between layers

        for idx, spec in enumerate(spectrum_evolution):
            # Sort by frequency
            sorted_indices = np.argsort(spec.frequencies)
            freqs_sorted = spec.frequencies[sorted_indices]
            amps_sorted = spec.amplitudes[sorted_indices]

            # Add vertical offset
            offset = idx * offset_step
            amps_offset = amps_sorted + offset

            # Plot with gradient color
            color = cm.viridis(idx / num_layers)
            ax.fill_between(freqs_sorted, offset, amps_offset, alpha=0.7, color=color)
            ax.plot(freqs_sorted, amps_offset, linewidth=1.5, color=color,
                   label=f'{spec.layer_name}')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (offset)')
        ax.set_title('Spectrum Evolution (Waterfall)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_spectral_metrics(
        self,
        spectrum_evolution: List[SpectrumSnapshot],
        shifts: Optional[List[SpectralShift]] = None,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot spectral metrics evolution.

        Args:
            spectrum_evolution: Spectrum snapshots
            shifts: Optional spectral shifts from compute_spectral_shift()
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Figure and axes
        """
        layer_names = [spec.layer_name for spec in spectrum_evolution]
        layer_indices = range(len(spectrum_evolution))

        if shifts is None and len(spectrum_evolution) > 1:
            shifts = self.compute_spectral_shift(spectrum_evolution)

        # Create subplots
        num_rows = 3 if shifts else 2
        fig, axes = plt.subplots(num_rows, 2, figsize=figsize)

        # Row 0: Spectral centroid and bandwidth
        ax = axes[0, 0]
        centroids = [spec.spectral_centroid for spec in spectrum_evolution]
        ax.plot(layer_indices, centroids, marker='o', linewidth=2, color='C0')
        ax.set_ylabel('Spectral Centroid (Hz)')
        ax.set_title('Spectral Centroid Evolution')
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        bandwidths = [spec.bandwidth for spec in spectrum_evolution]
        ax.plot(layer_indices, bandwidths, marker='o', linewidth=2, color='C1')
        ax.set_ylabel('Bandwidth (Hz)')
        ax.set_title('Spectral Bandwidth Evolution')
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Row 1: Peak frequency and total energy
        ax = axes[1, 0]
        peaks = [spec.peak_frequency for spec in spectrum_evolution]
        ax.plot(layer_indices, peaks, marker='o', linewidth=2, color='C2')
        ax.set_ylabel('Peak Frequency (Hz)')
        ax.set_title('Peak Frequency Evolution')
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        energies = [spec.total_energy for spec in spectrum_evolution]
        ax.plot(layer_indices, energies, marker='o', linewidth=2, color='C3')
        ax.set_ylabel('Total Energy')
        ax.set_title('Energy Evolution')
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Row 2: Shift metrics (if available)
        if shifts:
            transition_indices = range(len(shifts))
            transition_labels = [f'{layer_names[i]}→{layer_names[i+1]}'
                               for i in range(len(shifts))]

            ax = axes[2, 0]
            centroid_shifts = [shift.centroid_shift for shift in shifts]
            bandwidth_changes = [shift.bandwidth_change for shift in shifts]
            ax.plot(transition_indices, centroid_shifts, marker='o',
                   linewidth=2, label='Centroid Shift', color='C4')
            ax.plot(transition_indices, bandwidth_changes, marker='s',
                   linewidth=2, label='Bandwidth Change', color='C5')
            ax.set_ylabel('Change (Hz)')
            ax.set_title('Spectral Shifts Between Layers')
            ax.set_xticks(transition_indices)
            ax.set_xticklabels(transition_labels, rotation=45, ha='right', fontsize=8)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

            ax = axes[2, 1]
            energy_redist = [shift.energy_redistribution for shift in shifts]
            amp_ratios = [shift.amplitude_ratio for shift in shifts]
            ax.plot(transition_indices, energy_redist, marker='o',
                   linewidth=2, label='Energy Redistribution', color='C6')
            ax_twin = ax.twinx()
            ax_twin.plot(transition_indices, amp_ratios, marker='s',
                        linewidth=2, label='Amplitude Ratio', color='C7')
            ax.set_ylabel('Energy Redistribution')
            ax_twin.set_ylabel('Amplitude Ratio')
            ax.set_xlabel('Layer Transition')
            ax.set_title('Energy and Amplitude Changes')
            ax.set_xticks(transition_indices)
            ax.set_xticklabels(transition_labels, rotation=45, ha='right', fontsize=8)
            ax.legend(loc='upper left')
            ax_twin.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        fig.suptitle('Spectral Metrics Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def analyze_frequency_distribution_shift(
        self,
        spectrum_evolution: List[SpectrumSnapshot],
        num_bins: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how the frequency distribution changes through layers.

        Args:
            spectrum_evolution: Spectrum snapshots
            num_bins: Number of frequency bins for histogram

        Returns:
            Dictionary with:
                - 'bin_edges': Frequency bin edges
                - 'distributions': Array [num_layers, num_bins] of amplitude-weighted histograms
        """
        # Find global frequency range
        all_freqs = np.concatenate([spec.frequencies for spec in spectrum_evolution])
        freq_min, freq_max = all_freqs.min(), all_freqs.max()

        bin_edges = np.linspace(freq_min, freq_max, num_bins + 1)
        distributions = []

        for spec in spectrum_evolution:
            # Amplitude-weighted histogram
            hist, _ = np.histogram(spec.frequencies, bins=bin_edges,
                                  weights=spec.amplitudes, density=True)
            distributions.append(hist)

        return {
            'bin_edges': bin_edges,
            'distributions': np.array(distributions),  # [num_layers, num_bins]
        }

    def plot_frequency_distribution_evolution(
        self,
        distribution_analysis: Dict[str, np.ndarray],
        layer_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize frequency distribution evolution as a heatmap.

        Args:
            distribution_analysis: Output from analyze_frequency_distribution_shift()
            layer_names: Layer labels
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Figure and axis
        """
        bin_edges = distribution_analysis['bin_edges']
        distributions = distribution_analysis['distributions']  # [num_layers, num_bins]

        num_layers = distributions.shape[0]
        if layer_names is None:
            layer_names = [f'Layer {i}' for i in range(num_layers)]

        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        im = ax.imshow(distributions, aspect='auto', cmap='hot', interpolation='nearest',
                      origin='lower')

        ax.set_xlabel('Frequency Bin')
        ax.set_ylabel('Layer')
        ax.set_title('Frequency Distribution Evolution', fontsize=14, fontweight='bold')

        ax.set_yticks(range(num_layers))
        ax.set_yticklabels(layer_names)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Amplitude Density')

        # Add frequency bin labels
        num_ticks = min(10, len(bin_edges))
        tick_indices = np.linspace(0, len(bin_edges) - 1, num_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{bin_edges[i]:.1f}' for i in tick_indices], rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax
