"""
Wave trajectory tracking during generation.

This module tracks and analyzes the evolution of wave properties throughout
the autoregressive generation process, detecting patterns like mode collapse.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from ..core.wave_statistics import WaveStatistics


@dataclass
class TrajectoryStats:
    """Container for trajectory statistics at each generation step."""
    step: int
    energy: float
    spectral_centroid: float
    frequency_bandwidth: float
    amplitude_mean: float
    amplitude_std: float
    phase_coherence: float
    harmonic_entropy: float


class WaveTrajectoryTracker:
    """
    Track wave statistics evolution during autoregressive generation.

    Monitors wave properties at each generation step to understand how the
    model's internal representation evolves. Can detect mode collapse,
    repetitive patterns, and other generation anomalies.

    Example:
        >>> tracker = WaveTrajectoryTracker()
        >>> for step in range(max_length):
        ...     logits, wave = model(input_ids, return_encoder_outputs=True)
        ...     tracker.track_step(step, wave)
        >>> tracker.plot_trajectory()
        >>> collapse_detected = tracker.detect_mode_collapse()
    """

    def __init__(self, batch_idx: int = 0):
        """
        Initialize trajectory tracker.

        Args:
            batch_idx: Which batch element to track (for batched generation)
        """
        self.batch_idx = batch_idx
        self.trajectory: List[TrajectoryStats] = []
        self.wave_history: List = []  # Store Wave objects for post-hoc analysis

    def track_step(self, step: int, wave) -> TrajectoryStats:
        """
        Track wave statistics at a single generation step.

        Args:
            step: Current generation step number
            wave: Wave object from model encoder output

        Returns:
            TrajectoryStats for this step
        """
        # Compute statistics using WaveStatistics
        energy = WaveStatistics.compute_total_energy(
            wave, batch_idx=self.batch_idx, per_position=False
        ).item()

        centroid = WaveStatistics.compute_spectral_centroid(
            wave, batch_idx=self.batch_idx
        ).mean().item()

        bandwidth = WaveStatistics.compute_frequency_bandwidth(
            wave, percentile=90.0, batch_idx=self.batch_idx
        ).mean().item()

        coherence = WaveStatistics.compute_phase_coherence(
            wave, batch_idx=self.batch_idx
        ).mean().item()

        entropy = WaveStatistics.compute_harmonic_entropy(
            wave, batch_idx=self.batch_idx
        ).mean().item()

        # Basic amplitude statistics
        amps = wave.amplitudes[self.batch_idx]
        amp_mean = amps.mean().item()
        amp_std = amps.std().item()

        # Create stats object
        stats = TrajectoryStats(
            step=step,
            energy=energy,
            spectral_centroid=centroid,
            frequency_bandwidth=bandwidth,
            amplitude_mean=amp_mean,
            amplitude_std=amp_std,
            phase_coherence=coherence,
            harmonic_entropy=entropy
        )

        self.trajectory.append(stats)
        self.wave_history.append(wave)

        return stats

    @torch.no_grad()
    def track_generation(
        self,
        model: torch.nn.Module,
        initial_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, List[TrajectoryStats]]:
        """
        Track wave trajectory during full generation process.

        Args:
            model: Wave Transformer model
            initial_ids: Starting token IDs [1, initial_len]
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            device: Device for computation

        Returns:
            Tuple of (generated_ids, trajectory_stats_list)
        """
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        current_ids = initial_ids.to(device)
        self.trajectory.clear()
        self.wave_history.clear()

        for step in range(max_length):
            # Forward pass
            logits, wave = model(
                encoder_input={'token_ids': current_ids},
                return_encoder_outputs=True
            )

            # Track this step
            self.track_step(step, wave)

            # Sample next token
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            current_ids = torch.cat([current_ids, next_token], dim=1)

        return current_ids, self.trajectory

    def plot_trajectory(
        self,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize wave trajectory evolution across generation steps.

        Creates comprehensive plots showing how wave properties evolve.

        Args:
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Tuple of (figure, axes)
        """
        if len(self.trajectory) == 0:
            raise ValueError("No trajectory data available. Run track_generation first.")

        steps = [s.step for s in self.trajectory]
        energies = [s.energy for s in self.trajectory]
        centroids = [s.spectral_centroid for s in self.trajectory]
        bandwidths = [s.frequency_bandwidth for s in self.trajectory]
        amp_means = [s.amplitude_mean for s in self.trajectory]
        amp_stds = [s.amplitude_std for s in self.trajectory]
        coherences = [s.phase_coherence for s in self.trajectory]
        entropies = [s.harmonic_entropy for s in self.trajectory]

        fig, axes = plt.subplots(3, 3, figsize=figsize)

        # Row 1: Energy and amplitude
        axes[0, 0].plot(steps, energies, linewidth=2, color='purple')
        axes[0, 0].set_title('Total Wave Energy Evolution')
        axes[0, 0].set_xlabel('Generation Step')
        axes[0, 0].set_ylabel('Energy')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(steps, amp_means, linewidth=2, color='blue', label='Mean')
        axes[0, 1].fill_between(
            steps,
            np.array(amp_means) - np.array(amp_stds),
            np.array(amp_means) + np.array(amp_stds),
            alpha=0.3,
            color='blue',
            label='± Std'
        )
        axes[0, 1].set_title('Amplitude Statistics')
        axes[0, 1].set_xlabel('Generation Step')
        axes[0, 1].set_ylabel('Amplitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(steps, amp_stds, linewidth=2, color='cyan')
        axes[0, 2].set_title('Amplitude Variability')
        axes[0, 2].set_xlabel('Generation Step')
        axes[0, 2].set_ylabel('Std Dev')
        axes[0, 2].grid(True, alpha=0.3)

        # Row 2: Spectral properties
        axes[1, 0].plot(steps, centroids, linewidth=2, color='orange')
        axes[1, 0].set_title('Spectral Centroid Evolution')
        axes[1, 0].set_xlabel('Generation Step')
        axes[1, 0].set_ylabel('Centroid (Hz)')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(steps, bandwidths, linewidth=2, color='red')
        axes[1, 1].set_title('Frequency Bandwidth (90%)')
        axes[1, 1].set_xlabel('Generation Step')
        axes[1, 1].set_ylabel('Bandwidth (Hz)')
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(steps, entropies, linewidth=2, color='green')
        axes[1, 2].set_title('Harmonic Entropy')
        axes[1, 2].set_xlabel('Generation Step')
        axes[1, 2].set_ylabel('Entropy (nats)')
        axes[1, 2].grid(True, alpha=0.3)

        # Row 3: Phase and derived metrics
        axes[2, 0].plot(steps, coherences, linewidth=2, color='magenta')
        axes[2, 0].set_title('Phase Coherence')
        axes[2, 0].set_xlabel('Generation Step')
        axes[2, 0].set_ylabel('Coherence [0-1]')
        axes[2, 0].set_ylim([0, 1.05])
        axes[2, 0].grid(True, alpha=0.3)

        # Energy derivative (rate of change)
        energy_derivative = np.gradient(energies)
        axes[2, 1].plot(steps, energy_derivative, linewidth=2, color='brown')
        axes[2, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2, 1].set_title('Energy Rate of Change')
        axes[2, 1].set_xlabel('Generation Step')
        axes[2, 1].set_ylabel('dE/dt')
        axes[2, 1].grid(True, alpha=0.3)

        # Composite stability metric (lower is more stable)
        stability = np.array(amp_stds) / (np.array(amp_means) + 1e-8)
        axes[2, 2].plot(steps, stability, linewidth=2, color='purple')
        axes[2, 2].set_title('Stability Metric (CV)')
        axes[2, 2].set_xlabel('Generation Step')
        axes[2, 2].set_ylabel('Coefficient of Variation')
        axes[2, 2].grid(True, alpha=0.3)

        fig.suptitle('Wave Trajectory Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def detect_mode_collapse(
        self,
        window_size: int = 10,
        variance_threshold: float = 0.01
    ) -> Dict[str, bool]:
        """
        Detect mode collapse or repetitive patterns in generation.

        Mode collapse is indicated by:
        - Low variance in recent wave statistics
        - Stable/unchanging energy levels
        - Low harmonic entropy

        Args:
            window_size: Size of sliding window for variance computation
            variance_threshold: Threshold for considering variance "low"

        Returns:
            Dictionary with collapse detection results
        """
        if len(self.trajectory) < window_size:
            return {
                'mode_collapse_detected': False,
                'reason': 'Insufficient trajectory length'
            }

        # Extract recent statistics
        recent_trajectory = self.trajectory[-window_size:]

        recent_energies = [s.energy for s in recent_trajectory]
        recent_centroids = [s.spectral_centroid for s in recent_trajectory]
        recent_entropies = [s.harmonic_entropy for s in recent_trajectory]

        # Compute variances
        energy_var = np.var(recent_energies) / (np.mean(recent_energies) + 1e-8)
        centroid_var = np.var(recent_centroids) / (np.mean(recent_centroids) + 1e-8)
        entropy_mean = np.mean(recent_entropies)

        # Detection criteria
        low_energy_variance = energy_var < variance_threshold
        low_centroid_variance = centroid_var < variance_threshold
        low_entropy = entropy_mean < 1.0  # Threshold depends on num_harmonics

        # Overall detection
        collapse_detected = low_energy_variance and low_centroid_variance

        return {
            'mode_collapse_detected': collapse_detected,
            'low_energy_variance': low_energy_variance,
            'low_centroid_variance': low_centroid_variance,
            'low_entropy': low_entropy,
            'energy_variance': energy_var,
            'centroid_variance': centroid_var,
            'entropy_mean': entropy_mean,
            'window_size': window_size,
        }

    def get_trajectory_statistics(self) -> Dict:
        """
        Compute summary statistics over entire trajectory.

        Returns:
            Dictionary with aggregate statistics
        """
        if len(self.trajectory) == 0:
            return {}

        energies = np.array([s.energy for s in self.trajectory])
        centroids = np.array([s.spectral_centroid for s in self.trajectory])
        bandwidths = np.array([s.frequency_bandwidth for s in self.trajectory])
        coherences = np.array([s.phase_coherence for s in self.trajectory])
        entropies = np.array([s.harmonic_entropy for s in self.trajectory])

        return {
            'num_steps': len(self.trajectory),
            'energy': {
                'mean': float(energies.mean()),
                'std': float(energies.std()),
                'min': float(energies.min()),
                'max': float(energies.max()),
                'trend': float(np.polyfit(range(len(energies)), energies, 1)[0]),
            },
            'spectral_centroid': {
                'mean': float(centroids.mean()),
                'std': float(centroids.std()),
                'min': float(centroids.min()),
                'max': float(centroids.max()),
                'trend': float(np.polyfit(range(len(centroids)), centroids, 1)[0]),
            },
            'bandwidth': {
                'mean': float(bandwidths.mean()),
                'std': float(bandwidths.std()),
                'min': float(bandwidths.min()),
                'max': float(bandwidths.max()),
            },
            'phase_coherence': {
                'mean': float(coherences.mean()),
                'std': float(coherences.std()),
                'min': float(coherences.min()),
                'max': float(coherences.max()),
            },
            'harmonic_entropy': {
                'mean': float(entropies.mean()),
                'std': float(entropies.std()),
                'min': float(entropies.min()),
                'max': float(entropies.max()),
            },
        }

    def compare_segments(
        self,
        segment1_range: Tuple[int, int],
        segment2_range: Tuple[int, int]
    ) -> Dict:
        """
        Compare statistics between two segments of the trajectory.

        Useful for analyzing how generation behavior changes over time.

        Args:
            segment1_range: (start, end) indices for first segment
            segment2_range: (start, end) indices for second segment

        Returns:
            Dictionary comparing segments
        """
        s1_start, s1_end = segment1_range
        s2_start, s2_end = segment2_range

        seg1 = self.trajectory[s1_start:s1_end]
        seg2 = self.trajectory[s2_start:s2_end]

        if len(seg1) == 0 or len(seg2) == 0:
            raise ValueError("Segment ranges are empty")

        seg1_energies = np.array([s.energy for s in seg1])
        seg2_energies = np.array([s.energy for s in seg2])

        seg1_centroids = np.array([s.spectral_centroid for s in seg1])
        seg2_centroids = np.array([s.spectral_centroid for s in seg2])

        return {
            'segment1': {
                'range': segment1_range,
                'energy_mean': float(seg1_energies.mean()),
                'centroid_mean': float(seg1_centroids.mean()),
            },
            'segment2': {
                'range': segment2_range,
                'energy_mean': float(seg2_energies.mean()),
                'centroid_mean': float(seg2_centroids.mean()),
            },
            'differences': {
                'energy_change': float(seg2_energies.mean() - seg1_energies.mean()),
                'centroid_change': float(seg2_centroids.mean() - seg1_centroids.mean()),
                'energy_change_pct': float(
                    (seg2_energies.mean() - seg1_energies.mean()) / (seg1_energies.mean() + 1e-8) * 100
                ),
            }
        }

    def export_trajectory(self, save_path: str) -> None:
        """
        Export trajectory data to CSV file.

        Args:
            save_path: Path to save CSV file
        """
        import csv

        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'step', 'energy', 'spectral_centroid', 'frequency_bandwidth',
                'amplitude_mean', 'amplitude_std', 'phase_coherence', 'harmonic_entropy'
            ])

            # Data rows
            for stats in self.trajectory:
                writer.writerow([
                    stats.step,
                    stats.energy,
                    stats.spectral_centroid,
                    stats.frequency_bandwidth,
                    stats.amplitude_mean,
                    stats.amplitude_std,
                    stats.phase_coherence,
                    stats.harmonic_entropy
                ])

        print(f"Trajectory exported to {save_path}")
