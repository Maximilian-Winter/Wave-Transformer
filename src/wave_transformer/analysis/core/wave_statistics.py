"""
Core statistical analysis functions for Wave representations.

This module provides comprehensive statistical analysis tools for Wave objects,
including basic statistics, harmonic importance ranking, phase coherence,
spectral features, and energy analysis.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class WaveStats:
    """Container for wave statistics."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    variance: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'median': self.median,
            'variance': self.variance
        }


@dataclass
class HarmonicImportance:
    """Container for harmonic importance metrics."""
    indices: np.ndarray  # Harmonic indices sorted by importance
    scores: np.ndarray   # Importance scores
    metric: str          # Which metric was used ('amplitude', 'energy', 'variance')

    def top_k(self, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k most important harmonics.

        Args:
            k: Number of top harmonics to return

        Returns:
            Tuple of (indices, scores) for top k harmonics
        """
        k = min(k, len(self.indices))
        return self.indices[:k], self.scores[:k]


class WaveStatistics:
    """
    Static methods for computing comprehensive statistics on Wave representations.

    All methods are designed to be memory-efficient and work with batched data.
    Computations are performed with gradients disabled for efficiency.
    """

    @staticmethod
    @torch.no_grad()
    def compute_basic_stats(
        wave: 'Wave',
        component: str = 'all',
        dim: Optional[int] = None
    ) -> Dict[str, WaveStats]:
        """
        Compute basic statistics (mean, std, min, max, median) for wave components.

        Args:
            wave: Wave object with frequencies, amplitudes, phases [B, S, H]
            component: Which component to analyze ('frequencies', 'amplitudes',
                      'phases', or 'all')
            dim: Dimension along which to compute stats. If None, compute over all dims.
                 0=batch, 1=sequence, 2=harmonics

        Returns:
            Dictionary mapping component names to WaveStats objects
        """
        components_to_analyze = []
        if component == 'all':
            components_to_analyze = ['frequencies', 'amplitudes', 'phases']
        else:
            components_to_analyze = [component]

        results = {}

        for comp_name in components_to_analyze:
            tensor = getattr(wave, comp_name)

            if dim is not None:
                # Compute along specific dimension
                mean = tensor.mean(dim=dim)
                std = tensor.std(dim=dim)
                min_val = tensor.min(dim=dim)[0]
                max_val = tensor.max(dim=dim)[0]
                median = tensor.median(dim=dim)[0]
                variance = tensor.var(dim=dim)

                # Convert to scalars if result is 0-dim
                if mean.ndim == 0:
                    mean = mean.item()
                    std = std.item()
                    min_val = min_val.item()
                    max_val = max_val.item()
                    median = median.item()
                    variance = variance.item()

                    results[comp_name] = WaveStats(
                        mean=mean,
                        std=std,
                        min=min_val,
                        max=max_val,
                        median=median,
                        variance=variance
                    )
                else:
                    # Return tensors for multi-dimensional results
                    results[f'{comp_name}_mean'] = mean
                    results[f'{comp_name}_std'] = std
                    results[f'{comp_name}_min'] = min_val
                    results[f'{comp_name}_max'] = max_val
                    results[f'{comp_name}_median'] = median
                    results[f'{comp_name}_variance'] = variance
            else:
                # Compute over all dimensions
                results[comp_name] = WaveStats(
                    mean=tensor.mean().item(),
                    std=tensor.std().item(),
                    min=tensor.min().item(),
                    max=tensor.max().item(),
                    median=tensor.median().item(),
                    variance=tensor.var().item()
                )

        return results

    @staticmethod
    @torch.no_grad()
    def compute_harmonic_importance(
        wave: 'Wave',
        metric: str = 'amplitude',
        batch_idx: Optional[int] = None
    ) -> HarmonicImportance:
        """
        Rank harmonics by importance based on specified metric.

        Args:
            wave: Wave object [B, S, H]
            metric: Importance metric - 'amplitude', 'energy', or 'variance'
            batch_idx: If specified, analyze only this batch element.
                      If None, average across batch.

        Returns:
            HarmonicImportance object with sorted indices and scores
        """
        if batch_idx is not None:
            # Analyze single batch element
            amps = wave.amplitudes[batch_idx]  # [S, H]
            freqs = wave.frequencies[batch_idx]
            phases = wave.phases[batch_idx]
        else:
            # Average across batch
            amps = wave.amplitudes.mean(dim=0)  # [S, H]
            freqs = wave.frequencies.mean(dim=0)
            phases = wave.phases.mean(dim=0)

        # Compute importance scores
        if metric == 'amplitude':
            # Mean amplitude across sequence
            scores = amps.mean(dim=0)  # [H]
        elif metric == 'energy':
            # Mean energy (squared amplitude) across sequence
            scores = (amps ** 2).mean(dim=0)  # [H]
        elif metric == 'variance':
            # Variance of amplitude across sequence
            scores = amps.var(dim=0)  # [H]
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'amplitude', 'energy', or 'variance'")

        # Sort in descending order
        scores_np = scores.cpu().numpy()
        sorted_indices = np.argsort(scores_np)[::-1]  # Descending order
        sorted_scores = scores_np[sorted_indices]

        return HarmonicImportance(
            indices=sorted_indices,
            scores=sorted_scores,
            metric=metric
        )

    @staticmethod
    @torch.no_grad()
    def compute_phase_coherence(
        wave: 'Wave',
        batch_idx: Optional[int] = None,
        window_size: int = 1
    ) -> torch.Tensor:
        """
        Measure phase coherence across sequence positions.

        Phase coherence indicates how consistent phase relationships are.
        Computed as the magnitude of the mean complex exponential of phases.

        Args:
            wave: Wave object [B, S, H]
            batch_idx: If specified, analyze only this batch element
            window_size: Size of local window for coherence computation.
                        1 = global coherence, >1 = local coherence

        Returns:
            Coherence values [H] if batch_idx specified, else [B, H]
            Values range from 0 (random phases) to 1 (perfectly coherent)
        """
        if batch_idx is not None:
            phases = wave.phases[batch_idx]  # [S, H]
        else:
            phases = wave.phases  # [B, S, H]

        if window_size == 1:
            # Global coherence: mean across sequence
            # Coherence = |mean(exp(i*phase))|
            complex_phases = torch.exp(1j * phases)
            coherence = torch.abs(complex_phases.mean(dim=-2 if batch_idx is not None else -2))
        else:
            # Local coherence with sliding window
            if batch_idx is not None:
                S, H = phases.shape
                coherences = []
                for i in range(0, S - window_size + 1):
                    window_phases = phases[i:i+window_size]  # [window_size, H]
                    complex_phases = torch.exp(1j * window_phases)
                    coh = torch.abs(complex_phases.mean(dim=0))  # [H]
                    coherences.append(coh)
                coherence = torch.stack(coherences).mean(dim=0)  # [H]
            else:
                B, S, H = phases.shape
                coherences = []
                for i in range(0, S - window_size + 1):
                    window_phases = phases[:, i:i+window_size]  # [B, window_size, H]
                    complex_phases = torch.exp(1j * window_phases)
                    coh = torch.abs(complex_phases.mean(dim=1))  # [B, H]
                    coherences.append(coh)
                coherence = torch.stack(coherences, dim=1).mean(dim=1)  # [B, H]

        return coherence.real

    @staticmethod
    @torch.no_grad()
    def compute_spectral_centroid(
        wave: 'Wave',
        batch_idx: Optional[int] = None,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute frequency-weighted centroid of the spectrum.

        The spectral centroid indicates the "center of mass" of the spectrum,
        weighted by amplitude. Higher values indicate higher-frequency content.

        Args:
            wave: Wave object [B, S, H]
            batch_idx: If specified, analyze only this batch element
            eps: Small constant for numerical stability

        Returns:
            Spectral centroid per sequence position [S] if batch_idx specified,
            else [B, S]
        """
        if batch_idx is not None:
            freqs = wave.frequencies[batch_idx]  # [S, H]
            amps = wave.amplitudes[batch_idx]    # [S, H]
        else:
            freqs = wave.frequencies  # [B, S, H]
            amps = wave.amplitudes    # [B, S, H]

        # Spectral centroid = sum(freq * amplitude) / sum(amplitude)
        # Computed per sequence position
        weighted_freqs = freqs * amps
        centroid = weighted_freqs.sum(dim=-1) / (amps.sum(dim=-1) + eps)

        return centroid

    @staticmethod
    @torch.no_grad()
    def compute_total_energy(
        wave: 'Wave',
        batch_idx: Optional[int] = None,
        per_position: bool = False
    ) -> torch.Tensor:
        """
        Compute total energy as sum of squared amplitudes.

        Args:
            wave: Wave object [B, S, H]
            batch_idx: If specified, analyze only this batch element
            per_position: If True, return energy per sequence position.
                         If False, return total energy.

        Returns:
            Energy values - shape depends on batch_idx and per_position:
            - batch_idx=None, per_position=False: scalar
            - batch_idx=None, per_position=True: [B, S]
            - batch_idx=int, per_position=False: scalar
            - batch_idx=int, per_position=True: [S]
        """
        if batch_idx is not None:
            amps = wave.amplitudes[batch_idx]  # [S, H]
        else:
            amps = wave.amplitudes  # [B, S, H]

        # Energy = sum of squared amplitudes
        energy = amps ** 2

        if per_position:
            # Sum over harmonics only
            return energy.sum(dim=-1)
        else:
            # Sum over all dimensions
            return energy.sum()

    @staticmethod
    @torch.no_grad()
    def compute_frequency_bandwidth(
        wave: 'Wave',
        percentile: float = 90.0,
        batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute frequency bandwidth containing specified percentile of energy.

        Args:
            wave: Wave object [B, S, H]
            percentile: Percentile of energy to include (0-100)
            batch_idx: If specified, analyze only this batch element

        Returns:
            Bandwidth per sequence position [S] if batch_idx specified, else [B, S]
        """
        if batch_idx is not None:
            freqs = wave.frequencies[batch_idx]  # [S, H]
            amps = wave.amplitudes[batch_idx]    # [S, H]
        else:
            freqs = wave.frequencies  # [B, S, H]
            amps = wave.amplitudes    # [B, S, H]

        # Energy per harmonic
        energy = amps ** 2  # [S, H] or [B, S, H]

        # Process each sequence position independently
        if batch_idx is not None:
            S, H = freqs.shape
            bandwidths = []

            for s in range(S):
                # Sort by frequency for this position
                sorted_indices = torch.argsort(freqs[s])
                sorted_freqs = freqs[s][sorted_indices]
                sorted_energy = energy[s][sorted_indices]

                # Cumulative energy
                total_energy = sorted_energy.sum()
                cumsum_energy = torch.cumsum(sorted_energy, dim=0)

                # Find index where cumulative energy exceeds percentile
                threshold = (percentile / 100.0) * total_energy
                idx = torch.searchsorted(cumsum_energy, threshold)
                idx = min(idx.item(), H - 1)

                # Bandwidth = max_freq - min_freq in percentile range
                bandwidth = sorted_freqs[idx] - sorted_freqs[0]
                bandwidths.append(bandwidth)

            return torch.stack(bandwidths)
        else:
            B, S, H = freqs.shape
            bandwidths = []

            for b in range(B):
                batch_bandwidths = []
                for s in range(S):
                    # Sort by frequency
                    sorted_indices = torch.argsort(freqs[b, s])
                    sorted_freqs = freqs[b, s][sorted_indices]
                    sorted_energy = energy[b, s][sorted_indices]

                    # Cumulative energy
                    total_energy = sorted_energy.sum()
                    cumsum_energy = torch.cumsum(sorted_energy, dim=0)

                    # Find threshold
                    threshold = (percentile / 100.0) * total_energy
                    idx = torch.searchsorted(cumsum_energy, threshold)
                    idx = min(idx.item(), H - 1)

                    # Bandwidth
                    bandwidth = sorted_freqs[idx] - sorted_freqs[0]
                    batch_bandwidths.append(bandwidth)

                bandwidths.append(torch.stack(batch_bandwidths))

            return torch.stack(bandwidths)

    @staticmethod
    @torch.no_grad()
    def compute_harmonic_entropy(
        wave: 'Wave',
        batch_idx: Optional[int] = None,
        eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Compute Shannon entropy of amplitude distribution across harmonics.

        Higher entropy indicates more evenly distributed energy across harmonics.
        Lower entropy indicates energy concentrated in few harmonics.

        Args:
            wave: Wave object [B, S, H]
            batch_idx: If specified, analyze only this batch element
            eps: Small constant for numerical stability

        Returns:
            Entropy per sequence position [S] if batch_idx specified, else [B, S]
        """
        if batch_idx is not None:
            amps = wave.amplitudes[batch_idx]  # [S, H]
        else:
            amps = wave.amplitudes  # [B, S, H]

        # Normalize amplitudes to probabilities per position
        amp_sum = amps.sum(dim=-1, keepdim=True) + eps
        probs = amps / amp_sum  # [S, H] or [B, S, H]

        # Shannon entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)

        return entropy

    @staticmethod
    @torch.no_grad()
    def compute_phase_velocity(
        wave: 'Wave',
        batch_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute phase velocity (rate of phase change) across sequence.

        Args:
            wave: Wave object [B, S, H]
            batch_idx: If specified, analyze only this batch element

        Returns:
            Phase velocity [S-1, H] if batch_idx specified, else [B, S-1, H]
        """
        if batch_idx is not None:
            phases = wave.phases[batch_idx]  # [S, H]
        else:
            phases = wave.phases  # [B, S, H]

        # Phase difference between consecutive positions
        phase_diff = torch.diff(phases, dim=-2 if batch_idx is None else 0)

        # Wrap to [-pi, pi]
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

        return phase_diff

    @staticmethod
    @torch.no_grad()
    def compute_amplitude_envelope(
        wave: 'Wave',
        batch_idx: Optional[int] = None,
        smoothing_window: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute amplitude envelope (total amplitude per position).

        Args:
            wave: Wave object [B, S, H]
            batch_idx: If specified, analyze only this batch element
            smoothing_window: If specified, apply moving average smoothing

        Returns:
            Envelope [S] if batch_idx specified, else [B, S]
        """
        if batch_idx is not None:
            amps = wave.amplitudes[batch_idx]  # [S, H]
        else:
            amps = wave.amplitudes  # [B, S, H]

        # Sum across harmonics
        envelope = amps.sum(dim=-1)

        # Apply smoothing if requested
        if smoothing_window is not None and smoothing_window > 1:
            # Simple moving average
            if batch_idx is not None:
                # 1D convolution for smoothing
                kernel = torch.ones(smoothing_window, device=envelope.device) / smoothing_window
                padded = torch.nn.functional.pad(envelope, (smoothing_window // 2, smoothing_window // 2), mode='reflect')
                envelope = torch.nn.functional.conv1d(
                    padded.unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0)
                ).squeeze()
            else:
                # Batch processing
                B = envelope.shape[0]
                kernel = torch.ones(1, 1, smoothing_window, device=envelope.device) / smoothing_window
                padded = torch.nn.functional.pad(
                    envelope.unsqueeze(1),
                    (smoothing_window // 2, smoothing_window // 2),
                    mode='reflect'
                )
                envelope = torch.nn.functional.conv1d(padded, kernel).squeeze(1)

        return envelope
