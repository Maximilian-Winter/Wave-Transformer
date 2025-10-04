"""
Data collectors for capturing training dynamics and wave statistics.

This module provides abstract and concrete collectors for gathering data
during training, including wave statistics, gradients, and loss breakdowns.
Collectors support sampling to reduce overhead and memory-efficient storage.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import warnings

from .wave_statistics import WaveStatistics


class DataCollector(ABC):
    """
    Abstract base class for data collectors with sampling logic.

    Collectors can sample data at specified intervals to reduce overhead.
    Supports distributed training by checking rank.
    """

    def __init__(
        self,
        sample_interval: int = 1,
        max_samples: Optional[int] = None,
        collect_on_rank: int = 0
    ):
        """
        Initialize data collector.

        Args:
            sample_interval: Collect data every N calls (1 = every call)
            max_samples: Maximum number of samples to store (None = unlimited)
            collect_on_rank: Only collect on this rank (for distributed training)
        """
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.collect_on_rank = collect_on_rank

        self._call_count = 0
        self._sample_count = 0
        self.data = defaultdict(list)

    def should_collect(self) -> bool:
        """
        Determine if data should be collected on this call.

        Returns:
            True if data should be collected
        """
        # Check rank
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                if rank != self.collect_on_rank:
                    return False
        except (ImportError, RuntimeError):
            # Not in distributed mode
            pass

        # Check if we've reached max samples
        if self.max_samples is not None and self._sample_count >= self.max_samples:
            return False

        # Check sampling interval
        if self._call_count % self.sample_interval == 0:
            return True

        return False

    def _increment_counters(self):
        """Increment call and sample counters."""
        self._call_count += 1
        if self.should_collect():
            self._sample_count += 1

    @abstractmethod
    def collect(self, *args, **kwargs) -> None:
        """
        Collect data. Must be implemented by subclasses.

        This method should call should_collect() and _increment_counters().
        """
        pass

    def reset(self) -> None:
        """Reset collector state and clear data."""
        self._call_count = 0
        self._sample_count = 0
        self.data = defaultdict(list)

    def get_data(self) -> Dict[str, Any]:
        """
        Get collected data as dictionary.

        Returns:
            Dictionary of collected data
        """
        return dict(self.data)

    def __len__(self) -> int:
        """Return number of samples collected."""
        return self._sample_count


class WaveCollector(DataCollector):
    """
    Collector for wave statistics during training.

    Captures comprehensive wave statistics at specified intervals,
    including basic stats, harmonic importance, phase coherence, etc.
    """

    def __init__(
        self,
        sample_interval: int = 100,
        max_samples: Optional[int] = 1000,
        collect_on_rank: int = 0,
        statistics_to_collect: Optional[List[str]] = None,
        batch_reduction: str = 'mean'
    ):
        """
        Initialize wave collector.

        Args:
            sample_interval: Collect every N steps
            max_samples: Maximum samples to store
            collect_on_rank: Rank to collect on (for distributed training)
            statistics_to_collect: List of statistics to collect. Options:
                - 'basic_stats': Basic statistics (mean, std, etc.)
                - 'harmonic_importance': Harmonic ranking
                - 'phase_coherence': Phase coherence
                - 'spectral_centroid': Spectral centroid
                - 'total_energy': Total energy
                - 'frequency_bandwidth': Frequency bandwidth
                - 'harmonic_entropy': Amplitude entropy
                If None, collect all.
            batch_reduction: How to reduce batch dimension ('mean', 'first', 'all')
        """
        super().__init__(sample_interval, max_samples, collect_on_rank)

        if statistics_to_collect is None:
            statistics_to_collect = [
                'basic_stats',
                'harmonic_importance',
                'phase_coherence',
                'spectral_centroid',
                'total_energy'
            ]

        self.statistics_to_collect = statistics_to_collect
        self.batch_reduction = batch_reduction

    @torch.no_grad()
    def collect(
        self,
        wave: 'Wave',
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Collect wave statistics.

        Args:
            wave: Wave object to analyze
            step: Training step (for tracking)
            metadata: Additional metadata to store
        """
        if not self.should_collect():
            self._increment_counters()
            return

        # Determine batch index for analysis
        if self.batch_reduction == 'first':
            batch_idx = 0
        elif self.batch_reduction == 'mean':
            batch_idx = None  # Statistics will average across batch
        else:  # 'all'
            batch_idx = None

        # Store step
        if step is not None:
            self.data['step'].append(step)

        # Store metadata
        if metadata is not None:
            for key, value in metadata.items():
                self.data[f'metadata_{key}'].append(value)

        # Collect requested statistics
        if 'basic_stats' in self.statistics_to_collect:
            stats = WaveStatistics.compute_basic_stats(wave, component='all')
            for comp_name, stat_obj in stats.items():
                if hasattr(stat_obj, 'to_dict'):
                    for stat_name, value in stat_obj.to_dict().items():
                        self.data[f'{comp_name}_{stat_name}'].append(value)

        if 'harmonic_importance' in self.statistics_to_collect:
            importance = WaveStatistics.compute_harmonic_importance(
                wave, metric='amplitude', batch_idx=batch_idx
            )
            # Store top-10 harmonics
            top_indices, top_scores = importance.top_k(10)
            self.data['top_harmonics_indices'].append(top_indices.copy())
            self.data['top_harmonics_scores'].append(top_scores.copy())

        if 'phase_coherence' in self.statistics_to_collect:
            coherence = WaveStatistics.compute_phase_coherence(wave, batch_idx=batch_idx)
            self.data['phase_coherence_mean'].append(coherence.mean().item())
            self.data['phase_coherence_std'].append(coherence.std().item())

        if 'spectral_centroid' in self.statistics_to_collect:
            centroid = WaveStatistics.compute_spectral_centroid(wave, batch_idx=batch_idx)
            self.data['spectral_centroid_mean'].append(centroid.mean().item())
            self.data['spectral_centroid_std'].append(centroid.std().item())

        if 'total_energy' in self.statistics_to_collect:
            energy = WaveStatistics.compute_total_energy(wave, batch_idx=batch_idx, per_position=False)
            self.data['total_energy'].append(energy.item())

        if 'frequency_bandwidth' in self.statistics_to_collect:
            bandwidth = WaveStatistics.compute_frequency_bandwidth(wave, batch_idx=batch_idx)
            self.data['frequency_bandwidth_mean'].append(bandwidth.mean().item())
            self.data['frequency_bandwidth_std'].append(bandwidth.std().item())

        if 'harmonic_entropy' in self.statistics_to_collect:
            entropy = WaveStatistics.compute_harmonic_entropy(wave, batch_idx=batch_idx)
            self.data['harmonic_entropy_mean'].append(entropy.mean().item())
            self.data['harmonic_entropy_std'].append(entropy.std().item())

        self._increment_counters()


class GradientCollector(DataCollector):
    """
    Collector for gradient statistics during training.

    Captures gradient norms, distributions, and flow through the network.
    Useful for diagnosing training issues like vanishing/exploding gradients.
    """

    def __init__(
        self,
        sample_interval: int = 10,
        max_samples: Optional[int] = 1000,
        collect_on_rank: int = 0,
        track_layer_names: Optional[List[str]] = None,
        compute_histograms: bool = False,
        histogram_bins: int = 50
    ):
        """
        Initialize gradient collector.

        Args:
            sample_interval: Collect every N steps
            max_samples: Maximum samples to store
            collect_on_rank: Rank to collect on
            track_layer_names: List of layer name patterns to track (regex).
                              If None, track all layers.
            compute_histograms: Whether to compute gradient histograms
            histogram_bins: Number of bins for histograms
        """
        super().__init__(sample_interval, max_samples, collect_on_rank)

        self.track_layer_names = track_layer_names
        self.compute_histograms = compute_histograms
        self.histogram_bins = histogram_bins

    @torch.no_grad()
    def collect(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None
    ) -> None:
        """
        Collect gradient statistics from model.

        Args:
            model: Model to collect gradients from
            step: Training step
        """
        if not self.should_collect():
            self._increment_counters()
            return

        if step is not None:
            self.data['step'].append(step)

        # Collect gradient stats per parameter
        grad_norms = []
        grad_means = []
        grad_stds = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            # Check if we should track this layer
            if self.track_layer_names is not None:
                import re
                match = any(re.search(pattern, name) for pattern in self.track_layer_names)
                if not match:
                    continue

            grad = param.grad

            # Compute statistics
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()

            grad_norms.append(grad_norm)
            grad_means.append(grad_mean)
            grad_stds.append(grad_std)

            # Store per-layer stats
            self.data[f'{name}_grad_norm'].append(grad_norm)
            self.data[f'{name}_grad_mean'].append(grad_mean)
            self.data[f'{name}_grad_std'].append(grad_std)

            # Compute histograms if requested
            if self.compute_histograms:
                hist, bin_edges = np.histogram(
                    grad.cpu().numpy().flatten(),
                    bins=self.histogram_bins
                )
                self.data[f'{name}_grad_histogram'].append(hist)
                if f'{name}_grad_bin_edges' not in self.data:
                    self.data[f'{name}_grad_bin_edges'] = bin_edges

        # Store global statistics
        if grad_norms:
            self.data['global_grad_norm_mean'].append(np.mean(grad_norms))
            self.data['global_grad_norm_max'].append(np.max(grad_norms))
            self.data['global_grad_norm_min'].append(np.min(grad_norms))
            self.data['global_grad_mean'].append(np.mean(grad_means))
            self.data['global_grad_std'].append(np.mean(grad_stds))

        self._increment_counters()

    def get_gradient_flow_summary(self) -> Dict[str, Any]:
        """
        Get summary of gradient flow through network.

        Returns:
            Dictionary with gradient flow statistics
        """
        summary = {}

        # Get layer-wise gradient norms
        layer_norms = {}
        for key in self.data.keys():
            if key.endswith('_grad_norm') and not key.startswith('global'):
                layer_name = key.replace('_grad_norm', '')
                if self.data[key]:
                    layer_norms[layer_name] = {
                        'mean': np.mean(self.data[key]),
                        'std': np.std(self.data[key]),
                        'max': np.max(self.data[key]),
                        'min': np.min(self.data[key])
                    }

        summary['layer_norms'] = layer_norms

        # Global statistics
        if 'global_grad_norm_mean' in self.data and self.data['global_grad_norm_mean']:
            summary['global_mean_norm'] = np.mean(self.data['global_grad_norm_mean'])
            summary['global_max_norm'] = np.max(self.data['global_grad_norm_max'])

        return summary


class LossCollector(DataCollector):
    """
    Collector for per-position loss breakdown during training.

    Captures loss values at different sequence positions to identify
    which positions are hardest to predict.
    """

    def __init__(
        self,
        sample_interval: int = 50,
        max_samples: Optional[int] = 500,
        collect_on_rank: int = 0,
        reduction: str = 'mean'
    ):
        """
        Initialize loss collector.

        Args:
            sample_interval: Collect every N steps
            max_samples: Maximum samples to store
            collect_on_rank: Rank to collect on
            reduction: How to reduce batch dimension ('mean', 'sum', 'none')
        """
        super().__init__(sample_interval, max_samples, collect_on_rank)
        self.reduction = reduction

    @torch.no_grad()
    def collect(
        self,
        loss_per_position: torch.Tensor,
        step: Optional[int] = None,
        position_labels: Optional[List[str]] = None
    ) -> None:
        """
        Collect per-position loss values.

        Args:
            loss_per_position: Loss values [B, S] or [S]
            step: Training step
            position_labels: Optional labels for positions
        """
        if not self.should_collect():
            self._increment_counters()
            return

        if step is not None:
            self.data['step'].append(step)

        # Reduce batch dimension if needed
        if loss_per_position.ndim == 2:
            if self.reduction == 'mean':
                loss_per_position = loss_per_position.mean(dim=0)
            elif self.reduction == 'sum':
                loss_per_position = loss_per_position.sum(dim=0)
            # else: keep all batch elements

        # Store loss values
        self.data['loss_per_position'].append(loss_per_position.cpu().numpy())

        # Store position labels if provided
        if position_labels is not None and 'position_labels' not in self.data:
            self.data['position_labels'] = position_labels

        self._increment_counters()

    def get_hardest_positions(self, k: int = 10) -> List[int]:
        """
        Get k positions with highest average loss.

        Args:
            k: Number of positions to return

        Returns:
            List of position indices sorted by descending loss
        """
        if 'loss_per_position' not in self.data or not self.data['loss_per_position']:
            return []

        # Average across all samples
        all_losses = np.stack(self.data['loss_per_position'])
        mean_loss = all_losses.mean(axis=0)

        # Get top-k indices
        top_indices = np.argsort(mean_loss)[::-1][:k]
        return top_indices.tolist()

    def get_position_difficulty_profile(self) -> np.ndarray:
        """
        Get average loss per position across all samples.

        Returns:
            Array of average losses [S]
        """
        if 'loss_per_position' not in self.data or not self.data['loss_per_position']:
            return np.array([])

        all_losses = np.stack(self.data['loss_per_position'])
        return all_losses.mean(axis=0)


class ActivationCollector(DataCollector):
    """
    Collector for intermediate activations during forward pass.

    Useful for analyzing representation learning and debugging.
    """

    def __init__(
        self,
        sample_interval: int = 100,
        max_samples: Optional[int] = 100,
        collect_on_rank: int = 0,
        layer_names: Optional[List[str]] = None,
        compute_statistics: bool = True
    ):
        """
        Initialize activation collector.

        Args:
            sample_interval: Collect every N steps
            max_samples: Maximum samples to store
            collect_on_rank: Rank to collect on
            layer_names: Names of layers to collect from
            compute_statistics: Whether to compute activation statistics
        """
        super().__init__(sample_interval, max_samples, collect_on_rank)

        self.layer_names = layer_names or []
        self.compute_statistics = compute_statistics
        self.hooks = []

    @torch.no_grad()
    def collect(
        self,
        activations: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ) -> None:
        """
        Collect activations from specified layers.

        Args:
            activations: Dictionary mapping layer names to activation tensors
            step: Training step
        """
        if not self.should_collect():
            self._increment_counters()
            return

        if step is not None:
            self.data['step'].append(step)

        for layer_name, activation in activations.items():
            if self.layer_names and layer_name not in self.layer_names:
                continue

            if self.compute_statistics:
                # Store statistics instead of raw activations
                self.data[f'{layer_name}_mean'].append(activation.mean().item())
                self.data[f'{layer_name}_std'].append(activation.std().item())
                self.data[f'{layer_name}_max'].append(activation.max().item())
                self.data[f'{layer_name}_min'].append(activation.min().item())
                self.data[f'{layer_name}_norm'].append(activation.norm().item())
            else:
                # Store raw activations (memory intensive!)
                warnings.warn(
                    f"Storing raw activations for {layer_name}. This may use significant memory.",
                    ResourceWarning
                )
                self.data[f'{layer_name}_activations'].append(activation.cpu().numpy())

        self._increment_counters()

    def register_hooks(
        self,
        model: torch.nn.Module,
        forward: bool = True
    ) -> None:
        """
        Register forward hooks to automatically collect activations.

        Args:
            model: Model to register hooks on
            forward: If True, use forward hooks. If False, use backward hooks.
        """
        def create_hook(name):
            def hook(module, input, output):
                if self.should_collect():
                    self.collect({name: output}, step=None)
            return hook

        for name, module in model.named_modules():
            if self.layer_names and name not in self.layer_names:
                continue

            if forward:
                handle = module.register_forward_hook(create_hook(name))
            else:
                handle = module.register_backward_hook(create_hook(name))

            self.hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
