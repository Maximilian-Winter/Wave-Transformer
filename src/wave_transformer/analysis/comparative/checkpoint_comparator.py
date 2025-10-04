"""
Checkpoint Comparator

Compares multiple Wave Transformer checkpoints to analyze training evolution,
measure divergence, and identify critical checkpoints.
"""

import copy
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import warnings

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scipy.stats import wasserstein_distance
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available. Wasserstein distance will not be computed.")

from wave_transformer.core.wave import Wave
from wave_transformer.analysis.introspection.layer_analyzer import LayerWaveAnalyzer


class CheckpointComparator:
    """
    Compare multiple Wave Transformer checkpoints.

    This class loads multiple checkpoints and provides methods to:
    - Compare their outputs on the same inputs
    - Measure divergence between checkpoints (L2, cosine, KL, Wasserstein)
    - Visualize metric evolution across training
    - Identify critical checkpoints with large changes

    Args:
        checkpoint_paths: List of paths to checkpoint directories
        encoder_cls: Encoder class for loading (e.g., TokenToWaveEncoder)
        decoder_cls: Decoder class for loading (e.g., WaveToTokenDecoder)
        device: Device to load models on
        map_location: Optional device mapping for loading

    Example:
        >>> comparator = CheckpointComparator(
        ...     checkpoint_paths=['ckpt_1000', 'ckpt_2000', 'ckpt_3000'],
        ...     encoder_cls=TokenToWaveEncoder,
        ...     decoder_cls=WaveToTokenDecoder,
        ...     device='cuda'
        ... )
        >>> divergence = comparator.compute_checkpoint_divergence(input_data)
        >>> critical_points = comparator.identify_critical_checkpoints(divergence)
    """

    def __init__(
        self,
        checkpoint_paths: List[Union[str, Path]],
        encoder_cls: type,
        decoder_cls: type,
        device: str = 'cuda',
        map_location: Optional[str] = None
    ):
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.encoder_cls = encoder_cls
        self.decoder_cls = decoder_cls
        self.device = device
        self.map_location = map_location or device

        # Load all checkpoints
        self.models = []
        self.checkpoint_names = []

        for path in self.checkpoint_paths:
            try:
                from wave_transformer.core.transformer import WaveTransformer
                model = WaveTransformer.load(
                    path,
                    encoder_cls=encoder_cls,
                    decoder_cls=decoder_cls,
                    map_location=self.map_location
                )
                model.to(device)
                model.eval()
                self.models.append(model)
                self.checkpoint_names.append(path.name)
            except Exception as e:
                warnings.warn(f"Failed to load checkpoint {path}: {e}")

    @torch.no_grad()
    def compare_on_input(
        self,
        encoder_input: Dict[str, Any],
        attention_mask: Optional[torch.Tensor] = None,
        return_waves: bool = False
    ) -> Dict[str, Any]:
        """
        Run the same input through all checkpoints and extract wave representations.

        Args:
            encoder_input: Input dictionary for encoder (e.g., {'token_ids': tensor})
            attention_mask: Optional attention mask
            return_waves: If True, return full Wave objects; otherwise return statistics

        Returns:
            Dictionary containing:
            - 'waves': List of Wave objects from each checkpoint (if return_waves=True)
            - 'statistics': Dictionary of statistics per checkpoint
            - 'checkpoint_names': List of checkpoint names
        """
        waves = []
        statistics = []

        for model in self.models:
            model.eval()

            # Forward pass to get encoder output
            wave = model.wave_encoder(attention_mask=attention_mask, **encoder_input)
            waves.append(wave)

            # Compute statistics
            stats = wave.get_statistics(batch_idx=0)
            statistics.append(stats)

        result = {
            'checkpoint_names': self.checkpoint_names,
            'statistics': statistics,
        }

        if return_waves:
            result['waves'] = waves

        return result

    @torch.no_grad()
    def compare_on_dataset(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None,
        return_per_sample: bool = False
    ) -> Dict[str, Any]:
        """
        Compare checkpoints on multiple inputs from a dataloader.

        Args:
            dataloader: DataLoader providing inputs
            max_batches: Maximum number of batches to process (None = all)
            return_per_sample: If True, return per-sample divergences

        Returns:
            Dictionary containing aggregated statistics and divergences
        """
        all_divergences = {
            'l2_distance': [],
            'cosine_similarity': [],
            'kl_divergence': [],
            'wasserstein_distance': []
        }

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            # Move batch to device
            if isinstance(batch, dict):
                encoder_input = {k: v.to(self.device) if torch.is_tensor(v) else v
                               for k, v in batch.items() if k != 'attention_mask'}
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            else:
                encoder_input = {'token_ids': batch.to(self.device)}
                attention_mask = None

            # Get waves from all checkpoints
            result = self.compare_on_input(
                encoder_input=encoder_input,
                attention_mask=attention_mask,
                return_waves=True
            )
            waves = result['waves']

            # Compute pairwise divergences
            batch_divergences = self._compute_pairwise_divergences(waves)

            for metric, values in batch_divergences.items():
                all_divergences[metric].append(values)

        # Aggregate across batches
        aggregated = {}
        for metric, batch_list in all_divergences.items():
            aggregated[metric] = np.mean(batch_list, axis=0)

        result = {
            'divergences': aggregated,
            'checkpoint_names': self.checkpoint_names,
        }

        if return_per_sample:
            result['per_sample_divergences'] = all_divergences

        return result

    def compute_checkpoint_divergence(
        self,
        encoder_input: Dict[str, Any],
        attention_mask: Optional[torch.Tensor] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Measure divergence between consecutive checkpoints.

        Computes multiple divergence metrics:
        - L2 distance between wave representations
        - Cosine similarity of wave vectors
        - KL divergence of amplitude distributions
        - Wasserstein distance for frequency distributions

        Args:
            encoder_input: Input dictionary for encoder
            attention_mask: Optional attention mask
            metrics: List of metrics to compute. Options:
                     ['l2', 'cosine', 'kl', 'wasserstein']
                     If None, computes all metrics.

        Returns:
            Dictionary mapping metric names to arrays of shape [num_checkpoints-1]
            representing divergence from checkpoint i to i+1
        """
        if metrics is None:
            metrics = ['l2', 'cosine', 'kl', 'wasserstein']

        # Get waves from all checkpoints
        result = self.compare_on_input(
            encoder_input=encoder_input,
            attention_mask=attention_mask,
            return_waves=True
        )
        waves = result['waves']

        # Compute consecutive divergences
        divergences = {metric: [] for metric in metrics}

        for i in range(len(waves) - 1):
            wave1 = waves[i]
            wave2 = waves[i + 1]

            # L2 distance
            if 'l2' in metrics:
                repr1 = wave1.to_representation().detach().cpu().numpy()
                repr2 = wave2.to_representation().detach().cpu().numpy()
                l2_dist = np.linalg.norm(repr1 - repr2, axis=-1).mean()
                divergences['l2'].append(l2_dist)

            # Cosine similarity
            if 'cosine' in metrics:
                repr1 = wave1.to_representation().view(-1).detach().cpu().numpy()
                repr2 = wave2.to_representation().view(-1).detach().cpu().numpy()

                dot_product = np.dot(repr1, repr2)
                norm1 = np.linalg.norm(repr1)
                norm2 = np.linalg.norm(repr2)
                cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
                divergences['cosine'].append(cosine_sim)

            # KL divergence (using amplitude distributions)
            if 'kl' in metrics:
                amp1 = wave1.amplitudes.detach().cpu().numpy().flatten()
                amp2 = wave2.amplitudes.detach().cpu().numpy().flatten()

                # Normalize to probability distributions
                amp1 = amp1 / (amp1.sum() + 1e-8)
                amp2 = amp2 / (amp2.sum() + 1e-8)

                # Add small epsilon to avoid log(0)
                amp1 = amp1 + 1e-10
                amp2 = amp2 + 1e-10

                kl_div = np.sum(amp1 * np.log(amp1 / amp2))
                divergences['kl'].append(kl_div)

            # Wasserstein distance (using frequency distributions)
            if 'wasserstein' in metrics:
                if not HAS_SCIPY:
                    warnings.warn("scipy not available. Skipping Wasserstein distance.")
                    divergences['wasserstein'].append(np.nan)
                else:
                    freq1 = wave1.frequencies.detach().cpu().numpy().flatten()
                    freq2 = wave2.frequencies.detach().cpu().numpy().flatten()

                    wass_dist = wasserstein_distance(freq1, freq2)
                    divergences['wasserstein'].append(wass_dist)

        # Convert to numpy arrays
        for metric in metrics:
            divergences[metric] = np.array(divergences[metric])

        return divergences

    def _compute_pairwise_divergences(
        self,
        waves: List[Wave]
    ) -> Dict[str, np.ndarray]:
        """
        Compute pairwise divergences between all checkpoint pairs.

        Returns:
            Dictionary with divergence matrices [num_checkpoints, num_checkpoints]
        """
        n = len(waves)
        divergences = {
            'l2_distance': np.zeros((n, n)),
            'cosine_similarity': np.zeros((n, n)),
            'kl_divergence': np.zeros((n, n)),
            'wasserstein_distance': np.zeros((n, n))
        }

        for i in range(n):
            for j in range(n):
                if i == j:
                    divergences['cosine_similarity'][i, j] = 1.0
                    continue

                wave1 = waves[i]
                wave2 = waves[j]

                # L2 distance
                repr1 = wave1.to_representation().detach().cpu().numpy()
                repr2 = wave2.to_representation().detach().cpu().numpy()
                divergences['l2_distance'][i, j] = np.linalg.norm(repr1 - repr2, axis=-1).mean()

                # Cosine similarity
                repr1_flat = repr1.flatten()
                repr2_flat = repr2.flatten()
                dot_product = np.dot(repr1_flat, repr2_flat)
                norm1 = np.linalg.norm(repr1_flat)
                norm2 = np.linalg.norm(repr2_flat)
                divergences['cosine_similarity'][i, j] = dot_product / (norm1 * norm2 + 1e-8)

                # KL divergence
                amp1 = wave1.amplitudes.detach().cpu().numpy().flatten()
                amp2 = wave2.amplitudes.detach().cpu().numpy().flatten()
                amp1 = amp1 / (amp1.sum() + 1e-8) + 1e-10
                amp2 = amp2 / (amp2.sum() + 1e-8) + 1e-10
                divergences['kl_divergence'][i, j] = np.sum(amp1 * np.log(amp1 / amp2))

                # Wasserstein distance
                if HAS_SCIPY:
                    freq1 = wave1.frequencies.detach().cpu().numpy().flatten()
                    freq2 = wave2.frequencies.detach().cpu().numpy().flatten()
                    divergences['wasserstein_distance'][i, j] = wasserstein_distance(freq1, freq2)
                else:
                    divergences['wasserstein_distance'][i, j] = np.nan

        return divergences

    def plot_checkpoint_evolution(
        self,
        divergence_dict: Dict[str, np.ndarray],
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize metric evolution across checkpoints.

        Args:
            divergence_dict: Output from compute_checkpoint_divergence()
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Figure and axes array
        """
        num_metrics = len(divergence_dict)
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        checkpoint_indices = list(range(len(self.checkpoint_names) - 1))

        for idx, (metric_name, values) in enumerate(divergence_dict.items()):
            ax = axes[idx]

            ax.plot(checkpoint_indices, values, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Checkpoint Transition')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Evolution', fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Annotate checkpoint names
            transition_labels = [f'{self.checkpoint_names[i]} → {self.checkpoint_names[i+1]}'
                               for i in checkpoint_indices]
            ax.set_xticks(checkpoint_indices)
            ax.set_xticklabels(transition_labels, rotation=45, ha='right')

            # Highlight largest changes
            if len(values) > 0:
                if metric_name == 'cosine_similarity':
                    # For cosine similarity, lower is more divergent
                    critical_idx = np.argmin(values)
                else:
                    critical_idx = np.argmax(values)
                ax.axvline(critical_idx, color='red', linestyle='--', alpha=0.5,
                          label=f'Max change at transition {critical_idx}')
                ax.legend()

        fig.suptitle('Checkpoint Divergence Evolution', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def identify_critical_checkpoints(
        self,
        divergence_dict: Dict[str, np.ndarray],
        threshold_percentile: float = 75.0
    ) -> Dict[str, List[int]]:
        """
        Identify checkpoints with large changes (critical points in training).

        Args:
            divergence_dict: Output from compute_checkpoint_divergence()
            threshold_percentile: Percentile threshold for identifying large changes

        Returns:
            Dictionary mapping metric names to lists of critical checkpoint indices
        """
        critical_checkpoints = {}

        for metric_name, values in divergence_dict.items():
            if len(values) == 0:
                critical_checkpoints[metric_name] = []
                continue

            # For cosine similarity, invert (lower = more divergent)
            if metric_name == 'cosine_similarity':
                threshold = np.percentile(values, 100 - threshold_percentile)
                critical_indices = np.where(values <= threshold)[0].tolist()
            else:
                threshold = np.percentile(values, threshold_percentile)
                critical_indices = np.where(values >= threshold)[0].tolist()

            critical_checkpoints[metric_name] = critical_indices

        return critical_checkpoints

    def compare_layer_by_layer(
        self,
        encoder_input: Dict[str, Any],
        attention_mask: Optional[torch.Tensor] = None,
        checkpoint_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Compare checkpoints using layer-wise analysis.

        Args:
            encoder_input: Input dictionary for encoder
            attention_mask: Optional attention mask
            checkpoint_indices: Indices of checkpoints to compare (None = all)

        Returns:
            Dictionary containing layer-wise comparison results
        """
        if checkpoint_indices is None:
            checkpoint_indices = list(range(len(self.models)))

        layer_comparisons = []

        for idx in checkpoint_indices:
            model = self.models[idx]
            analyzer = LayerWaveAnalyzer(model, device=self.device)
            analyzer.register_extraction_hooks()

            snapshots = analyzer.analyze_input(
                encoder_input=encoder_input,
                attention_mask=attention_mask
            )

            layer_comparisons.append({
                'checkpoint_idx': idx,
                'checkpoint_name': self.checkpoint_names[idx],
                'snapshots': snapshots
            })

            analyzer.cleanup()

        return {
            'layer_comparisons': layer_comparisons,
            'checkpoint_names': [self.checkpoint_names[i] for i in checkpoint_indices]
        }

    def plot_pairwise_divergence_matrix(
        self,
        divergence_matrix: np.ndarray,
        metric_name: str = 'L2 Distance',
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot pairwise divergence matrix as a heatmap.

        Args:
            divergence_matrix: Matrix of pairwise divergences [n_ckpts, n_ckpts]
            metric_name: Name of the metric for labeling
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            divergence_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r' if metric_name != 'Cosine Similarity' else 'RdYlBu',
            xticklabels=self.checkpoint_names,
            yticklabels=self.checkpoint_names,
            ax=ax,
            cbar_kws={'label': metric_name}
        )

        ax.set_title(f'Pairwise {metric_name} Between Checkpoints',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def __len__(self) -> int:
        """Return number of loaded checkpoints"""
        return len(self.models)

    def __getitem__(self, idx: int) -> nn.Module:
        """Get a specific checkpoint model"""
        return self.models[idx]
