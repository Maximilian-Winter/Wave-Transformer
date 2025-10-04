"""
Harmonic Importance Analysis

Analyzes which harmonics contribute most to model predictions and supports
model compression through harmonic pruning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm

from wave_transformer.core.wave import Wave


class HarmonicImportanceAnalyzer:
    """
    Analyzes importance of individual harmonics for model compression and interpretation.

    Provides multiple methods for measuring harmonic importance:
    - Amplitude-based: Average amplitude across dataset
    - Energy-based: Sum of squared amplitudes
    - Variance-based: Variance of harmonic values
    - Gradient-based: Sensitivity via ablation studies

    Args:
        model: WaveTransformer model to analyze
        criterion: Loss function for gradient-based analysis
        device: Device for computation

    Example:
        >>> analyzer = HarmonicImportanceAnalyzer(model, criterion)
        >>> importance = analyzer.analyze_harmonic_importance(dataloader, method='energy')
        >>> analyzer.plot_harmonic_importance(importance)
        >>> mask = analyzer.get_sparse_harmonic_mask(importance, top_k=32)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: Optional[Callable] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.device = device

    @torch.no_grad()
    def analyze_harmonic_importance(
        self,
        dataloader: torch.utils.data.DataLoader,
        method: str = 'energy',
        layer_name: str = 'encoder',
        max_batches: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute harmonic importance across a dataset.

        Args:
            dataloader: DataLoader providing (input, target) batches
            method: Importance metric to use:
                - 'amplitude': Mean amplitude per harmonic
                - 'energy': Sum of squared amplitudes (default)
                - 'variance': Variance across dataset
                - 'max_amplitude': Maximum amplitude observed
            layer_name: Which layer to analyze ('encoder', 'layer_0', etc.)
            max_batches: Limit number of batches (for faster analysis)

        Returns:
            Dictionary with:
                - 'importance': Array of shape [num_harmonics] with importance scores
                - 'mean_amplitude': Mean amplitude per harmonic
                - 'std_amplitude': Std of amplitude per harmonic
                - 'method': Method used
        """
        self.model.eval()

        # Collect harmonic statistics
        harmonic_amplitudes = []
        hook_outputs = {}

        # Register hook to extract wave from specified layer
        def extraction_hook(module, input, output):
            if isinstance(output, Wave):
                hook_outputs['wave'] = output
            else:
                # Tensor representation - convert to Wave
                hook_outputs['wave'] = Wave.from_representation(output)

        # Find and hook the appropriate layer
        hook_handle = None
        if layer_name == 'encoder' and hasattr(self.model, 'wave_encoder'):
            hook_handle = self.model.wave_encoder.register_forward_hook(extraction_hook)
        elif layer_name.startswith('layer_') and hasattr(self.model, 'layers'):
            layer_idx = int(layer_name.split('_')[1])
            hook_handle = self.model.layers[layer_idx].register_forward_hook(extraction_hook)
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")

        try:
            num_batches = 0
            for batch in tqdm(dataloader, desc=f"Analyzing harmonics ({method})"):
                if max_batches and num_batches >= max_batches:
                    break

                # Extract input and move to device
                if isinstance(batch, dict):
                    encoder_input = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                   for k, v in batch.items() if k != 'labels'}
                    attention_mask = batch.get('attention_mask', None)
                else:
                    # Assume tuple (input, target)
                    inputs = batch[0].to(self.device)
                    encoder_input = {'token_ids': inputs}
                    attention_mask = None

                # Forward pass
                hook_outputs = {}
                with torch.no_grad():
                    _ = self.model(encoder_input, attention_mask=attention_mask)

                # Extract amplitudes from hooked wave
                if 'wave' in hook_outputs:
                    wave = hook_outputs['wave']
                    # Shape: [B, S, H]
                    amps = wave.amplitudes.detach().cpu().numpy()
                    harmonic_amplitudes.append(amps)

                num_batches += 1

        finally:
            if hook_handle:
                hook_handle.remove()

        # Concatenate all batches: [total_samples, seq_len, num_harmonics]
        all_amplitudes = np.concatenate(harmonic_amplitudes, axis=0)

        # Compute importance based on method
        if method == 'amplitude':
            # Mean amplitude per harmonic across all positions and samples
            importance = all_amplitudes.mean(axis=(0, 1))  # [H]
        elif method == 'energy':
            # Sum of squared amplitudes (energy)
            importance = (all_amplitudes ** 2).sum(axis=(0, 1))  # [H]
        elif method == 'variance':
            # Variance across dataset
            importance = all_amplitudes.reshape(-1, all_amplitudes.shape[-1]).var(axis=0)  # [H]
        elif method == 'max_amplitude':
            # Maximum amplitude observed
            importance = all_amplitudes.max(axis=(0, 1))  # [H]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Compute statistics
        mean_amplitude = all_amplitudes.mean(axis=(0, 1))
        std_amplitude = all_amplitudes.reshape(-1, all_amplitudes.shape[-1]).std(axis=0)

        return {
            'importance': importance,
            'mean_amplitude': mean_amplitude,
            'std_amplitude': std_amplitude,
            'method': method,
            'num_harmonics': len(importance),
        }

    def compute_gradient_sensitivity(
        self,
        dataloader: torch.utils.data.DataLoader,
        harmonic_indices: Optional[List[int]] = None,
        max_batches: int = 10,
        ablation_value: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Measure harmonic importance via gradient-based ablation study.

        For each harmonic, measures the change in loss when that harmonic
        is ablated (set to zero or another value).

        Args:
            dataloader: DataLoader with (input, target) batches
            harmonic_indices: Specific harmonics to test (or None for all)
            max_batches: Number of batches to use for estimation
            ablation_value: Value to set ablated harmonics to (default: 0.0)

        Returns:
            Dictionary with:
                - 'sensitivity': Array [num_harmonics] with loss increase per harmonic
                - 'baseline_loss': Loss without ablation
        """
        self.model.eval()

        # Get number of harmonics from model
        num_harmonics = self.model.num_harmonics
        if harmonic_indices is None:
            harmonic_indices = list(range(num_harmonics))

        sensitivity_scores = np.zeros(num_harmonics)

        # Compute baseline loss
        baseline_loss = self._compute_average_loss(dataloader, max_batches)

        # Test ablating each harmonic
        for h_idx in tqdm(harmonic_indices, desc="Computing gradient sensitivity"):
            ablated_loss = self._compute_average_loss(
                dataloader,
                max_batches,
                ablate_harmonic=h_idx,
                ablation_value=ablation_value
            )
            # Sensitivity = increase in loss when harmonic is removed
            sensitivity_scores[h_idx] = max(0, ablated_loss - baseline_loss)

        return {
            'sensitivity': sensitivity_scores,
            'baseline_loss': baseline_loss,
            'harmonic_indices_tested': harmonic_indices,
        }

    def _compute_average_loss(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int,
        ablate_harmonic: Optional[int] = None,
        ablation_value: float = 0.0
    ) -> float:
        """
        Compute average loss over batches, optionally ablating a harmonic.

        Args:
            dataloader: Data loader
            max_batches: Max batches to use
            ablate_harmonic: Harmonic index to ablate (or None)
            ablation_value: Value to set ablated harmonic to

        Returns:
            Average loss
        """
        total_loss = 0.0
        num_batches = 0

        # Hook to ablate harmonic in encoder output
        hook_outputs = {}

        def ablation_hook(module, input, output):
            if ablate_harmonic is not None:
                if isinstance(output, Wave):
                    # Ablate in Wave object
                    output.amplitudes[:, :, ablate_harmonic] = ablation_value
                    # Could also ablate frequencies/phases, but amplitude is most direct
            return output

        # Register hook on encoder
        hook_handle = None
        if hasattr(self.model, 'wave_encoder'):
            hook_handle = self.model.wave_encoder.register_forward_hook(ablation_hook)

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= max_batches:
                        break

                    # Parse batch
                    if isinstance(batch, dict):
                        encoder_input = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                       for k, v in batch.items() if k not in ['labels', 'label']}
                        targets = batch.get('labels', batch.get('label')).to(self.device)
                        attention_mask = batch.get('attention_mask', None)
                    else:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        encoder_input = {'token_ids': inputs}
                        attention_mask = None

                    # Forward pass
                    outputs = self.model(encoder_input, attention_mask=attention_mask)

                    # Compute loss
                    if outputs.dim() == 3:  # [B, S, V]
                        loss = self.criterion(
                            outputs.reshape(-1, outputs.size(-1)),
                            targets.reshape(-1)
                        )
                    else:
                        loss = self.criterion(outputs, targets)

                    total_loss += loss.item()
                    num_batches += 1

        finally:
            if hook_handle:
                hook_handle.remove()

        return total_loss / max(num_batches, 1)

    def plot_harmonic_importance(
        self,
        importance_results: Dict[str, np.ndarray],
        top_k: Optional[int] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize harmonic importance scores.

        Args:
            importance_results: Output from analyze_harmonic_importance()
            top_k: Highlight top-k most important harmonics (or None for all)
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Figure and axes
        """
        importance = importance_results['importance']
        num_harmonics = len(importance)
        method = importance_results.get('method', 'unknown')

        # Normalize importance to [0, 1]
        importance_normalized = importance / (importance.max() + 1e-8)

        # Get top-k indices
        top_indices = np.argsort(importance)[::-1]
        if top_k is not None:
            top_k = min(top_k, num_harmonics)
            top_indices = top_indices[:top_k]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: Bar chart of importance scores
        ax = axes[0, 0]
        colors = ['red' if i in top_indices else 'steelblue' for i in range(num_harmonics)]
        ax.bar(range(num_harmonics), importance_normalized, color=colors, alpha=0.7)
        ax.set_xlabel('Harmonic Index')
        ax.set_ylabel('Normalized Importance')
        ax.set_title(f'Harmonic Importance ({method})')
        ax.grid(True, alpha=0.3)

        # Plot 2: Top-k harmonics
        ax = axes[0, 1]
        if top_k:
            top_k_values = importance[top_indices]
            ax.barh(range(len(top_indices)), top_k_values, color='coral', alpha=0.7)
            ax.set_yticks(range(len(top_indices)))
            ax.set_yticklabels([f'H{i}' for i in top_indices])
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Top {len(top_indices)} Most Important Harmonics')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

        # Plot 3: Cumulative importance
        ax = axes[1, 0]
        sorted_importance = np.sort(importance)[::-1]
        cumulative = np.cumsum(sorted_importance) / sorted_importance.sum()
        ax.plot(range(num_harmonics), cumulative, linewidth=2, color='green')
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
        ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95% threshold')
        ax.set_xlabel('Number of Harmonics (sorted by importance)')
        ax.set_ylabel('Cumulative Importance')
        ax.set_title('Cumulative Harmonic Importance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Mean amplitude with std if available
        ax = axes[1, 1]
        if 'mean_amplitude' in importance_results:
            mean_amp = importance_results['mean_amplitude']
            std_amp = importance_results.get('std_amplitude', None)

            ax.plot(range(num_harmonics), mean_amp, linewidth=2, color='purple', label='Mean')
            if std_amp is not None:
                ax.fill_between(range(num_harmonics),
                               mean_amp - std_amp,
                               mean_amp + std_amp,
                               alpha=0.3, color='purple', label='±1 std')
            ax.set_xlabel('Harmonic Index')
            ax.set_ylabel('Amplitude')
            ax.set_title('Mean Harmonic Amplitudes')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Amplitude statistics not available',
                   ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def get_sparse_harmonic_mask(
        self,
        importance_results: Dict[str, np.ndarray],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        cumulative_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Create binary mask for model compression via harmonic pruning.

        Supports three selection strategies:
        1. top_k: Keep the k most important harmonics
        2. threshold: Keep harmonics above importance threshold
        3. cumulative_threshold: Keep harmonics until cumulative importance reaches threshold

        Args:
            importance_results: Output from analyze_harmonic_importance()
            top_k: Number of top harmonics to keep
            threshold: Absolute importance threshold
            cumulative_threshold: Cumulative importance threshold (e.g., 0.95 for 95%)

        Returns:
            Binary mask of shape [num_harmonics] where 1=keep, 0=prune
        """
        importance = importance_results['importance']
        num_harmonics = len(importance)
        mask = np.zeros(num_harmonics, dtype=bool)

        if top_k is not None:
            # Keep top-k harmonics
            top_indices = np.argsort(importance)[::-1][:top_k]
            mask[top_indices] = True

        elif threshold is not None:
            # Keep harmonics above threshold
            mask = importance >= threshold

        elif cumulative_threshold is not None:
            # Keep harmonics until cumulative importance reaches threshold
            sorted_indices = np.argsort(importance)[::-1]
            cumulative = 0.0
            total_importance = importance.sum()

            for idx in sorted_indices:
                mask[idx] = True
                cumulative += importance[idx]
                if cumulative / total_importance >= cumulative_threshold:
                    break
        else:
            raise ValueError("Must specify one of: top_k, threshold, or cumulative_threshold")

        return mask.astype(np.float32)

    def apply_harmonic_mask(
        self,
        wave: Wave,
        mask: np.ndarray,
        zero_frequencies: bool = True,
        zero_amplitudes: bool = True,
        zero_phases: bool = False
    ) -> Wave:
        """
        Apply harmonic mask to a Wave object for compression testing.

        Args:
            wave: Input Wave object
            mask: Binary mask [num_harmonics] where 1=keep, 0=zero
            zero_frequencies: Whether to zero masked frequencies
            zero_amplitudes: Whether to zero masked amplitudes
            zero_phases: Whether to zero masked phases

        Returns:
            Masked Wave object
        """
        mask_tensor = torch.from_numpy(mask).to(wave.frequencies.device)
        # Reshape for broadcasting: [1, 1, H]
        mask_tensor = mask_tensor.view(1, 1, -1)

        new_freqs = wave.frequencies.clone()
        new_amps = wave.amplitudes.clone()
        new_phases = wave.phases.clone()

        if zero_frequencies:
            new_freqs = new_freqs * mask_tensor
        if zero_amplitudes:
            new_amps = new_amps * mask_tensor
        if zero_phases:
            new_phases = new_phases * mask_tensor

        return Wave(new_freqs, new_amps, new_phases)
