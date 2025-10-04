"""
Layer-by-Layer Wave Analysis

Provides tools for extracting and analyzing wave representations at each layer
of the WaveTransformer architecture.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from wave_transformer.core.wave import Wave


@dataclass
class LayerWaveSnapshot:
    """Container for wave data extracted from a specific layer"""
    layer_idx: int
    layer_name: str
    wave: Wave
    is_encoder: bool = False
    is_decoder: bool = False


class LayerWaveAnalyzer:
    """
    Analyzes wave evolution through transformer layers.

    This analyzer registers hooks to extract wave representations at each layer,
    allowing detailed study of how waves transform through the network.

    Args:
        model: WaveTransformer instance to analyze
        device: Device to perform analysis on

    Example:
        >>> analyzer = LayerWaveAnalyzer(model)
        >>> analyzer.register_extraction_hooks()
        >>> results = analyzer.analyze_input(input_data)
        >>> analyzer.plot_layer_evolution(results)
        >>> analyzer.cleanup()
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.hooks = []
        self.layer_outputs = {}

    def register_extraction_hooks(self) -> None:
        """
        Register forward hooks on encoder, transformer layers, and decoder.

        Hooks extract:
        - Encoder output (Wave object)
        - Each transformer layer output (tensor representation)
        - Pre-decoder representation (tensor)
        """
        self.layer_outputs = {}
        self.hooks = []

        # Hook encoder output (Wave object)
        def encoder_hook(module, input, output):
            if isinstance(output, Wave):
                self.layer_outputs['encoder'] = output

        if hasattr(self.model, 'wave_encoder'):
            handle = self.model.wave_encoder.register_forward_hook(encoder_hook)
            self.hooks.append(handle)

        # Hook each transformer layer (tensor outputs)
        if hasattr(self.model, 'layers'):
            for idx, layer in enumerate(self.model.layers):
                def make_layer_hook(layer_idx):
                    def hook(module, input, output):
                        self.layer_outputs[f'layer_{layer_idx}'] = output
                    return hook

                handle = layer.register_forward_hook(make_layer_hook(idx))
                self.hooks.append(handle)

        # Hook final norm before decoder
        def pre_decoder_hook(module, input, output):
            self.layer_outputs['pre_decoder'] = output

        if hasattr(self.model, 'norm_f'):
            handle = self.model.norm_f.register_forward_hook(pre_decoder_hook)
            self.hooks.append(handle)

    def cleanup(self) -> None:
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = {}

    @torch.no_grad()
    def analyze_input(
        self,
        encoder_input: Dict,
        attention_mask: Optional[torch.Tensor] = None,
        batch_idx: int = 0
    ) -> List[LayerWaveSnapshot]:
        """
        Run forward pass and extract wave representations from all layers.

        Args:
            encoder_input: Input dictionary for encoder (e.g., {'token_ids': ...})
            attention_mask: Optional attention mask
            batch_idx: Which batch element to analyze

        Returns:
            List of LayerWaveSnapshot objects containing wave data from each layer
        """
        self.model.eval()
        self.layer_outputs = {}

        # Forward pass to populate layer_outputs
        with torch.no_grad():
            _ = self.model(encoder_input, attention_mask=attention_mask)

        # Extract waves from captured outputs
        snapshots = []

        # Encoder wave (already a Wave object)
        if 'encoder' in self.layer_outputs:
            encoder_wave = self.layer_outputs['encoder']
            snapshots.append(LayerWaveSnapshot(
                layer_idx=-1,
                layer_name='encoder',
                wave=encoder_wave,
                is_encoder=True
            ))

        # Transformer layer waves (need conversion from tensor representation)
        num_layers = len([k for k in self.layer_outputs.keys() if k.startswith('layer_')])
        for idx in range(num_layers):
            key = f'layer_{idx}'
            if key in self.layer_outputs:
                tensor_repr = self.layer_outputs[key]
                wave = Wave.from_representation(tensor_repr)
                snapshots.append(LayerWaveSnapshot(
                    layer_idx=idx,
                    layer_name=f'transformer_layer_{idx}',
                    wave=wave,
                    is_encoder=False,
                    is_decoder=False
                ))

        # Pre-decoder representation
        if 'pre_decoder' in self.layer_outputs:
            tensor_repr = self.layer_outputs['pre_decoder']
            wave = Wave.from_representation(tensor_repr)
            snapshots.append(LayerWaveSnapshot(
                layer_idx=num_layers,
                layer_name='pre_decoder',
                wave=wave,
                is_decoder=True
            ))

        return snapshots

    def compare_layers(
        self,
        snapshots: List[LayerWaveSnapshot],
        batch_idx: int = 0,
        metrics: List[str] = ['amplitude_mean', 'frequency_mean', 'phase_std']
    ) -> Dict[str, np.ndarray]:
        """
        Compare wave evolution metrics across layers.

        Args:
            snapshots: List of LayerWaveSnapshot from analyze_input()
            batch_idx: Which batch element to analyze
            metrics: Metrics to compute. Options:
                - 'amplitude_mean': Mean amplitude across harmonics
                - 'amplitude_energy': Total energy (sum of squared amplitudes)
                - 'frequency_mean': Mean frequency across harmonics
                - 'frequency_std': Frequency diversity
                - 'phase_std': Phase diversity
                - 'spectral_centroid': Amplitude-weighted mean frequency

        Returns:
            Dictionary mapping metric names to arrays of shape [num_layers, seq_len]
        """
        results = {metric: [] for metric in metrics}

        for snapshot in snapshots:
            wave = snapshot.wave

            # Extract tensors for single batch element
            freqs = wave.frequencies[batch_idx].detach().cpu().numpy()  # [S, H]
            amps = wave.amplitudes[batch_idx].detach().cpu().numpy()    # [S, H]
            phases = wave.phases[batch_idx].detach().cpu().numpy()      # [S, H]

            for metric in metrics:
                if metric == 'amplitude_mean':
                    values = amps.mean(axis=1)  # [S]
                elif metric == 'amplitude_energy':
                    values = (amps ** 2).sum(axis=1)  # [S]
                elif metric == 'frequency_mean':
                    values = freqs.mean(axis=1)  # [S]
                elif metric == 'frequency_std':
                    values = freqs.std(axis=1)  # [S]
                elif metric == 'phase_std':
                    values = phases.std(axis=1)  # [S]
                elif metric == 'spectral_centroid':
                    # Amplitude-weighted mean frequency
                    total_amp = amps.sum(axis=1, keepdims=True) + 1e-8
                    values = (freqs * amps / total_amp).sum(axis=1)  # [S]
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                results[metric].append(values)

        # Convert to numpy arrays [num_layers, seq_len]
        for metric in metrics:
            results[metric] = np.array(results[metric])

        return results

    def plot_layer_evolution(
        self,
        snapshots: List[LayerWaveSnapshot],
        batch_idx: int = 0,
        seq_slice: Optional[Tuple[int, int]] = None,
        figsize: Tuple[int, int] = (18, 12),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize how wave components evolve through layers.

        Creates a comprehensive visualization showing:
        - Amplitude heatmaps across layers
        - Frequency evolution
        - Phase evolution
        - Energy distribution

        Args:
            snapshots: Layer snapshots from analyze_input()
            batch_idx: Which batch element to visualize
            seq_slice: Optional (start, end) for sequence subset
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Figure and axes array
        """
        num_layers = len(snapshots)

        # Determine sequence slice
        first_wave = snapshots[0].wave
        S = first_wave.frequencies.shape[1]
        if seq_slice is None:
            start, end = 0, min(S, 50)  # Default to first 50 positions
        else:
            start, end = seq_slice

        fig, axes = plt.subplots(3, num_layers, figsize=figsize, squeeze=False)

        for col_idx, snapshot in enumerate(snapshots):
            wave = snapshot.wave

            # Extract data for batch and sequence slice
            freqs = wave.frequencies[batch_idx, start:end].detach().cpu().numpy().T  # [H, S_slice]
            amps = wave.amplitudes[batch_idx, start:end].detach().cpu().numpy().T
            phases = wave.phases[batch_idx, start:end].detach().cpu().numpy().T

            # Row 0: Amplitude heatmap
            im0 = axes[0, col_idx].imshow(amps, aspect='auto', cmap='plasma', interpolation='nearest')
            axes[0, col_idx].set_title(f'{snapshot.layer_name}\nAmplitudes', fontsize=9)
            axes[0, col_idx].set_ylabel('Harmonic' if col_idx == 0 else '')
            plt.colorbar(im0, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

            # Row 1: Frequency heatmap
            im1 = axes[1, col_idx].imshow(freqs, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[1, col_idx].set_title('Frequencies', fontsize=9)
            axes[1, col_idx].set_ylabel('Harmonic' if col_idx == 0 else '')
            plt.colorbar(im1, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

            # Row 2: Phase heatmap
            im2 = axes[2, col_idx].imshow(phases, aspect='auto', cmap='twilight',
                                          vmin=-np.pi, vmax=np.pi, interpolation='nearest')
            axes[2, col_idx].set_title('Phases', fontsize=9)
            axes[2, col_idx].set_xlabel('Sequence Position')
            axes[2, col_idx].set_ylabel('Harmonic' if col_idx == 0 else '')
            plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.04)

        fig.suptitle(f'Wave Evolution Through Layers [Batch {batch_idx}, Pos {start}-{end}]',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def plot_metric_evolution(
        self,
        comparison_results: Dict[str, np.ndarray],
        layer_names: List[str],
        seq_positions: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot how specific metrics evolve across layers for selected sequence positions.

        Args:
            comparison_results: Output from compare_layers()
            layer_names: Names of layers for x-axis labels
            seq_positions: Sequence positions to track (or None for auto-select)
            figsize: Figure size
            save_path: Optional save path

        Returns:
            Figure and axes
        """
        num_metrics = len(comparison_results)

        # Auto-select sequence positions if not provided
        if seq_positions is None:
            seq_len = list(comparison_results.values())[0].shape[1]
            num_pos = min(5, seq_len)
            seq_positions = list(np.linspace(0, seq_len - 1, num_pos, dtype=int))

        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, (metric_name, metric_data) in enumerate(comparison_results.items()):
            ax = axes[idx]

            # metric_data shape: [num_layers, seq_len]
            for pos in seq_positions:
                values = metric_data[:, pos]
                ax.plot(range(len(layer_names)), values, marker='o',
                       label=f'Pos {pos}', alpha=0.7)

            ax.set_xlabel('Layer')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Evolution Across Layers')
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels(layer_names, rotation=45, ha='right')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, axes

    def __enter__(self):
        """Context manager entry"""
        self.register_extraction_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup hooks"""
        self.cleanup()
