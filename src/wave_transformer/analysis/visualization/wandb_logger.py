"""
Weights & Biases Integration for Wave Transformer Analysis

Provides utilities for logging experiments, visualizations, and results to W&B
with graceful handling of optional wandb dependency.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Union, Any
from pathlib import Path
import warnings

# Optional wandb import with graceful fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from wave_transformer.core.wave import Wave


class WaveWandbLogger:
    """
    Weights & Biases logger with specialized support for Wave representations.

    Provides methods to log:
    - Wave statistics and metrics
    - Visualizations (heatmaps, spectra, animations)
    - Generation examples with trajectories
    - Layer-wise analysis
    - Ablation study results as tables

    Note: Requires wandb to be installed. Install with: pip install wandb

    Args:
        project: W&B project name
        name: Run name (optional)
        config: Configuration dictionary to log
        tags: List of tags for the run
        notes: Notes about the run
        entity: W&B entity (username or team)
        reinit: Whether to allow reinitialization
        resume: Resume mode ('allow', 'must', 'never', or run_id)

    Example:
        >>> logger = WaveWandbLogger(project='wave-transformer', name='experiment-1')
        >>> logger.log_wave_statistics(wave, prefix='encoder', step=100)
        >>> logger.log_wave_visualizations(wave, prefix='layer_0', step=100)
        >>> logger.finish()
    """

    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        entity: Optional[str] = None,
        reinit: bool = False,
        resume: Optional[str] = None
    ):
        if not WANDB_AVAILABLE:
            warnings.warn(
                "wandb is not installed. Install with 'pip install wandb'. "
                "WaveWandbLogger will not log anything.",
                ImportWarning
            )
            self.enabled = False
            self.run = None
            return

        self.enabled = True

        # Initialize wandb run
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            entity=entity,
            reinit=reinit,
            resume=resume
        )

    def log_wave_statistics(
        self,
        wave: Wave,
        prefix: str = 'wave',
        step: Optional[int] = None,
        batch_idx: Optional[int] = None,
        component: str = 'all'
    ):
        """
        Log wave statistics to W&B.

        Args:
            wave: Wave object to analyze [B, S, H]
            prefix: Prefix for metric names (e.g., 'encoder', 'layer_0')
            step: Step number (optional, uses wandb's internal step if None)
            batch_idx: If specified, analyze only this batch element
            component: Which components to log ('all', 'frequencies', 'amplitudes', 'phases')
        """
        if not self.enabled:
            return

        from wave_transformer.analysis.core.wave_statistics import WaveStatistics

        # Compute basic statistics
        stats = WaveStatistics.compute_basic_stats(wave, component=component, dim=None)

        # Flatten statistics for logging
        log_dict = {}
        for comp_name, wave_stats in stats.items():
            if hasattr(wave_stats, 'to_dict'):
                stats_dict = wave_stats.to_dict()
                for stat_name, value in stats_dict.items():
                    log_dict[f'{prefix}/{comp_name}/{stat_name}'] = value

        # Compute additional spectral metrics
        spectral_centroid = WaveStatistics.compute_spectral_centroid(
            wave, batch_idx=batch_idx
        )
        log_dict[f'{prefix}/spectral_centroid_mean'] = spectral_centroid.mean().item()

        total_energy = WaveStatistics.compute_total_energy(
            wave, batch_idx=batch_idx, per_position=False
        )
        log_dict[f'{prefix}/total_energy'] = (
            total_energy.item() if torch.is_tensor(total_energy) else total_energy
        )

        entropy = WaveStatistics.compute_harmonic_entropy(wave, batch_idx=batch_idx)
        log_dict[f'{prefix}/harmonic_entropy_mean'] = entropy.mean().item()

        phase_coherence = WaveStatistics.compute_phase_coherence(wave, batch_idx=batch_idx)
        log_dict[f'{prefix}/phase_coherence_mean'] = phase_coherence.mean().item()

        # Log to W&B
        wandb.log(log_dict, step=step)

    def log_wave_visualizations(
        self,
        wave: Wave,
        prefix: str = 'wave',
        step: Optional[int] = None,
        batch_idx: int = 0,
        seq_position: int = 0,
        max_seq_len: Optional[int] = None
    ):
        """
        Log wave visualizations (heatmaps and spectra) to W&B.

        Args:
            wave: Wave object [B, S, H]
            prefix: Prefix for image names
            step: Step number
            batch_idx: Which batch element to visualize
            seq_position: Which sequence position for spectrum plot
            max_seq_len: Maximum sequence length for heatmaps
        """
        if not self.enabled:
            return

        log_dict = {}

        # Create heatmaps
        freqs = wave.frequencies[batch_idx].detach().cpu().numpy()  # [S, H]
        amps = wave.amplitudes[batch_idx].detach().cpu().numpy()
        phases = wave.phases[batch_idx].detach().cpu().numpy()

        if max_seq_len is not None:
            freqs = freqs[:max_seq_len]
            amps = amps[:max_seq_len]
            phases = phases[:max_seq_len]

        # Heatmap figures
        fig_freq = self._create_heatmap(freqs, 'Frequencies', 'viridis')
        fig_amp = self._create_heatmap(amps, 'Amplitudes', 'hot')
        fig_phase = self._create_heatmap(phases, 'Phases', 'twilight', vmin=-np.pi, vmax=np.pi)

        log_dict[f'{prefix}/frequencies_heatmap'] = wandb.Image(fig_freq)
        log_dict[f'{prefix}/amplitudes_heatmap'] = wandb.Image(fig_amp)
        log_dict[f'{prefix}/phases_heatmap'] = wandb.Image(fig_phase)

        plt.close(fig_freq)
        plt.close(fig_amp)
        plt.close(fig_phase)

        # Spectrum visualization
        fig_spectrum = self._create_spectrum_figure(wave, batch_idx, seq_position)
        log_dict[f'{prefix}/spectrum'] = wandb.Image(fig_spectrum)
        plt.close(fig_spectrum)

        # Log all images
        wandb.log(log_dict, step=step)

    def log_generation_example(
        self,
        input_text: str,
        generated_text: str,
        wave_trajectory: Optional[List[Wave]] = None,
        confidence_scores: Optional[np.ndarray] = None,
        prefix: str = 'generation',
        step: Optional[int] = None
    ):
        """
        Log a generation example with optional wave animation.

        Args:
            input_text: Input prompt
            generated_text: Generated output
            wave_trajectory: List of Wave objects at each generation step
            confidence_scores: Confidence scores per token [num_tokens]
            prefix: Prefix for log entries
            step: Step number
        """
        if not self.enabled:
            return

        log_dict = {
            f'{prefix}/input': input_text,
            f'{prefix}/output': generated_text
        }

        # Log confidence scores if available
        if confidence_scores is not None:
            log_dict[f'{prefix}/avg_confidence'] = confidence_scores.mean()
            log_dict[f'{prefix}/min_confidence'] = confidence_scores.min()

            # Create confidence plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(confidence_scores, marker='o', linewidth=2)
            ax.set_xlabel('Token Position')
            ax.set_ylabel('Confidence')
            ax.set_title('Generation Confidence')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            log_dict[f'{prefix}/confidence_plot'] = wandb.Image(fig)
            plt.close(fig)

        # Log wave trajectory as animation if available
        if wave_trajectory is not None and len(wave_trajectory) > 0:
            animation_frames = []
            for i, wave in enumerate(wave_trajectory):
                fig = self._create_wave_frame(wave, i)
                # Convert to image array
                fig.canvas.draw()
                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                animation_frames.append(img_array)
                plt.close(fig)

            # Log as video (requires imageio or similar)
            try:
                log_dict[f'{prefix}/wave_animation'] = wandb.Video(
                    np.array(animation_frames).transpose(0, 3, 1, 2),  # [T, C, H, W]
                    fps=2,
                    format='mp4'
                )
            except Exception as e:
                warnings.warn(f"Could not create wave animation: {e}")

        wandb.log(log_dict, step=step)

    def log_layer_analysis(
        self,
        layer_snapshots: List[Dict[str, Any]],
        prefix: str = 'layers',
        step: Optional[int] = None
    ):
        """
        Log layer-wise analysis metrics and visualizations.

        Args:
            layer_snapshots: List of dicts with 'layer_name' and 'wave'
            prefix: Prefix for log entries
            step: Step number
        """
        if not self.enabled:
            return

        from wave_transformer.analysis.core.wave_statistics import WaveStatistics

        # Compute metrics for each layer
        metrics_by_layer = []

        for snap in layer_snapshots:
            wave = snap['wave']
            layer_name = snap['layer_name']

            centroid = WaveStatistics.compute_spectral_centroid(wave).mean().item()
            energy = WaveStatistics.compute_total_energy(wave, per_position=False)
            energy = energy.item() if torch.is_tensor(energy) else energy
            entropy = WaveStatistics.compute_harmonic_entropy(wave).mean().item()

            metrics_by_layer.append({
                'layer': layer_name,
                'spectral_centroid': centroid,
                'total_energy': energy,
                'harmonic_entropy': entropy
            })

        # Create layer evolution plot
        fig = self._create_layer_evolution_plot(metrics_by_layer)

        wandb.log({
            f'{prefix}/evolution': wandb.Image(fig),
            f'{prefix}/num_layers': len(layer_snapshots)
        }, step=step)

        plt.close(fig)

    def log_ablation_results(
        self,
        results: List[Dict[str, Any]],
        table_name: str = 'ablation_study',
        step: Optional[int] = None
    ):
        """
        Log ablation study results as a W&B table.

        Args:
            results: List of dicts with ablation results
                Each dict should have keys like: 'config', 'metric', 'value'
            table_name: Name for the W&B table
            step: Step number
        """
        if not self.enabled:
            return

        # Create W&B table
        columns = list(results[0].keys()) if results else []
        table = wandb.Table(columns=columns)

        for result in results:
            row = [result.get(col, None) for col in columns]
            table.add_data(*row)

        wandb.log({table_name: table}, step=step)

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """
        Log arbitrary metrics dictionary.

        Args:
            metrics: Dictionary of metric name -> value
            step: Step number
        """
        if not self.enabled:
            return

        # Flatten nested dictionaries
        flattened = self._flatten_dict(metrics)
        wandb.log(flattened, step=step)

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '/'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary with separator."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _create_heatmap(
        self,
        data: np.ndarray,
        title: str,
        cmap: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> plt.Figure:
        """Create heatmap figure."""
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest',
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('Harmonic Index')
        ax.set_ylabel('Sequence Position')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig

    def _create_spectrum_figure(
        self,
        wave: Wave,
        batch_idx: int,
        seq_position: int
    ) -> plt.Figure:
        """Create spectrum visualization figure."""
        freqs = wave.frequencies[batch_idx, seq_position].detach().cpu().numpy()
        amps = wave.amplitudes[batch_idx, seq_position].detach().cpu().numpy()
        phases = wave.phases[batch_idx, seq_position].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Amplitude spectrum
        ax = axes[0]
        sorted_indices = np.argsort(freqs)
        ax.stem(freqs[sorted_indices], amps[sorted_indices], basefmt=' ')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Amplitude')
        ax.set_title('Amplitude Spectrum')
        ax.grid(True, alpha=0.3)

        # Amplitude histogram
        ax = axes[1]
        ax.hist(amps, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Count')
        ax.set_title('Amplitude Distribution')
        ax.grid(True, alpha=0.3)

        # Phase scatter
        ax = axes[2]
        scatter = ax.scatter(freqs, phases, c=amps, cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Phase (rad)')
        ax.set_title('Phase vs Frequency')
        ax.set_ylim(-np.pi, np.pi)
        plt.colorbar(scatter, ax=ax, label='Amplitude')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _create_wave_frame(self, wave: Wave, step_idx: int) -> plt.Figure:
        """Create a single frame for wave animation."""
        # Take first batch element, average across sequence
        amps = wave.amplitudes[0].mean(dim=0).detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(amps)), amps, color='steelblue', alpha=0.7)
        ax.set_xlabel('Harmonic Index')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Wave State - Step {step_idx}')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _create_layer_evolution_plot(
        self,
        metrics_by_layer: List[Dict[str, Any]]
    ) -> plt.Figure:
        """Create layer evolution visualization."""
        layer_names = [m['layer'] for m in metrics_by_layer]
        num_layers = len(layer_names)

        # Extract metrics
        centroids = [m['spectral_centroid'] for m in metrics_by_layer]
        energies = [m['total_energy'] for m in metrics_by_layer]
        entropies = [m['harmonic_entropy'] for m in metrics_by_layer]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Spectral centroid
        ax = axes[0]
        ax.plot(range(num_layers), centroids, marker='o', linewidth=2)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Spectral Centroid')
        ax.set_title('Centroid Evolution')
        ax.set_xticks(range(num_layers))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Total energy
        ax = axes[1]
        ax.plot(range(num_layers), energies, marker='o', linewidth=2, color='C1')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Total Energy')
        ax.set_title('Energy Evolution')
        ax.set_xticks(range(num_layers))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        # Harmonic entropy
        ax = axes[2]
        ax.plot(range(num_layers), entropies, marker='o', linewidth=2, color='C2')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Harmonic Entropy')
        ax.set_title('Entropy Evolution')
        ax.set_xticks(range(num_layers))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def finish(self):
        """Finish the W&B run."""
        if self.enabled and self.run is not None:
            wandb.finish()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
