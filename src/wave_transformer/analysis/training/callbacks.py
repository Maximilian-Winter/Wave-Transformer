"""
Training callbacks for Wave Transformer analysis.

This module provides callback classes that integrate with training loops to:
- Track wave statistics evolution during training
- Monitor gradient flow
- Analyze loss breakdown
- Visualize training dynamics

All callbacks follow a consistent lifecycle with hooks for different training events.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None

from wave_transformer.core.wave import Wave


class AnalysisCallback(ABC):
    """
    Abstract base class for analysis callbacks.

    All callbacks follow a consistent lifecycle:
    - on_train_begin: Called once at the start of training
    - on_epoch_begin: Called at the start of each epoch
    - on_batch_begin: Called before processing each batch
    - on_batch_end: Called after processing each batch
    - on_epoch_end: Called at the end of each epoch
    - on_train_end: Called once at the end of training
    """

    def __init__(self, frequency: int = 1):
        """
        Args:
            frequency: How often to run the callback (in steps/epochs)
        """
        self.frequency = frequency
        self.current_step = 0
        self.current_epoch = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the beginning of each epoch."""
        self.current_epoch = epoch

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called before processing each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None):
        """Called after processing each batch."""
        self.current_step += 1
        if self.current_step % self.frequency == 0:
            self._on_step(logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of each epoch."""
        if epoch % self.frequency == 0:
            self._on_epoch(logs)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None):
        """Called at the end of training."""
        pass

    @abstractmethod
    def _on_step(self, logs: Optional[Dict[str, Any]] = None):
        """Internal method called every `frequency` steps."""
        pass

    def _on_epoch(self, logs: Optional[Dict[str, Any]] = None):
        """Internal method called every `frequency` epochs."""
        pass


class WaveEvolutionCallback(AnalysisCallback):
    """
    Tracks evolution of wave statistics during training.

    Monitors how wave components (frequencies, amplitudes, phases) change
    over the course of training, helping identify:
    - Mode collapse (low variance)
    - Exploding/vanishing components
    - Convergence patterns
    """

    def __init__(
        self,
        model: nn.Module,
        wave_module_names: List[str],
        frequency: int = 100,
        track_gradients: bool = True,
        tb_writer: Optional[Any] = None,
        wandb_log: bool = False,
        save_dir: Optional[str] = None,
    ):
        """
        Args:
            model: The model being trained
            wave_module_names: Names of modules that output Wave objects
            frequency: How often to collect statistics (in steps)
            track_gradients: Whether to track gradient statistics
            tb_writer: TensorBoard SummaryWriter (optional)
            wandb_log: Whether to log to Weights & Biases
            save_dir: Directory to save statistics (optional)
        """
        super().__init__(frequency=frequency)
        self.model = model
        self.wave_module_names = wave_module_names
        self.track_gradients = track_gradients
        self.tb_writer = tb_writer
        self.wandb_log = wandb_log and wandb is not None
        self.save_dir = Path(save_dir) if save_dir else None

        # Statistics storage
        self.statistics_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        # Create save directory if needed
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self, logs: Optional[Dict[str, Any]] = None):
        """Collect and log wave statistics."""
        logs = logs or {}

        # Extract wave objects from logs or model
        wave_stats = self._collect_wave_statistics(logs)

        # Log statistics
        self._log_statistics(wave_stats)

        # Store in history
        for module_name, stats in wave_stats.items():
            self.statistics_history[module_name].append(stats)

    def _collect_wave_statistics(self, logs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Collect statistics from wave representations."""
        wave_stats = {}

        # Try to get waves from logs first
        for module_name in self.wave_module_names:
            if module_name in logs:
                wave = logs[module_name]
                if isinstance(wave, Wave):
                    wave_stats[module_name] = self._compute_wave_stats(wave)

        return wave_stats

    @staticmethod
    def _compute_wave_stats(wave: Wave) -> Dict[str, float]:
        """Compute comprehensive statistics for a Wave object."""
        freq = wave.frequencies.detach()
        amp = wave.amplitudes.detach()
        phase = wave.phases.detach()

        stats = {
            # Frequency statistics
            'freq_mean': freq.mean().item(),
            'freq_std': freq.std().item(),
            'freq_min': freq.min().item(),
            'freq_max': freq.max().item(),
            'freq_median': freq.median().item(),

            # Amplitude statistics
            'amp_mean': amp.mean().item(),
            'amp_std': amp.std().item(),
            'amp_min': amp.min().item(),
            'amp_max': amp.max().item(),
            'amp_median': amp.median().item(),
            'amp_energy': (amp ** 2).sum().item(),

            # Phase statistics
            'phase_mean': phase.mean().item(),
            'phase_std': phase.std().item(),
            'phase_min': phase.min().item(),
            'phase_max': phase.max().item(),

            # Cross-component statistics
            'freq_amp_corr': torch.corrcoef(torch.stack([freq.flatten(), amp.flatten()]))[0, 1].item(),
            'spectral_entropy': WaveEvolutionCallback._compute_spectral_entropy(amp),
        }

        return stats

    @staticmethod
    def _compute_spectral_entropy(amplitudes: torch.Tensor) -> float:
        """Compute spectral entropy to measure frequency distribution."""
        # Normalize amplitudes to get probability distribution
        amp_flat = amplitudes.flatten()
        amp_normalized = amp_flat / (amp_flat.sum() + 1e-10)

        # Compute entropy
        entropy = -(amp_normalized * torch.log(amp_normalized + 1e-10)).sum()
        return entropy.item()

    def _log_statistics(self, wave_stats: Dict[str, Dict[str, float]]):
        """Log statistics to TensorBoard and/or W&B."""
        for module_name, stats in wave_stats.items():
            prefix = f"wave_evolution/{module_name}"

            # Log to TensorBoard
            if self.tb_writer is not None:
                for stat_name, value in stats.items():
                    self.tb_writer.add_scalar(f"{prefix}/{stat_name}", value, self.current_step)

            # Log to W&B
            if self.wandb_log:
                wandb_dict = {f"{prefix}/{k}": v for k, v in stats.items()}
                wandb.log(wandb_dict, step=self.current_step)

    def _on_epoch(self, logs: Optional[Dict[str, Any]] = None):
        """Save statistics at epoch end."""
        if self.save_dir:
            # Save statistics history
            save_path = self.save_dir / f"wave_stats_epoch_{self.current_epoch}.pt"
            torch.save(self.statistics_history, save_path)

    def get_statistics_summary(self, module_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific module."""
        if module_name not in self.statistics_history:
            return {}

        history = self.statistics_history[module_name]
        if not history:
            return {}

        # Compute trends
        summary = {}
        for key in history[0].keys():
            values = [h[key] for h in history]
            summary[key] = {
                'current': values[-1],
                'mean': np.mean(values),
                'std': np.std(values),
                'trend': values[-1] - values[0] if len(values) > 1 else 0.0,
            }

        return summary


class GradientFlowCallback(AnalysisCallback):
    """
    Monitors gradient flow through the network.

    Tracks:
    - Gradient norms per layer
    - Gradient ratios (to detect vanishing/exploding gradients)
    - Wave component gradients separately
    """

    def __init__(
        self,
        model: nn.Module,
        frequency: int = 100,
        log_histogram: bool = False,
        tb_writer: Optional[Any] = None,
        wandb_log: bool = False,
        alert_on_anomaly: bool = True,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0,
    ):
        """
        Args:
            model: The model being trained
            frequency: How often to check gradients (in steps)
            log_histogram: Whether to log gradient histograms
            tb_writer: TensorBoard SummaryWriter
            wandb_log: Whether to log to W&B
            alert_on_anomaly: Whether to print warnings for gradient anomalies
            vanishing_threshold: Threshold for vanishing gradient detection
            exploding_threshold: Threshold for exploding gradient detection
        """
        super().__init__(frequency=frequency)
        self.model = model
        self.log_histogram = log_histogram
        self.tb_writer = tb_writer
        self.wandb_log = wandb_log and wandb is not None
        self.alert_on_anomaly = alert_on_anomaly
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold

        # Gradient statistics storage
        self.gradient_norms: Dict[str, List[float]] = defaultdict(list)
        self.gradient_ratios: List[float] = []

    def _on_step(self, logs: Optional[Dict[str, Any]] = None):
        """Collect and analyze gradients."""
        gradient_stats = self._collect_gradient_stats()

        # Log statistics
        self._log_gradient_stats(gradient_stats)

        # Check for anomalies
        if self.alert_on_anomaly:
            self._check_gradient_anomalies(gradient_stats)

    def _collect_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """Collect gradient statistics from model parameters."""
        stats = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()

                # Compute gradient norm
                grad_norm = grad.norm().item()
                self.gradient_norms[name].append(grad_norm)

                # Compute statistics
                stats[name] = {
                    'norm': grad_norm,
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                    'min': grad.abs().min().item(),
                }

                # Compute gradient-to-parameter ratio
                param_norm = param.data.norm().item()
                if param_norm > 0:
                    stats[name]['grad_param_ratio'] = grad_norm / param_norm

        # Compute layer-wise gradient ratios
        if len(stats) > 1:
            norms = [s['norm'] for s in stats.values()]
            if norms:
                max_norm = max(norms)
                min_norm = min([n for n in norms if n > 0] or [1e-10])
                ratio = max_norm / min_norm if min_norm > 0 else float('inf')
                self.gradient_ratios.append(ratio)
                stats['_global'] = {'norm_ratio': ratio}

        return stats

    def _log_gradient_stats(self, gradient_stats: Dict[str, Dict[str, float]]):
        """Log gradient statistics."""
        for param_name, stats in gradient_stats.items():
            if param_name.startswith('_'):
                continue

            prefix = f"gradient_flow/{param_name}"

            # Log to TensorBoard
            if self.tb_writer is not None:
                for stat_name, value in stats.items():
                    self.tb_writer.add_scalar(f"{prefix}/{stat_name}", value, self.current_step)

                # Log histograms
                if self.log_histogram:
                    param = dict(self.model.named_parameters())[param_name]
                    if param.grad is not None:
                        self.tb_writer.add_histogram(f"{prefix}/gradient", param.grad, self.current_step)

            # Log to W&B
            if self.wandb_log:
                wandb_dict = {f"{prefix}/{k}": v for k, v in stats.items()}
                wandb.log(wandb_dict, step=self.current_step)

        # Log global statistics
        if '_global' in gradient_stats:
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("gradient_flow/norm_ratio",
                                         gradient_stats['_global']['norm_ratio'],
                                         self.current_step)
            if self.wandb_log:
                wandb.log({"gradient_flow/norm_ratio": gradient_stats['_global']['norm_ratio']},
                         step=self.current_step)

    def _check_gradient_anomalies(self, gradient_stats: Dict[str, Dict[str, float]]):
        """Check for vanishing or exploding gradients."""
        for param_name, stats in gradient_stats.items():
            if param_name.startswith('_'):
                continue

            norm = stats['norm']

            if norm < self.vanishing_threshold:
                print(f"WARNING: Vanishing gradient detected in {param_name}: norm = {norm:.2e}")
            elif norm > self.exploding_threshold:
                print(f"WARNING: Exploding gradient detected in {param_name}: norm = {norm:.2e}")

    def get_gradient_flow_report(self) -> str:
        """Generate a text report of gradient flow."""
        report_lines = ["=" * 80, "Gradient Flow Report", "=" * 80]

        for param_name, norms in self.gradient_norms.items():
            if norms:
                avg_norm = np.mean(norms)
                std_norm = np.std(norms)
                min_norm = np.min(norms)
                max_norm = np.max(norms)

                report_lines.append(f"\n{param_name}:")
                report_lines.append(f"  Average norm: {avg_norm:.6e}")
                report_lines.append(f"  Std dev:      {std_norm:.6e}")
                report_lines.append(f"  Min norm:     {min_norm:.6e}")
                report_lines.append(f"  Max norm:     {max_norm:.6e}")

        if self.gradient_ratios:
            report_lines.append(f"\nGlobal gradient ratio (max/min): {np.mean(self.gradient_ratios):.2f}")

        report_lines.append("=" * 80)
        return "\n".join(report_lines)


class LossAnalysisCallback(AnalysisCallback):
    """
    Analyzes loss breakdown and training dynamics.

    Tracks:
    - Overall loss trends
    - Loss component breakdown (if available)
    - Loss variance and stability
    - Convergence metrics
    """

    def __init__(
        self,
        frequency: int = 10,
        track_components: bool = True,
        compute_moving_average: bool = True,
        window_size: int = 100,
        tb_writer: Optional[Any] = None,
        wandb_log: bool = False,
    ):
        """
        Args:
            frequency: How often to analyze loss (in steps)
            track_components: Whether to track individual loss components
            compute_moving_average: Whether to compute moving average
            window_size: Window size for moving average
            tb_writer: TensorBoard SummaryWriter
            wandb_log: Whether to log to W&B
        """
        super().__init__(frequency=frequency)
        self.track_components = track_components
        self.compute_moving_average = compute_moving_average
        self.window_size = window_size
        self.tb_writer = tb_writer
        self.wandb_log = wandb_log and wandb is not None

        # Loss tracking
        self.loss_history: List[float] = []
        self.component_history: Dict[str, List[float]] = defaultdict(list)
        self.moving_averages: List[float] = []

    def _on_step(self, logs: Optional[Dict[str, Any]] = None):
        """Analyze loss."""
        logs = logs or {}

        # Extract loss values
        loss = logs.get('loss', None)
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()

            self.loss_history.append(loss)

            # Compute moving average
            if self.compute_moving_average:
                window = self.loss_history[-self.window_size:]
                ma = np.mean(window)
                self.moving_averages.append(ma)

                # Log moving average
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("loss/moving_average", ma, self.current_step)
                if self.wandb_log:
                    wandb.log({"loss/moving_average": ma}, step=self.current_step)

        # Track loss components
        if self.track_components:
            for key, value in logs.items():
                if 'loss' in key.lower() and key != 'loss':
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    self.component_history[key].append(value)

                    # Log component
                    if self.tb_writer is not None:
                        self.tb_writer.add_scalar(f"loss_components/{key}", value, self.current_step)
                    if self.wandb_log:
                        wandb.log({f"loss_components/{key}": value}, step=self.current_step)

    def _on_epoch(self, logs: Optional[Dict[str, Any]] = None):
        """Compute and log epoch-level statistics."""
        if not self.loss_history:
            return

        # Get losses for this epoch (approximate)
        epoch_losses = self.loss_history[-len(self.loss_history) // (self.current_epoch + 1):]

        if epoch_losses:
            stats = {
                'mean': np.mean(epoch_losses),
                'std': np.std(epoch_losses),
                'min': np.min(epoch_losses),
                'max': np.max(epoch_losses),
            }

            # Log epoch statistics
            if self.tb_writer is not None:
                for stat_name, value in stats.items():
                    self.tb_writer.add_scalar(f"loss_epoch/{stat_name}", value, self.current_epoch)

            if self.wandb_log:
                wandb_dict = {f"loss_epoch/{k}": v for k, v in stats.items()}
                wandb.log(wandb_dict, step=self.current_step)

    def get_convergence_metrics(self) -> Dict[str, float]:
        """Compute convergence metrics."""
        if len(self.loss_history) < 2:
            return {}

        recent_window = min(100, len(self.loss_history) // 10)
        recent_losses = self.loss_history[-recent_window:]
        early_losses = self.loss_history[:recent_window]

        metrics = {
            'current_loss': self.loss_history[-1],
            'recent_mean': np.mean(recent_losses),
            'recent_std': np.std(recent_losses),
            'improvement': np.mean(early_losses) - np.mean(recent_losses) if early_losses else 0.0,
            'stability': np.std(recent_losses) / (np.mean(recent_losses) + 1e-10),
        }

        return metrics

    def plot_loss_curves(self, save_path: Optional[str] = None):
        """Plot loss curves (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot main loss
            axes[0].plot(self.loss_history, alpha=0.6, label='Loss')
            if self.moving_averages:
                axes[0].plot(self.moving_averages, linewidth=2, label='Moving Average')
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot loss components
            if self.component_history:
                for component_name, values in self.component_history.items():
                    axes[1].plot(values, label=component_name, alpha=0.7)
                axes[1].set_xlabel('Step')
                axes[1].set_ylabel('Component Loss')
                axes[1].set_title('Loss Components')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")
