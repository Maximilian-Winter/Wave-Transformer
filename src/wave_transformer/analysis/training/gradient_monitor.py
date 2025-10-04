"""
Gradient monitoring and visualization for Wave Transformer.

Provides comprehensive gradient flow analysis including:
- Gradient norm tracking per layer
- Vanishing/exploding gradient detection
- Component-wise gradient analysis (freq/amp/phase)
- Gradient flow visualization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, OrderedDict
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class GradientMonitor:
    """
    Monitors gradient flow through the network during training.

    Tracks gradient statistics, detects anomalies, and provides visualization
    tools for understanding gradient dynamics.

    Example usage:
        monitor = GradientMonitor(model)
        monitor.register_hooks()

        # Training loop
        for batch in dataloader:
            loss = model(batch)
            loss.backward()

            # Get gradient report
            report = monitor.get_gradient_flow_report()
            print(report)

            # Visualize gradient flow
            monitor.plot_gradient_flow(save_path='grad_flow.png')

            optimizer.step()
            optimizer.zero_grad()

        monitor.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        track_wave_components: bool = True,
        num_harmonics: Optional[int] = None,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0,
    ):
        """
        Args:
            model: The model to monitor
            track_wave_components: Whether to separate freq/amp/phase gradients
            num_harmonics: Number of harmonics (for component separation)
            vanishing_threshold: Threshold for vanishing gradient detection
            exploding_threshold: Threshold for exploding gradient detection
        """
        self.model = model
        self.track_wave_components = track_wave_components
        self.num_harmonics = num_harmonics
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold

        # Storage for gradient statistics
        self.gradient_norms: OrderedDict[str, List[float]] = OrderedDict()
        self.gradient_stats: OrderedDict[str, Dict[str, float]] = OrderedDict()
        self.component_norms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        # Hook handles for cleanup
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Anomaly tracking
        self.anomalies: List[Dict[str, Any]] = []

    def register_hooks(self):
        """Register gradient hooks on all model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                handle = param.register_hook(self._make_gradient_hook(name, param))
                self.hook_handles.append(handle)

    def _make_gradient_hook(self, name: str, param: nn.Parameter) -> callable:
        """Create a gradient hook for a specific parameter."""

        def hook(grad: torch.Tensor):
            if grad is None:
                return

            # Compute gradient norm
            grad_norm = grad.norm().item()

            # Initialize storage for this parameter
            if name not in self.gradient_norms:
                self.gradient_norms[name] = []

            self.gradient_norms[name].append(grad_norm)

            # Compute detailed statistics
            self.gradient_stats[name] = {
                'norm': grad_norm,
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item(),
                'min': grad.abs().min().item(),
            }

            # Compute gradient-to-parameter ratio
            param_norm = param.data.norm().item()
            if param_norm > 0:
                self.gradient_stats[name]['grad_param_ratio'] = grad_norm / param_norm

            # Check for anomalies
            self._check_anomaly(name, grad_norm)

            # Track wave components if enabled
            if self.track_wave_components and self.num_harmonics is not None:
                self._track_wave_components(name, grad)

        return hook

    def _check_anomaly(self, name: str, grad_norm: float):
        """Check for gradient anomalies (vanishing/exploding)."""
        if grad_norm < self.vanishing_threshold:
            self.anomalies.append({
                'type': 'vanishing',
                'parameter': name,
                'norm': grad_norm,
            })
        elif grad_norm > self.exploding_threshold:
            self.anomalies.append({
                'type': 'exploding',
                'parameter': name,
                'norm': grad_norm,
            })

    def _track_wave_components(self, name: str, grad: torch.Tensor):
        """Track gradients for frequency, amplitude, and phase components separately."""
        # Check if this gradient has the right shape for wave components
        if grad.shape[-1] == 3 * self.num_harmonics:
            try:
                # Split into components
                chunks = grad.chunk(3, dim=-1)
                components = {
                    'frequencies': chunks[0],
                    'amplitudes': chunks[1],
                    'phases': chunks[2]
                }

                # Compute and store component norms
                for comp_name, comp_grad in components.items():
                    comp_norm = comp_grad.norm().item()
                    self.component_norms[name][comp_name].append(comp_norm)

            except Exception:
                # If splitting fails, skip component tracking
                pass

    def remove_hooks(self):
        """Remove all registered gradient hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def get_gradient_flow_report(
        self,
        top_k: int = 10,
        include_components: bool = True,
    ) -> str:
        """
        Generate a comprehensive gradient flow report.

        Args:
            top_k: Number of top/bottom parameters to show
            include_components: Whether to include wave component analysis

        Returns:
            Formatted string report
        """
        if not self.gradient_stats:
            return "No gradient statistics available. Ensure hooks are registered and backward() has been called."

        lines = ["=" * 100]
        lines.append("GRADIENT FLOW REPORT")
        lines.append("=" * 100)

        # Overall statistics
        all_norms = [stats['norm'] for stats in self.gradient_stats.values()]
        lines.append(f"\nOverall Statistics:")
        lines.append(f"  Total parameters: {len(all_norms)}")
        lines.append(f"  Mean gradient norm: {np.mean(all_norms):.6e}")
        lines.append(f"  Std gradient norm: {np.std(all_norms):.6e}")
        lines.append(f"  Max gradient norm: {np.max(all_norms):.6e}")
        lines.append(f"  Min gradient norm: {np.min(all_norms):.6e}")

        # Gradient ratio (max/min)
        max_norm = np.max(all_norms)
        min_norm = np.min([n for n in all_norms if n > 0] or [1e-10])
        ratio = max_norm / min_norm if min_norm > 0 else float('inf')
        lines.append(f"  Gradient ratio (max/min): {ratio:.2f}")

        # Anomalies
        if self.anomalies:
            lines.append(f"\nAnomalies Detected: {len(self.anomalies)}")
            vanishing = sum(1 for a in self.anomalies if a['type'] == 'vanishing')
            exploding = sum(1 for a in self.anomalies if a['type'] == 'exploding')
            lines.append(f"  Vanishing gradients: {vanishing}")
            lines.append(f"  Exploding gradients: {exploding}")

            # Show recent anomalies
            if self.anomalies:
                lines.append(f"\nRecent Anomalies:")
                for anomaly in self.anomalies[-5:]:
                    lines.append(f"  {anomaly['type'].upper()}: {anomaly['parameter']} (norm={anomaly['norm']:.2e})")

        # Top parameters by gradient norm
        sorted_params = sorted(self.gradient_stats.items(), key=lambda x: x[1]['norm'], reverse=True)

        lines.append(f"\nTop {top_k} Parameters by Gradient Norm:")
        for i, (name, stats) in enumerate(sorted_params[:top_k]):
            lines.append(f"  {i+1}. {name}")
            lines.append(f"     Norm: {stats['norm']:.6e}, Mean: {stats['mean']:.6e}, "
                        f"Std: {stats['std']:.6e}, Ratio: {stats.get('grad_param_ratio', 0.0):.6e}")

        # Bottom parameters by gradient norm
        lines.append(f"\nBottom {top_k} Parameters by Gradient Norm:")
        for i, (name, stats) in enumerate(reversed(sorted_params[-top_k:])):
            lines.append(f"  {i+1}. {name}")
            lines.append(f"     Norm: {stats['norm']:.6e}, Mean: {stats['mean']:.6e}, "
                        f"Std: {stats['std']:.6e}, Ratio: {stats.get('grad_param_ratio', 0.0):.6e}")

        # Wave component analysis
        if include_components and self.component_norms:
            lines.append(f"\nWave Component Gradient Analysis:")
            for param_name, components in self.component_norms.items():
                if components:
                    lines.append(f"\n  {param_name}:")
                    for comp_name, norms in components.items():
                        if norms:
                            mean_norm = np.mean(norms)
                            lines.append(f"    {comp_name}: {mean_norm:.6e}")

        lines.append("=" * 100)
        return "\n".join(lines)

    def plot_gradient_flow(
        self,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        show_components: bool = True,
    ):
        """
        Visualize gradient flow through the network.

        Creates a comprehensive visualization including:
        - Gradient norms per layer (bar plot)
        - Gradient norm distribution (histogram)
        - Gradient evolution over time (line plot)
        - Component-wise gradients (if available)

        Args:
            figsize: Figure size
            save_path: Path to save the figure (optional)
            show_components: Whether to show wave component analysis
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available for plotting")
            return

        if not self.gradient_stats:
            print("No gradient statistics to plot")
            return

        # Determine subplot layout
        n_plots = 4 if (show_components and self.component_norms) else 3
        fig = plt.figure(figsize=figsize)

        # Plot 1: Gradient norms per parameter (bar plot)
        ax1 = plt.subplot(2, 2, 1)
        param_names = list(self.gradient_stats.keys())
        grad_norms = [self.gradient_stats[name]['norm'] for name in param_names]

        # Shorten parameter names for display
        short_names = [self._shorten_name(name) for name in param_names]

        ax1.barh(range(len(short_names)), grad_norms, alpha=0.7, color='steelblue')
        ax1.set_yticks(range(len(short_names)))
        ax1.set_yticklabels(short_names, fontsize=6)
        ax1.set_xlabel('Gradient Norm', fontsize=10)
        ax1.set_title('Gradient Norms per Parameter', fontsize=12, fontweight='bold')
        ax1.axvline(self.vanishing_threshold, color='red', linestyle='--', alpha=0.5, label='Vanishing threshold')
        ax1.axvline(self.exploding_threshold, color='orange', linestyle='--', alpha=0.5, label='Exploding threshold')
        ax1.set_xscale('log')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis='x')

        # Plot 2: Gradient norm distribution
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(grad_norms, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(np.mean(grad_norms), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(grad_norms):.2e}')
        ax2.set_xlabel('Gradient Norm', fontsize=10)
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Gradient Norm Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Gradient evolution over time
        ax3 = plt.subplot(2, 2, 3)
        # Plot gradient norms over time for select parameters
        num_params_to_plot = min(10, len(self.gradient_norms))
        for i, (name, norms) in enumerate(list(self.gradient_norms.items())[:num_params_to_plot]):
            ax3.plot(norms, label=self._shorten_name(name), alpha=0.7)

        ax3.set_xlabel('Step', fontsize=10)
        ax3.set_ylabel('Gradient Norm', fontsize=10)
        ax3.set_title('Gradient Evolution Over Time', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=6, ncol=2)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Wave component analysis (if available)
        if show_components and self.component_norms:
            ax4 = plt.subplot(2, 2, 4)

            # Collect component data
            component_data = defaultdict(list)
            component_labels = []

            for param_name, components in self.component_norms.items():
                for comp_name, norms in components.items():
                    if norms:
                        component_data[comp_name].append(np.mean(norms))
                        if param_name not in component_labels:
                            component_labels.append(self._shorten_name(param_name))

            # Create grouped bar plot
            if component_data:
                x = np.arange(len(component_labels))
                width = 0.25
                colors = {'frequencies': 'steelblue', 'amplitudes': 'coral', 'phases': 'mediumseagreen'}

                for i, (comp_name, values) in enumerate(component_data.items()):
                    offset = width * (i - 1)
                    ax4.bar(x + offset, values[:len(component_labels)], width,
                           label=comp_name, alpha=0.7, color=colors.get(comp_name, 'gray'))

                ax4.set_xlabel('Parameter', fontsize=10)
                ax4.set_ylabel('Mean Gradient Norm', fontsize=10)
                ax4.set_title('Wave Component Gradients', fontsize=12, fontweight='bold')
                ax4.set_xticks(x)
                ax4.set_xticklabels(component_labels, rotation=45, ha='right', fontsize=6)
                ax4.legend(fontsize=8)
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Gradient flow plot saved to {save_path}")
        else:
            plt.show()

        return fig

    @staticmethod
    def _shorten_name(name: str, max_length: int = 30) -> str:
        """Shorten parameter names for display."""
        if len(name) <= max_length:
            return name

        # Try to keep the most informative parts
        parts = name.split('.')
        if len(parts) > 2:
            return f"{parts[0]}...{parts[-1]}"
        return name[:max_length-3] + "..."

    def get_layer_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get gradient statistics organized by layer.

        Returns:
            Dictionary mapping layer names to their gradient statistics
        """
        layer_stats = defaultdict(lambda: {'norms': [], 'means': [], 'stds': []})

        for param_name, stats in self.gradient_stats.items():
            # Extract layer name (first part before first dot)
            layer_name = param_name.split('.')[0] if '.' in param_name else param_name

            layer_stats[layer_name]['norms'].append(stats['norm'])
            layer_stats[layer_name]['means'].append(stats['mean'])
            layer_stats[layer_name]['stds'].append(stats['std'])

        # Compute aggregated statistics per layer
        aggregated_stats = {}
        for layer_name, stats in layer_stats.items():
            aggregated_stats[layer_name] = {
                'mean_norm': np.mean(stats['norms']),
                'max_norm': np.max(stats['norms']),
                'min_norm': np.min(stats['norms']),
                'mean_mean': np.mean(stats['means']),
                'mean_std': np.mean(stats['stds']),
            }

        return aggregated_stats

    def reset_statistics(self):
        """Reset all collected statistics."""
        self.gradient_norms.clear()
        self.gradient_stats.clear()
        self.component_norms.clear()
        self.anomalies.clear()

    def save_statistics(self, save_path: str):
        """Save gradient statistics to disk."""
        save_data = {
            'gradient_norms': dict(self.gradient_norms),
            'gradient_stats': dict(self.gradient_stats),
            'component_norms': dict(self.component_norms),
            'anomalies': self.anomalies,
        }
        torch.save(save_data, save_path)
        print(f"Gradient statistics saved to {save_path}")

    def load_statistics(self, load_path: str):
        """Load gradient statistics from disk."""
        save_data = torch.load(load_path)
        self.gradient_norms = OrderedDict(save_data['gradient_norms'])
        self.gradient_stats = OrderedDict(save_data['gradient_stats'])
        self.component_norms = defaultdict(lambda: defaultdict(list), save_data['component_norms'])
        self.anomalies = save_data['anomalies']
        print(f"Gradient statistics loaded from {load_path}")

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
