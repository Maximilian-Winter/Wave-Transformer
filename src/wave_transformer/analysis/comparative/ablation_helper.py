"""
Ablation Helper

Systematic ablation studies to understand component importance in Wave Transformers.
Supports ablating harmonics, layers, and wave components.
"""

import copy
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from wave_transformer.core.wave import Wave


class AblationHelper:
    """
    Perform systematic ablation studies on Wave Transformer models.

    This class provides methods to:
    - Ablate specific harmonics (zero, randomize, mean)
    - Ablate layers (convert to identity/skip connections)
    - Ablate entire wave components (freq/amp/phase)
    - Run comprehensive ablation studies with multiple configurations
    - Restore model to original state
    - Visualize ablation impact

    Args:
        model: WaveTransformer model to ablate
        device: Device to perform computations on

    Example:
        >>> ablator = AblationHelper(model)
        >>> results = ablator.run_ablation_study(
        ...     ablation_configs=[
        ...         {'type': 'harmonics', 'indices': [0, 1, 2], 'mode': 'zero'},
        ...         {'type': 'layer', 'layer_idx': 2, 'mode': 'identity'}
        ...     ],
        ...     eval_fn=evaluate_model,
        ...     dataloader=val_loader
        ... )
        >>> ablator.plot_ablation_results(results)
        >>> ablator.restore_model()
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)

        # Store original state for restoration
        self.original_state_dict = None
        self._save_original_state()

        # Track current ablations
        self.active_ablations = []

    def _save_original_state(self):
        """Save original model state for restoration"""
        self.original_state_dict = copy.deepcopy(self.model.state_dict())

    def restore_model(self):
        """Restore model to original state before any ablations"""
        if self.original_state_dict is not None:
            self.model.load_state_dict(self.original_state_dict)
            self.active_ablations = []
        else:
            warnings.warn("No original state saved. Cannot restore.")

    def ablate_harmonics(
        self,
        harmonic_indices: List[int],
        mode: str = 'zero',
        component: str = 'all'
    ) -> None:
        """
        Ablate specific harmonics in the wave encoder.

        Args:
            harmonic_indices: List of harmonic indices to ablate
            mode: Ablation mode:
                 - 'zero': Set to 0
                 - 'random': Randomize with same distribution
                 - 'mean': Replace with mean value
                 - 'noise': Add Gaussian noise
            component: Which component to ablate:
                      - 'all': All components (freq, amp, phase)
                      - 'freq': Only frequencies
                      - 'amp': Only amplitudes
                      - 'phase': Only phases

        Note: This modifies the encoder's projection layers to zero out
              specific harmonic outputs.
        """
        encoder = self.model.wave_encoder

        # Create hooks to modify encoder output
        def create_ablation_hook(mode, component, indices):
            def hook(module, input, output):
                if not isinstance(output, Wave):
                    return output

                # Create modified wave
                new_freq = output.frequencies.clone()
                new_amp = output.amplitudes.clone()
                new_phase = output.phases.clone()

                B, S, H = new_freq.shape

                for idx in indices:
                    if idx >= H:
                        warnings.warn(f"Harmonic index {idx} out of range (max {H-1})")
                        continue

                    if component in ['all', 'freq']:
                        if mode == 'zero':
                            new_freq[:, :, idx] = 0.0
                        elif mode == 'mean':
                            new_freq[:, :, idx] = new_freq[:, :, idx].mean()
                        elif mode == 'random':
                            new_freq[:, :, idx] = torch.randn_like(new_freq[:, :, idx]) * new_freq[:, :, idx].std()
                        elif mode == 'noise':
                            new_freq[:, :, idx] += torch.randn_like(new_freq[:, :, idx]) * 0.1

                    if component in ['all', 'amp']:
                        if mode == 'zero':
                            new_amp[:, :, idx] = 0.0
                        elif mode == 'mean':
                            new_amp[:, :, idx] = new_amp[:, :, idx].mean()
                        elif mode == 'random':
                            new_amp[:, :, idx] = torch.randn_like(new_amp[:, :, idx]).abs() * new_amp[:, :, idx].std()
                        elif mode == 'noise':
                            new_amp[:, :, idx] += torch.randn_like(new_amp[:, :, idx]) * 0.1

                    if component in ['all', 'phase']:
                        if mode == 'zero':
                            new_phase[:, :, idx] = 0.0
                        elif mode == 'mean':
                            new_phase[:, :, idx] = new_phase[:, :, idx].mean()
                        elif mode == 'random':
                            new_phase[:, :, idx] = torch.randn_like(new_phase[:, :, idx]) * np.pi
                        elif mode == 'noise':
                            new_phase[:, :, idx] += torch.randn_like(new_phase[:, :, idx]) * 0.1

                return Wave(new_freq, new_amp, new_phase)

            return hook

        # Register hook
        handle = encoder.register_forward_hook(
            create_ablation_hook(mode, component, harmonic_indices)
        )

        self.active_ablations.append({
            'type': 'harmonics',
            'indices': harmonic_indices,
            'mode': mode,
            'component': component,
            'handle': handle
        })

    def ablate_layers(
        self,
        layer_indices: List[int],
        mode: str = 'identity'
    ) -> None:
        """
        Ablate transformer layers by converting them to identity or skip connections.

        Args:
            layer_indices: List of layer indices to ablate
            mode: Ablation mode:
                 - 'identity': Make layer output = input (skip connection)
                 - 'zero': Zero out layer output
                 - 'random': Replace with random noise matching input distribution

        Note: This uses hooks to modify layer outputs during forward pass.
        """
        def create_layer_hook(mode):
            def hook(module, input, output):
                input_tensor = input[0] if isinstance(input, tuple) else input

                if mode == 'identity':
                    return input_tensor
                elif mode == 'zero':
                    return torch.zeros_like(output)
                elif mode == 'random':
                    return torch.randn_like(output) * output.std()
                else:
                    return output

            return hook

        for layer_idx in layer_indices:
            if layer_idx >= len(self.model.layers):
                warnings.warn(f"Layer index {layer_idx} out of range (max {len(self.model.layers)-1})")
                continue

            layer = self.model.layers[layer_idx]
            handle = layer.register_forward_hook(create_layer_hook(mode))

            self.active_ablations.append({
                'type': 'layer',
                'layer_idx': layer_idx,
                'mode': mode,
                'handle': handle
            })

    def ablate_wave_component(
        self,
        component: str,
        mode: str = 'zero'
    ) -> None:
        """
        Ablate entire wave component (frequencies, amplitudes, or phases).

        Args:
            component: Component to ablate ('freq', 'amp', or 'phase')
            mode: Ablation mode:
                 - 'zero': Set all values to 0
                 - 'mean': Replace with mean value
                 - 'random': Randomize with same distribution
                 - 'constant': Set to a constant value

        Note: This completely removes information from one wave component.
        """
        encoder = self.model.wave_encoder

        def create_component_hook(component, mode):
            def hook(module, input, output):
                if not isinstance(output, Wave):
                    return output

                new_freq = output.frequencies.clone()
                new_amp = output.amplitudes.clone()
                new_phase = output.phases.clone()

                if component == 'freq':
                    if mode == 'zero':
                        new_freq = torch.zeros_like(new_freq)
                    elif mode == 'mean':
                        new_freq = torch.full_like(new_freq, new_freq.mean())
                    elif mode == 'random':
                        new_freq = torch.randn_like(new_freq) * new_freq.std() + new_freq.mean()
                    elif mode == 'constant':
                        new_freq = torch.full_like(new_freq, 1.0)

                elif component == 'amp':
                    if mode == 'zero':
                        new_amp = torch.zeros_like(new_amp)
                    elif mode == 'mean':
                        new_amp = torch.full_like(new_amp, new_amp.mean())
                    elif mode == 'random':
                        new_amp = torch.randn_like(new_amp).abs() * new_amp.std() + new_amp.mean()
                    elif mode == 'constant':
                        new_amp = torch.full_like(new_amp, 1.0)

                elif component == 'phase':
                    if mode == 'zero':
                        new_phase = torch.zeros_like(new_phase)
                    elif mode == 'mean':
                        new_phase = torch.full_like(new_phase, new_phase.mean())
                    elif mode == 'random':
                        new_phase = torch.randn_like(new_phase) * np.pi
                    elif mode == 'constant':
                        new_phase = torch.zeros_like(new_phase)

                return Wave(new_freq, new_amp, new_phase)

            return hook

        handle = encoder.register_forward_hook(
            create_component_hook(component, mode)
        )

        self.active_ablations.append({
            'type': 'wave_component',
            'component': component,
            'mode': mode,
            'handle': handle
        })

    def clear_ablations(self):
        """Remove all active ablation hooks"""
        for ablation in self.active_ablations:
            if 'handle' in ablation:
                ablation['handle'].remove()
        self.active_ablations = []

    def run_ablation_study(
        self,
        ablation_configs: List[Dict[str, Any]],
        eval_fn: Optional[Callable] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        metrics: Optional[List[str]] = None,
        include_baseline: bool = True
    ) -> pd.DataFrame:
        """
        Run systematic ablation study with multiple configurations.

        Args:
            ablation_configs: List of ablation configurations. Each config is a dict:
                             {'type': 'harmonics'|'layer'|'wave_component', ...}
            eval_fn: Evaluation function that takes (model, dataloader) and returns
                    a dict of metrics. If None, uses default evaluation.
            dataloader: DataLoader for evaluation
            metrics: List of metric names to track
            include_baseline: If True, include baseline (no ablation) results

        Returns:
            pandas DataFrame with ablation results

        Example config:
            [
                {'type': 'harmonics', 'indices': [0, 1, 2], 'mode': 'zero'},
                {'type': 'layer', 'layer_idx': 2, 'mode': 'identity'},
                {'type': 'wave_component', 'component': 'phase', 'mode': 'zero'}
            ]
        """
        if eval_fn is None:
            eval_fn = self._default_eval_fn

        results = []

        # Baseline evaluation (no ablation)
        if include_baseline:
            self.restore_model()
            self.clear_ablations()

            baseline_metrics = eval_fn(self.model, dataloader)
            result_row = {
                'ablation_id': 0,
                'ablation_type': 'baseline',
                'description': 'No ablation (baseline)',
                **baseline_metrics
            }
            results.append(result_row)

        # Run each ablation configuration
        for config_idx, config in enumerate(ablation_configs):
            # Restore model and clear previous ablations
            self.restore_model()
            self.clear_ablations()

            # Apply ablation
            ablation_type = config['type']

            if ablation_type == 'harmonics':
                self.ablate_harmonics(
                    harmonic_indices=config['indices'],
                    mode=config.get('mode', 'zero'),
                    component=config.get('component', 'all')
                )
                description = f"Harmonics {config['indices']} - {config.get('mode', 'zero')}"

            elif ablation_type == 'layer':
                if isinstance(config.get('layer_idx'), list):
                    layer_indices = config['layer_idx']
                else:
                    layer_indices = [config['layer_idx']]

                self.ablate_layers(
                    layer_indices=layer_indices,
                    mode=config.get('mode', 'identity')
                )
                description = f"Layer(s) {layer_indices} - {config.get('mode', 'identity')}"

            elif ablation_type == 'wave_component':
                self.ablate_wave_component(
                    component=config['component'],
                    mode=config.get('mode', 'zero')
                )
                description = f"Component {config['component']} - {config.get('mode', 'zero')}"

            else:
                warnings.warn(f"Unknown ablation type: {ablation_type}")
                continue

            # Evaluate
            ablation_metrics = eval_fn(self.model, dataloader)

            result_row = {
                'ablation_id': config_idx + 1,
                'ablation_type': ablation_type,
                'description': description,
                **ablation_metrics
            }
            results.append(result_row)

        # Restore model after all ablations
        self.restore_model()
        self.clear_ablations()

        # Create DataFrame
        df = pd.DataFrame(results)

        return df

    @torch.no_grad()
    def _default_eval_fn(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Default evaluation function that computes basic metrics.

        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation

        Returns:
            Dictionary of metrics
        """
        model.eval()

        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        total_tokens = 0

        criterion = nn.CrossEntropyLoss(reduction='mean')

        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, dict):
                encoder_input = {k: v.to(self.device) if torch.is_tensor(v) else v
                               for k, v in batch.items()
                               if k not in ['attention_mask', 'labels', 'targets']}
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Get targets (try different common key names)
                targets = None
                for key in ['labels', 'targets', 'target', 'token_ids']:
                    if key in batch:
                        targets = batch[key].to(self.device)
                        break
            else:
                # Assume batch is (input, target) tuple or just input
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    encoder_input = {'token_ids': batch[0].to(self.device)}
                    targets = batch[1].to(self.device)
                    attention_mask = None
                else:
                    encoder_input = {'token_ids': batch.to(self.device)}
                    targets = batch.to(self.device)
                    attention_mask = None

            # Forward pass
            try:
                logits = model(encoder_input, attention_mask=attention_mask)

                # Compute loss if targets available
                if targets is not None:
                    # Reshape for loss computation
                    if logits.dim() == 3:  # [B, S, V]
                        B, S, V = logits.shape
                        logits_flat = logits.view(-1, V)
                        targets_flat = targets.view(-1)

                        # Filter out padding if needed
                        if attention_mask is not None:
                            mask_flat = attention_mask.view(-1).bool()
                            logits_flat = logits_flat[mask_flat]
                            targets_flat = targets_flat[mask_flat]

                        loss = criterion(logits_flat, targets_flat)
                        total_loss += loss.item() * targets_flat.size(0)

                        # Compute accuracy
                        predictions = logits_flat.argmax(dim=-1)
                        total_correct += (predictions == targets_flat).sum().item()
                        total_tokens += targets_flat.size(0)
                    else:
                        loss = criterion(logits, targets)
                        total_loss += loss.item() * targets.size(0)

                total_samples += targets.size(0) if targets is not None else encoder_input['token_ids'].size(0)

            except Exception as e:
                warnings.warn(f"Error during evaluation: {e}")
                continue

        # Compute final metrics
        metrics = {}

        if total_samples > 0:
            metrics['avg_loss'] = total_loss / total_samples if total_samples > 0 else float('inf')

        if total_tokens > 0:
            metrics['perplexity'] = np.exp(total_loss / total_tokens)
            metrics['accuracy'] = total_correct / total_tokens

        return metrics

    def plot_ablation_results(
        self,
        results_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize ablation study results.

        Args:
            results_df: DataFrame from run_ablation_study()
            metrics: List of metric columns to plot (None = all numeric columns)
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        # Identify metric columns
        if metrics is None:
            metrics = [col for col in results_df.columns
                      if col not in ['ablation_id', 'ablation_type', 'description']
                      and pd.api.types.is_numeric_dtype(results_df[col])]

        if not metrics:
            warnings.warn("No numeric metrics found in results DataFrame")
            return None

        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Get baseline value
            baseline_row = results_df[results_df['ablation_type'] == 'baseline']
            baseline_value = baseline_row[metric].values[0] if len(baseline_row) > 0 else None

            # Plot bars
            ablation_data = results_df[results_df['ablation_type'] != 'baseline']
            x_pos = range(len(ablation_data))

            bars = ax.bar(x_pos, ablation_data[metric].values, alpha=0.7)

            # Color bars by performance relative to baseline
            if baseline_value is not None:
                for i, (bar, value) in enumerate(zip(bars, ablation_data[metric].values)):
                    if 'loss' in metric.lower() or 'perplexity' in metric.lower():
                        # Lower is better
                        if value < baseline_value:
                            bar.set_color('green')
                        elif value > baseline_value:
                            bar.set_color('red')
                    else:
                        # Higher is better (e.g., accuracy)
                        if value > baseline_value:
                            bar.set_color('green')
                        elif value < baseline_value:
                            bar.set_color('red')

                # Draw baseline line
                ax.axhline(baseline_value, color='blue', linestyle='--',
                          linewidth=2, label=f'Baseline: {baseline_value:.4f}')

            ax.set_xlabel('Ablation Configuration')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} - Ablation Impact',
                        fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(ablation_data['description'].values,
                              rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()

        fig.suptitle('Ablation Study Results', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_impact_heatmap(
        self,
        results_df: pd.DataFrame,
        metric: str = 'avg_loss',
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create heatmap showing relative impact of each ablation.

        Args:
            results_df: DataFrame from run_ablation_study()
            metric: Metric to visualize
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            Figure object
        """
        # Get baseline value
        baseline_row = results_df[results_df['ablation_type'] == 'baseline']
        baseline_value = baseline_row[metric].values[0] if len(baseline_row) > 0 else 0

        # Compute relative change from baseline
        ablation_data = results_df[results_df['ablation_type'] != 'baseline'].copy()
        ablation_data['relative_change'] = (
            (ablation_data[metric] - baseline_value) / (abs(baseline_value) + 1e-8)
        ) * 100

        # Create matrix for heatmap (group by ablation type)
        types = ablation_data['ablation_type'].unique()
        matrix_data = []
        labels = []

        for ablation_type in types:
            type_data = ablation_data[ablation_data['ablation_type'] == ablation_type]
            for _, row in type_data.iterrows():
                matrix_data.append(row['relative_change'])
                labels.append(row['description'])

        # Create single-row heatmap
        matrix = np.array(matrix_data).reshape(1, -1)

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r' if 'loss' in metric.lower() else 'RdYlGn',
            xticklabels=labels,
            yticklabels=['Impact'],
            ax=ax,
            cbar_kws={'label': f'% Change in {metric}'},
            center=0
        )

        ax.set_title(f'Ablation Impact on {metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def __enter__(self):
        """Context manager entry"""
        self._save_original_state()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore model"""
        self.restore_model()
        self.clear_ablations()
