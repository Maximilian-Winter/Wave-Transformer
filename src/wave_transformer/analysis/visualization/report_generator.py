"""
Publication-Quality Report Generator for Wave Transformer Analysis

Generates publication-ready figures and LaTeX tables for research papers.
Supports IEEE, Nature, Science, and arXiv formatting styles.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Optional, List, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from wave_transformer.core.wave import Wave


class PublicationStyle(Enum):
    """Publication formatting styles"""
    IEEE = 'ieee'
    NATURE = 'nature'
    SCIENCE = 'science'
    ARXIV = 'arxiv'
    CUSTOM = 'custom'


@dataclass
class FigureConfig:
    """Configuration for publication figures"""
    style: PublicationStyle = PublicationStyle.IEEE
    font_size: int = 10
    figure_width: float = 3.5  # inches (IEEE 2-column)
    figure_height: float = 2.5  # inches
    dpi: int = 300
    use_tex: bool = False
    color_scheme: str = 'tab10'


class PaperReportGenerator:
    """
    Generate publication-quality reports and figures for Wave Transformer analysis.

    Provides methods to create:
    - Training curves (loss, perplexity over time)
    - Layer analysis figures (evolution visualization)
    - Harmonic importance figures (ranking and distribution)
    - Generation analysis (trajectory + confidence)
    - Comparison figures (checkpoint/input comparison)
    - LaTeX tables for results

    Args:
        output_dir: Directory for saving figures and reports
        style: Publication style (IEEE, Nature, Science, arXiv, or custom)
        config: Optional FigureConfig for custom styling

    Example:
        >>> generator = PaperReportGenerator(
        ...     output_dir='paper_figures',
        ...     style=PublicationStyle.IEEE
        ... )
        >>> generator.create_training_curve_figure(
        ...     train_losses, val_losses, save_name='training_curves.pdf'
        ... )
        >>> generator.generate_full_report(all_analysis_data)
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        style: PublicationStyle = PublicationStyle.IEEE,
        config: Optional[FigureConfig] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.style = style
        self.config = config if config else self._get_default_config(style)

        # Setup matplotlib style
        self._setup_matplotlib_style()

    def _get_default_config(self, style: PublicationStyle) -> FigureConfig:
        """Get default configuration for publication style."""
        configs = {
            PublicationStyle.IEEE: FigureConfig(
                style=PublicationStyle.IEEE,
                font_size=10,
                figure_width=3.5,  # 2-column width
                figure_height=2.5,
                dpi=300,
                use_tex=False
            ),
            PublicationStyle.NATURE: FigureConfig(
                style=PublicationStyle.NATURE,
                font_size=8,
                figure_width=3.5,
                figure_height=2.5,
                dpi=600,
                use_tex=False
            ),
            PublicationStyle.SCIENCE: FigureConfig(
                style=PublicationStyle.SCIENCE,
                font_size=8,
                figure_width=3.5,
                figure_height=2.5,
                dpi=600,
                use_tex=False
            ),
            PublicationStyle.ARXIV: FigureConfig(
                style=PublicationStyle.ARXIV,
                font_size=11,
                figure_width=6.0,
                figure_height=4.0,
                dpi=150,
                use_tex=False
            )
        }
        return configs.get(style, FigureConfig())

    def _setup_matplotlib_style(self):
        """Configure matplotlib for publication-quality output."""
        # Font settings
        mpl.rcParams['font.size'] = self.config.font_size
        mpl.rcParams['axes.labelsize'] = self.config.font_size
        mpl.rcParams['axes.titlesize'] = self.config.font_size + 1
        mpl.rcParams['xtick.labelsize'] = self.config.font_size - 1
        mpl.rcParams['ytick.labelsize'] = self.config.font_size - 1
        mpl.rcParams['legend.fontsize'] = self.config.font_size - 1

        # Figure settings
        mpl.rcParams['figure.dpi'] = self.config.dpi
        mpl.rcParams['savefig.dpi'] = self.config.dpi
        mpl.rcParams['figure.figsize'] = [self.config.figure_width, self.config.figure_height]

        # Line and marker settings
        mpl.rcParams['lines.linewidth'] = 1.5
        mpl.rcParams['lines.markersize'] = 4

        # LaTeX settings
        if self.config.use_tex:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['font.family'] = 'serif'
        else:
            mpl.rcParams['text.usetex'] = False
            mpl.rcParams['font.family'] = 'sans-serif'

        # Grid and axes
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['grid.alpha'] = 0.3
        mpl.rcParams['axes.axisbelow'] = True

        # Color scheme
        if self.config.color_scheme:
            plt.style.use('seaborn-v0_8-colorblind' if 'seaborn' in plt.style.available else 'default')

    def create_training_curve_figure(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        train_perplexities: Optional[List[float]] = None,
        val_perplexities: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        save_name: str = 'training_curves',
        save_formats: List[str] = ['pdf', 'png']
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create training curve figure with loss and perplexity.

        Args:
            train_losses: Training loss values
            val_losses: Validation loss values (optional)
            train_perplexities: Training perplexity values (optional)
            val_perplexities: Validation perplexity values (optional)
            steps: Step numbers (if None, uses indices)
            save_name: Base filename for saving
            save_formats: Formats to save ('pdf', 'png', etc.)

        Returns:
            Figure and axes array
        """
        has_perplexity = train_perplexities is not None
        num_plots = 2 if has_perplexity else 1

        # Determine figure size
        fig_width = self.config.figure_width * (2 if num_plots == 2 else 1)
        fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, self.config.figure_height))
        if num_plots == 1:
            axes = [axes]

        if steps is None:
            steps = list(range(len(train_losses)))

        # Plot 1: Loss curves
        ax = axes[0]
        ax.plot(steps, train_losses, label='Train', linewidth=2, marker='o',
                markevery=max(1, len(steps) // 10))
        if val_losses is not None:
            ax.plot(steps, val_losses, label='Validation', linewidth=2, marker='s',
                    markevery=max(1, len(steps) // 10))
        ax.set_xlabel('Training Steps' if 'step' in save_name.lower() else 'Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3)

        # Plot 2: Perplexity curves (if available)
        if has_perplexity:
            ax = axes[1]
            ax.plot(steps, train_perplexities, label='Train', linewidth=2, marker='o',
                    markevery=max(1, len(steps) // 10))
            if val_perplexities is not None:
                ax.plot(steps, val_perplexities, label='Validation', linewidth=2, marker='s',
                        markevery=max(1, len(steps) // 10))
            ax.set_xlabel('Training Steps' if 'step' in save_name.lower() else 'Epochs')
            ax.set_ylabel('Perplexity')
            ax.set_title('Perplexity')
            ax.legend(frameon=True, fancybox=False, edgecolor='black')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save in requested formats
        for fmt in save_formats:
            save_path = self.output_dir / f"{save_name}.{fmt}"
            plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=self.config.dpi)

        return fig, np.array(axes)

    def create_layer_analysis_figure(
        self,
        layer_metrics: List[Dict[str, Any]],
        metrics_to_plot: List[str] = ['spectral_centroid', 'total_energy', 'harmonic_entropy'],
        save_name: str = 'layer_evolution',
        save_formats: List[str] = ['pdf', 'png']
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create layer evolution visualization.

        Args:
            layer_metrics: List of dicts with 'layer_name' and metric values
            metrics_to_plot: Which metrics to visualize
            save_name: Base filename
            save_formats: Formats to save

        Returns:
            Figure and axes array
        """
        layer_names = [m['layer_name'] for m in layer_metrics]
        num_layers = len(layer_names)
        num_metrics = len(metrics_to_plot)

        # Determine layout
        num_cols = min(3, num_metrics)
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig_width = self.config.figure_width * min(num_cols, 2)
        fig_height = self.config.figure_height * num_rows

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            # Extract metric values
            values = [m.get(metric, 0) for m in layer_metrics]

            ax.plot(range(num_layers), values, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Layer Index')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xticks(range(num_layers))
            ax.set_xticklabels([ln[:8] for ln in layer_names], rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        # Save
        for fmt in save_formats:
            save_path = self.output_dir / f"{save_name}.{fmt}"
            plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=self.config.dpi)

        return fig, axes

    def create_harmonic_importance_figure(
        self,
        importance_scores: np.ndarray,
        top_k: int = 16,
        save_name: str = 'harmonic_importance',
        save_formats: List[str] = ['pdf', 'png']
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create harmonic importance visualization.

        Args:
            importance_scores: Array of importance scores [num_harmonics]
            top_k: Number of top harmonics to highlight
            save_name: Base filename
            save_formats: Formats to save

        Returns:
            Figure and axes array
        """
        num_harmonics = len(importance_scores)
        scores_norm = importance_scores / (importance_scores.max() + 1e-8)
        top_indices = np.argsort(importance_scores)[::-1][:top_k]

        fig_width = self.config.figure_width * 2
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, self.config.figure_height))

        # Panel 1: Bar chart
        ax = axes[0]
        colors = ['#d62728' if i in top_indices else '#1f77b4'
                  for i in range(num_harmonics)]
        ax.bar(range(num_harmonics), scores_norm, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Harmonic Index')
        ax.set_ylabel('Normalized Importance')
        ax.set_title(f'Harmonic Importance (Top-{top_k} Highlighted)')
        ax.grid(True, alpha=0.3, axis='y')

        # Panel 2: Cumulative importance
        ax = axes[1]
        sorted_scores = np.sort(importance_scores)[::-1]
        cumulative = np.cumsum(sorted_scores) / sorted_scores.sum()
        ax.plot(range(num_harmonics), cumulative, linewidth=2, color='#2ca02c')
        ax.axhline(y=0.9, color='#d62728', linestyle='--', linewidth=1.5, label='90%')
        ax.axhline(y=0.95, color='#ff7f0e', linestyle='--', linewidth=1.5, label='95%')
        ax.set_xlabel('Number of Harmonics (Sorted)')
        ax.set_ylabel('Cumulative Importance')
        ax.set_title('Cumulative Harmonic Contribution')
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        for fmt in save_formats:
            save_path = self.output_dir / f"{save_name}.{fmt}"
            plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=self.config.dpi)

        return fig, axes

    def create_generation_analysis_figure(
        self,
        wave_trajectory: List[Wave],
        confidence_scores: Optional[np.ndarray] = None,
        generated_tokens: Optional[List[str]] = None,
        save_name: str = 'generation_analysis',
        save_formats: List[str] = ['pdf', 'png']
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create generation analysis with trajectory and confidence.

        Args:
            wave_trajectory: List of Wave objects at each generation step
            confidence_scores: Confidence scores per token
            generated_tokens: Token strings (optional, for labeling)
            save_name: Base filename
            save_formats: Formats to save

        Returns:
            Figure and axes array
        """
        num_steps = len(wave_trajectory)

        # Extract amplitude evolution
        amp_evolution = []
        for wave in wave_trajectory:
            # Average amplitude across batch and sequence
            avg_amp = wave.amplitudes.mean(dim=(0, 1)).detach().cpu().numpy()
            amp_evolution.append(avg_amp)
        amp_evolution = np.array(amp_evolution)  # [num_steps, num_harmonics]

        has_confidence = confidence_scores is not None
        num_plots = 2 if has_confidence else 1

        fig_width = self.config.figure_width * 2
        fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, self.config.figure_height))
        if num_plots == 1:
            axes = [axes]

        # Panel 1: Amplitude trajectory heatmap
        ax = axes[0]
        im = ax.imshow(amp_evolution.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Generation Step')
        ax.set_ylabel('Harmonic Index')
        ax.set_title('Wave Amplitude Evolution')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude', rotation=270, labelpad=15)

        # Panel 2: Confidence scores (if available)
        if has_confidence:
            ax = axes[1]
            ax.plot(confidence_scores, marker='o', linewidth=2, markersize=4)
            ax.set_xlabel('Token Position')
            ax.set_ylabel('Confidence')
            ax.set_title('Generation Confidence')
            ax.grid(True, alpha=0.3)

            # Annotate tokens if provided
            if generated_tokens is not None and len(generated_tokens) == len(confidence_scores):
                # Annotate every few tokens to avoid clutter
                step = max(1, len(generated_tokens) // 5)
                for i in range(0, len(generated_tokens), step):
                    ax.annotate(
                        generated_tokens[i][:5],  # Truncate long tokens
                        (i, confidence_scores[i]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center',
                        fontsize=7
                    )

        plt.tight_layout()

        # Save
        for fmt in save_formats:
            save_path = self.output_dir / f"{save_name}.{fmt}"
            plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=self.config.dpi)

        return fig, axes

    def create_comparison_figure(
        self,
        comparison_data: Dict[str, Dict[str, float]],
        metric_names: Optional[List[str]] = None,
        save_name: str = 'model_comparison',
        save_formats: List[str] = ['pdf', 'png']
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create comparison figure (e.g., different checkpoints or configurations).

        Args:
            comparison_data: Dict of {model_name: {metric: value}}
            metric_names: List of metrics to compare (if None, use all)
            save_name: Base filename
            save_formats: Formats to save

        Returns:
            Figure and axes
        """
        model_names = list(comparison_data.keys())
        if metric_names is None:
            metric_names = list(comparison_data[model_names[0]].keys())

        num_metrics = len(metric_names)
        num_models = len(model_names)

        fig, ax = plt.subplots(figsize=(self.config.figure_width * 1.5, self.config.figure_height * 1.2))

        # Prepare data for grouped bar chart
        x = np.arange(num_metrics)
        width = 0.8 / num_models

        for idx, model_name in enumerate(model_names):
            values = [comparison_data[model_name].get(metric, 0) for metric in metric_names]
            offset = (idx - num_models / 2) * width + width / 2
            ax.bar(x + offset, values, width, label=model_name, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45, ha='right')
        ax.legend(frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save
        for fmt in save_formats:
            save_path = self.output_dir / f"{save_name}.{fmt}"
            plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=self.config.dpi)

        return fig, ax

    def generate_latex_table(
        self,
        data: List[Dict[str, Any]],
        column_names: Optional[List[str]] = None,
        caption: str = 'Results Table',
        label: str = 'tab:results',
        format_specs: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate LaTeX table code.

        Args:
            data: List of dicts, each representing a row
            column_names: Column headers (if None, use dict keys)
            caption: Table caption
            label: LaTeX label for referencing
            format_specs: Dict of {column: format_string} for custom formatting

        Returns:
            LaTeX table code as string
        """
        if not data:
            return ""

        if column_names is None:
            column_names = list(data[0].keys())

        num_cols = len(column_names)

        # Build LaTeX table
        latex = []
        latex.append(r"\begin{table}[t]")
        latex.append(r"\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")

        # Column alignment (c for all)
        latex.append(r"\begin{tabular}{" + "c" * num_cols + "}")
        latex.append(r"\toprule")

        # Header
        header = " & ".join([col.replace('_', r'\_') for col in column_names])
        latex.append(header + r" \\")
        latex.append(r"\midrule")

        # Data rows
        for row_dict in data:
            row_values = []
            for col in column_names:
                value = row_dict.get(col, '-')

                # Apply custom formatting if specified
                if format_specs and col in format_specs:
                    fmt = format_specs[col]
                    if isinstance(value, float):
                        value = fmt.format(value)

                # Default formatting
                elif isinstance(value, float):
                    value = f"{value:.4f}"
                elif isinstance(value, str):
                    value = value.replace('_', r'\_')

                row_values.append(str(value))

            latex.append(" & ".join(row_values) + r" \\")

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        return "\n".join(latex)

    def save_latex_table(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        **kwargs
    ):
        """
        Generate and save LaTeX table to file.

        Args:
            data: Table data
            filename: Output filename (without extension)
            **kwargs: Additional arguments for generate_latex_table
        """
        latex_code = self.generate_latex_table(data, **kwargs)
        output_path = self.output_dir / f"{filename}.tex"

        with open(output_path, 'w') as f:
            f.write(latex_code)

        print(f"LaTeX table saved to: {output_path}")

    def generate_full_report(
        self,
        analysis_data: Dict[str, Any],
        report_name: str = 'wave_transformer_analysis'
    ):
        """
        Generate comprehensive report with all figures and tables.

        Args:
            analysis_data: Dictionary containing all analysis results
                Expected keys:
                - 'training_curves': {train_losses, val_losses, etc.}
                - 'layer_analysis': List of layer metrics
                - 'harmonic_importance': importance scores array
                - 'generation_examples': List of generation results
                - 'comparison_results': Comparison data
                - 'ablation_results': Ablation study data
            report_name: Base name for report files
        """
        print(f"Generating full report: {report_name}")

        # Create training curves
        if 'training_curves' in analysis_data:
            tc_data = analysis_data['training_curves']
            self.create_training_curve_figure(
                train_losses=tc_data.get('train_losses', []),
                val_losses=tc_data.get('val_losses'),
                train_perplexities=tc_data.get('train_perplexities'),
                val_perplexities=tc_data.get('val_perplexities'),
                save_name=f'{report_name}_training'
            )
            print("  ✓ Training curves generated")

        # Create layer analysis
        if 'layer_analysis' in analysis_data:
            self.create_layer_analysis_figure(
                layer_metrics=analysis_data['layer_analysis'],
                save_name=f'{report_name}_layers'
            )
            print("  ✓ Layer analysis generated")

        # Create harmonic importance
        if 'harmonic_importance' in analysis_data:
            self.create_harmonic_importance_figure(
                importance_scores=analysis_data['harmonic_importance'],
                save_name=f'{report_name}_harmonics'
            )
            print("  ✓ Harmonic importance generated")

        # Create comparison
        if 'comparison_results' in analysis_data:
            self.create_comparison_figure(
                comparison_data=analysis_data['comparison_results'],
                save_name=f'{report_name}_comparison'
            )
            print("  ✓ Comparison figure generated")

        # Generate LaTeX tables
        if 'ablation_results' in analysis_data:
            self.save_latex_table(
                data=analysis_data['ablation_results'],
                filename=f'{report_name}_ablation',
                caption='Ablation Study Results',
                label='tab:ablation'
            )
            print("  ✓ Ablation table generated")

        print(f"\nReport generation complete. Files saved to: {self.output_dir}")
