"""
Demonstration of Wave Transformer Visualization and Reporting Tools

This script demonstrates how to use the visualization components:
- TensorBoard logging
- Weights & Biases integration
- Publication-quality report generation
- Configuration management
"""

import torch
import numpy as np
from pathlib import Path

# Import Wave Transformer components
from wave_transformer.core.wave import Wave
from wave_transformer.analysis.visualization import (
    WaveTensorBoardWriter,
    WaveWandbLogger,
    PaperReportGenerator,
    PublicationStyle
)
from wave_transformer.analysis.utils import (
    AnalysisConfig,
    create_default_config
)


def create_dummy_wave(batch_size=4, seq_len=32, num_harmonics=64):
    """Create a dummy wave for demonstration."""
    frequencies = torch.randn(batch_size, seq_len, num_harmonics) * 0.5 + 1.0
    amplitudes = torch.abs(torch.randn(batch_size, seq_len, num_harmonics)) * 0.5
    phases = torch.randn(batch_size, seq_len, num_harmonics) * np.pi
    return Wave(frequencies, amplitudes, phases)


def demo_tensorboard_logging():
    """Demonstrate TensorBoard logging."""
    print("\n" + "="*70)
    print("TensorBoard Logging Demo")
    print("="*70)

    # Create TensorBoard writer
    writer = WaveTensorBoardWriter(
        log_dir='runs/visualization_demo',
        comment='demo_run'
    )

    # Generate dummy data
    wave = create_dummy_wave()

    # Log wave statistics
    print("Logging wave statistics...")
    writer.add_wave_statistics(wave, tag='demo/wave', step=0)

    # Log wave heatmaps
    print("Logging wave heatmaps...")
    writer.add_wave_heatmaps(wave, tag='demo/wave', step=0, batch_idx=0)

    # Log wave spectrum
    print("Logging wave spectrum...")
    writer.add_wave_spectrum(wave, tag='demo/wave', step=0, batch_idx=0, seq_position=0)

    # Log layer comparison
    print("Logging layer comparison...")
    layer_snapshots = [
        {'layer_name': 'layer_0', 'wave': create_dummy_wave()},
        {'layer_name': 'layer_1', 'wave': create_dummy_wave()},
        {'layer_name': 'layer_2', 'wave': create_dummy_wave()},
    ]
    writer.add_layer_comparison(layer_snapshots, tag='demo/layers', step=0)

    # Log harmonic importance
    print("Logging harmonic importance...")
    importance_scores = np.random.exponential(scale=2.0, size=64)
    writer.add_harmonic_importance(importance_scores, tag='demo/harmonics', step=0)

    # Close writer
    writer.close()

    print("\n✓ TensorBoard logs saved to: runs/visualization_demo")
    print("  View with: tensorboard --logdir runs/visualization_demo")


def demo_wandb_logging():
    """Demonstrate Weights & Biases logging."""
    print("\n" + "="*70)
    print("Weights & Biases Logging Demo")
    print("="*70)

    try:
        import wandb

        # Note: This will create a W&B run. Comment out if not desired.
        print("Initializing W&B logger...")
        logger = WaveWandbLogger(
            project='wave-transformer-demo',
            name='visualization_demo',
            tags=['demo', 'visualization']
        )

        # Generate dummy data
        wave = create_dummy_wave()

        # Log wave statistics
        print("Logging wave statistics...")
        logger.log_wave_statistics(wave, prefix='demo', step=0)

        # Log wave visualizations
        print("Logging wave visualizations...")
        logger.log_wave_visualizations(wave, prefix='demo', step=0)

        # Log generation example
        print("Logging generation example...")
        logger.log_generation_example(
            input_text="Once upon a time",
            generated_text="Once upon a time in a land far away...",
            confidence_scores=np.random.rand(10) * 0.3 + 0.7
        )

        # Log layer analysis
        print("Logging layer analysis...")
        layer_snapshots = [
            {'layer_name': 'layer_0', 'wave': create_dummy_wave()},
            {'layer_name': 'layer_1', 'wave': create_dummy_wave()},
            {'layer_name': 'layer_2', 'wave': create_dummy_wave()},
        ]
        logger.log_layer_analysis(layer_snapshots, prefix='demo', step=0)

        # Log ablation results
        print("Logging ablation results...")
        ablation_results = [
            {'config': 'baseline', 'metric': 'perplexity', 'value': 15.2},
            {'config': 'no_phase', 'metric': 'perplexity', 'value': 18.5},
            {'config': 'no_freq_learning', 'metric': 'perplexity', 'value': 20.1},
        ]
        logger.log_ablation_results(ablation_results, table_name='ablation_study')

        # Finish W&B run
        logger.finish()

        print("\n✓ W&B logs saved successfully")
        print("  View at: https://wandb.ai")

    except ImportError:
        print("\n⚠ wandb not installed. Skipping W&B demo.")
        print("  Install with: pip install wandb")


def demo_report_generation():
    """Demonstrate publication-quality report generation."""
    print("\n" + "="*70)
    print("Publication Report Generation Demo")
    print("="*70)

    # Create report generator
    output_dir = Path('demo_reports')
    generator = PaperReportGenerator(
        output_dir=output_dir,
        style=PublicationStyle.IEEE
    )

    print(f"Report output directory: {output_dir}")

    # 1. Training curves
    print("\nGenerating training curves...")
    train_losses = [4.5, 3.8, 3.2, 2.9, 2.6, 2.4, 2.2, 2.1, 2.0, 1.95]
    val_losses = [4.6, 3.9, 3.4, 3.0, 2.8, 2.6, 2.5, 2.4, 2.35, 2.3]
    train_perplexities = [np.exp(l) for l in train_losses]
    val_perplexities = [np.exp(l) for l in val_losses]

    generator.create_training_curve_figure(
        train_losses=train_losses,
        val_losses=val_losses,
        train_perplexities=train_perplexities,
        val_perplexities=val_perplexities,
        save_name='training_curves'
    )

    # 2. Layer analysis
    print("Generating layer analysis figure...")
    layer_metrics = [
        {'layer_name': 'encoder', 'spectral_centroid': 1.2, 'total_energy': 45.3, 'harmonic_entropy': 2.8},
        {'layer_name': 'layer_0', 'spectral_centroid': 1.5, 'total_energy': 42.1, 'harmonic_entropy': 3.1},
        {'layer_name': 'layer_1', 'spectral_centroid': 1.8, 'total_energy': 39.5, 'harmonic_entropy': 3.4},
        {'layer_name': 'layer_2', 'spectral_centroid': 2.0, 'total_energy': 37.2, 'harmonic_entropy': 3.6},
    ]
    generator.create_layer_analysis_figure(layer_metrics, save_name='layer_evolution')

    # 3. Harmonic importance
    print("Generating harmonic importance figure...")
    importance_scores = np.random.exponential(scale=2.0, size=64)
    generator.create_harmonic_importance_figure(
        importance_scores,
        top_k=16,
        save_name='harmonic_importance'
    )

    # 4. Generation analysis
    print("Generating generation analysis figure...")
    # Create dummy wave trajectory
    wave_trajectory = [create_dummy_wave(batch_size=1, seq_len=16) for _ in range(10)]
    confidence_scores = np.random.rand(10) * 0.3 + 0.7

    generator.create_generation_analysis_figure(
        wave_trajectory=wave_trajectory,
        confidence_scores=confidence_scores,
        save_name='generation_analysis'
    )

    # 5. Model comparison
    print("Generating model comparison figure...")
    comparison_data = {
        'Baseline': {'perplexity': 15.2, 'accuracy': 0.72, 'speed': 1.0},
        'Wave-32': {'perplexity': 12.8, 'accuracy': 0.76, 'speed': 0.95},
        'Wave-64': {'perplexity': 11.5, 'accuracy': 0.78, 'speed': 0.88},
    }
    generator.create_comparison_figure(
        comparison_data,
        save_name='model_comparison'
    )

    # 6. LaTeX table
    print("Generating LaTeX table...")
    ablation_data = [
        {'Model': 'Full Model', 'Perplexity': 11.5, 'Accuracy': 0.78, 'Params (M)': 124},
        {'Model': 'No Phase', 'Perplexity': 13.2, 'Accuracy': 0.75, 'Params (M)': 124},
        {'Model': 'Fixed Freq', 'Perplexity': 14.8, 'Accuracy': 0.73, 'Params (M)': 120},
        {'Model': 'Baseline', 'Perplexity': 15.2, 'Accuracy': 0.72, 'Params (M)': 125},
    ]
    generator.save_latex_table(
        ablation_data,
        filename='ablation_table',
        caption='Ablation Study Results',
        label='tab:ablation'
    )

    print("\n✓ All figures generated successfully")
    print(f"  Output directory: {output_dir.absolute()}")


def demo_configuration():
    """Demonstrate configuration management."""
    print("\n" + "="*70)
    print("Configuration Management Demo")
    print("="*70)

    # 1. Create default configuration
    print("\n1. Creating default configuration...")
    config = create_default_config(
        output_dir='analysis_results',
        experiment_name='demo_experiment',
        enable_tensorboard=True,
        enable_wandb=False
    )
    print(config)

    # 2. Save to YAML
    try:
        config_path = Path('demo_config.yaml')
        print(f"\n2. Saving configuration to {config_path}...")
        config.to_yaml(str(config_path))
        print("   ✓ Configuration saved")

        # 3. Load from YAML
        print(f"\n3. Loading configuration from {config_path}...")
        loaded_config = AnalysisConfig.from_yaml(str(config_path))
        print("   ✓ Configuration loaded")
        print(f"   Experiment name: {loaded_config.experiment_name}")

        # 4. Update configuration
        print("\n4. Updating configuration...")
        loaded_config.update(wave_sampling_rate=0.5)
        print(f"   ✓ Updated wave_sampling_rate to {loaded_config.wave_sampling_rate}")

    except ImportError:
        print("\n⚠ pyyaml not installed. YAML demo skipped.")
        print("  Install with: pip install pyyaml")

    # 5. Access component configs
    print("\n5. Accessing component configurations...")
    viz_config = config.get_component_config('visualization')
    print(f"   TensorBoard enabled: {viz_config.use_tensorboard}")
    print(f"   W&B enabled: {viz_config.use_wandb}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("Wave Transformer Visualization & Reporting Demo")
    print("="*70)

    # Demo 1: TensorBoard
    demo_tensorboard_logging()

    # Demo 2: Weights & Biases (optional - requires wandb)
    # Uncomment to enable:
    # demo_wandb_logging()

    # Demo 3: Publication reports
    demo_report_generation()

    # Demo 4: Configuration
    demo_configuration()

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nGenerated outputs:")
    print("  - TensorBoard logs: runs/visualization_demo/")
    print("  - Publication figures: demo_reports/")
    print("  - Config file: demo_config.yaml")
    print("\nNext steps:")
    print("  1. View TensorBoard: tensorboard --logdir runs/visualization_demo")
    print("  2. Check publication figures in demo_reports/")
    print("  3. Customize demo_config.yaml for your experiments")


if __name__ == '__main__':
    main()
