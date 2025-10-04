"""
Comprehensive Demonstration of Wave Transformer Introspection Tools

This example shows how to use all four introspection modules:
1. LayerWaveAnalyzer - Track wave evolution through layers
2. HarmonicImportanceAnalyzer - Identify important harmonics
3. WaveInterferenceAnalyzer - Analyze wave interference patterns
4. SpectrumEvolutionTracker - Track frequency spectrum changes

Run this script to generate analysis visualizations for your model.
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import introspection tools
from wave_transformer.analysis import (
    LayerWaveAnalyzer,
    HarmonicImportanceAnalyzer,
    WaveInterferenceAnalyzer,
    SpectrumEvolutionTracker,
)


def demo_layer_analyzer(model, sample_input, output_dir):
    """Demonstrate LayerWaveAnalyzer usage"""
    print("\n" + "="*80)
    print("LAYER WAVE ANALYZER DEMO")
    print("="*80)

    # Create analyzer with context manager for automatic cleanup
    with LayerWaveAnalyzer(model, device='cuda') as analyzer:
        # Extract wave representations from all layers
        print("\nExtracting wave snapshots from all layers...")
        snapshots = analyzer.analyze_input(
            encoder_input=sample_input,
            batch_idx=0
        )

        print(f"Captured {len(snapshots)} layer snapshots:")
        for snap in snapshots:
            print(f"  - {snap.layer_name} (idx={snap.layer_idx})")

        # Compare layers using multiple metrics
        print("\nComputing layer comparison metrics...")
        comparison = analyzer.compare_layers(
            snapshots,
            batch_idx=0,
            metrics=['amplitude_mean', 'frequency_mean', 'phase_std',
                    'amplitude_energy', 'spectral_centroid']
        )

        # Visualize layer evolution
        print("\nGenerating layer evolution visualization...")
        analyzer.plot_layer_evolution(
            snapshots,
            batch_idx=0,
            save_path=output_dir / 'layer_evolution.png'
        )

        # Plot metric evolution
        layer_names = [s.layer_name for s in snapshots]
        analyzer.plot_metric_evolution(
            comparison,
            layer_names=layer_names,
            save_path=output_dir / 'metric_evolution.png'
        )

    print("Layer analysis complete! Saved visualizations to:", output_dir)
    return snapshots


def demo_harmonic_analyzer(model, dataloader, output_dir):
    """Demonstrate HarmonicImportanceAnalyzer usage"""
    print("\n" + "="*80)
    print("HARMONIC IMPORTANCE ANALYZER DEMO")
    print("="*80)

    criterion = nn.CrossEntropyLoss()
    analyzer = HarmonicImportanceAnalyzer(model, criterion, device='cuda')

    # Analyze harmonic importance using energy method
    print("\nAnalyzing harmonic importance (energy-based)...")
    importance_energy = analyzer.analyze_harmonic_importance(
        dataloader,
        method='energy',
        layer_name='encoder',
        max_batches=50
    )

    print(f"Found {importance_energy['num_harmonics']} harmonics")
    print(f"Top 5 most important harmonics: {importance_energy['importance'].argsort()[::-1][:5]}")

    # Plot importance
    print("\nGenerating importance visualization...")
    analyzer.plot_harmonic_importance(
        importance_energy,
        top_k=32,
        save_path=output_dir / 'harmonic_importance.png'
    )

    # Create sparse mask for model compression
    print("\nCreating sparse harmonic masks...")
    mask_top32 = analyzer.get_sparse_harmonic_mask(
        importance_energy,
        top_k=32
    )
    print(f"Top-32 mask: {mask_top32.sum():.0f} harmonics kept")

    mask_95 = analyzer.get_sparse_harmonic_mask(
        importance_energy,
        cumulative_threshold=0.95
    )
    print(f"95% cumulative mask: {mask_95.sum():.0f} harmonics kept")

    # Optional: Gradient-based sensitivity (slower)
    print("\nComputing gradient-based sensitivity (this may take a while)...")
    sensitivity = analyzer.compute_gradient_sensitivity(
        dataloader,
        harmonic_indices=list(range(0, 64, 4)),  # Sample every 4th harmonic
        max_batches=5
    )
    print(f"Baseline loss: {sensitivity['baseline_loss']:.4f}")
    print(f"Max sensitivity: {sensitivity['sensitivity'].max():.4f}")

    print("Harmonic analysis complete! Saved visualizations to:", output_dir)
    return importance_energy, mask_top32


def demo_interference_analyzer(model, snapshots, output_dir):
    """Demonstrate WaveInterferenceAnalyzer usage"""
    print("\n" + "="*80)
    print("WAVE INTERFERENCE ANALYZER DEMO")
    print("="*80)

    analyzer = WaveInterferenceAnalyzer(model, device='cuda')

    # Analyze interference between layers
    print("\nAnalyzing interference patterns between consecutive layers...")
    interference_results = analyzer.analyze_layer_interference(
        snapshots,
        modes=['constructive', 'destructive', 'modulate'],
        batch_idx=0
    )

    print(f"Analyzed {len(interference_results['constructive'])} layer transitions")

    # Print sample metrics
    if interference_results['constructive']:
        first_metrics = interference_results['constructive'][0]
        print(f"\nFirst transition metrics (constructive mode):")
        print(f"  Phase alignment: {first_metrics.phase_alignment:.3f}")
        print(f"  Frequency coupling: {first_metrics.frequency_coupling:.3f}")
        print(f"  Energy transfer: {first_metrics.energy_transfer:.3f}")
        print(f"  Spectral overlap: {first_metrics.spectral_overlap:.3f}")

    # Visualize interference patterns
    print("\nGenerating interference pattern visualization...")
    layer_names = [f"{snapshots[i].layer_name}→{snapshots[i+1].layer_name}"
                  for i in range(len(snapshots)-1)]
    analyzer.plot_interference_patterns(
        interference_results,
        layer_names=layer_names,
        save_path=output_dir / 'interference_patterns.png'
    )

    # Detailed component visualization for first two layers
    if len(snapshots) >= 2:
        print("\nGenerating detailed interference component visualization...")
        analyzer.visualize_interference_components(
            snapshots[0].wave,
            snapshots[1].wave,
            batch_idx=0,
            seq_position=0,
            save_path=output_dir / 'interference_components.png'
        )

    # Compute pairwise interference matrix
    print("\nComputing pairwise interference matrix...")
    matrix = analyzer.compute_pairwise_interference_matrix(
        snapshots,
        mode='constructive',
        metric='energy_transfer'
    )

    analyzer.plot_interference_matrix(
        matrix,
        layer_names=[s.layer_name for s in snapshots],
        metric_name='Energy Transfer',
        save_path=output_dir / 'interference_matrix.png'
    )

    print("Interference analysis complete! Saved visualizations to:", output_dir)
    return interference_results


def demo_spectrum_tracker(model, snapshots, output_dir):
    """Demonstrate SpectrumEvolutionTracker usage"""
    print("\n" + "="*80)
    print("SPECTRUM EVOLUTION TRACKER DEMO")
    print("="*80)

    tracker = SpectrumEvolutionTracker(model, device='cuda')

    # Extract spectrum evolution
    print("\nExtracting spectrum evolution for sequence position 0...")
    spectrum_evolution = tracker.extract_spectrum_evolution(
        snapshots,
        batch_idx=0,
        seq_position=0
    )

    print(f"Extracted spectrum from {len(spectrum_evolution)} layers:")
    for spec in spectrum_evolution:
        print(f"  - {spec.layer_name}: "
              f"centroid={spec.spectral_centroid:.2f}Hz, "
              f"bandwidth={spec.bandwidth:.2f}Hz, "
              f"energy={spec.total_energy:.2f}")

    # Compute spectral shifts
    print("\nComputing spectral shifts between layers...")
    shifts = tracker.compute_spectral_shift(spectrum_evolution)

    for i, shift in enumerate(shifts):
        print(f"  Shift {i}: "
              f"centroid_shift={shift.centroid_shift:.2f}Hz, "
              f"energy_redist={shift.energy_redistribution:.3f}")

    # Generate visualizations
    print("\nGenerating 3D spectrum visualization...")
    tracker.plot_spectrum_evolution(
        spectrum_evolution,
        mode='3d',
        save_path=output_dir / 'spectrum_3d.png'
    )

    print("\nGenerating stacked 2D spectrum visualization...")
    tracker.plot_spectrum_evolution(
        spectrum_evolution,
        mode='2d_stacked',
        save_path=output_dir / 'spectrum_2d_stacked.png'
    )

    print("\nGenerating waterfall visualization...")
    tracker.plot_spectrum_evolution(
        spectrum_evolution,
        mode='waterfall',
        save_path=output_dir / 'spectrum_waterfall.png'
    )

    # Plot spectral metrics
    print("\nGenerating spectral metrics plot...")
    tracker.plot_spectral_metrics(
        spectrum_evolution,
        shifts=shifts,
        save_path=output_dir / 'spectral_metrics.png'
    )

    # Analyze frequency distribution shift
    print("\nAnalyzing frequency distribution evolution...")
    dist_analysis = tracker.analyze_frequency_distribution_shift(
        spectrum_evolution,
        num_bins=30
    )

    tracker.plot_frequency_distribution_evolution(
        dist_analysis,
        layer_names=[s.layer_name for s in spectrum_evolution],
        save_path=output_dir / 'frequency_distribution_evolution.png'
    )

    print("Spectrum tracking complete! Saved visualizations to:", output_dir)
    return spectrum_evolution, shifts


def run_complete_introspection_demo(model, dataloader, output_dir='introspection_results'):
    """
    Run complete introspection analysis demo

    Args:
        model: WaveTransformer model
        dataloader: DataLoader with sample data
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "="*80)
    print("WAVE TRANSFORMER INTROSPECTION DEMO")
    print("="*80)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Number of harmonics: {model.num_harmonics}")
    print(f"Number of transformer layers: {model.transformer_num_layers}")
    print(f"Output directory: {output_dir}")

    # Get a sample batch
    sample_batch = next(iter(dataloader))
    if isinstance(sample_batch, dict):
        sample_input = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v
                       for k, v in sample_batch.items() if k not in ['labels', 'label']}
    else:
        sample_input = {'token_ids': sample_batch[0].to('cuda')}

    # Run all demos
    print("\n\nStarting comprehensive introspection analysis...")

    # 1. Layer Analysis
    snapshots = demo_layer_analyzer(model, sample_input, output_dir)

    # 2. Harmonic Analysis
    importance, mask = demo_harmonic_analyzer(model, dataloader, output_dir)

    # 3. Interference Analysis
    interference = demo_interference_analyzer(model, snapshots, output_dir)

    # 4. Spectrum Tracking
    spectrum, shifts = demo_spectrum_tracker(model, snapshots, output_dir)

    print("\n" + "="*80)
    print("INTROSPECTION DEMO COMPLETE!")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*.png')):
        print(f"  - {file.name}")

    return {
        'snapshots': snapshots,
        'importance': importance,
        'mask': mask,
        'interference': interference,
        'spectrum': spectrum,
        'shifts': shifts,
    }


# Example usage
if __name__ == '__main__':
    """
    To use this demo with your model:

    ```python
    from wave_transformer.core.transformer import WaveTransformer
    from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoder
    from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder
    import torch.utils.data as data

    # Create your model
    encoder = TokenToWaveEncoder(vocab_size=50000, num_harmonics=64)
    decoder = WaveToTokenDecoder(vocab_size=50000, num_harmonics=64)
    model = WaveTransformer(encoder, decoder, num_harmonics=64).cuda()

    # Create a dataloader
    # ... your dataloader setup ...

    # Run introspection
    results = run_complete_introspection_demo(model, dataloader)
    ```
    """
    print(__doc__)
    print("\nTo run this demo, instantiate your model and dataloader, then call:")
    print("  results = run_complete_introspection_demo(model, dataloader)")
