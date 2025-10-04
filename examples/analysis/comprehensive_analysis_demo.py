"""
Comprehensive Wave Transformer Analysis Demo

This script demonstrates how to use the full Wave Transformer analysis suite
during and after training to gain deep insights into model behavior.

Author: Wave Transformer Analysis Suite
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import os

# Import Wave Transformer components
from wave_transformer.core.transformer import WaveTransformer
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoderSlim
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder

# Import analysis suite
from wave_transformer.analysis import (
    # Core statistics
    WaveStatistics,
    WaveCollector,
    GradientCollector,
    AnalysisExporter,

    # Training monitoring
    WaveForwardHook,
    GradientMonitor,
    WaveEvolutionCallback,
    GradientFlowCallback,

    # Introspection
    LayerWaveAnalyzer,
    HarmonicImportanceAnalyzer,
    WaveInterferenceAnalyzer,
    SpectrumEvolutionTracker,

    # Generation analysis
    LiveGenerationVisualizer,
    WaveTrajectoryTracker,
    GenerationConfidenceTracker,
    RoundTripAnalyzer,

    # Comparative analysis
    CheckpointComparator,
    InputComparator,
    AblationHelper,

    # Visualization
    WaveTensorBoardWriter,
    WaveWandbLogger,
    PaperReportGenerator,
    PublicationStyle,

    # Configuration
    create_default_config,
)


def demo_basic_wave_statistics():
    """Demonstrate basic wave statistics computation."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Wave Statistics")
    print("="*80)

    # Create synthetic wave for demonstration
    from wave_transformer.core.wave import Wave

    B, S, H = 2, 64, 32  # batch, sequence, harmonics
    wave = Wave(
        frequencies=torch.rand(B, S, H) * 10 + 0.5,
        amplitudes=torch.rand(B, S, H),
        phases=torch.rand(B, S, H) * 6.28 - 3.14
    )

    # Compute comprehensive statistics
    stats = WaveStatistics.compute_basic_stats(wave, batch_idx=0)
    print(f"\nWave Statistics (batch 0):")
    print(f"  Frequencies: mean={stats['frequencies']['mean']:.4f}, std={stats['frequencies']['std']:.4f}")
    print(f"  Amplitudes:  mean={stats['amplitudes']['mean']:.4f}, std={stats['amplitudes']['std']:.4f}")
    print(f"  Phases:      mean={stats['phases']['mean']:.4f}, std={stats['phases']['std']:.4f}")

    # Harmonic importance
    importance = WaveStatistics.compute_harmonic_importance(wave, method='energy')
    print(f"\nTop 5 important harmonics (by energy): {importance.top_k(5).tolist()}")

    # Spectral centroid
    centroid = WaveStatistics.compute_spectral_centroid(wave)
    print(f"Spectral centroid shape: {centroid.shape}")
    print(f"Mean spectral centroid: {centroid.mean():.4f}")

    # Total energy
    energy = WaveStatistics.compute_total_energy(wave)
    print(f"Mean energy per position: {energy.mean():.4f}")

    print("\n✓ Basic statistics computation complete")


def demo_training_monitoring(model, dataloader, device, output_dir='analysis_results/training'):
    """Demonstrate training-time monitoring tools."""
    print("\n" + "="*80)
    print("DEMO 2: Training Monitoring")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Setup collectors
    print("\n1. Setting up collectors...")
    wave_collector = WaveCollector(
        sample_interval=10,  # Collect every 10 batches
        max_samples=100,
        collect_on_rank=0
    )

    gradient_collector = GradientCollector(
        sample_interval=10,
        max_samples=100
    )

    # 2. Setup gradient monitor
    print("2. Setting up gradient monitor...")
    gradient_monitor = GradientMonitor(
        model=model,
        track_norms=True,
        track_histograms=False  # Set True for detailed analysis
    )
    gradient_monitor.register_hooks()

    # 3. Setup TensorBoard writer
    print("3. Setting up TensorBoard writer...")
    try:
        tb_writer = WaveTensorBoardWriter(log_dir=f"{output_dir}/tensorboard")
        use_tensorboard = True
    except ImportError:
        print("   TensorBoard not available, skipping...")
        use_tensorboard = False

    # 4. Mini training loop to demonstrate monitoring
    print("4. Running monitored training steps...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:  # Just 5 batches for demo
            break

        # Move to device
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass
        logits, encoder_wave = model(
            encoder_input={'token_ids': inputs},
            attention_mask=attention_mask,
            return_encoder_outputs=True
        )

        # Simple loss (for demo)
        loss = logits.sum() / logits.numel()

        # Collect wave statistics
        if wave_collector.should_collect():
            wave_collector.collect(
                wave=encoder_wave,
                step=batch_idx,
                layer_name='encoder'
            )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log to TensorBoard
        if use_tensorboard and batch_idx % 2 == 0:
            tb_writer.add_wave_statistics(encoder_wave, step=batch_idx, tag='encoder')
            if batch_idx == 0:  # Log heatmaps once
                tb_writer.add_wave_heatmaps(encoder_wave, step=batch_idx, tag='encoder')

        print(f"   Batch {batch_idx}: loss={loss.item():.4f}")

    # 5. Get gradient flow report
    print("\n5. Generating gradient flow report...")
    grad_report = gradient_monitor.get_gradient_flow_report()
    print(grad_report[:500] + "...\n")  # Print first 500 chars

    # 6. Aggregate collected statistics
    print("6. Aggregating collected statistics...")
    wave_stats_agg = wave_collector.aggregate()
    print(f"   Collected {len(wave_collector.data)} wave samples")
    print(f"   Mean amplitude across all samples: {wave_stats_agg['mean_amplitude']:.4f}")

    # 7. Export results
    print("7. Exporting results...")
    AnalysisExporter.to_json(wave_stats_agg, f"{output_dir}/wave_statistics.json")

    if use_tensorboard:
        tb_writer.close()

    print("\n✓ Training monitoring demo complete")
    print(f"   Results saved to: {output_dir}")


def demo_model_introspection(model, sample_input, device, output_dir='analysis_results/introspection'):
    """Demonstrate model introspection tools."""
    print("\n" + "="*80)
    print("DEMO 3: Model Introspection")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Layer-wise analysis
    print("\n1. Layer-wise wave analysis...")
    with LayerWaveAnalyzer(model) as analyzer:
        layer_waves = analyzer.analyze_input(
            encoder_input={'token_ids': sample_input['input_ids'].to(device)},
            attention_mask=sample_input['attention_mask'].to(device)
        )

        print(f"   Extracted waves from {len(layer_waves)} layers")

        # Compare layers
        comparison = analyzer.compare_layers(layer_waves)
        print(f"   Layer comparison metrics: {list(comparison.keys())}")

        # Visualize
        analyzer.plot_layer_evolution(
            layer_waves,
            save_path=f"{output_dir}/layer_evolution.png"
        )
        print(f"   Saved layer evolution plot")

    # 2. Harmonic importance analysis
    print("\n2. Harmonic importance analysis...")
    # Note: This would normally use a dataloader, using single batch for demo
    mini_dataloader = [sample_input]  # Simulated dataloader

    harmonic_analyzer = HarmonicImportanceAnalyzer(
        model=model,
        criterion=nn.CrossEntropyLoss()
    )

    importance = harmonic_analyzer.analyze_harmonic_importance(
        dataloader=mini_dataloader,
        num_batches=1,
        method='amplitude'
    )

    print(f"   Top 10 harmonics: {importance['top_harmonics'][:10].tolist()}")

    # Visualize
    harmonic_analyzer.plot_harmonic_importance(
        importance,
        save_path=f"{output_dir}/harmonic_importance.png"
    )
    print(f"   Saved harmonic importance plot")

    # 3. Wave interference analysis
    print("\n3. Wave interference analysis...")
    interference_analyzer = WaveInterferenceAnalyzer(model)

    interference_results = interference_analyzer.analyze_layer_interference(layer_waves)
    print(f"   Mean phase alignment: {interference_results['mean_phase_alignment']:.4f}")
    print(f"   Mean frequency coupling: {interference_results['mean_frequency_coupling']:.4f}")

    # Visualize
    interference_analyzer.plot_interference_patterns(
        layer_waves,
        save_path=f"{output_dir}/interference_patterns.png"
    )
    print(f"   Saved interference patterns plot")

    # 4. Spectrum evolution tracking
    print("\n4. Spectrum evolution tracking...")
    spectrum_tracker = SpectrumEvolutionTracker(model)

    spectra = spectrum_tracker.extract_spectrum_evolution(layer_waves)
    print(f"   Extracted spectra from {len(spectra)} layers")

    # Visualize
    spectrum_tracker.plot_spectrum_evolution(
        spectra,
        save_path=f"{output_dir}/spectrum_evolution.png",
        mode='waterfall'
    )
    print(f"   Saved spectrum evolution plot")

    print("\n✓ Model introspection demo complete")
    print(f"   Results saved to: {output_dir}")


def demo_generation_analysis(model, tokenizer, device, output_dir='analysis_results/generation'):
    """Demonstrate generation analysis tools."""
    print("\n" + "="*80)
    print("DEMO 4: Generation Analysis")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    prompt = "The way that can be told"
    encoded = tokenizer.encode(prompt)
    prompt_ids = torch.tensor([encoded.ids]).to(device)

    # 1. Wave trajectory tracking
    print("\n1. Tracking wave trajectory during generation...")
    trajectory_tracker = WaveTrajectoryTracker()

    trajectory_data = trajectory_tracker.track_generation(
        model=model,
        prompt_ids=prompt_ids,
        max_length=20,
        temperature=0.8
    )

    print(f"   Tracked {len(trajectory_data['energy'])} generation steps")
    print(f"   Mean energy: {trajectory_data['energy'].mean():.4f}")

    # Visualize
    trajectory_tracker.plot_trajectory(
        trajectory_data,
        save_path=f"{output_dir}/trajectory.png"
    )
    print(f"   Saved trajectory plot")

    # Detect mode collapse
    mode_collapse = trajectory_tracker.detect_mode_collapse(trajectory_data)
    print(f"   Mode collapse detected: {mode_collapse}")

    # 2. Confidence tracking
    print("\n2. Tracking generation confidence...")
    confidence_tracker = GenerationConfidenceTracker()

    confidence_data = confidence_tracker.track_generation(
        model=model,
        prompt_ids=prompt_ids,
        max_length=20,
        temperature=0.8
    )

    print(f"   Mean confidence: {confidence_data['confidence'].mean():.4f}")
    print(f"   Mean entropy: {confidence_data['entropy'].mean():.4f}")

    # Visualize
    confidence_tracker.plot_confidence_trajectory(
        confidence_data,
        save_path=f"{output_dir}/confidence.png"
    )
    print(f"   Saved confidence plot")

    # Identify uncertain regions
    uncertain = confidence_tracker.identify_uncertain_regions(
        confidence_data,
        threshold=0.5
    )
    print(f"   Uncertain token positions: {uncertain}")

    # 3. Round-trip analysis
    print("\n3. Analyzing token → wave → token round-trip...")
    roundtrip_analyzer = RoundTripAnalyzer(model, tokenizer)

    test_text = "The way that can be told is not the eternal way"
    roundtrip_results = roundtrip_analyzer.analyze_roundtrip(test_text)

    print(f"   Reconstruction accuracy: {roundtrip_results['reconstruction_accuracy']:.2%}")
    print(f"   Mean per-position loss: {roundtrip_results['per_position_loss'].mean():.4f}")

    # Visualize
    roundtrip_analyzer.plot_roundtrip_analysis(
        roundtrip_results,
        save_path=f"{output_dir}/roundtrip.png"
    )
    print(f"   Saved round-trip analysis plot")

    print("\n✓ Generation analysis demo complete")
    print(f"   Results saved to: {output_dir}")


def demo_comparative_analysis(model, sample_input, device, output_dir='analysis_results/comparative'):
    """Demonstrate comparative analysis tools."""
    print("\n" + "="*80)
    print("DEMO 5: Comparative Analysis")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Input comparison
    print("\n1. Comparing different inputs...")
    input_comparator = InputComparator(model)

    # Create multiple synthetic inputs for demo
    inputs = [
        {'token_ids': sample_input['input_ids'][:1].to(device),
         'attention_mask': sample_input['attention_mask'][:1].to(device)},
        {'token_ids': sample_input['input_ids'][:1].to(device) + 1,  # Slightly different
         'attention_mask': sample_input['attention_mask'][:1].to(device)},
    ]
    labels = ['Input 1', 'Input 2']

    comparison = input_comparator.compare_inputs(
        input_list=inputs,
        labels=labels
    )

    print(f"   Compared {len(comparison)} inputs")

    # Compute similarity
    similarity = input_comparator.compute_input_similarity(comparison)
    print(f"   Similarity matrix:\n{similarity}")

    # Visualize
    input_comparator.plot_input_comparison(
        comparison,
        labels=labels,
        save_path=f"{output_dir}/input_comparison.png"
    )
    print(f"   Saved input comparison plot")

    # 2. Ablation study
    print("\n2. Running ablation study...")

    with AblationHelper(model) as ablation_helper:
        # Define ablation configurations
        ablation_configs = [
            {'name': 'baseline', 'ablation': None},
            {'name': 'no_harmonics_0-4', 'ablation': ('harmonics', list(range(5)), 'zero')},
            {'name': 'no_phase', 'ablation': ('component', 'phases', 'zero')},
        ]

        # Simple evaluation function for demo
        def simple_eval(model_to_eval):
            model_to_eval.eval()
            with torch.no_grad():
                logits = model_to_eval(
                    encoder_input={'token_ids': sample_input['input_ids'][:1].to(device)},
                    attention_mask=sample_input['attention_mask'][:1].to(device)
                )
                return {'loss': logits.abs().mean().item()}

        # Run ablation study
        results = ablation_helper.run_ablation_study(
            ablation_configs=ablation_configs,
            evaluation_fn=simple_eval,
            metrics=['loss']
        )

        print(f"\n   Ablation Results:")
        print(results)

        # Visualize
        ablation_helper.plot_ablation_results(
            results,
            save_path=f"{output_dir}/ablation_results.png"
        )
        print(f"   Saved ablation results plot")

    print("\n✓ Comparative analysis demo complete")
    print(f"   Results saved to: {output_dir}")


def demo_publication_report(output_dir='analysis_results/publication'):
    """Demonstrate publication-quality report generation."""
    print("\n" + "="*80)
    print("DEMO 6: Publication Report Generation")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Create report generator
    generator = PaperReportGenerator(
        output_dir=output_dir,
        style=PublicationStyle.IEEE
    )

    # Generate synthetic data for demo
    import numpy as np

    # 1. Training curves
    print("\n1. Creating training curve figure...")
    train_losses = np.random.randn(100).cumsum() / 10 + 2.0
    val_losses = train_losses + np.random.randn(100) * 0.1

    generator.create_training_curve_figure(
        train_metrics={'loss': train_losses, 'perplexity': np.exp(train_losses)},
        val_metrics={'loss': val_losses, 'perplexity': np.exp(val_losses)},
        save_path=f"{output_dir}/training_curves"
    )
    print(f"   Saved training curves (IEEE style)")

    # 2. Generate LaTeX table
    print("\n2. Generating LaTeX table...")
    import pandas as pd

    results_df = pd.DataFrame({
        'Model': ['Baseline', 'Wave-32', 'Wave-64', 'Wave-128'],
        'Loss': [2.45, 2.31, 2.18, 2.09],
        'Perplexity': [11.6, 10.1, 8.8, 8.1],
        'Parameters (M)': [45.2, 47.8, 52.1, 60.3]
    })

    latex_code = generator.generate_latex_table(
        results_df,
        caption='Comparison of Wave Transformer variants',
        label='tab:results'
    )

    with open(f"{output_dir}/table_results.tex", 'w') as f:
        f.write(latex_code)

    print(f"   Saved LaTeX table code")
    print(f"\n{latex_code}\n")

    # 3. Generate full report
    print("\n3. Generating comprehensive report...")
    analysis_results = {
        'experiment_name': 'wave_transformer_analysis',
        'training_data': {
            'train_losses': train_losses.tolist(),
            'val_losses': val_losses.tolist()
        },
        'model_config': {
            'num_harmonics': 64,
            'num_layers': 6,
            'd_model': 512
        }
    }

    report_path = generator.generate_full_report(
        analysis_results,
        output_dir=output_dir
    )

    print(f"   Full report generated at: {report_path}")

    print("\n✓ Publication report generation complete")
    print(f"   Results saved to: {output_dir}")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("WAVE TRANSFORMER COMPREHENSIVE ANALYSIS DEMO")
    print("="*80)
    print("\nThis demo showcases the complete analysis suite capabilities.")
    print("Each section demonstrates different analysis tools.\n")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create a small model for demonstration
    print("Creating demo model...")
    vocab_size = 1000
    num_harmonics = 32
    d_model = 128

    encoder = TokenToWaveEncoderSlim(
        vocab_size=vocab_size,
        num_harmonics=num_harmonics,
        d_model=d_model,
        num_heads=4,
        num_heads_kv=4,
        num_layers=2
    )

    decoder = WaveToTokenDecoder(
        vocab_size=vocab_size,
        num_harmonics=num_harmonics,
        d_model=d_model,
        num_heads=4,
        num_heads_kv=4,
        num_layers=2
    )

    model = WaveTransformer(
        wave_encoder=encoder,
        wave_decoder=decoder,
        num_harmonics=num_harmonics,
        transformer_num_heads=4,
        transformer_heads_kv=4,
        transformer_num_layers=2,
        dropout=0.1
    ).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Create synthetic data
    batch_size = 4
    seq_len = 32

    sample_input = {
        'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len)
    }

    # Simulate a mini dataloader
    class MiniDataset:
        def __init__(self, data, num_batches=10):
            self.data = data
            self.num_batches = num_batches

        def __iter__(self):
            for _ in range(self.num_batches):
                yield self.data

        def __len__(self):
            return self.num_batches

    dataloader = MiniDataset(sample_input)

    # Create a simple tokenizer for demo
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size

        def encode(self, text):
            class Encoding:
                def __init__(self, ids):
                    self.ids = ids
            # Simple character-based encoding for demo
            return Encoding([ord(c) % self.vocab_size for c in text])

        def decode(self, ids):
            return ''.join([chr(i % 128 + 32) for i in ids])

    tokenizer = SimpleTokenizer(vocab_size)

    # Run demos
    try:
        demo_basic_wave_statistics()
        demo_training_monitoring(model, dataloader, device)
        demo_model_introspection(model, sample_input, device)
        demo_generation_analysis(model, tokenizer, device)
        demo_comparative_analysis(model, sample_input, device)
        demo_publication_report()

        print("\n" + "="*80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nResults saved to: analysis_results/")
        print("\nExplore the generated files:")
        print("  - training/: Training monitoring results")
        print("  - introspection/: Model introspection visualizations")
        print("  - generation/: Generation analysis plots")
        print("  - comparative/: Comparative analysis results")
        print("  - publication/: Publication-ready figures and tables")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
