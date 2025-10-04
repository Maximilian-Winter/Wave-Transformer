# Introspection Tools - Quick Start Guide

Get started with Wave Transformer introspection in 5 minutes.

## Installation

```python
# Already installed if you have wave_transformer
from wave_transformer.analysis import (
    LayerWaveAnalyzer,
    HarmonicImportanceAnalyzer,
    WaveInterferenceAnalyzer,
    SpectrumEvolutionTracker,
)
```

## Basic Usage

### 1. Analyze Layer Evolution (30 seconds)

```python
from wave_transformer.analysis import LayerWaveAnalyzer

# Automatic hook cleanup with context manager
with LayerWaveAnalyzer(model, device='cuda') as analyzer:
    # Extract waves from all layers
    snapshots = analyzer.analyze_input(
        encoder_input={'token_ids': tokens},
        batch_idx=0
    )

    # Visualize (creates 'layer_evolution.png')
    analyzer.plot_layer_evolution(snapshots, save_path='layer_evolution.png')
```

**What you get:**
- Heatmaps showing amplitude, frequency, and phase evolution through layers
- One column per layer
- Visual understanding of wave transformations

### 2. Find Important Harmonics (2 minutes)

```python
from wave_transformer.analysis import HarmonicImportanceAnalyzer

analyzer = HarmonicImportanceAnalyzer(model, criterion, device='cuda')

# Analyze over dataset
importance = analyzer.analyze_harmonic_importance(
    dataloader,
    method='energy',  # or 'amplitude', 'variance', 'max_amplitude'
    max_batches=50    # Limit for speed
)

# Visualize
analyzer.plot_harmonic_importance(importance, top_k=32, save_path='importance.png')

# Get compression mask
mask = analyzer.get_sparse_harmonic_mask(importance, top_k=32)
print(f"Keeping {mask.sum()} / {len(mask)} harmonics")
```

**What you get:**
- Bar chart of harmonic importance scores
- Top-k most important harmonics highlighted
- Cumulative importance curve
- Ready-to-use compression mask

### 3. Analyze Wave Interference (1 minute)

```python
from wave_transformer.analysis import WaveInterferenceAnalyzer

analyzer = WaveInterferenceAnalyzer(model, device='cuda')

# Analyze interference between layers (uses snapshots from step 1)
interference = analyzer.analyze_layer_interference(
    snapshots,
    modes=['constructive', 'destructive', 'modulate']
)

# Visualize
analyzer.plot_interference_patterns(interference, save_path='interference.png')
```

**What you get:**
- Phase alignment across layers
- Frequency coupling strength
- Energy transfer ratios
- Spectral overlap metrics

### 4. Track Spectrum Evolution (1 minute)

```python
from wave_transformer.analysis import SpectrumEvolutionTracker

tracker = SpectrumEvolutionTracker(model, device='cuda')

# Extract spectrum at each layer
spectrum = tracker.extract_spectrum_evolution(
    snapshots,
    batch_idx=0,
    seq_position=0  # Which token to analyze
)

# Create 3D visualization
tracker.plot_spectrum_evolution(spectrum, mode='3d', save_path='spectrum_3d.png')

# Or waterfall plot
tracker.plot_spectrum_evolution(spectrum, mode='waterfall', save_path='waterfall.png')
```

**What you get:**
- 3D surface showing frequency spectrum evolution
- Spectral centroid and bandwidth tracking
- Energy redistribution visualization

## Complete 5-Minute Analysis

```python
import torch
from pathlib import Path
from wave_transformer.analysis import (
    LayerWaveAnalyzer,
    HarmonicImportanceAnalyzer,
    WaveInterferenceAnalyzer,
    SpectrumEvolutionTracker,
)

def quick_analysis(model, dataloader, output_dir='analysis_output'):
    """Run complete analysis in ~5 minutes"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get sample input
    sample_batch = next(iter(dataloader))
    sample_input = {'token_ids': sample_batch[0].cuda()}

    # 1. Layer analysis
    print("Analyzing layers...")
    with LayerWaveAnalyzer(model) as layer_analyzer:
        snapshots = layer_analyzer.analyze_input(sample_input)
        layer_analyzer.plot_layer_evolution(
            snapshots,
            save_path=output_dir / 'layers.png'
        )

    # 2. Harmonic importance
    print("Computing harmonic importance...")
    harmonic_analyzer = HarmonicImportanceAnalyzer(model, torch.nn.CrossEntropyLoss())
    importance = harmonic_analyzer.analyze_harmonic_importance(
        dataloader,
        method='energy',
        max_batches=50
    )
    harmonic_analyzer.plot_harmonic_importance(
        importance,
        top_k=32,
        save_path=output_dir / 'importance.png'
    )

    # 3. Interference
    print("Analyzing interference...")
    interference_analyzer = WaveInterferenceAnalyzer(model)
    interference = interference_analyzer.analyze_layer_interference(snapshots)
    interference_analyzer.plot_interference_patterns(
        interference,
        save_path=output_dir / 'interference.png'
    )

    # 4. Spectrum
    print("Tracking spectrum...")
    spectrum_tracker = SpectrumEvolutionTracker(model)
    spectrum = spectrum_tracker.extract_spectrum_evolution(snapshots)
    spectrum_tracker.plot_spectrum_evolution(
        spectrum,
        mode='waterfall',
        save_path=output_dir / 'spectrum.png'
    )

    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    return snapshots, importance, interference, spectrum

# Run it!
results = quick_analysis(model, dataloader)
```

## Common Patterns

### Pattern 1: Layer Comparison

```python
with LayerWaveAnalyzer(model) as analyzer:
    snapshots = analyzer.analyze_input(input_dict)

    # Compare specific metrics
    comparison = analyzer.compare_layers(
        snapshots,
        metrics=['amplitude_mean', 'spectral_centroid']
    )

    # Plot evolution
    analyzer.plot_metric_evolution(comparison, layer_names=[s.layer_name for s in snapshots])
```

### Pattern 2: Model Compression

```python
# Find important harmonics
analyzer = HarmonicImportanceAnalyzer(model, criterion)
importance = analyzer.analyze_harmonic_importance(dataloader, method='energy')

# Create 50% sparse model
mask = analyzer.get_sparse_harmonic_mask(importance, cumulative_threshold=0.5)

# Test performance with mask
masked_wave = analyzer.apply_harmonic_mask(wave, mask, zero_amplitudes=True)
```

### Pattern 3: Multi-Position Spectrum Analysis

```python
tracker = SpectrumEvolutionTracker(model)

for pos in [0, 10, 20]:
    spectrum = tracker.extract_spectrum_evolution(snapshots, seq_position=pos)
    tracker.plot_spectrum_evolution(
        spectrum,
        mode='3d',
        save_path=f'spectrum_pos_{pos}.png'
    )
```

## Tips & Tricks

### Faster Analysis
- Use `max_batches` parameter to limit dataset iteration
- Sample harmonics for gradient sensitivity: `harmonic_indices=list(range(0, 64, 4))`
- Use 2D visualizations instead of 3D for speed

### Better Visualizations
- Use `seq_slice=(0, 50)` to focus on specific sequence regions
- Set `figsize=(20, 12)` for larger, more detailed plots
- Combine multiple `batch_idx` to see batch variation

### Memory Efficiency
- Always use `with LayerWaveAnalyzer(model) as analyzer:` for automatic cleanup
- Set `max_batches=10` for quick tests before full runs
- Use `@torch.no_grad()` around analysis calls (already done internally)

## Troubleshooting

**Q: Hooks not capturing outputs?**
```python
# Make sure to call analyze_input() after register_extraction_hooks()
with LayerWaveAnalyzer(model) as analyzer:
    # Hooks automatically registered by context manager
    snapshots = analyzer.analyze_input(input_dict)
```

**Q: Out of memory?**
```python
# Reduce max_batches
importance = analyzer.analyze_harmonic_importance(
    dataloader,
    max_batches=10  # Start small
)
```

**Q: Want to analyze specific layers?**
```python
# Filter snapshots
encoder_snapshot = [s for s in snapshots if s.is_encoder][0]
transformer_snapshots = [s for s in snapshots if not s.is_encoder and not s.is_decoder]
```

**Q: Need metrics for specific harmonics?**
```python
# Access raw data from snapshots
wave = snapshots[0].wave
harmonic_5_amplitude = wave.amplitudes[batch_idx, :, 5]  # All positions, harmonic 5
```

## Next Steps

- See [README.md](README.md) for complete documentation
- Check [../../../examples/introspection_demo.py](../../../examples/introspection_demo.py) for full example
- Explore advanced use cases in the README

## One-Liner Examples

```python
# Quick layer visualization
with LayerWaveAnalyzer(model) as a: a.plot_layer_evolution(a.analyze_input({'token_ids': tokens}), save_path='out.png')

# Quick importance check
print(HarmonicImportanceAnalyzer(model, criterion).analyze_harmonic_importance(dl, max_batches=10)['importance'].argsort()[::-1][:5])

# Quick spectrum view
SpectrumEvolutionTracker(model).plot_spectrum_evolution(SpectrumEvolutionTracker(model).extract_spectrum_evolution(snapshots), mode='waterfall', save_path='s.png')
```
