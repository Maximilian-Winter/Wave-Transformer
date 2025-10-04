# Wave Transformer Introspection Tools

This module provides advanced introspection and analysis capabilities for Wave Transformer models. It enables deep inspection of how wave representations evolve through the network, which harmonics are most important, how waves interfere, and how frequency spectra change across layers.

## Modules

### 1. LayerWaveAnalyzer (`layer_analyzer.py`)

Extracts and analyzes wave representations at each layer of the WaveTransformer.

**Key Features:**
- Registers hooks on encoder, transformer layers, and decoder
- Extracts wave representations from all layers in a single forward pass
- Compares wave evolution using multiple metrics
- Visualizes layer-by-layer transformations

**Usage:**
```python
from wave_transformer.analysis import LayerWaveAnalyzer

# Use as context manager for automatic cleanup
with LayerWaveAnalyzer(model, device='cuda') as analyzer:
    # Extract waves from all layers
    snapshots = analyzer.analyze_input(
        encoder_input={'token_ids': input_tokens},
        batch_idx=0
    )

    # Compare layers
    comparison = analyzer.compare_layers(
        snapshots,
        metrics=['amplitude_mean', 'frequency_mean', 'spectral_centroid']
    )

    # Visualize evolution
    analyzer.plot_layer_evolution(snapshots, save_path='layer_evolution.png')
```

**Available Metrics:**
- `amplitude_mean`: Mean amplitude across harmonics
- `amplitude_energy`: Total energy (sum of squared amplitudes)
- `frequency_mean`: Mean frequency across harmonics
- `frequency_std`: Frequency diversity
- `phase_std`: Phase diversity
- `spectral_centroid`: Amplitude-weighted mean frequency

### 2. HarmonicImportanceAnalyzer (`harmonic_analyzer.py`)

Identifies which harmonics contribute most to model predictions and supports model compression.

**Key Features:**
- Multiple importance metrics (amplitude, energy, variance, gradient sensitivity)
- Efficient batch processing over datasets
- Harmonic pruning mask generation for model compression
- Gradient-based ablation studies

**Usage:**
```python
from wave_transformer.analysis import HarmonicImportanceAnalyzer

analyzer = HarmonicImportanceAnalyzer(model, criterion, device='cuda')

# Analyze importance using energy metric
importance = analyzer.analyze_harmonic_importance(
    dataloader,
    method='energy',
    layer_name='encoder',
    max_batches=100
)

# Visualize importance
analyzer.plot_harmonic_importance(importance, top_k=32)

# Create sparse mask for compression
mask = analyzer.get_sparse_harmonic_mask(
    importance,
    top_k=32  # Keep top 32 harmonics
)

# Or use cumulative threshold
mask = analyzer.get_sparse_harmonic_mask(
    importance,
    cumulative_threshold=0.95  # Keep harmonics that account for 95% of importance
)

# Gradient-based sensitivity analysis (slower but more accurate)
sensitivity = analyzer.compute_gradient_sensitivity(
    dataloader,
    max_batches=10
)
```

**Importance Methods:**
- `amplitude`: Mean amplitude per harmonic across dataset
- `energy`: Sum of squared amplitudes (default)
- `variance`: Variance of harmonic values across dataset
- `max_amplitude`: Maximum amplitude observed

### 3. WaveInterferenceAnalyzer (`interference_analyzer.py`)

Analyzes wave interference patterns between layers using the Wave's built-in interference methods.

**Key Features:**
- Uses `Wave.interfere_with()` for three interference modes
- Computes phase alignment, frequency coupling, and energy transfer metrics
- Pairwise layer interference matrix
- Detailed component-level visualization

**Usage:**
```python
from wave_transformer.analysis import WaveInterferenceAnalyzer

analyzer = WaveInterferenceAnalyzer(model, device='cuda')

# Analyze interference between consecutive layers
interference_results = analyzer.analyze_layer_interference(
    snapshots,
    modes=['constructive', 'destructive', 'modulate'],
    batch_idx=0
)

# Visualize patterns
analyzer.plot_interference_patterns(
    interference_results,
    save_path='interference.png'
)

# Detailed component visualization
analyzer.visualize_interference_components(
    wave1, wave2,
    batch_idx=0,
    seq_position=0,
    save_path='components.png'
)

# Compute pairwise interference matrix
matrix = analyzer.compute_pairwise_interference_matrix(
    snapshots,
    mode='constructive',
    metric='energy_transfer'
)
```

**Interference Modes:**
- `constructive`: Phases align, amplitudes add
- `destructive`: Phases oppose, amplitudes subtract
- `modulate`: Frequency modulation, amplitude multiplication

**Metrics Computed:**
- `phase_alignment`: Cosine similarity of phases (-1 to 1)
- `frequency_coupling`: Correlation of input frequencies
- `amplitude_correlation`: Correlation of input amplitudes
- `energy_transfer`: Ratio of output to input energy
- `spectral_overlap`: Frequency distribution overlap (Bhattacharyya coefficient)

### 4. SpectrumEvolutionTracker (`spectrum_tracker.py`)

Tracks how frequency spectra evolve through transformer layers with advanced visualizations.

**Key Features:**
- Extracts frequency spectrum at each layer
- Computes spectral centroids, bandwidth, peak frequencies
- Measures spectral shifts between layers
- 3D, stacked 2D, and waterfall visualizations
- Frequency distribution evolution tracking

**Usage:**
```python
from wave_transformer.analysis import SpectrumEvolutionTracker

tracker = SpectrumEvolutionTracker(model, device='cuda')

# Extract spectrum evolution for a sequence position
spectrum_evolution = tracker.extract_spectrum_evolution(
    snapshots,
    batch_idx=0,
    seq_position=0
)

# Compute spectral shifts
shifts = tracker.compute_spectral_shift(spectrum_evolution)

# 3D visualization
tracker.plot_spectrum_evolution(
    spectrum_evolution,
    mode='3d',
    save_path='spectrum_3d.png'
)

# Waterfall plot
tracker.plot_spectrum_evolution(
    spectrum_evolution,
    mode='waterfall',
    save_path='spectrum_waterfall.png'
)

# Plot all metrics
tracker.plot_spectral_metrics(
    spectrum_evolution,
    shifts=shifts,
    save_path='metrics.png'
)

# Analyze frequency distribution evolution
dist_analysis = tracker.analyze_frequency_distribution_shift(
    spectrum_evolution,
    num_bins=30
)
tracker.plot_frequency_distribution_evolution(dist_analysis)
```

**Visualization Modes:**
- `3d`: 3D surface plot (layer × harmonic × amplitude)
- `2d_stacked`: Stacked 2D spectra with offset
- `waterfall`: Waterfall plot with color gradient

**Metrics Tracked:**
- `spectral_centroid`: Amplitude-weighted mean frequency
- `bandwidth`: Amplitude-weighted frequency spread
- `peak_frequency`: Frequency with maximum amplitude
- `total_energy`: Sum of squared amplitudes
- `centroid_shift`: Change in spectral centroid between layers
- `bandwidth_change`: Change in spectral bandwidth
- `energy_redistribution`: How much energy moves between harmonics
- `frequency_drift`: Average frequency change

## Complete Workflow Example

```python
from wave_transformer.analysis import (
    LayerWaveAnalyzer,
    HarmonicImportanceAnalyzer,
    WaveInterferenceAnalyzer,
    SpectrumEvolutionTracker,
)

# Step 1: Extract layer-wise wave representations
with LayerWaveAnalyzer(model) as layer_analyzer:
    snapshots = layer_analyzer.analyze_input(
        encoder_input={'token_ids': input_tokens}
    )
    layer_analyzer.plot_layer_evolution(snapshots, save_path='layers.png')

# Step 2: Identify important harmonics
harmonic_analyzer = HarmonicImportanceAnalyzer(model, criterion)
importance = harmonic_analyzer.analyze_harmonic_importance(
    dataloader,
    method='energy',
    max_batches=100
)
harmonic_analyzer.plot_harmonic_importance(importance, save_path='importance.png')

# Step 3: Analyze wave interference
interference_analyzer = WaveInterferenceAnalyzer(model)
interference = interference_analyzer.analyze_layer_interference(snapshots)
interference_analyzer.plot_interference_patterns(interference, save_path='interference.png')

# Step 4: Track spectrum evolution
spectrum_tracker = SpectrumEvolutionTracker(model)
spectrum = spectrum_tracker.extract_spectrum_evolution(snapshots)
spectrum_tracker.plot_spectrum_evolution(spectrum, mode='3d', save_path='spectrum.png')
```

## Advanced Use Cases

### Model Compression via Harmonic Pruning

```python
# Analyze importance
importance = harmonic_analyzer.analyze_harmonic_importance(dataloader, method='energy')

# Create sparse mask keeping top 50% of harmonics
mask = harmonic_analyzer.get_sparse_harmonic_mask(
    importance,
    cumulative_threshold=0.5
)

# Apply mask to wave for testing
masked_wave = harmonic_analyzer.apply_harmonic_mask(
    wave,
    mask,
    zero_amplitudes=True
)
```

### Layer-by-Layer Metric Tracking

```python
with LayerWaveAnalyzer(model) as analyzer:
    snapshots = analyzer.analyze_input(encoder_input)

    # Compare multiple metrics
    comparison = analyzer.compare_layers(
        snapshots,
        metrics=['amplitude_mean', 'amplitude_energy', 'spectral_centroid']
    )

    # Plot metric evolution
    layer_names = [s.layer_name for s in snapshots]
    analyzer.plot_metric_evolution(comparison, layer_names)
```

### Interference Matrix Analysis

```python
# Compute pairwise interference for all layer pairs
matrix = interference_analyzer.compute_pairwise_interference_matrix(
    snapshots,
    mode='constructive',
    metric='energy_transfer'
)

# Visualize as heatmap
interference_analyzer.plot_interference_matrix(
    matrix,
    layer_names=[s.layer_name for s in snapshots],
    save_path='matrix.png'
)
```

### Multi-Resolution Spectrum Analysis

```python
# Analyze spectrum at multiple sequence positions
positions = [0, 10, 20, 30]

for pos in positions:
    spectrum = tracker.extract_spectrum_evolution(snapshots, seq_position=pos)
    tracker.plot_spectrum_evolution(
        spectrum,
        mode='waterfall',
        save_path=f'spectrum_pos_{pos}.png'
    )
```

## Performance Considerations

### Memory Efficiency

- **LayerWaveAnalyzer**: Hooks store references to layer outputs. Use context manager to ensure cleanup.
- **HarmonicImportanceAnalyzer**: Use `max_batches` parameter to limit memory usage during analysis.
- **All analyzers**: Use `@torch.no_grad()` internally for memory-efficient evaluation.

### Speed Optimization

- **Gradient sensitivity analysis** is slower than other methods. Sample harmonics or use fewer batches for faster results.
- **Interference matrix computation** scales as O(L²) where L is the number of layers. For large models, analyze specific layer pairs.
- **3D visualizations** can be slow for large numbers of layers. Use 2D modes for faster rendering.

### Batch Processing

```python
# Efficient batch processing for harmonic analysis
importance = harmonic_analyzer.analyze_harmonic_importance(
    dataloader,
    method='energy',
    max_batches=50,  # Limit batches for faster analysis
    layer_name='encoder'
)
```

## Integration with Training

These tools can be integrated into training loops for continuous monitoring:

```python
from wave_transformer.analysis.training import AnalysisCallback

class IntrospectionCallback(AnalysisCallback):
    def __init__(self, model, dataloader, output_dir):
        super().__init__()
        self.layer_analyzer = LayerWaveAnalyzer(model)
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        with self.layer_analyzer:
            snapshots = self.layer_analyzer.analyze_input(sample_input)
            self.layer_analyzer.plot_layer_evolution(
                snapshots,
                save_path=self.output_dir / f'epoch_{epoch}_layers.png'
            )

# Use in training
callback = IntrospectionCallback(model, val_dataloader, Path('checkpoints/analysis'))
trainer.fit(model, callbacks=[callback])
```

## API Reference

See individual module docstrings for complete API documentation:
- `LayerWaveAnalyzer`: Layer-wise wave extraction and comparison
- `HarmonicImportanceAnalyzer`: Harmonic importance and compression
- `WaveInterferenceAnalyzer`: Wave interference pattern analysis
- `SpectrumEvolutionTracker`: Frequency spectrum evolution tracking

## Citation

If you use these introspection tools in your research, please cite:

```bibtex
@software{wave_transformer_introspection,
  title={Wave Transformer Introspection Tools},
  author={Wave Transformer Team},
  year={2025},
  url={https://github.com/your-repo/wave-transformer}
}
```
