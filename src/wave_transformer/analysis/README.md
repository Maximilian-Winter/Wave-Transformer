# Wave Transformer Analysis Suite

A comprehensive toolkit for analyzing Wave Transformer representations, training dynamics, and model behavior.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Core Modules](#core-modules)
- [Quick Start](#quick-start)
- [Detailed Examples](#detailed-examples)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

## Overview

The Wave Transformer Analysis Suite provides production-ready tools for:

1. **Statistical Analysis**: Comprehensive statistics on wave representations (frequencies, amplitudes, phases)
2. **Data Collection**: Efficient collectors for capturing training dynamics with minimal overhead
3. **Export Utilities**: Multi-format export (JSON, HDF5, TensorBoard, W&B) with publication-quality figures
4. **Memory Efficiency**: Specialized data structures for long-running experiments

## Installation

The analysis suite is included with the Wave Transformer package. Optional dependencies:

```bash
# For HDF5 export (recommended for large datasets)
pip install h5py

# For Weights & Biases logging
pip install wandb

# For TensorBoard support
pip install tensorboard
```

## Core Modules

### 1. Wave Statistics (`core.wave_statistics`)

Compute comprehensive statistics on Wave objects:

**Key Features:**
- Basic statistics (mean, std, min, max, median, variance)
- Harmonic importance ranking by amplitude/energy/variance
- Phase coherence measurements (global and local)
- Spectral features (centroid, bandwidth, entropy)
- Energy analysis (total, per-position, per-harmonic)
- Phase velocity and amplitude envelope

**Main Class:** `WaveStatistics` - All static methods, works with `@torch.no_grad()`

### 2. Data Collectors (`core.collectors`)

Efficient data collection during training:

**Available Collectors:**
- `WaveCollector`: Capture wave statistics at intervals
- `GradientCollector`: Track gradient flow and norms
- `LossCollector`: Monitor per-position loss breakdown
- `ActivationCollector`: Record intermediate activations

**Key Features:**
- Configurable sampling intervals to reduce overhead
- Distributed training support (collect on specific rank)
- Maximum sample limits for memory management
- Abstract base class for custom collectors

### 3. Export Utilities (`core.exporters`)

Multi-format export with intelligent type handling:

**Supported Formats:**
- **JSON**: Human-readable, handles numpy/torch types automatically
- **HDF5**: Efficient for large datasets, supports compression
- **TensorBoard**: Direct integration with SummaryWriter
- **Weights & Biases**: One-line logging to W&B

**Publication Features:**
- `create_paper_figure()`: Publication-quality matplotlib figures
- `save_figure()`: Multi-format export (PNG, PDF, SVG, EPS)
- Automatic metadata embedding

### 4. Memory-Efficient Storage (`utils.memory_efficient`)

Specialized data structures for long experiments:

**Available Classes:**
- `CircularBuffer`: Fixed-size rolling window (constant memory)
- `DownsampledStorage`: Multi-resolution storage (100x+ compression)
- `StreamingStatistics`: Online mean/variance (Welford's algorithm)
- `ExponentialMovingAverage`: Smoothed metrics
- `SlidingWindowStatistics`: Recent history statistics

## Quick Start

### Basic Wave Analysis

```python
from wave_transformer.analysis import WaveStatistics

# Analyze a wave representation
wave = model.get_wave_output(input_data)

# Basic statistics
stats = WaveStatistics.compute_basic_stats(wave, component='all')
print(f"Mean amplitude: {stats['amplitudes'].mean}")

# Harmonic importance
importance = WaveStatistics.compute_harmonic_importance(wave, metric='energy')
top_10_indices, top_10_scores = importance.top_k(10)

# Spectral features
centroid = WaveStatistics.compute_spectral_centroid(wave)
coherence = WaveStatistics.compute_phase_coherence(wave)
```

### Training Monitoring

```python
from wave_transformer.analysis import WaveCollector, GradientCollector

# Initialize collectors
wave_collector = WaveCollector(
    sample_interval=100,  # Collect every 100 steps
    max_samples=1000,     # Keep last 1000 samples
    statistics_to_collect=['basic_stats', 'harmonic_importance', 'total_energy']
)

grad_collector = GradientCollector(
    sample_interval=10,
    track_layer_names=['.*attn.*', '.*ffn.*']  # Regex patterns
)

# During training
for step, batch in enumerate(dataloader):
    wave = model(batch)
    loss.backward()

    # Collect data
    wave_collector.collect(wave, step=step)
    grad_collector.collect(model, step=step)

    optimizer.step()

# Get results
wave_data = wave_collector.get_data()
grad_data = grad_collector.get_data()
```

### Export Results

```python
from wave_transformer.analysis import AnalysisExporter

# Export to JSON
AnalysisExporter.to_json(results, 'results.json')

# Export to HDF5
AnalysisExporter.to_hdf5(results, 'results.h5', compression='gzip')

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
AnalysisExporter.to_tensorboard(results, writer, step=1000)

# Weights & Biases
import wandb
wandb.init(project='wave-transformer')
AnalysisExporter.to_wandb(results, step=1000)

# Publication figure
fig, ax = AnalysisExporter.create_paper_figure(figsize=(8, 6))
ax.plot(data)
AnalysisExporter.save_figure(fig, 'figure', formats=['png', 'pdf'])
```

### Memory-Efficient Storage

```python
from wave_transformer.analysis import (
    CircularBuffer,
    StreamingStatistics,
    DownsampledStorage
)

# Circular buffer for recent history
buffer = CircularBuffer(capacity=1000)
for value in data_stream:
    buffer.append(value)

recent_mean = buffer.mean(last_n=100)

# Streaming statistics (constant memory)
stats = StreamingStatistics(shape=(64,))
for batch in dataloader:
    stats.update(batch.mean(dim=0))

print(f"Running mean: {stats.get_mean()}")
print(f"Running std: {stats.get_std()}")

# Downsampled storage (100x compression)
storage = DownsampledStorage(
    full_resolution_size=1000,
    downsample_factor=10,
    num_levels=3
)
for i in range(100000):
    storage.append(data[i])

all_data = storage.get_all()  # Only ~1000 samples stored
```

## Detailed Examples

### Example 1: Analyzing Harmonic Evolution

```python
from wave_transformer.analysis import WaveCollector, AnalysisExporter
import matplotlib.pyplot as plt

# Collect wave statistics over training
collector = WaveCollector(
    sample_interval=50,
    statistics_to_collect=['harmonic_importance', 'total_energy', 'spectral_centroid']
)

# Training loop
for step, batch in enumerate(dataloader):
    wave = model(batch)
    collector.collect(wave, step=step, metadata={'epoch': epoch, 'lr': current_lr})
    # ... training code ...

# Analyze results
data = collector.get_data()

# Plot energy evolution
fig, ax = AnalysisExporter.create_paper_figure()
ax.plot(data['step'], data['total_energy'])
ax.set_xlabel('Training Step')
ax.set_ylabel('Total Wave Energy')
ax.set_title('Wave Energy Evolution During Training')
AnalysisExporter.save_figure(fig, 'energy_evolution', formats=['png', 'pdf'])

# Export full results
AnalysisExporter.to_hdf5(data, 'training_analysis.h5')
```

### Example 2: Gradient Flow Analysis

```python
from wave_transformer.analysis import GradientCollector

# Track gradients
grad_collector = GradientCollector(
    sample_interval=10,
    compute_histograms=True,
    histogram_bins=50
)

# After backward pass
for step in range(num_steps):
    loss.backward()
    grad_collector.collect(model, step=step)
    optimizer.step()

# Analyze gradient flow
summary = grad_collector.get_gradient_flow_summary()

print("Gradient Flow Summary:")
for layer_name, stats in summary['layer_norms'].items():
    print(f"{layer_name}:")
    print(f"  Mean norm: {stats['mean']:.4f}")
    print(f"  Max norm:  {stats['max']:.4f}")
```

### Example 3: Per-Position Loss Analysis

```python
from wave_transformer.analysis import LossCollector
import torch.nn.functional as F

# Collect per-position losses
loss_collector = LossCollector(sample_interval=20, reduction='mean')

for step, (input_ids, target_ids) in enumerate(dataloader):
    logits = model(input_ids)

    # Compute per-position loss (no reduction)
    loss_per_position = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction='none'
    ).reshape(batch_size, seq_len).mean(dim=0)

    loss_collector.collect(loss_per_position, step=step)

    # ... backward pass ...

# Find hardest positions
hardest_positions = loss_collector.get_hardest_positions(k=10)
difficulty_profile = loss_collector.get_position_difficulty_profile()

print(f"Hardest positions: {hardest_positions}")
```

### Example 4: Long-Running Experiment Monitoring

```python
from wave_transformer.analysis import (
    DownsampledStorage,
    StreamingStatistics,
    ExponentialMovingAverage
)

# For 1M+ steps, use memory-efficient storage
energy_storage = DownsampledStorage(
    full_resolution_size=1000,
    downsample_factor=10,
    num_levels=3
)

running_stats = StreamingStatistics()
ema = ExponentialMovingAverage(alpha=0.01)

# Training loop
for step in range(1_000_000):
    wave = model(batch)
    energy = WaveStatistics.compute_total_energy(wave).item()

    # Update all trackers
    energy_storage.append(energy)
    running_stats.update(energy)
    ema.update(energy)

    if step % 1000 == 0:
        print(f"Step {step}:")
        print(f"  Running mean: {running_stats.get_mean():.4f}")
        print(f"  EMA: {ema.get():.4f}")

# Final statistics
storage_stats = energy_storage.get_statistics()
print(f"Compression: {storage_stats['compression_ratio']:.1f}x")
```

## API Reference

### WaveStatistics

All methods are static and decorated with `@torch.no_grad()`.

```python
# Basic statistics
stats = WaveStatistics.compute_basic_stats(
    wave: Wave,
    component: str = 'all',  # 'frequencies', 'amplitudes', 'phases', 'all'
    dim: Optional[int] = None  # Dimension to reduce (None = all)
) -> Dict[str, WaveStats]

# Harmonic importance
importance = WaveStatistics.compute_harmonic_importance(
    wave: Wave,
    metric: str = 'amplitude',  # 'amplitude', 'energy', 'variance'
    batch_idx: Optional[int] = None
) -> HarmonicImportance

# Phase coherence
coherence = WaveStatistics.compute_phase_coherence(
    wave: Wave,
    batch_idx: Optional[int] = None,
    window_size: int = 1  # 1 = global, >1 = local
) -> torch.Tensor

# Spectral centroid
centroid = WaveStatistics.compute_spectral_centroid(
    wave: Wave,
    batch_idx: Optional[int] = None,
    eps: float = 1e-8
) -> torch.Tensor

# Total energy
energy = WaveStatistics.compute_total_energy(
    wave: Wave,
    batch_idx: Optional[int] = None,
    per_position: bool = False
) -> torch.Tensor

# Frequency bandwidth
bandwidth = WaveStatistics.compute_frequency_bandwidth(
    wave: Wave,
    percentile: float = 90.0,
    batch_idx: Optional[int] = None
) -> torch.Tensor

# Additional methods
entropy = WaveStatistics.compute_harmonic_entropy(wave, ...)
phase_vel = WaveStatistics.compute_phase_velocity(wave, ...)
envelope = WaveStatistics.compute_amplitude_envelope(wave, ...)
```

### Data Collectors

```python
# WaveCollector
collector = WaveCollector(
    sample_interval: int = 100,
    max_samples: Optional[int] = 1000,
    collect_on_rank: int = 0,
    statistics_to_collect: Optional[List[str]] = None,
    batch_reduction: str = 'mean'  # 'mean', 'first', 'all'
)
collector.collect(wave, step=step, metadata=metadata)
data = collector.get_data()

# GradientCollector
grad_collector = GradientCollector(
    sample_interval: int = 10,
    track_layer_names: Optional[List[str]] = None,  # Regex patterns
    compute_histograms: bool = False
)
grad_collector.collect(model, step=step)
summary = grad_collector.get_gradient_flow_summary()

# LossCollector
loss_collector = LossCollector(
    sample_interval: int = 50,
    reduction: str = 'mean'  # 'mean', 'sum', 'none'
)
loss_collector.collect(loss_per_position, step=step)
hardest = loss_collector.get_hardest_positions(k=10)
```

### Export Utilities

```python
# JSON
AnalysisExporter.to_json(data, filepath, indent=2, compress=False)
data = AnalysisExporter.from_json(filepath)

# HDF5
AnalysisExporter.to_hdf5(data, filepath, compression='gzip', compression_opts=4)
data = AnalysisExporter.from_hdf5(filepath)

# TensorBoard
AnalysisExporter.to_tensorboard(data, writer, step=step, prefix='')

# Weights & Biases
AnalysisExporter.to_wandb(data, step=step, prefix='', commit=True)

# Publication figures
fig, ax = AnalysisExporter.create_paper_figure(
    figsize=(8, 6),
    dpi=300,
    style='seaborn-v0_8-paper',
    use_latex=False
)
AnalysisExporter.save_figure(
    fig, filepath,
    formats=['png', 'pdf'],
    metadata={'Title': 'My Figure'}
)
```

### Memory-Efficient Storage

```python
# CircularBuffer
buffer = CircularBuffer(capacity=1000, dtype=np.float32)
buffer.append(value)
data = buffer.get(last_n=100)
mean = buffer.mean()

# StreamingStatistics
stats = StreamingStatistics(shape=(64,))
stats.update(value)
mean = stats.get_mean()
std = stats.get_std()
merged = stats.merge(other_stats)

# DownsampledStorage
storage = DownsampledStorage(
    full_resolution_size=1000,
    downsample_factor=10,
    num_levels=3
)
storage.append(value)
all_data = storage.get_all()
stats = storage.get_statistics()

# ExponentialMovingAverage
ema = ExponentialMovingAverage(alpha=0.1)
ema.update(value)
smoothed = ema.get()
```

## Best Practices

### 1. Sampling for Efficiency

Always use sampling intervals for production training:

```python
# Good: Sample every 100 steps
collector = WaveCollector(sample_interval=100)

# Bad: Collect every step (overhead!)
collector = WaveCollector(sample_interval=1)
```

### 2. Memory Management

Set `max_samples` to prevent unbounded growth:

```python
collector = WaveCollector(
    sample_interval=100,
    max_samples=1000  # Stop after 1000 samples
)
```

### 3. Distributed Training

Collectors automatically detect distributed mode and collect only on specified rank:

```python
collector = WaveCollector(
    collect_on_rank=0  # Only collect on rank 0
)
```

### 4. Export Format Selection

- **JSON**: Small datasets, human-readable, easy debugging
- **HDF5**: Large datasets (>100MB), numerical data, compression
- **TensorBoard/W&B**: Real-time monitoring during training

### 5. Long Experiments

For experiments >100K steps, use memory-efficient storage:

```python
# Instead of lists
metrics = []
for i in range(1_000_000):
    metrics.append(value)  # Uses 8MB+ RAM

# Use downsampled storage
storage = DownsampledStorage(full_resolution_size=1000)
for i in range(1_000_000):
    storage.append(value)  # Uses ~80KB RAM (100x less!)
```

## Performance Notes

- All statistical computations use `@torch.no_grad()` for efficiency
- Collectors use lazy evaluation and sampling to minimize overhead
- Memory-efficient storage provides 10-100x compression
- Export utilities handle large datasets efficiently with streaming/chunking

## Contributing

When adding new analysis features:

1. Use `@torch.no_grad()` for analysis computations
2. Support both batched and single-sample analysis
3. Add type hints and comprehensive docstrings
4. Include examples in the module docstring
5. Test with various tensor shapes and devices (CPU/GPU)

## Citation

If you use this analysis suite in your research, please cite:

```bibtex
@software{wave_transformer_analysis,
  title={Wave Transformer Analysis Suite},
  author={Wave Transformer Team},
  year={2025},
  url={https://github.com/your-repo/wave-transformer}
}
```
