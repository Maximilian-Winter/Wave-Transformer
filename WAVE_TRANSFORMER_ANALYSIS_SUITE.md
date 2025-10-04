# Wave Transformer Analysis & Visualization Suite

## 🌊 Complete Analysis Toolkit for Wave-Based Neural Architectures

A comprehensive, production-ready analysis and visualization suite specifically designed for the Wave Transformer architecture. This suite provides deep insights into wave-based representations, training dynamics, and model behavior through specialized tools for statistics, monitoring, introspection, generation analysis, and comparative studies.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Core Modules](#core-modules)
6. [Usage Examples](#usage-examples)
7. [Integration with Training](#integration-with-training)
8. [API Reference](#api-reference)
9. [Advanced Features](#advanced-features)
10. [Performance Considerations](#performance-considerations)
11. [Contributing](#contributing)

---

## 🎯 Overview

### What is the Wave Transformer Analysis Suite?

The Wave Transformer uses dense wave representations (frequencies, amplitudes, phases) instead of traditional embeddings. This creates unique analysis challenges and opportunities. Our analysis suite provides:

- **Wave-Specific Statistics**: Harmonic importance, spectral analysis, phase coherence
- **Training Monitoring**: Real-time wave evolution, gradient flow, loss breakdown
- **Model Introspection**: Layer-wise analysis, interference patterns, spectrum evolution
- **Generation Analysis**: Trajectory tracking, confidence monitoring, round-trip fidelity
- **Comparative Tools**: Checkpoint comparison, ablation studies, input similarity
- **Visualization**: TensorBoard/W&B integration, publication-quality figures

### Key Features

✅ **Comprehensive**: 15+ analysis modules covering all aspects
✅ **Production-Ready**: Type hints, docstrings, error handling
✅ **Efficient**: Minimal overhead, memory-conscious design
✅ **Distributed**: Full DDP support for multi-GPU training
✅ **Flexible**: Modular design, use any component independently
✅ **Well-Documented**: Extensive examples and API documentation

---

## 🏗️ Architecture

### Module Organization

```
src/wave_transformer/analysis/
├── core/                          # Foundation
│   ├── wave_statistics.py         # Statistical computations
│   ├── collectors.py              # Data collection abstractions
│   └── exporters.py               # Multi-format export
│
├── training/                      # Training-time monitoring
│   ├── hooks.py                   # PyTorch hooks
│   ├── callbacks.py               # Training callbacks
│   └── gradient_monitor.py        # Gradient flow analysis
│
├── introspection/                 # Model analysis
│   ├── layer_analyzer.py          # Layer-wise wave extraction
│   ├── harmonic_analyzer.py       # Harmonic importance
│   ├── interference_analyzer.py   # Wave interference
│   └── spectrum_tracker.py        # Frequency spectrum evolution
│
├── generation/                    # Generation analysis
│   ├── live_visualizer.py         # Real-time visualization
│   ├── trajectory_tracker.py      # Wave trajectory
│   ├── confidence_tracker.py      # Token confidence
│   └── roundtrip_analyzer.py      # Reconstruction fidelity
│
├── comparative/                   # Comparative studies
│   ├── checkpoint_comparator.py   # Checkpoint comparison
│   ├── input_comparator.py        # Input similarity
│   └── ablation_helper.py         # Ablation studies
│
├── visualization/                 # Reporting
│   ├── tensorboard_writer.py      # TensorBoard integration
│   ├── wandb_logger.py            # W&B integration
│   └── report_generator.py        # Publication figures
│
└── utils/                         # Utilities
    ├── memory_efficient.py        # Memory management
    ├── distributed_utils.py       # DDP support
    └── config.py                  # Configuration
```

---

## 📦 Installation

The analysis suite is included with the Wave Transformer package:

```bash
# Basic installation (included with Wave Transformer)
pip install -e .

# Optional dependencies for full functionality
pip install tensorboard wandb pyyaml h5py scipy scikit-learn umap-learn
```

### Dependencies

**Required** (included with Wave Transformer):
- PyTorch >= 1.12
- NumPy >= 1.20
- Matplotlib >= 3.5
- Pandas >= 1.3
- Seaborn >= 0.11

**Optional** (enhanced features):
- `tensorboard` - TensorBoard logging
- `wandb` - Weights & Biases logging
- `pyyaml` - YAML configuration files
- `h5py` - HDF5 export for large datasets
- `scipy` - Statistical tests, Wasserstein distance
- `scikit-learn` - Clustering, dimensionality reduction
- `umap-learn` - UMAP projections

---

## 🚀 Quick Start

### 1. Basic Wave Statistics

```python
from wave_transformer.core.wave import Wave
from wave_transformer.analysis import WaveStatistics
import torch

# Your wave from model
wave = Wave(
    frequencies=torch.randn(B, S, H),
    amplitudes=torch.randn(B, S, H),
    phases=torch.randn(B, S, H)
)

# Compute comprehensive statistics
stats = WaveStatistics.compute_basic_stats(wave, batch_idx=0)
print(f"Mean frequency: {stats['frequencies']['mean']:.4f}")
print(f"Mean amplitude: {stats['amplitudes']['mean']:.4f}")

# Harmonic importance
importance = WaveStatistics.compute_harmonic_importance(wave, method='energy')
print(f"Top 5 harmonics: {importance.top_k(5)}")
```

### 2. Training Monitoring

```python
from wave_transformer.analysis import (
    WaveEvolutionCallback,
    GradientFlowCallback,
    WaveTensorBoardWriter
)

# Setup
writer = WaveTensorBoardWriter(log_dir='runs/experiment')
callbacks = [
    WaveEvolutionCallback(plot_every=500),
    GradientFlowCallback(log_every=100)
]

# Training loop
for epoch in range(num_epochs):
    for callback in callbacks:
        callback.on_epoch_begin(epoch)

    for batch_idx, batch in enumerate(dataloader):
        loss = train_step(model, batch, optimizer)

        # Callbacks handle monitoring
        for callback in callbacks:
            callback.on_batch_end(
                batch_idx=batch_idx,
                loss=loss,
                model=model,
                writer=writer
            )
```

### 3. Model Introspection

```python
from wave_transformer.analysis import LayerWaveAnalyzer

# Analyze layer-wise wave evolution
with LayerWaveAnalyzer(model) as analyzer:
    layer_waves = analyzer.analyze_input({'token_ids': tokens})

    # Compare layers
    comparison = analyzer.compare_layers(layer_waves)
    print(f"Energy change from encoder to layer 0: {comparison['energy_delta'][0]:.4f}")

    # Visualize
    analyzer.plot_layer_evolution(layer_waves, save_path='layer_evolution.png')
```

### 4. Generation Analysis

```python
from wave_transformer.analysis import (
    WaveTrajectoryTracker,
    GenerationConfidenceTracker
)

# Track wave trajectory during generation
tracker = WaveTrajectoryTracker()
trajectory = tracker.track_generation(
    model=model,
    prompt_ids=prompt_ids,
    max_length=50
)

# Plot trajectory
tracker.plot_trajectory(trajectory, save_path='trajectory.png')

# Detect mode collapse
if tracker.detect_mode_collapse(trajectory):
    print("Warning: Mode collapse detected!")

# Track confidence
confidence_tracker = GenerationConfidenceTracker()
confidence_data = confidence_tracker.track_generation(
    model=model,
    prompt_ids=prompt_ids,
    max_length=50
)

# Find uncertain tokens
uncertain = confidence_tracker.identify_uncertain_regions(confidence_data)
print(f"Low-confidence positions: {uncertain}")
```

### 5. Comparative Analysis

```python
from wave_transformer.analysis import CheckpointComparator, AblationHelper

# Compare checkpoints
comparator = CheckpointComparator(
    checkpoint_paths=['ckpt_epoch_1', 'ckpt_epoch_5', 'ckpt_epoch_10'],
    encoder_cls=TokenToWaveEncoder,
    decoder_cls=WaveToTokenDecoder
)

comparison = comparator.compare_on_input(test_tokens)
comparator.plot_checkpoint_evolution(comparison, save_path='ckpt_evolution.png')

# Ablation study
with AblationHelper(model) as ablation:
    configs = [
        {'name': 'baseline', 'ablation': None},
        {'name': 'no_phase', 'ablation': ('component', 'phases', 'zero')},
        {'name': 'sparse_harmonics', 'ablation': ('harmonics', range(10), 'zero')}
    ]

    results = ablation.run_ablation_study(
        ablation_configs=configs,
        evaluation_fn=evaluate_model,
        metrics=['loss', 'perplexity']
    )

    ablation.plot_ablation_results(results, save_path='ablation.png')
```

---

## 🔧 Core Modules

### 1. Wave Statistics (`core/wave_statistics.py`)

Comprehensive statistical analysis of wave representations.

**Key Methods:**
- `compute_basic_stats()` - Mean, std, min, max, median for all components
- `compute_harmonic_importance()` - Rank harmonics by amplitude/energy/variance
- `compute_phase_coherence()` - Measure phase synchronization
- `compute_spectral_centroid()` - Frequency-weighted center of mass
- `compute_total_energy()` - Wave energy per position
- `compute_frequency_bandwidth()` - Bandwidth at percentile threshold

**Example:**
```python
stats = WaveStatistics.compute_basic_stats(wave)
centroid = WaveStatistics.compute_spectral_centroid(wave)
energy = WaveStatistics.compute_total_energy(wave)
```

### 2. Data Collectors (`core/collectors.py`)

Memory-efficient data collection during training.

**Available Collectors:**
- `WaveCollector` - Collect wave statistics
- `GradientCollector` - Track gradient flow
- `LossCollector` - Per-position loss breakdown
- `ActivationCollector` - Intermediate activations

**Example:**
```python
collector = WaveCollector(
    sample_interval=100,  # Sample every 100 steps
    max_samples=1000,     # Maximum samples to store
    collect_on_rank=0     # Only on main process
)

collector.collect(wave=wave, step=step, layer_name='encoder')
aggregated = collector.aggregate()
```

### 3. Training Hooks (`training/hooks.py`)

PyTorch hooks for capturing intermediate data.

**Available Hooks:**
- `WaveForwardHook` - Capture wave representations
- `WaveGradientHook` - Capture gradients
- `AttentionHook` - Capture attention patterns (non-flash)

**Example:**
```python
from wave_transformer.analysis.training import HookManager

manager = HookManager()
manager.register_forward_hook(model.wave_encoder, 'encoder')
manager.register_forward_hook(model.layers[0], 'layer_0')

# After forward pass
captured = manager.get_captured_data()
manager.remove_all_hooks()
```

### 4. Layer Analyzer (`introspection/layer_analyzer.py`)

Extract and compare wave representations across layers.

**Key Methods:**
- `analyze_input()` - Extract waves from all layers
- `compare_layers()` - Compute layer-wise deltas
- `plot_layer_evolution()` - Visualize evolution

**Example:**
```python
with LayerWaveAnalyzer(model) as analyzer:
    waves = analyzer.analyze_input({'token_ids': tokens})
    comparison = analyzer.compare_layers(waves)
    analyzer.plot_layer_evolution(waves, save_path='evolution.png')
```

### 5. Harmonic Importance (`introspection/harmonic_analyzer.py`)

Identify which harmonics contribute most to model performance.

**Key Methods:**
- `analyze_harmonic_importance()` - Compute importance scores
- `compute_gradient_sensitivity()` - Ablation-based sensitivity
- `get_sparse_harmonic_mask()` - For model compression

**Example:**
```python
analyzer = HarmonicImportanceAnalyzer(model, criterion)
importance = analyzer.analyze_harmonic_importance(
    dataloader,
    num_batches=10,
    method='amplitude'
)

# Get top-K harmonics
top_harmonics = importance['top_harmonics'][:10]

# Create sparse mask for compression
mask = analyzer.get_sparse_harmonic_mask(
    importance['importance_scores'],
    sparsity=0.5  # Keep top 50%
)
```

### 6. Generation Visualizer (`generation/live_visualizer.py`)

Real-time visualization during autoregressive generation.

**Key Methods:**
- `generate_with_visualization()` - Generate with live updates
- `create_animation()` - Export as video/GIF

**Example:**
```python
visualizer = LiveGenerationVisualizer(model, update_interval=1)
generated_ids, wave_history = visualizer.generate_with_visualization(
    prompt_ids=prompt_ids,
    max_length=50,
    temperature=0.8
)

visualizer.create_animation(wave_history, save_path='generation.mp4')
```

### 7. Checkpoint Comparator (`comparative/checkpoint_comparator.py`)

Compare models from different training stages.

**Key Methods:**
- `compare_on_input()` - Compare single input
- `compare_on_dataset()` - Compare on dataset
- `compute_checkpoint_divergence()` - Divergence metrics
- `identify_critical_checkpoints()` - Find major changes

**Example:**
```python
comparator = CheckpointComparator(
    checkpoint_paths=['epoch_1', 'epoch_5', 'epoch_10'],
    encoder_cls=TokenToWaveEncoder,
    decoder_cls=WaveToTokenDecoder
)

comparison = comparator.compare_on_input(tokens)
divergence = comparator.compute_checkpoint_divergence(comparison)
comparator.plot_checkpoint_evolution(comparison)
```

### 8. TensorBoard Writer (`visualization/tensorboard_writer.py`)

Comprehensive TensorBoard logging for Wave Transformers.

**Key Methods:**
- `add_wave_statistics()` - Log scalars
- `add_wave_heatmaps()` - Log heatmap images
- `add_wave_spectrum()` - Log spectrum plots
- `add_layer_comparison()` - Log layer analysis
- `add_gradient_flow()` - Log gradient flow

**Example:**
```python
with WaveTensorBoardWriter(log_dir='runs/exp1') as writer:
    writer.add_wave_statistics(wave, step=100, tag='encoder')
    writer.add_wave_heatmaps(wave, step=100, tag='encoder')
    writer.add_gradient_flow(gradient_data, step=100)
```

---

## 💡 Usage Examples

### Example 1: Complete Training Monitoring

```python
from wave_transformer.analysis import (
    create_default_config,
    WaveEvolutionCallback,
    GradientFlowCallback,
    WaveTensorBoardWriter
)

# Load configuration
config = create_default_config(
    enable_wave_tracking=True,
    enable_tensorboard=True,
    plot_frequency=500
)

# Setup monitoring
writer = WaveTensorBoardWriter(log_dir=config.output_dir)
callbacks = [
    WaveEvolutionCallback(
        plot_every=config.plot_frequency,
        save_dir=config.output_dir
    ),
    GradientFlowCallback(
        log_every=config.gradient_log_frequency
    )
]

# Training loop with monitoring
for epoch in range(num_epochs):
    # Epoch start
    for callback in callbacks:
        callback.on_epoch_begin(epoch)

    # Training
    for batch_idx, batch in enumerate(dataloader):
        loss = train_step(model, batch, optimizer)

        # Batch end callbacks
        for callback in callbacks:
            callback.on_batch_end(
                batch_idx=batch_idx,
                loss=loss,
                model=model,
                optimizer=optimizer,
                writer=writer
            )

    # Epoch end
    for callback in callbacks:
        callback.on_epoch_end(
            epoch=epoch,
            model=model,
            dataloader=val_dataloader,
            writer=writer
        )

writer.close()
```

### Example 2: Deep Model Introspection

```python
from wave_transformer.analysis import (
    LayerWaveAnalyzer,
    HarmonicImportanceAnalyzer,
    WaveInterferenceAnalyzer,
    SpectrumEvolutionTracker
)

# 1. Layer-wise analysis
with LayerWaveAnalyzer(model) as layer_analyzer:
    layer_waves = layer_analyzer.analyze_input({'token_ids': tokens})
    layer_comparison = layer_analyzer.compare_layers(layer_waves)
    layer_analyzer.plot_layer_evolution(layer_waves, save_path='layers.png')

# 2. Harmonic importance
harmonic_analyzer = HarmonicImportanceAnalyzer(model, criterion)
importance = harmonic_analyzer.analyze_harmonic_importance(
    dataloader,
    num_batches=50,
    method='energy'
)
harmonic_analyzer.plot_harmonic_importance(importance, save_path='harmonics.png')

# 3. Interference patterns
interference_analyzer = WaveInterferenceAnalyzer(model)
interference = interference_analyzer.analyze_layer_interference(layer_waves)
interference_analyzer.plot_interference_patterns(
    layer_waves,
    save_path='interference.png'
)

# 4. Spectrum evolution
spectrum_tracker = SpectrumEvolutionTracker(model)
spectra = spectrum_tracker.extract_spectrum_evolution(layer_waves)
spectrum_tracker.plot_spectrum_evolution(
    spectra,
    save_path='spectrum.png',
    mode='waterfall'
)
```

### Example 3: Comprehensive Generation Analysis

```python
from wave_transformer.analysis import (
    LiveGenerationVisualizer,
    WaveTrajectoryTracker,
    GenerationConfidenceTracker,
    RoundTripAnalyzer
)

prompt = "The way that can be told"
prompt_ids = tokenizer.encode(prompt)

# 1. Live visualization
visualizer = LiveGenerationVisualizer(model, update_interval=5)
generated_ids, wave_history = visualizer.generate_with_visualization(
    prompt_ids=prompt_ids,
    max_length=100,
    temperature=0.8
)
visualizer.create_animation(wave_history, save_path='generation.gif')

# 2. Trajectory tracking
trajectory_tracker = WaveTrajectoryTracker()
trajectory = trajectory_tracker.track_generation(
    model, prompt_ids, max_length=100
)
trajectory_tracker.plot_trajectory(trajectory, save_path='trajectory.png')

# Check for mode collapse
if trajectory_tracker.detect_mode_collapse(trajectory):
    print("Warning: Mode collapse detected!")

# 3. Confidence tracking
confidence_tracker = GenerationConfidenceTracker()
confidence = confidence_tracker.track_generation(
    model, prompt_ids, max_length=100
)
confidence_tracker.plot_confidence_trajectory(
    confidence,
    save_path='confidence.png'
)

# Identify uncertain regions
uncertain = confidence_tracker.identify_uncertain_regions(confidence)
print(f"Low-confidence tokens: {uncertain}")

# 4. Round-trip analysis
roundtrip_analyzer = RoundTripAnalyzer(model, tokenizer)
roundtrip = roundtrip_analyzer.analyze_roundtrip(prompt)
print(f"Reconstruction accuracy: {roundtrip['reconstruction_accuracy']:.2%}")
roundtrip_analyzer.plot_roundtrip_analysis(roundtrip, save_path='roundtrip.png')
```

### Example 4: Systematic Ablation Study

```python
from wave_transformer.analysis import AblationHelper
import pandas as pd

# Define evaluation function
def evaluate_model(model_to_eval):
    model_to_eval.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            logits = model_to_eval(batch)
            loss = criterion(logits, batch['targets'])
            total_loss += loss.item()
    return {
        'loss': total_loss / len(eval_dataloader),
        'perplexity': np.exp(total_loss / len(eval_dataloader))
    }

# Run ablation study
with AblationHelper(model) as ablation:
    configs = [
        # Baseline
        {'name': 'baseline', 'ablation': None},

        # Harmonic ablations
        {'name': 'no_low_freq', 'ablation': ('harmonics', range(0, 5), 'zero')},
        {'name': 'no_high_freq', 'ablation': ('harmonics', range(59, 64), 'zero')},
        {'name': 'sparse_50%', 'ablation': ('harmonics', range(0, 32), 'zero')},

        # Component ablations
        {'name': 'no_phase', 'ablation': ('component', 'phases', 'zero')},
        {'name': 'no_amplitude', 'ablation': ('component', 'amplitudes', 'zero')},
        {'name': 'no_frequency', 'ablation': ('component', 'frequencies', 'mean')},

        # Layer ablations
        {'name': 'no_layer_0', 'ablation': ('layer', 0, 'identity')},
        {'name': 'no_layer_5', 'ablation': ('layer', 5, 'identity')},
    ]

    results = ablation.run_ablation_study(
        ablation_configs=configs,
        evaluation_fn=evaluate_model,
        metrics=['loss', 'perplexity']
    )

    # Visualize
    ablation.plot_ablation_results(results, save_path='ablation.png')

    # Export
    results.to_csv('ablation_results.csv', index=False)
    print(results)
```

### Example 5: Publication-Quality Report

```python
from wave_transformer.analysis import PaperReportGenerator, PublicationStyle

# Create report generator
generator = PaperReportGenerator(
    output_dir='paper_figures',
    style=PublicationStyle.IEEE
)

# 1. Training curves
generator.create_training_curve_figure(
    train_metrics={'loss': train_losses, 'perplexity': train_ppls},
    val_metrics={'loss': val_losses, 'perplexity': val_ppls},
    save_path='training_curves'
)

# 2. Layer analysis
generator.create_layer_analysis_figure(
    layer_waves=layer_waves,
    save_path='layer_analysis'
)

# 3. Harmonic importance
generator.create_harmonic_importance_figure(
    importance_data=importance,
    save_path='harmonic_importance'
)

# 4. Generation analysis
generator.create_generation_analysis_figure(
    trajectory_data=trajectory,
    confidence_data=confidence,
    save_path='generation_analysis'
)

# 5. Comparison figure
generator.create_comparison_figure(
    comparison_data=checkpoint_comparison,
    save_path='checkpoint_comparison'
)

# 6. LaTeX table
latex_table = generator.generate_latex_table(
    results_df,
    caption='Model performance comparison',
    label='tab:results'
)

# 7. Full report
generator.generate_full_report(
    analysis_results={
        'experiment_name': 'wave_transformer_analysis',
        'training_data': training_data,
        'introspection_data': introspection_data,
        'generation_data': generation_data
    },
    output_dir='paper_figures'
)
```

---

## 🔗 Integration with Training

### Method 1: Callback-Based (Recommended)

```python
from wave_transformer.analysis import (
    WaveEvolutionCallback,
    GradientFlowCallback,
    LossAnalysisCallback
)

callbacks = [
    WaveEvolutionCallback(plot_every=500),
    GradientFlowCallback(log_every=100),
    LossAnalysisCallback(log_every=50)
]

for epoch in range(num_epochs):
    for callback in callbacks:
        callback.on_epoch_begin(epoch)

    for batch_idx, batch in enumerate(dataloader):
        loss = train_step(model, batch, optimizer)

        for callback in callbacks:
            callback.on_batch_end(
                batch_idx=batch_idx,
                loss=loss,
                model=model,
                optimizer=optimizer
            )

    for callback in callbacks:
        callback.on_epoch_end(epoch, model=model)
```

### Method 2: Manual Collection

```python
from wave_transformer.analysis import WaveCollector, GradientMonitor

# Setup
wave_collector = WaveCollector(sample_interval=100, max_samples=1000)
grad_monitor = GradientMonitor(model, track_norms=True)
grad_monitor.register_hooks()

# Training loop
for batch in dataloader:
    logits, wave = model(batch, return_encoder_outputs=True)
    loss = criterion(logits, targets)

    # Collect wave statistics
    wave_collector.collect(wave=wave, step=global_step)

    # Backward
    loss.backward()
    optimizer.step()

# Get results
wave_stats = wave_collector.aggregate()
grad_report = grad_monitor.get_gradient_flow_report()
```

### Method 3: Hook-Based

```python
from wave_transformer.analysis.training import HookManager

# Register hooks
hook_manager = HookManager()
hook_manager.register_forward_hook(model.wave_encoder, 'encoder')
for i, layer in enumerate(model.layers):
    hook_manager.register_forward_hook(layer, f'layer_{i}')

# Training step
output = model(batch)
loss = criterion(output, targets)
loss.backward()

# Access captured data
captured = hook_manager.get_captured_data()
encoder_wave = captured['encoder']

# Clean up
hook_manager.remove_all_hooks()
```

---

## 📚 API Reference

### Core Statistics

```python
class WaveStatistics:
    @staticmethod
    def compute_basic_stats(wave: Wave, batch_idx: int = 0) -> dict

    @staticmethod
    def compute_harmonic_importance(
        wave: Wave,
        method: str = 'amplitude'  # 'amplitude', 'energy', 'variance'
    ) -> HarmonicImportance

    @staticmethod
    def compute_phase_coherence(
        wave: Wave,
        window_size: int = 5
    ) -> torch.Tensor

    @staticmethod
    def compute_spectral_centroid(wave: Wave) -> torch.Tensor

    @staticmethod
    def compute_total_energy(wave: Wave) -> torch.Tensor

    @staticmethod
    def compute_frequency_bandwidth(
        wave: Wave,
        percentile: float = 95
    ) -> torch.Tensor
```

### Training Callbacks

```python
class AnalysisCallback:
    def on_train_begin(self, **kwargs): ...
    def on_epoch_begin(self, epoch: int, **kwargs): ...
    def on_batch_begin(self, batch_idx: int, **kwargs): ...
    def on_batch_end(self, batch_idx: int, loss, **kwargs): ...
    def on_epoch_end(self, epoch: int, **kwargs): ...
    def on_train_end(self, **kwargs): ...

class WaveEvolutionCallback(AnalysisCallback):
    def __init__(
        self,
        plot_every: int = 1000,
        save_dir: str = 'analysis_results'
    ): ...

class GradientFlowCallback(AnalysisCallback):
    def __init__(
        self,
        log_every: int = 100
    ): ...
```

### Introspection

```python
class LayerWaveAnalyzer:
    def __init__(self, model: WaveTransformer): ...

    def analyze_input(
        self,
        encoder_input: dict,
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict[str, Wave]: ...

    def compare_layers(self, layer_waves: dict) -> dict: ...

    def plot_layer_evolution(
        self,
        layer_waves: dict,
        save_path: Optional[str] = None
    ): ...

class HarmonicImportanceAnalyzer:
    def __init__(
        self,
        model: WaveTransformer,
        criterion: nn.Module
    ): ...

    def analyze_harmonic_importance(
        self,
        dataloader,
        num_batches: int = 10,
        method: str = 'amplitude'
    ) -> dict: ...

    def plot_harmonic_importance(
        self,
        importance_data: dict,
        save_path: Optional[str] = None
    ): ...
```

### Generation Analysis

```python
class WaveTrajectoryTracker:
    def track_generation(
        self,
        model: WaveTransformer,
        prompt_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0
    ) -> dict: ...

    def plot_trajectory(
        self,
        trajectory_data: dict,
        save_path: Optional[str] = None
    ): ...

    def detect_mode_collapse(
        self,
        trajectory_data: dict,
        threshold: float = 0.1
    ) -> bool: ...

class GenerationConfidenceTracker:
    def track_generation(
        self,
        model: WaveTransformer,
        prompt_ids: torch.Tensor,
        max_length: int = 50
    ) -> dict: ...

    def identify_uncertain_regions(
        self,
        confidence_data: dict,
        threshold: float = 0.5
    ) -> list[int]: ...
```

---

## 🚀 Advanced Features

### 1. Memory-Efficient Storage

For long training runs, use memory-efficient storage:

```python
from wave_transformer.analysis import DownsampledStorage, StreamingStatistics

# Downsampled storage (100x compression)
storage = DownsampledStorage(
    full_resolution_size=1000,
    downsample_factor=10,
    num_levels=3
)

for step in range(100000):
    storage.store(step, data)
# Stores ~1000 samples instead of 100,000

# Streaming statistics (constant memory)
stats = StreamingStatistics()
for value in data_stream:
    stats.update(value)
mean, std = stats.get_statistics()['mean'], stats.get_statistics()['std']
```

### 2. Distributed Training Support

All collectors and callbacks support distributed training:

```python
from wave_transformer.analysis import (
    WaveCollector,
    DistributedCallback,
    DistributedAnalysisHelper
)

# Collector automatically handles DDP
collector = WaveCollector(
    collect_on_rank=0  # Only main process collects
)

# Wrap callbacks for distributed
callback = WaveEvolutionCallback()
distributed_callback = DistributedCallback(callback)

# Helper functions
if DistributedAnalysisHelper.is_main_process():
    # Run expensive analysis only on rank 0
    plot_results()

# Reduce metrics across processes
metrics = {'loss': local_loss, 'accuracy': local_acc}
global_metrics = DistributedAnalysisHelper.reduce_dict(metrics, reduction='mean')
```

### 3. Custom Metrics

Add custom metrics to collectors:

```python
from wave_transformer.analysis import WaveCollector

class CustomWaveCollector(WaveCollector):
    def collect(self, wave, step, **kwargs):
        # Call parent
        super().collect(wave, step, **kwargs)

        # Add custom metric
        custom_metric = self.compute_custom_metric(wave)
        if self.data:
            self.data[-1]['custom_metric'] = custom_metric

    def compute_custom_metric(self, wave):
        # Your custom computation
        return wave.frequencies.max().item()

collector = CustomWaveCollector()
```

### 4. Export Pipelines

Multi-format export:

```python
from wave_transformer.analysis import AnalysisExporter

# JSON (small datasets)
AnalysisExporter.to_json(data, 'results.json', compress=True)

# HDF5 (large datasets)
AnalysisExporter.to_hdf5(data, 'results.h5')

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
AnalysisExporter.to_tensorboard(data, writer, step=100)

# Weights & Biases
import wandb
wandb.init(project='wave-transformer')
AnalysisExporter.to_wandb(data, step=100, prefix='train/')

# Multi-format report
AnalysisExporter.export_report(
    data,
    output_dir='results',
    formats=['json', 'hdf5']
)
```

---

## ⚡ Performance Considerations

### Overhead Analysis

| Component | Overhead | Recommended Frequency |
|-----------|----------|----------------------|
| Wave Statistics | <1% | Every batch |
| Gradient Monitor | 2-5% | Every 100 batches |
| Layer Analysis | 10-20% | Every 500 batches |
| Live Visualization | 20-30% | Eval only |
| TensorBoard Logging | <2% | Every 100 batches |

### Optimization Tips

1. **Sampling**: Use `sample_interval` to reduce collection frequency
```python
collector = WaveCollector(sample_interval=100)  # Collect every 100 batches
```

2. **Max Samples**: Limit stored samples
```python
collector = WaveCollector(max_samples=1000)  # Stop after 1000 samples
```

3. **Downsampling**: Use multi-resolution storage
```python
storage = DownsampledStorage(downsample_factor=10, num_levels=3)
```

4. **Distributed**: Collect only on rank 0
```python
collector = WaveCollector(collect_on_rank=0)
```

5. **Disable Gradients**: Use `@torch.no_grad()` for analysis
```python
with torch.no_grad():
    stats = WaveStatistics.compute_basic_stats(wave)
```

6. **Detach and Move**: Reduce GPU memory
```python
hook = WaveForwardHook(collector, detach=True, to_cpu=True)
```

---

## 📊 Example Workflow

### Complete Analysis Pipeline

```python
# 1. Setup
from wave_transformer.analysis import *

config = create_default_config(
    enable_wave_tracking=True,
    enable_tensorboard=True,
    enable_generation_tracking=True,
    plot_frequency=500
)

# 2. Training Monitoring
writer = WaveTensorBoardWriter(log_dir=config.output_dir)
callbacks = [
    WaveEvolutionCallback(plot_every=500),
    GradientFlowCallback(log_every=100)
]

for epoch in range(num_epochs):
    train_with_callbacks(model, dataloader, callbacks, writer)

    # Epoch-end analysis
    if epoch % 5 == 0:
        # Model introspection
        with LayerWaveAnalyzer(model) as analyzer:
            layer_waves = analyzer.analyze_input(sample_input)
            analyzer.plot_layer_evolution(layer_waves)

        # Generation analysis
        trajectory_tracker = WaveTrajectoryTracker()
        trajectory = trajectory_tracker.track_generation(
            model, sample_prompt, max_length=50
        )
        trajectory_tracker.plot_trajectory(trajectory)

# 3. Final Analysis
# Checkpoint comparison
comparator = CheckpointComparator(checkpoint_paths, encoder_cls, decoder_cls)
comparison = comparator.compare_on_dataset(test_dataloader)
comparator.plot_checkpoint_evolution(comparison)

# Ablation study
with AblationHelper(model) as ablation:
    results = ablation.run_ablation_study(ablation_configs, evaluate_fn)
    ablation.plot_ablation_results(results)

# 4. Publication Report
generator = PaperReportGenerator(
    output_dir='paper_figures',
    style=PublicationStyle.IEEE
)
generator.generate_full_report(all_analysis_results)
```

---

## 🤝 Contributing

We welcome contributions! Areas for expansion:

- Additional wave statistics
- New visualization types
- Distributed training optimizations
- Integration with other frameworks (JAX, MLX)
- Additional export formats

---

## 📝 Citation

If you use this analysis suite in your research, please cite:

```bibtex
@software{wave_transformer_analysis_suite,
  title={Wave Transformer Analysis Suite},
  author={Wave Transformer Team},
  year={2024},
  url={https://github.com/your-repo/wave-transformer}
}
```

---

## 📧 Support

- **Documentation**: See module-specific READMEs
- **Examples**: `examples/analysis/` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

---

## 🎓 Learn More

- [Wave Transformer Paper](link)
- [Tutorial Notebooks](link)
- [API Documentation](link)
- [Video Tutorials](link)

---

**Built with ❤️ for deep understanding of Wave Transformer architectures**
