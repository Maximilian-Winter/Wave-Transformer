"""
Wave Transformer Analysis Suite

A comprehensive toolkit for analyzing Wave Transformer representations,
training dynamics, and model behavior.

## Core Components

### Statistics (`wave_transformer.analysis.core.wave_statistics`)
Comprehensive statistical analysis of wave representations:
- Basic statistics (mean, std, min, max, median)
- Harmonic importance ranking
- Phase coherence measurements
- Spectral features (centroid, bandwidth, entropy)
- Energy analysis

### Data Collectors (`wave_transformer.analysis.core.collectors`)
Efficient data collection during training:
- WaveCollector: Capture wave statistics at intervals
- GradientCollector: Track gradient flow and norms
- LossCollector: Monitor per-position loss breakdown
- ActivationCollector: Record intermediate activations

### Export Utilities (`wave_transformer.analysis.core.exporters`)
Multi-format export support:
- JSON with numpy/torch handling
- HDF5 for large datasets
- TensorBoard integration
- Weights & Biases logging
- Publication-quality figure generation

### Memory-Efficient Storage (`wave_transformer.analysis.utils.memory_efficient`)
Efficient data structures for long-running experiments:
- CircularBuffer: Fixed-size rolling history
- DownsampledStorage: Multi-resolution storage
- StreamingStatistics: Online mean/variance computation
- ExponentialMovingAverage: Smoothed metrics
- SlidingWindowStatistics: Recent history stats

### Generation Analysis (`wave_transformer.analysis.generation`)
Comprehensive tools for analyzing autoregressive generation:
- LiveGenerationVisualizer: Real-time wave evolution during generation
- WaveTrajectoryTracker: Track wave statistics across generation steps
- GenerationConfidenceTracker: Monitor model confidence and uncertainty
- RoundTripAnalyzer: Token-wave-token reconstruction analysis

## Quick Start

```python
from wave_transformer.analysis import (
    WaveStatistics,
    WaveCollector,
    AnalysisExporter
)

# Analyze a wave representation
wave = model.get_wave_output(input_data)
stats = WaveStatistics.compute_basic_stats(wave)
importance = WaveStatistics.compute_harmonic_importance(wave)

# Collect data during training
collector = WaveCollector(sample_interval=100)
for step, batch in enumerate(dataloader):
    wave = model(batch)
    collector.collect(wave, step=step)

# Export results
data = collector.get_data()
AnalysisExporter.to_json(data, 'analysis_results.json')
AnalysisExporter.to_wandb(data, step=step)
```
"""

# Import introspection components
from .introspection import (
    LayerWaveAnalyzer,
    HarmonicImportanceAnalyzer,
    WaveInterferenceAnalyzer,
    SpectrumEvolutionTracker,
)

# Import core analysis components
from .core import (
    # Statistics
    WaveStatistics,
    WaveStats,
    HarmonicImportance,

    # Collectors
    DataCollector,
    WaveCollector,
    GradientCollector,
    LossCollector,
    ActivationCollector,

    # Exporters
    AnalysisExporter,
    NumpyEncoder,
)

# Import memory-efficient utilities and configuration
from .utils import (
    CircularBuffer,
    DownsampledStorage,
    StreamingStatistics,
    ExponentialMovingAverage,
    SlidingWindowStatistics,
    AnalysisConfig,
    create_default_config,
    create_minimal_config,
    create_full_config,
)

# Import training monitoring components
from .training.hooks import (
    HookStorage,
    WaveForwardHook,
    WaveGradientHook,
    AttentionHook,
    HookManager,
)

from .training.callbacks import (
    AnalysisCallback,
    WaveEvolutionCallback,
    GradientFlowCallback,
    LossAnalysisCallback,
)

from .training.gradient_monitor import GradientMonitor

from .utils.distributed_utils import (
    DistributedAnalysisHelper,
    DistributedCallback,
    DistributedMetricsAggregator,
    main_process_only,
    synchronized,
)

# Import comparative analysis components
from .comparative import (
    CheckpointComparator,
    InputComparator,
    AblationHelper,
)

# Import generation analysis components
from .generation import (
    LiveGenerationVisualizer,
    WaveTrajectoryTracker,
    GenerationConfidenceTracker,
    RoundTripAnalyzer,
)

# Import visualization and reporting components
from .visualization import (
    WaveTensorBoardWriter,
    WaveWandbLogger,
    PaperReportGenerator,
    PublicationStyle,
    FigureConfig,
)

__version__ = '0.1.0'

__all__ = [
    # Introspection
    'LayerWaveAnalyzer',
    'HarmonicImportanceAnalyzer',
    'WaveInterferenceAnalyzer',
    'SpectrumEvolutionTracker',

    # Statistics
    'WaveStatistics',
    'WaveStats',
    'HarmonicImportance',

    # Collectors
    'DataCollector',
    'WaveCollector',
    'GradientCollector',
    'LossCollector',
    'ActivationCollector',

    # Exporters
    'AnalysisExporter',
    'NumpyEncoder',

    # Memory-efficient utilities
    'CircularBuffer',
    'DownsampledStorage',
    'StreamingStatistics',
    'ExponentialMovingAverage',
    'SlidingWindowStatistics',

    # Configuration
    'AnalysisConfig',
    'create_default_config',
    'create_minimal_config',
    'create_full_config',

    # Training Hooks
    'HookStorage',
    'WaveForwardHook',
    'WaveGradientHook',
    'AttentionHook',
    'HookManager',

    # Training Callbacks
    'AnalysisCallback',
    'WaveEvolutionCallback',
    'GradientFlowCallback',
    'LossAnalysisCallback',

    # Gradient Monitoring
    'GradientMonitor',

    # Distributed Utils
    'DistributedAnalysisHelper',
    'DistributedCallback',
    'DistributedMetricsAggregator',
    'main_process_only',
    'synchronized',

    # Comparative Analysis
    'CheckpointComparator',
    'InputComparator',
    'AblationHelper',

    # Generation Analysis
    'LiveGenerationVisualizer',
    'WaveTrajectoryTracker',
    'GenerationConfidenceTracker',
    'RoundTripAnalyzer',

    # Visualization and Reporting
    'WaveTensorBoardWriter',
    'WaveWandbLogger',
    'PaperReportGenerator',
    'PublicationStyle',
    'FigureConfig',
]
