"""
Utility components for memory-efficient analysis and configuration.

This module provides:
- Memory-efficient storage and computation utilities
- Configuration management with YAML support
- Distributed training utilities

Components:
- Circular buffers for fixed-size history
- Downsampled storage for long sequences
- Streaming statistics (Welford's algorithm)
- Exponential moving averages
- Sliding window statistics
- AnalysisConfig and component-specific configurations
"""

from .memory_efficient import (
    CircularBuffer,
    DownsampledStorage,
    StreamingStatistics,
    ExponentialMovingAverage,
    SlidingWindowStatistics
)

from .config import (
    AnalysisConfig,
    CollectorConfig,
    VisualizationConfig,
    IntrospectionConfig,
    TrainingConfig,
    ExportConfig,
    MemoryConfig,
    create_default_config,
    create_minimal_config,
    create_full_config,
)

__all__ = [
    # Memory-efficient utilities
    'CircularBuffer',
    'DownsampledStorage',
    'StreamingStatistics',
    'ExponentialMovingAverage',
    'SlidingWindowStatistics',

    # Configuration
    'AnalysisConfig',
    'CollectorConfig',
    'VisualizationConfig',
    'IntrospectionConfig',
    'TrainingConfig',
    'ExportConfig',
    'MemoryConfig',
    'create_default_config',
    'create_minimal_config',
    'create_full_config',
]
