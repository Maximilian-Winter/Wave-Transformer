"""
Utility components for memory-efficient analysis.

This module provides memory-efficient storage and computation utilities:
- Circular buffers for fixed-size history
- Downsampled storage for long sequences
- Streaming statistics (Welford's algorithm)
- Exponential moving averages
- Sliding window statistics
"""

from .memory_efficient import (
    CircularBuffer,
    DownsampledStorage,
    StreamingStatistics,
    ExponentialMovingAverage,
    SlidingWindowStatistics
)

__all__ = [
    'CircularBuffer',
    'DownsampledStorage',
    'StreamingStatistics',
    'ExponentialMovingAverage',
    'SlidingWindowStatistics',
]
