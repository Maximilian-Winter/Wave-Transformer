"""
Memory-efficient storage utilities for long-running analysis.

This module provides efficient data structures for storing analysis data
with limited memory usage, including circular buffers, downsampled storage,
and streaming statistics computation.
"""

import numpy as np
import torch
from typing import Optional, Union, Tuple, Any
from collections import deque


class CircularBuffer:
    """
    Fixed-size circular buffer that overwrites oldest data when full.

    Efficient for storing recent history with constant memory usage.
    """

    def __init__(self, capacity: int, dtype: type = np.float32):
        """
        Initialize circular buffer.

        Args:
            capacity: Maximum number of elements to store
            dtype: Data type for numpy arrays
        """
        self.capacity = capacity
        self.dtype = dtype
        self.buffer = None
        self.index = 0
        self.size = 0

    def append(self, value: Union[float, np.ndarray, torch.Tensor]) -> None:
        """
        Add value to buffer.

        Args:
            value: Value to add (scalar, array, or tensor)
        """
        # Convert to numpy
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array(value, dtype=self.dtype)

        # Initialize buffer on first append
        if self.buffer is None:
            if value.ndim == 0:
                # Scalar
                self.buffer = np.zeros(self.capacity, dtype=self.dtype)
            else:
                # Array
                self.buffer = np.zeros((self.capacity,) + value.shape, dtype=self.dtype)

        # Store value
        self.buffer[self.index] = value

        # Update index and size
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, last_n: Optional[int] = None) -> np.ndarray:
        """
        Get buffer contents.

        Args:
            last_n: If specified, return only last n elements.
                   If None, return all stored elements.

        Returns:
            Array of stored values
        """
        if self.size == 0:
            return np.array([], dtype=self.dtype)

        if self.size < self.capacity:
            # Buffer not yet full
            data = self.buffer[:self.size]
        else:
            # Buffer is full, need to reorder
            data = np.concatenate([
                self.buffer[self.index:],
                self.buffer[:self.index]
            ])

        if last_n is not None:
            data = data[-last_n:]

        return data

    def mean(self, last_n: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute mean of stored values."""
        data = self.get(last_n)
        return data.mean(axis=0) if len(data) > 0 else 0.0

    def std(self, last_n: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute standard deviation of stored values."""
        data = self.get(last_n)
        return data.std(axis=0) if len(data) > 0 else 0.0

    def max(self, last_n: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute maximum of stored values."""
        data = self.get(last_n)
        return data.max(axis=0) if len(data) > 0 else 0.0

    def min(self, last_n: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute minimum of stored values."""
        data = self.get(last_n)
        return data.min(axis=0) if len(data) > 0 else 0.0

    def __len__(self) -> int:
        """Return number of stored elements."""
        return self.size

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size == self.capacity

    def clear(self) -> None:
        """Clear buffer contents."""
        self.index = 0
        self.size = 0


class DownsampledStorage:
    """
    Adaptive downsampling storage for long sequences.

    Stores data at multiple resolutions:
    - Recent data at full resolution
    - Older data at progressively lower resolutions

    This allows efficient storage of long training runs while preserving
    both recent details and long-term trends.
    """

    def __init__(
        self,
        full_resolution_size: int = 1000,
        downsample_factor: int = 10,
        num_levels: int = 3,
        dtype: type = np.float32
    ):
        """
        Initialize downsampled storage.

        Args:
            full_resolution_size: Number of recent samples to keep at full resolution
            downsample_factor: Factor by which to downsample at each level
            num_levels: Number of downsampling levels
            dtype: Data type for storage
        """
        self.full_resolution_size = full_resolution_size
        self.downsample_factor = downsample_factor
        self.num_levels = num_levels
        self.dtype = dtype

        # Storage levels: level 0 is full resolution
        self.levels = [[] for _ in range(num_levels + 1)]
        self.total_samples = 0

    def append(self, value: Union[float, np.ndarray, torch.Tensor]) -> None:
        """
        Add value to storage.

        Args:
            value: Value to add
        """
        # Convert to numpy
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().astype(self.dtype)
        elif not isinstance(value, np.ndarray):
            value = np.array(value, dtype=self.dtype)

        # Add to full resolution level
        self.levels[0].append(value)
        self.total_samples += 1

        # Check if we need to downsample
        if len(self.levels[0]) > self.full_resolution_size:
            self._downsample_level(0)

    def _downsample_level(self, level: int) -> None:
        """
        Downsample a level and propagate to next level.

        Args:
            level: Level to downsample
        """
        if level >= self.num_levels:
            return

        # Take excess samples from this level
        excess = len(self.levels[level]) - self.full_resolution_size
        if excess <= 0:
            return

        # Extract samples to downsample
        samples_to_downsample = self.levels[level][:excess]
        self.levels[level] = self.levels[level][excess:]

        # Downsample by averaging blocks
        num_blocks = len(samples_to_downsample) // self.downsample_factor
        if num_blocks > 0:
            for i in range(num_blocks):
                start = i * self.downsample_factor
                end = start + self.downsample_factor
                block = samples_to_downsample[start:end]

                # Compute mean
                if isinstance(block[0], np.ndarray):
                    downsampled = np.mean(np.stack(block), axis=0)
                else:
                    downsampled = np.mean(block)

                # Add to next level
                self.levels[level + 1].append(downsampled)

            # Check if next level needs downsampling
            if len(self.levels[level + 1]) > self.full_resolution_size:
                self._downsample_level(level + 1)

    def get_all(self, as_single_array: bool = True) -> Union[np.ndarray, list]:
        """
        Get all stored data across all levels.

        Args:
            as_single_array: If True, concatenate all levels into single array

        Returns:
            All stored data
        """
        if as_single_array:
            all_data = []
            # Reverse order: oldest (most downsampled) first
            for level in reversed(self.levels):
                if level:
                    all_data.extend(level)

            if not all_data:
                return np.array([], dtype=self.dtype)

            if isinstance(all_data[0], np.ndarray):
                return np.stack(all_data)
            else:
                return np.array(all_data, dtype=self.dtype)
        else:
            return self.levels

    def get_statistics(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage info
        """
        level_sizes = [len(level) for level in self.levels]
        total_stored = sum(level_sizes)

        return {
            'total_samples_seen': self.total_samples,
            'total_samples_stored': total_stored,
            'compression_ratio': self.total_samples / max(total_stored, 1),
            'level_sizes': level_sizes,
            'full_resolution_size': self.full_resolution_size,
            'downsample_factor': self.downsample_factor,
            'num_levels': self.num_levels
        }

    def __len__(self) -> int:
        """Return total number of stored samples."""
        return sum(len(level) for level in self.levels)


class StreamingStatistics:
    """
    Compute running statistics using Welford's online algorithm.

    Allows computing mean and variance/std without storing all values.
    More numerically stable than naive online computation.

    Based on: Welford, B. P. (1962). "Note on a method for calculating
              corrected sums of squares and products". Technometrics.
    """

    def __init__(self, shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize streaming statistics.

        Args:
            shape: Shape of values to track. If None, assumes scalars.
        """
        self.shape = shape
        self.n = 0

        if shape is None:
            self.mean = 0.0
            self.m2 = 0.0
            self.min_val = float('inf')
            self.max_val = float('-inf')
        else:
            self.mean = np.zeros(shape, dtype=np.float64)
            self.m2 = np.zeros(shape, dtype=np.float64)
            self.min_val = np.full(shape, float('inf'), dtype=np.float64)
            self.max_val = np.full(shape, float('-inf'), dtype=np.float64)

    def update(self, value: Union[float, np.ndarray, torch.Tensor]) -> None:
        """
        Update statistics with new value.

        Args:
            value: New value to incorporate
        """
        # Convert to numpy
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        # Welford's algorithm
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

        # Update min/max
        if isinstance(value, np.ndarray):
            self.min_val = np.minimum(self.min_val, value)
            self.max_val = np.maximum(self.max_val, value)
        else:
            self.min_val = min(self.min_val, value)
            self.max_val = max(self.max_val, value)

    def get_mean(self) -> Union[float, np.ndarray]:
        """Get current mean."""
        return self.mean

    def get_variance(self, ddof: int = 0) -> Union[float, np.ndarray]:
        """
        Get current variance.

        Args:
            ddof: Delta degrees of freedom (0 for population, 1 for sample)

        Returns:
            Variance
        """
        if self.n < ddof + 1:
            return np.nan if self.shape is None else np.full(self.shape, np.nan)

        return self.m2 / (self.n - ddof)

    def get_std(self, ddof: int = 0) -> Union[float, np.ndarray]:
        """
        Get current standard deviation.

        Args:
            ddof: Delta degrees of freedom

        Returns:
            Standard deviation
        """
        variance = self.get_variance(ddof)
        if isinstance(variance, np.ndarray):
            return np.sqrt(variance)
        else:
            return np.sqrt(variance) if not np.isnan(variance) else np.nan

    def get_min(self) -> Union[float, np.ndarray]:
        """Get minimum value seen."""
        return self.min_val

    def get_max(self) -> Union[float, np.ndarray]:
        """Get maximum value seen."""
        return self.max_val

    def get_statistics(self) -> dict:
        """
        Get all statistics as dictionary.

        Returns:
            Dictionary with mean, variance, std, min, max
        """
        return {
            'count': self.n,
            'mean': self.get_mean(),
            'variance': self.get_variance(ddof=1),
            'std': self.get_std(ddof=1),
            'min': self.get_min(),
            'max': self.get_max()
        }

    def merge(self, other: 'StreamingStatistics') -> 'StreamingStatistics':
        """
        Merge with another StreamingStatistics object.

        Uses Chan's parallel algorithm for combining statistics.

        Args:
            other: Another StreamingStatistics object

        Returns:
            New StreamingStatistics with merged values
        """
        if self.shape != other.shape:
            raise ValueError("Cannot merge statistics with different shapes")

        merged = StreamingStatistics(shape=self.shape)

        n_a = self.n
        n_b = other.n
        merged.n = n_a + n_b

        if merged.n == 0:
            return merged

        # Combine means
        delta = other.mean - self.mean
        merged.mean = self.mean + delta * n_b / merged.n

        # Combine M2 (for variance)
        merged.m2 = self.m2 + other.m2 + delta ** 2 * n_a * n_b / merged.n

        # Combine min/max
        if isinstance(self.min_val, np.ndarray):
            merged.min_val = np.minimum(self.min_val, other.min_val)
            merged.max_val = np.maximum(self.max_val, other.max_val)
        else:
            merged.min_val = min(self.min_val, other.min_val)
            merged.max_val = max(self.max_val, other.max_val)

        return merged

    def __len__(self) -> int:
        """Return number of values processed."""
        return self.n


class ExponentialMovingAverage:
    """
    Exponential moving average for smoothed metrics.

    Gives more weight to recent values while maintaining history.
    """

    def __init__(self, alpha: float = 0.1, shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize exponential moving average.

        Args:
            alpha: Smoothing factor (0 < alpha <= 1). Smaller = more smoothing.
            shape: Shape of values to track
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

        self.alpha = alpha
        self.shape = shape
        self.value = None
        self.initialized = False

    def update(self, value: Union[float, np.ndarray, torch.Tensor]) -> None:
        """
        Update EMA with new value.

        Args:
            value: New value
        """
        # Convert to numpy
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array(value)

        if not self.initialized:
            self.value = value
            self.initialized = True
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value

    def get(self) -> Union[float, np.ndarray]:
        """Get current EMA value."""
        if not self.initialized:
            return 0.0 if self.shape is None else np.zeros(self.shape)
        return self.value

    def reset(self) -> None:
        """Reset EMA."""
        self.value = None
        self.initialized = False


class SlidingWindowStatistics:
    """
    Compute statistics over a sliding window of recent values.

    More efficient than storing all values and recomputing.
    """

    def __init__(self, window_size: int, dtype: type = np.float32):
        """
        Initialize sliding window statistics.

        Args:
            window_size: Size of sliding window
            dtype: Data type for storage
        """
        self.window_size = window_size
        self.dtype = dtype
        self.window = deque(maxlen=window_size)

    def update(self, value: Union[float, np.ndarray, torch.Tensor]) -> None:
        """
        Add value to window.

        Args:
            value: New value
        """
        # Convert to numpy
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array(value, dtype=self.dtype)

        self.window.append(value)

    def get_mean(self) -> Union[float, np.ndarray]:
        """Get mean over window."""
        if not self.window:
            return 0.0

        if isinstance(self.window[0], np.ndarray):
            return np.mean(np.stack(list(self.window)), axis=0)
        else:
            return np.mean(list(self.window))

    def get_std(self) -> Union[float, np.ndarray]:
        """Get standard deviation over window."""
        if not self.window:
            return 0.0

        if isinstance(self.window[0], np.ndarray):
            return np.std(np.stack(list(self.window)), axis=0)
        else:
            return np.std(list(self.window))

    def get_min(self) -> Union[float, np.ndarray]:
        """Get minimum over window."""
        if not self.window:
            return 0.0

        if isinstance(self.window[0], np.ndarray):
            return np.min(np.stack(list(self.window)), axis=0)
        else:
            return min(self.window)

    def get_max(self) -> Union[float, np.ndarray]:
        """Get maximum over window."""
        if not self.window:
            return 0.0

        if isinstance(self.window[0], np.ndarray):
            return np.max(np.stack(list(self.window)), axis=0)
        else:
            return max(self.window)

    def __len__(self) -> int:
        """Return current window size."""
        return len(self.window)
