"""
Example usage of Wave Transformer Analysis Suite.

This script demonstrates how to use the core analysis modules:
- Wave statistics computation
- Data collection during training
- Export utilities
- Memory-efficient storage

Run this script to verify the analysis suite is working correctly.
"""

import torch
import numpy as np
from pathlib import Path

# Import Wave class
from wave_transformer.core.wave import Wave

# Import analysis components
from wave_transformer.analysis import (
    # Statistics
    WaveStatistics,
    WaveStats,
    HarmonicImportance,

    # Collectors
    WaveCollector,
    GradientCollector,
    LossCollector,

    # Exporters
    AnalysisExporter,

    # Memory-efficient utilities
    CircularBuffer,
    DownsampledStorage,
    StreamingStatistics,
    ExponentialMovingAverage,
)


def create_sample_wave(batch_size=4, seq_len=16, num_harmonics=64):
    """Create a sample Wave object for testing."""
    frequencies = torch.randn(batch_size, seq_len, num_harmonics).abs() * 10
    amplitudes = torch.randn(batch_size, seq_len, num_harmonics).abs()
    phases = torch.randn(batch_size, seq_len, num_harmonics) * 2 * np.pi

    return Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)


def example_1_basic_statistics():
    """Example 1: Compute basic statistics on a wave."""
    print("\n" + "="*70)
    print("Example 1: Basic Wave Statistics")
    print("="*70)

    # Create sample wave
    wave = create_sample_wave(batch_size=4, seq_len=16, num_harmonics=64)
    print(f"Created wave with shape: B={wave.frequencies.shape[0]}, "
          f"S={wave.frequencies.shape[1]}, H={wave.frequencies.shape[2]}")

    # Compute basic statistics
    stats = WaveStatistics.compute_basic_stats(wave, component='all')

    print("\nFrequency Statistics:")
    freq_stats = stats['frequencies']
    print(f"  Mean: {freq_stats.mean:.4f} Hz")
    print(f"  Std:  {freq_stats.std:.4f} Hz")
    print(f"  Min:  {freq_stats.min:.4f} Hz")
    print(f"  Max:  {freq_stats.max:.4f} Hz")

    print("\nAmplitude Statistics:")
    amp_stats = stats['amplitudes']
    print(f"  Mean: {amp_stats.mean:.4f}")
    print(f"  Std:  {amp_stats.std:.4f}")
    print(f"  Min:  {amp_stats.min:.4f}")
    print(f"  Max:  {amp_stats.max:.4f}")

    print("\nPhase Statistics:")
    phase_stats = stats['phases']
    print(f"  Mean: {phase_stats.mean:.4f} rad")
    print(f"  Std:  {phase_stats.std:.4f} rad")


def example_2_harmonic_importance():
    """Example 2: Rank harmonics by importance."""
    print("\n" + "="*70)
    print("Example 2: Harmonic Importance Ranking")
    print("="*70)

    wave = create_sample_wave()

    # Rank by amplitude
    importance_amp = WaveStatistics.compute_harmonic_importance(
        wave, metric='amplitude'
    )
    top_10_indices, top_10_scores = importance_amp.top_k(10)

    print("\nTop 10 Harmonics by Amplitude:")
    for i, (idx, score) in enumerate(zip(top_10_indices, top_10_scores)):
        print(f"  {i+1}. Harmonic {idx}: {score:.4f}")

    # Rank by energy
    importance_energy = WaveStatistics.compute_harmonic_importance(
        wave, metric='energy'
    )
    top_10_indices, top_10_scores = importance_energy.top_k(10)

    print("\nTop 10 Harmonics by Energy:")
    for i, (idx, score) in enumerate(zip(top_10_indices, top_10_scores)):
        print(f"  {i+1}. Harmonic {idx}: {score:.4f}")


def example_3_spectral_features():
    """Example 3: Compute spectral features."""
    print("\n" + "="*70)
    print("Example 3: Spectral Features")
    print("="*70)

    wave = create_sample_wave(batch_size=1)

    # Phase coherence
    coherence = WaveStatistics.compute_phase_coherence(wave, batch_idx=0)
    print(f"\nPhase Coherence:")
    print(f"  Mean: {coherence.mean().item():.4f}")
    print(f"  Std:  {coherence.std().item():.4f}")

    # Spectral centroid
    centroid = WaveStatistics.compute_spectral_centroid(wave, batch_idx=0)
    print(f"\nSpectral Centroid (per position):")
    print(f"  Mean: {centroid.mean().item():.4f} Hz")
    print(f"  Std:  {centroid.std().item():.4f} Hz")

    # Total energy
    energy = WaveStatistics.compute_total_energy(wave, batch_idx=0, per_position=True)
    print(f"\nTotal Energy (per position):")
    print(f"  Mean: {energy.mean().item():.4f}")
    print(f"  Max:  {energy.max().item():.4f}")

    # Frequency bandwidth
    bandwidth = WaveStatistics.compute_frequency_bandwidth(wave, batch_idx=0, percentile=90)
    print(f"\nFrequency Bandwidth (90th percentile):")
    print(f"  Mean: {bandwidth.mean().item():.4f} Hz")


def example_4_wave_collector():
    """Example 4: Collect wave statistics during training simulation."""
    print("\n" + "="*70)
    print("Example 4: Wave Data Collection")
    print("="*70)

    # Create collector
    collector = WaveCollector(
        sample_interval=10,  # Collect every 10 steps
        max_samples=100,
        statistics_to_collect=['basic_stats', 'total_energy']
    )

    print(f"Created collector with sample_interval={collector.sample_interval}")

    # Simulate training
    num_steps = 100
    for step in range(num_steps):
        wave = create_sample_wave()
        collector.collect(wave, step=step)

    # Get collected data
    data = collector.get_data()

    print(f"\nCollected {len(collector)} samples over {num_steps} steps")
    print(f"Keys in collected data: {list(data.keys())}")

    if 'total_energy' in data:
        energies = data['total_energy']
        print(f"\nTotal Energy over time:")
        print(f"  Mean: {np.mean(energies):.4f}")
        print(f"  Std:  {np.std(energies):.4f}")
        print(f"  Trend: {energies[0]:.4f} -> {energies[-1]:.4f}")


def example_5_export_utilities():
    """Example 5: Export analysis results."""
    print("\n" + "="*70)
    print("Example 5: Export Utilities")
    print("="*70)

    # Create sample data
    data = {
        'statistics': {
            'mean_frequency': 5.234,
            'mean_amplitude': 1.456,
            'total_energy': 234.567
        },
        'harmonics': {
            'top_indices': [5, 12, 23, 34, 45],
            'top_scores': [0.95, 0.87, 0.76, 0.65, 0.54]
        },
        'training_metrics': {
            'steps': list(range(0, 100, 10)),
            'energy': list(np.random.randn(10) + 100)
        }
    }

    # Create output directory
    output_dir = Path("E:/WaveML/Wave-Transformer/analysis_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export to JSON
    json_path = output_dir / "example_results.json"
    AnalysisExporter.to_json(data, json_path)
    print(f"\nExported to JSON: {json_path}")

    # Load back
    loaded_data = AnalysisExporter.from_json(json_path)
    print(f"Loaded data keys: {list(loaded_data.keys())}")

    # Export to HDF5 (if h5py is available)
    try:
        h5_path = output_dir / "example_results.h5"
        AnalysisExporter.to_hdf5(data, h5_path)
        print(f"Exported to HDF5: {h5_path}")

        # Load back
        loaded_h5 = AnalysisExporter.from_hdf5(h5_path)
        print(f"Loaded HDF5 data keys: {list(loaded_h5.keys())}")
    except ImportError:
        print("h5py not available, skipping HDF5 export")


def example_6_memory_efficient_storage():
    """Example 6: Memory-efficient storage utilities."""
    print("\n" + "="*70)
    print("Example 6: Memory-Efficient Storage")
    print("="*70)

    # Circular buffer
    print("\nCircular Buffer:")
    buffer = CircularBuffer(capacity=10)

    for i in range(25):
        buffer.append(float(i))

    print(f"  Capacity: {buffer.capacity}")
    print(f"  Size: {len(buffer)}")
    print(f"  Recent values: {buffer.get(last_n=5)}")
    print(f"  Mean: {buffer.mean():.4f}")

    # Streaming statistics
    print("\nStreaming Statistics:")
    stats = StreamingStatistics()

    for i in range(1000):
        value = np.random.randn() * 10 + 50
        stats.update(value)

    print(f"  Samples processed: {len(stats)}")
    print(f"  Mean: {stats.get_mean():.4f}")
    print(f"  Std:  {stats.get_std():.4f}")
    print(f"  Min:  {stats.get_min():.4f}")
    print(f"  Max:  {stats.get_max():.4f}")

    # Downsampled storage
    print("\nDownsampled Storage:")
    storage = DownsampledStorage(
        full_resolution_size=100,
        downsample_factor=10,
        num_levels=3
    )

    for i in range(10000):
        storage.append(float(i))

    storage_stats = storage.get_statistics()
    print(f"  Total samples seen: {storage_stats['total_samples_seen']}")
    print(f"  Total samples stored: {storage_stats['total_samples_stored']}")
    print(f"  Compression ratio: {storage_stats['compression_ratio']:.2f}x")
    print(f"  Level sizes: {storage_stats['level_sizes']}")

    # Exponential moving average
    print("\nExponential Moving Average:")
    ema = ExponentialMovingAverage(alpha=0.1)

    values = []
    ema_values = []
    for i in range(100):
        value = 10 + 5 * np.sin(i * 0.1) + np.random.randn()
        ema.update(value)
        values.append(value)
        ema_values.append(ema.get())

    print(f"  Last raw value: {values[-1]:.4f}")
    print(f"  Last EMA value: {ema_values[-1]:.4f}")
    print(f"  Smoothing effect: {abs(values[-1] - ema_values[-1]):.4f}")


def main():
    """Run all examples."""
    print("\n")
    print("="*70)
    print("Wave Transformer Analysis Suite - Usage Examples")
    print("="*70)

    # Run examples
    example_1_basic_statistics()
    example_2_harmonic_importance()
    example_3_spectral_features()
    example_4_wave_collector()
    example_5_export_utilities()
    example_6_memory_efficient_storage()

    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
