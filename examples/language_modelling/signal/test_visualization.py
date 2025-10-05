"""
Quick Test Script for Signal Visualization

Run this to verify the visualization tools work correctly!
"""

import torch
import numpy as np
import sys
from pathlib import Path

# You'll need to adjust these imports based on where you put the files
# from wave_transformer.core.signal_core import SignalConfig, MultiSignal
# from wave_transformer.core.normalization import linear_norm

# For standalone testing, we'll create a simple MultiSignal mock
class SimpleMultiSignal:
    """Simplified MultiSignal for testing."""
    def __init__(self, signals):
        self.signals = signals
        self.representation_data = torch.cat(signals, dim=-1)
    
    def get_all_signals(self):
        return self.signals
    
    @property
    def shape(self):
        return self.representation_data.shape


def test_signal_visualization():
    """Test the signal visualizer with synthetic data."""
    print("🔥 Testing Signal Visualization...")
    
    # Import the visualizer
    try:
        from signal_visualizer import SignalVisualizer, extract_signal_stats
        from signal_training_monitor import SignalTrainingMonitor
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure signal_visualizer.py and signal_training_monitor.py are in the same directory!")
        return
    
    # Create synthetic signal data
    batch_size, seq_len = 2, 20
    dim = 32
    
    print(f"Creating synthetic signals: batch={batch_size}, seq_len={seq_len}, dim={dim}")
    
    # Frequency: sigmoid-like values (0-20)
    frequency = torch.sigmoid(torch.randn(batch_size, seq_len, dim)) * 20 + 0.1
    
    # Amplitude: positive values
    amplitude = torch.abs(torch.randn(batch_size, seq_len, dim)) * 0.5
    
    # Phase: -pi to pi
    phase = torch.tanh(torch.randn(batch_size, seq_len, dim)) * np.pi
    
    # Create MultiSignal
    multi_signal = SimpleMultiSignal([frequency, amplitude, phase])
    
    print(f"✓ Created MultiSignal with shape: {multi_signal.shape}")
    
    # Test visualizer
    viz = SignalVisualizer(['frequency', 'amplitude', 'phase'])
    
    # Test 1: Comprehensive dashboard
    print("\n📊 Creating comprehensive dashboard...")
    tokens = ['The', 'tao', 'that', 'can', 'be', 'told', 'is', 'not']
    viz.plot_comprehensive_dashboard(
        multi_signal,
        tokens=tokens[:min(seq_len, len(tokens))],
        save_path='test_dashboard.png',
        title='Test Signal Dashboard'
    )
    print("✓ Saved: test_dashboard.png")
    
    # Test 2: Signal evolution
    print("\n📈 Creating signal evolution plot...")
    viz.plot_signal_evolution(
        multi_signal,
        seq_indices=[0, 5, 10, 15],
        save_path='test_evolution.png',
        title='Test Signal Evolution'
    )
    print("✓ Saved: test_evolution.png")
    
    # Test 3: Heatmaps
    print("\n🔥 Creating signal heatmaps...")
    viz.plot_signal_heatmaps(
        multi_signal,
        tokens=tokens[:min(seq_len, len(tokens))],
        save_path='test_heatmaps.png',
        title='Test Signal Heatmaps'
    )
    print("✓ Saved: test_heatmaps.png")
    
    # Test 4: Distributions
    print("\n📊 Creating signal distributions...")
    viz.plot_signal_distribution(
        multi_signal,
        save_path='test_distributions.png',
        title='Test Signal Distributions'
    )
    print("✓ Saved: test_distributions.png")
    
    # Test 5: Extract stats
    print("\n📋 Extracting signal statistics...")
    stats = extract_signal_stats(multi_signal)
    print("Signal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 6: Training monitor
    print("\n🎯 Testing training monitor...")
    monitor = SignalTrainingMonitor(
        save_dir='test_viz_output',
        signal_names=['frequency', 'amplitude', 'phase'],
        log_interval=10,
        visualize_interval=50
    )
    
    # Simulate some training steps
    for step in [0, 50, 100, 150]:
        # Add some drift to simulate training
        noise = torch.randn_like(frequency) * 0.1
        freq_step = frequency + noise * step * 0.001
        amp_step = amplitude + noise * step * 0.0005
        phase_step = phase + noise * step * 0.0002
        
        test_signal = SimpleMultiSignal([freq_step, amp_step, phase_step])
        monitor.log_batch_signals(test_signal, step, tokens[:8])
    
    # Plot training evolution
    monitor.plot_training_history()
    print("✓ Saved training evolution")
    
    print("\n✨ All tests completed successfully!")
    print("\nGenerated files:")
    print("  • test_dashboard.png")
    print("  • test_evolution.png")
    print("  • test_heatmaps.png")
    print("  • test_distributions.png")
    print("  • test_viz_output/signal_training_evolution.png")
    print("  • test_viz_output/signals_step_*.png")
    

if __name__ == "__main__":
    test_signal_visualization()
