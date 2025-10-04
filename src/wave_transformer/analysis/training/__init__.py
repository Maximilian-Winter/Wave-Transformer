"""
Training monitoring components for Wave Transformer.

Provides hooks, callbacks, and gradient monitoring for training analysis.
"""

from .hooks import (
    HookStorage,
    WaveForwardHook,
    WaveGradientHook,
    AttentionHook,
    HookManager,
)

from .callbacks import (
    AnalysisCallback,
    WaveEvolutionCallback,
    GradientFlowCallback,
    LossAnalysisCallback,
)

from .gradient_monitor import GradientMonitor

__all__ = [
    # Hooks
    'HookStorage',
    'WaveForwardHook',
    'WaveGradientHook',
    'AttentionHook',
    'HookManager',
    # Callbacks
    'AnalysisCallback',
    'WaveEvolutionCallback',
    'GradientFlowCallback',
    'LossAnalysisCallback',
    # Gradient Monitoring
    'GradientMonitor',
]
