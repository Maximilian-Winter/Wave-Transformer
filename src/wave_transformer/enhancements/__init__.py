"""
Wave Transformer Enhancements

This module contains advanced enhancements for the Wave Transformer architecture:
- Adaptive Harmonic Sparsification (AHS): Learnable top-k harmonic selection
- Frequency Curriculum Learning (FCL): Progressive frequency unmasking
- Phase-Coherent Cross-Attention (PCCA): Phase-aware attention mechanism
- Multi-Resolution Wave Pyramid (MRWP): Multi-scale wave processing
"""

from wave_transformer.enhancements.adaptive_sparsification import (
    AdaptiveHarmonicSelector,
    HarmonicSparsificationLoss,
)
from wave_transformer.enhancements.curriculum_learning import (
    FrequencyCurriculumScheduler,
    FrequencyMask,
)
from wave_transformer.enhancements.phase_coherent_attention import (
    PhaseCoherentAttention,
    PhaseCoherenceComputer,
)
from wave_transformer.enhancements.multi_resolution_pyramid import (
    MultiResolutionWavePyramid,
    FrequencyBandDecomposer,
    BandProcessor,
    CrossBandAttention,
    AdaptiveFusion,
)

__all__ = [
    # Adaptive Harmonic Sparsification
    "AdaptiveHarmonicSelector",
    "HarmonicSparsificationLoss",
    # Frequency Curriculum Learning
    "FrequencyCurriculumScheduler",
    "FrequencyMask",
    # Phase-Coherent Cross-Attention
    "PhaseCoherentAttention",
    "PhaseCoherenceComputer",
    # Multi-Resolution Wave Pyramid
    "MultiResolutionWavePyramid",
    "FrequencyBandDecomposer",
    "BandProcessor",
    "CrossBandAttention",
    "AdaptiveFusion",
]
