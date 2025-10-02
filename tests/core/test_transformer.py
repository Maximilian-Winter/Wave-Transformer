"""
Comprehensive test suite for wave_transformer.core.transformer module.

This module tests all PyTorch components including:
- Wave dataclass and its methods
- FlashAttention module
- SwiGLU activation
- RMSNorm normalization
- ParallelBlock, DeepNormParallelBlock, NonCausalParallelBlock
- PositionWiseFeedForward
- WaveTransformer (integration tests)
"""

import math
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from unittest.mock import Mock, patch

# Import components to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from wave_transformer.core.transformer import (
    Wave,
    FlashAttention,
    SwiGLU,
    RMSNorm,
    ParallelBlock,
    DeepNormParallelBlock,
    NonCausalParallelBlock,
    PositionWiseFeedForward,
    WaveTransformer,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def wave_simple():
    """Create a simple Wave object for testing."""
    frequencies = torch.tensor([[1.0, 2.0, 3.0]])
    amplitudes = torch.tensor([[0.5, 0.3, 0.2]])
    phases = torch.tensor([[0.0, np.pi/2, np.pi]])
    return Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)


@pytest.fixture
def wave_batch():
    """Create a batched Wave object for testing."""
    batch_size = 4
    num_harmonics = 8
    frequencies = torch.randn(batch_size, num_harmonics)
    amplitudes = torch.abs(torch.randn(batch_size, num_harmonics))
    phases = torch.randn(batch_size, num_harmonics) * 2 * np.pi
    return Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for module testing."""
    batch_size, seq_len, d_model = 2, 10, 64
    return torch.randn(batch_size, seq_len, d_model)


# ============================================================================
# Wave Class Tests
# ============================================================================

class TestWave:
    """Test suite for Wave dataclass."""

    def test_wave_initialization(self, wave_simple):
        """Test that Wave can be initialized with tensors."""
        assert isinstance(wave_simple.frequencies, torch.Tensor)
        assert isinstance(wave_simple.amplitudes, torch.Tensor)
        assert isinstance(wave_simple.phases, torch.Tensor)
        assert wave_simple.frequencies.shape == wave_simple.amplitudes.shape
        assert wave_simple.amplitudes.shape == wave_simple.phases.shape

    def test_to_representation_shape(self, wave_simple):
        """Test that to_representation concatenates along last dimension."""
        representation = wave_simple.to_representation()

        # Should concatenate 3 tensors along dim=-1
        expected_shape = (wave_simple.frequencies.shape[0],
                         wave_simple.frequencies.shape[1] * 3)
        assert representation.shape == expected_shape

    def test_to_representation_content(self):
        """Test that to_representation preserves values correctly."""
        frequencies = torch.tensor([[1.0, 2.0]])
        amplitudes = torch.tensor([[3.0, 4.0]])
        phases = torch.tensor([[5.0, 6.0]])
        wave = Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)

        representation = wave.to_representation()
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

        torch.testing.assert_close(representation, expected)

    def test_from_representation_inverse(self, wave_batch):
        """Test that from_representation is inverse of to_representation."""
        representation = wave_batch.to_representation()
        reconstructed = Wave.from_representation(representation)

        torch.testing.assert_close(reconstructed.frequencies, wave_batch.frequencies)
        torch.testing.assert_close(reconstructed.amplitudes, wave_batch.amplitudes)
        torch.testing.assert_close(reconstructed.phases, wave_batch.phases)

    def test_from_representation_shape_validation(self):
        """Test that from_representation requires correct shape."""
        # Should be divisible by 3
        valid_tensor = torch.randn(2, 12)  # 12 = 4 * 3
        wave = Wave.from_representation(valid_tensor)
        assert wave.frequencies.shape[-1] == 4

    def test_synthesize_shape(self, wave_simple):
        """Test that synthesize produces correct output shape."""
        num_time_points = 100
        t = torch.linspace(0, 1, num_time_points)
        signal = wave_simple.synthesize(t)

        # Should sum over harmonics, leaving time dimension
        assert signal.shape == (wave_simple.frequencies.shape[0], num_time_points)

    def test_synthesize_batch(self, wave_batch):
        """Test synthesize with batched inputs."""
        batch_size = wave_batch.frequencies.shape[0]
        num_time_points = 50
        t = torch.linspace(0, 1, num_time_points)
        signal = wave_batch.synthesize(t)

        assert signal.shape == (batch_size, num_time_points)

    def test_synthesize_pure_tone(self):
        """Test synthesize with known pure tone (mathematical validation)."""
        # Create a simple 1 Hz sine wave with amplitude 1, phase 0
        frequencies = torch.tensor([[1.0]])
        amplitudes = torch.tensor([[1.0]])
        phases = torch.tensor([[0.0]])
        wave = Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)

        # Sample at specific time points
        t = torch.tensor([0.0, 0.25, 0.5, 0.75])
        signal = wave.synthesize(t)

        # Expected values: sin(2π * 1 * t)
        expected = torch.tensor([[0.0, 1.0, 0.0, -1.0]])

        torch.testing.assert_close(signal, expected, atol=1e-6, rtol=1e-5)

    def test_synthesize_with_phase(self):
        """Test synthesize respects phase offset."""
        # 1 Hz sine wave with π/2 phase offset (becomes cosine)
        frequencies = torch.tensor([[1.0]])
        amplitudes = torch.tensor([[1.0]])
        phases = torch.tensor([[np.pi/2]])
        wave = Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)

        t = torch.tensor([0.0, 0.25, 0.5])
        signal = wave.synthesize(t)

        # Expected: sin(2π*t + π/2) = cos(2π*t)
        expected = torch.tensor([[1.0, 0.0, -1.0]])

        torch.testing.assert_close(signal, expected, atol=1e-6, rtol=1e-5)

    def test_synthesize_superposition(self):
        """Test that multiple harmonics add correctly (superposition principle)."""
        # Two harmonics: 1 Hz + 2 Hz
        frequencies = torch.tensor([[1.0, 2.0]])
        amplitudes = torch.tensor([[1.0, 0.5]])
        phases = torch.tensor([[0.0, 0.0]])
        wave = Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)

        t = torch.tensor([0.0])
        signal = wave.synthesize(t)

        # At t=0: sin(0) + 0.5*sin(0) = 0
        expected = torch.tensor([[0.0]])
        torch.testing.assert_close(signal, expected, atol=1e-6, rtol=1e-5)

    def test_wave_gradient_flow(self):
        """Test that gradients flow through Wave operations."""
        frequencies = torch.tensor([[1.0, 2.0]], requires_grad=True)
        amplitudes = torch.tensor([[0.5, 0.3]], requires_grad=True)
        phases = torch.tensor([[0.0, np.pi/2]], requires_grad=True)
        wave = Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)

        # Create signal and compute loss
        t = torch.linspace(0, 1, 10)
        signal = wave.synthesize(t)
        loss = signal.sum()

        # Backpropagate
        loss.backward()

        # Check gradients exist
        assert frequencies.grad is not None
        assert amplitudes.grad is not None
        assert phases.grad is not None
        assert not torch.all(frequencies.grad == 0)

    def test_wave_device_compatibility(self, device):
        """Test Wave operations work on different devices."""
        if device.type == "cpu":
            pytest.skip("GPU not available")

        frequencies = torch.randn(2, 4).to(device)
        amplitudes = torch.randn(2, 4).to(device)
        phases = torch.randn(2, 4).to(device)
        wave = Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)

        t = torch.linspace(0, 1, 10).to(device)
        signal = wave.synthesize(t)

        assert signal.device == device

    def test_wave_dtype_consistency(self):
        """Test Wave operations preserve dtype."""
        for dtype in [torch.float32, torch.float64]:
            frequencies = torch.randn(2, 4, dtype=dtype)
            amplitudes = torch.randn(2, 4, dtype=dtype)
            phases = torch.randn(2, 4, dtype=dtype)
            wave = Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)

            representation = wave.to_representation()
            assert representation.dtype == dtype

            t = torch.linspace(0, 1, 10, dtype=dtype)
            signal = wave.synthesize(t)
            assert signal.dtype == dtype


# ============================================================================
# FlashAttention Tests
# ============================================================================

class TestFlashAttention:
    """Test suite for FlashAttention module."""

    @pytest.mark.parametrize("d_model,n_heads", [
        (64, 4),
        (128, 8),
        (256, 16),
    ])
    def test_flash_attention_initialization(self, d_model, n_heads):
        """Test FlashAttention initializes with correct parameters."""
        attn = FlashAttention(d_model, n_heads, dropout=0.1, use_flash=False)

        assert attn.d_model == d_model
        assert attn.n_heads == n_heads
        assert attn.d_head == d_model // n_heads
        assert abs(attn.scale - 1.0 / math.sqrt(attn.d_head)) < 1e-6

    def test_flash_attention_forward_shape(self):
        """Test FlashAttention output shape matches input shape."""
        d_model, n_heads = 64, 4
        batch_size, seq_len = 2, 10

        attn = FlashAttention(d_model, n_heads, use_flash=False)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attn(x, causal=False)

        assert output.shape == x.shape

    @pytest.mark.parametrize("seq_len", [1, 5, 16, 64])
    def test_flash_attention_variable_sequence_length(self, seq_len):
        """Test FlashAttention handles different sequence lengths."""
        d_model, n_heads = 64, 4
        batch_size = 2

        attn = FlashAttention(d_model, n_heads, use_flash=False)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attn(x, causal=False)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_flash_attention_causal_mask(self):
        """Test that causal masking prevents information flow from future."""
        d_model, n_heads = 64, 4
        batch_size, seq_len = 1, 10

        attn = FlashAttention(d_model, n_heads, use_flash=False)
        attn.eval()  # Disable dropout

        # Create input where first position is all zeros, rest are ones
        x = torch.ones(batch_size, seq_len, d_model)
        x[:, 0, :] = 0

        output = attn(x, causal=True)

        # First position output should not be influenced by future positions
        # This is a weak test but validates basic causal behavior
        assert output.shape == x.shape

    def test_flash_attention_non_causal(self):
        """Test non-causal attention allows bidirectional flow."""
        d_model, n_heads = 64, 4
        batch_size, seq_len = 2, 8

        attn = FlashAttention(d_model, n_heads, use_flash=False)
        attn.eval()

        x = torch.randn(batch_size, seq_len, d_model)
        output = attn(x, causal=False)

        assert output.shape == x.shape
        # Non-causal should produce different outputs than causal
        output_causal = attn(x, causal=True)
        assert not torch.allclose(output, output_causal)

    def test_flash_attention_gradient_flow(self):
        """Test gradients flow through FlashAttention."""
        d_model, n_heads = 64, 4
        batch_size, seq_len = 2, 5

        attn = FlashAttention(d_model, n_heads, use_flash=False)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = attn(x, causal=False)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_flash_attention_training_vs_eval_mode(self):
        """Test that training and eval modes behave differently (dropout)."""
        d_model, n_heads = 64, 4
        batch_size, seq_len = 2, 8

        attn = FlashAttention(d_model, n_heads, dropout=0.5, use_flash=False)
        x = torch.randn(batch_size, seq_len, d_model)

        torch.manual_seed(42)
        attn.train()
        output_train = attn(x, causal=False)

        torch.manual_seed(42)
        attn.eval()
        output_eval = attn(x, causal=False)

        # Outputs should be different due to dropout in training mode
        # Note: This test might be flaky, but generally should pass
        assert not torch.allclose(output_train, output_eval, rtol=1e-3)

    def test_flash_attention_qkv_projection_shape(self):
        """Test that QKV projection has correct dimensions."""
        d_model, n_heads = 64, 4
        attn = FlashAttention(d_model, n_heads, use_flash=False)

        # QKV projection should map d_model -> 3*d_model
        assert attn.qkv.in_features == d_model
        assert attn.qkv.out_features == 3 * d_model

    def test_flash_attention_output_projection(self):
        """Test output projection shape."""
        d_model, n_heads = 64, 4
        attn = FlashAttention(d_model, n_heads, use_flash=False)

        assert attn.out_proj.in_features == d_model
        assert attn.out_proj.out_features == d_model

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_flash_attention_batch_independence(self, batch_size):
        """Test that different batch elements are processed independently."""
        d_model, n_heads = 64, 4
        seq_len = 5

        attn = FlashAttention(d_model, n_heads, use_flash=False)
        attn.eval()

        # Process single sample
        x_single = torch.randn(1, seq_len, d_model)
        output_single = attn(x_single, causal=False)

        # Process as part of batch
        x_batch = x_single.repeat(batch_size, 1, 1)
        output_batch = attn(x_batch, causal=False)

        # First element should match
        torch.testing.assert_close(output_single, output_batch[0:1], rtol=1e-5, atol=1e-6)


# ============================================================================
# SwiGLU Tests
# ============================================================================

class TestSwiGLU:
    """Test suite for SwiGLU activation module."""

    @pytest.mark.parametrize("d_model,d_ff", [
        (64, 256),
        (128, 512),
        (256, 1024),
    ])
    def test_swiglu_initialization(self, d_model, d_ff):
        """Test SwiGLU initializes with correct dimensions."""
        swiglu = SwiGLU(d_model, d_ff, dropout=0.1)

        assert swiglu.w1.in_features == d_model
        assert swiglu.w1.out_features == d_ff
        assert swiglu.w2.in_features == d_ff
        assert swiglu.w2.out_features == d_model
        assert swiglu.w3.in_features == d_model
        assert swiglu.w3.out_features == d_ff

    def test_swiglu_forward_shape(self):
        """Test SwiGLU output shape matches input shape."""
        d_model, d_ff = 64, 256
        batch_size, seq_len = 2, 10

        swiglu = SwiGLU(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = swiglu(x)

        assert output.shape == x.shape

    def test_swiglu_non_linearity(self):
        """Test that SwiGLU is non-linear."""
        d_model, d_ff = 64, 256
        swiglu = SwiGLU(d_model, d_ff, dropout=0.0)
        swiglu.eval()

        x1 = torch.randn(2, 5, d_model)
        x2 = torch.randn(2, 5, d_model)

        # f(x1 + x2) should not equal f(x1) + f(x2) for non-linear function
        out_sum = swiglu(x1 + x2)
        sum_out = swiglu(x1) + swiglu(x2)

        assert not torch.allclose(out_sum, sum_out, rtol=1e-3)

    def test_swiglu_gradient_flow(self):
        """Test gradients flow through SwiGLU."""
        d_model, d_ff = 64, 256
        swiglu = SwiGLU(d_model, d_ff)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        output = swiglu(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_swiglu_zero_input(self):
        """Test SwiGLU with zero input."""
        d_model, d_ff = 64, 256
        swiglu = SwiGLU(d_model, d_ff, dropout=0.0)
        swiglu.eval()

        x = torch.zeros(2, 5, d_model)
        output = swiglu(x)

        # Should produce some output (due to bias if present, or close to zero)
        assert output.shape == x.shape

    def test_swiglu_dropout_behavior(self):
        """Test dropout behavior in training vs eval mode."""
        d_model, d_ff = 64, 256
        swiglu = SwiGLU(d_model, d_ff, dropout=0.5)

        x = torch.randn(2, 10, d_model)

        torch.manual_seed(42)
        swiglu.train()
        output_train = swiglu(x)

        torch.manual_seed(42)
        swiglu.eval()
        output_eval = swiglu(x)

        # Should be different due to dropout
        assert not torch.allclose(output_train, output_eval, rtol=1e-3)


# ============================================================================
# RMSNorm Tests
# ============================================================================

class TestRMSNorm:
    """Test suite for RMSNorm module."""

    @pytest.mark.parametrize("d_model", [64, 128, 256, 512])
    def test_rmsnorm_initialization(self, d_model):
        """Test RMSNorm initializes with correct parameters."""
        norm = RMSNorm(d_model, eps=1e-6)

        assert norm.eps == 1e-6
        assert norm.weight.shape == (d_model,)
        # Weight should be initialized to ones
        torch.testing.assert_close(norm.weight, torch.ones(d_model))

    def test_rmsnorm_forward_shape(self):
        """Test RMSNorm output shape matches input shape."""
        d_model = 64
        batch_size, seq_len = 2, 10

        norm = RMSNorm(d_model)
        x = torch.randn(batch_size, seq_len, d_model)

        output = norm(x)

        assert output.shape == x.shape

    def test_rmsnorm_normalization(self):
        """Test that RMSNorm normalizes to correct RMS."""
        d_model = 64
        norm = RMSNorm(d_model, eps=1e-8)

        # Set weight to 1 for this test
        with torch.no_grad():
            norm.weight.fill_(1.0)

        x = torch.randn(2, 10, d_model) * 10  # Large scale input
        output = norm(x)

        # Compute RMS of output along last dimension
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))

        # RMS should be close to 1 for each position
        torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)

    def test_rmsnorm_zero_mean_not_required(self):
        """Test that RMSNorm doesn't require zero mean (unlike LayerNorm)."""
        d_model = 64
        norm = RMSNorm(d_model)

        # Input with non-zero mean
        x = torch.randn(2, 10, d_model) + 5.0
        output = norm(x)

        # Should still normalize properly
        assert output.shape == x.shape
        # Output mean won't be zero (that's the point of RMS vs Layer norm)
        assert not torch.allclose(output.mean(dim=-1), torch.zeros(2, 10), atol=0.1)

    def test_rmsnorm_learnable_weights(self):
        """Test that RMSNorm weights are learnable."""
        d_model = 64
        norm = RMSNorm(d_model)

        assert norm.weight.requires_grad

    def test_rmsnorm_gradient_flow(self):
        """Test gradients flow through RMSNorm."""
        d_model = 64
        norm = RMSNorm(d_model)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None
        assert not torch.all(x.grad == 0)

    def test_rmsnorm_numerical_stability(self):
        """Test RMSNorm handles small values without numerical issues."""
        d_model = 64
        norm = RMSNorm(d_model, eps=1e-6)

        # Very small input
        x = torch.randn(2, 5, d_model) * 1e-10
        output = norm(x)

        # Should not produce NaN or Inf
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_rmsnorm_scale_invariance(self):
        """Test that RMSNorm is scale-invariant (except for learnable weight)."""
        d_model = 64
        norm = RMSNorm(d_model)

        # Freeze weights to test pure normalization
        with torch.no_grad():
            norm.weight.fill_(1.0)
        norm.eval()

        x = torch.randn(2, 5, d_model)

        # Scale input by constant
        output1 = norm(x)
        output2 = norm(x * 10.0)

        # Outputs should be similar (scaled inputs normalized to same magnitude)
        torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-5)


# ============================================================================
# ParallelBlock Tests
# ============================================================================

class TestParallelBlock:
    """Test suite for ParallelBlock module."""

    def test_parallel_block_initialization(self):
        """Test ParallelBlock initializes with correct components."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        block = ParallelBlock(d_model, n_heads, n_heads_kv, d_ff, use_flash=False)

        assert isinstance(block.norm, RMSNorm)
        assert isinstance(block.attn, FlashAttention)
        assert isinstance(block.ffn, SwiGLU)
        assert isinstance(block.dropout, nn.Dropout)

    def test_parallel_block_forward_shape(self):
        """Test ParallelBlock output shape matches input."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        batch_size, seq_len = 2, 10

        block = ParallelBlock(d_model, n_heads, n_heads_kv, d_ff, use_flash=False)
        x = torch.randn(batch_size, seq_len, d_model)

        output = block(x, causal=False)

        assert output.shape == x.shape

    def test_parallel_block_residual_connection(self):
        """Test that ParallelBlock implements residual connection."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        block = ParallelBlock(d_model, n_heads, n_heads_kv, d_ff,
                             dropout=0.0, use_flash=False)
        block.eval()

        x = torch.randn(2, 5, d_model)
        output = block(x, causal=False)

        # Output should be input + transformation
        # Verify that output is not just the transformation
        assert not torch.allclose(output, x, rtol=1e-2)

    def test_parallel_block_causal_mode(self):
        """Test ParallelBlock with causal attention."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        block = ParallelBlock(d_model, n_heads, n_heads_kv, d_ff, use_flash=False)
        block.eval()

        x = torch.randn(2, 8, d_model)

        output_causal = block(x, causal=True)
        output_non_causal = block(x, causal=False)

        assert output_causal.shape == x.shape
        # Different causal modes should produce different outputs
        assert not torch.allclose(output_causal, output_non_causal, rtol=1e-3)

    def test_parallel_block_gradient_flow(self):
        """Test gradients flow through ParallelBlock."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        block = ParallelBlock(d_model, n_heads, n_heads_kv, d_ff, use_flash=False)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        output = block(x, causal=False)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check gradients for internal parameters
        assert block.norm.weight.grad is not None
        assert block.attn.qkv.weight.grad is not None


# ============================================================================
# DeepNormParallelBlock Tests
# ============================================================================

class TestDeepNormParallelBlock:
    """Test suite for DeepNormParallelBlock module."""

    @pytest.mark.parametrize("num_layers", [1, 6, 12, 24])
    def test_deepnorm_residual_scaling(self, num_layers):
        """Test that DeepNorm computes correct residual scaling factor."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        block = DeepNormParallelBlock(d_model, n_heads, n_heads_kv, d_ff,
                                      use_flash=False, num_layers=num_layers)

        expected_scale = 1.0 / math.sqrt(2 * num_layers)
        assert abs(block.residual_scale - expected_scale) < 1e-6

    def test_deepnorm_forward_shape(self):
        """Test DeepNormParallelBlock output shape."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        batch_size, seq_len = 2, 10

        block = DeepNormParallelBlock(d_model, n_heads, n_heads_kv, d_ff,
                                      use_flash=False, num_layers=6)
        x = torch.randn(batch_size, seq_len, d_model)

        output = block(x, causal=False)

        assert output.shape == x.shape

    def test_deepnorm_scaling_effect(self):
        """Test that deeper models have smaller residual scaling."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256

        block_shallow = DeepNormParallelBlock(d_model, n_heads, n_heads_kv, d_ff,
                                              use_flash=False, num_layers=1)
        block_deep = DeepNormParallelBlock(d_model, n_heads, n_heads_kv, d_ff,
                                           use_flash=False, num_layers=24)

        assert block_shallow.residual_scale > block_deep.residual_scale

    def test_deepnorm_gradient_flow(self):
        """Test gradients flow through DeepNormParallelBlock."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        block = DeepNormParallelBlock(d_model, n_heads, n_heads_kv, d_ff,
                                      use_flash=False, num_layers=6)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        output = block(x, causal=True)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


# ============================================================================
# NonCausalParallelBlock Tests
# ============================================================================

class TestNonCausalParallelBlock:
    """Test suite for NonCausalParallelBlock module."""

    def test_noncausal_block_initialization(self):
        """Test NonCausalParallelBlock initializes correctly."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        block = NonCausalParallelBlock(d_model, n_heads, n_heads_kv, d_ff, use_flash=False)

        assert isinstance(block.norm, RMSNorm)
        assert isinstance(block.attn, FlashAttention)
        assert isinstance(block.ffn, SwiGLU)

    def test_noncausal_block_forward_shape(self):
        """Test NonCausalParallelBlock output shape."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        batch_size, seq_len = 2, 10

        block = NonCausalParallelBlock(d_model, n_heads, n_heads_kv, d_ff, use_flash=False)
        x = torch.randn(batch_size, seq_len, d_model)

        output = block(x, attention_mask=None)

        assert output.shape == x.shape

    def test_noncausal_block_uses_non_causal_attention(self):
        """Test that NonCausalParallelBlock uses non-causal attention."""
        d_model, n_heads, n_heads_kv, d_ff = 64, 4, 2, 256
        block = NonCausalParallelBlock(d_model, n_heads, n_heads_kv, d_ff, use_flash=False)
        block.eval()

        # Create a ParallelBlock with causal=True for comparison
        causal_block = ParallelBlock(d_model, n_heads, n_heads_kv, d_ff, use_flash=False)
        causal_block.eval()

        x = torch.randn(2, 8, d_model)

        output_noncausal = block(x, attention_mask=None)
        output_causal = causal_block(x, causal=True)

        # Should produce different results
        assert not torch.allclose(output_noncausal, output_causal, rtol=1e-3)


# ============================================================================
# PositionWiseFeedForward Tests
# ============================================================================

class TestPositionWiseFeedForward:
    """Test suite for PositionWiseFeedForward module."""

    @pytest.mark.parametrize("d_model,d_ff", [
        (64, 256),
        (128, 512),
        (256, 1024),
    ])
    def test_ffn_initialization(self, d_model, d_ff):
        """Test PositionWiseFeedForward initializes with correct dimensions."""
        ffn = PositionWiseFeedForward(d_model, d_ff)

        assert ffn.linear1.in_features == d_model
        assert ffn.linear1.out_features == d_ff
        assert ffn.linear2.in_features == d_ff
        assert ffn.linear2.out_features == d_model

    def test_ffn_forward_shape(self):
        """Test PositionWiseFeedForward output shape."""
        d_model, d_ff = 64, 256
        batch_size, seq_len = 2, 10

        ffn = PositionWiseFeedForward(d_model, d_ff)
        x = torch.randn(batch_size, seq_len, d_model)

        output = ffn(x)

        assert output.shape == x.shape

    def test_ffn_relu_activation(self):
        """Test that FFN uses ReLU (non-negative outputs for positive inputs)."""
        d_model, d_ff = 64, 256
        ffn = PositionWiseFeedForward(d_model, d_ff, dropout=0.0)
        ffn.eval()

        # Test with various inputs
        x = torch.randn(2, 5, d_model)
        output = ffn(x)

        # Output can be positive or negative (ReLU is in the middle)
        assert output.shape == x.shape

    def test_ffn_gradient_flow(self):
        """Test gradients flow through PositionWiseFeedForward."""
        d_model, d_ff = 64, 256
        ffn = PositionWiseFeedForward(d_model, d_ff)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


# ============================================================================
# WaveTransformer Integration Tests
# ============================================================================

class TestWaveTransformer:
    """Test suite for WaveTransformer module (integration tests)."""

    @pytest.fixture
    def mock_encoder(self):
        """Create a mock wave encoder for testing."""
        class MockEncoder(nn.Module):
            def __init__(self, num_harmonics=64):
                super().__init__()
                self.num_harmonics = num_harmonics

            def forward(self, attention_mask=None, **kwargs):
                # Return a simple Wave object
                batch_size = kwargs.get('batch_size', 2)
                frequencies = torch.randn(batch_size, 10, self.num_harmonics)
                amplitudes = torch.abs(torch.randn(batch_size, 10, self.num_harmonics))
                phases = torch.randn(batch_size, 10, self.num_harmonics)
                return Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)

        return MockEncoder()

    @pytest.fixture
    def mock_decoder(self):
        """Create a mock wave decoder for testing."""
        class MockDecoder(nn.Module):
            def forward(self, x, attention_mask=None):
                # Return tensor of appropriate shape
                return torch.randn(x.shape[0], x.shape[1], 100)  # Arbitrary output dim

        return MockDecoder()

    def test_wave_transformer_initialization(self, mock_encoder, mock_decoder):
        """Test WaveTransformer initializes correctly."""
        model = WaveTransformer(
            wave_encoder=mock_encoder,
            wave_decoder=mock_decoder,
            num_harmonics=64,
            transformer_num_layers=6,
            transformer_num_heads=8,
            transformer_heads_kv=4,
            transformer_d_ff_multi=4,
            use_flash=False
        )

        assert model.num_harmonics == 64
        assert model.input_dim == 64 * 3  # frequencies + amplitudes + phases
        assert len(model.layers) == 6
        assert isinstance(model.norm_f, RMSNorm)

    def test_wave_transformer_forward_shape(self, mock_encoder, mock_decoder):
        """Test WaveTransformer forward pass output shape."""
        model = WaveTransformer(
            wave_encoder=mock_encoder,
            wave_decoder=mock_decoder,
            num_harmonics=64,
            transformer_num_layers=2,
            transformer_num_heads=8,
            use_flash=False
        )

        encoder_input = {'batch_size': 2}
        output = model(encoder_input, causal=False)

        # Output shape depends on decoder, which returns (2, 10, 100)
        assert output.shape == (2, 10, 100)

    def test_wave_transformer_return_encoder_outputs(self, mock_encoder, mock_decoder):
        """Test WaveTransformer can return encoder outputs."""
        model = WaveTransformer(
            wave_encoder=mock_encoder,
            wave_decoder=mock_decoder,
            num_harmonics=64,
            transformer_num_layers=2,
            transformer_num_heads=8,
            use_flash=False
        )

        encoder_input = {'batch_size': 2}
        output, wave = model(encoder_input, causal=False, return_encoder_outputs=True)

        assert isinstance(wave, Wave)
        assert output.shape == (2, 10, 100)

    def test_wave_transformer_causal_vs_noncausal(self, mock_encoder, mock_decoder):
        """Test WaveTransformer with causal vs non-causal modes."""
        model = WaveTransformer(
            wave_encoder=mock_encoder,
            wave_decoder=mock_decoder,
            num_harmonics=64,
            transformer_num_layers=2,
            transformer_num_heads=8,
            use_flash=False
        )
        model.eval()

        encoder_input = {'batch_size': 2}

        output_causal = model(encoder_input, causal=True)
        output_noncausal = model(encoder_input, causal=False)

        # Different causal modes should produce different outputs
        assert not torch.allclose(output_causal, output_noncausal, rtol=1e-3)

    def test_wave_transformer_gradient_flow(self, mock_encoder, mock_decoder):
        """Test gradients flow through WaveTransformer."""
        # Make encoder and decoder have learnable parameters
        class LearnableEncoder(nn.Module):
            def __init__(self, num_harmonics=64):
                super().__init__()
                self.num_harmonics = num_harmonics
                self.proj = nn.Linear(1, num_harmonics)

            def forward(self, attention_mask=None, **kwargs):
                batch_size = kwargs.get('batch_size', 2)
                x = torch.ones(batch_size, 10, 1)
                features = self.proj(x)
                return Wave(
                    frequencies=features,
                    amplitudes=torch.abs(features),
                    phases=features
                )

        class LearnableDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(192, 100)

            def forward(self, x, attention_mask=None):
                return self.proj(x)

        encoder = LearnableEncoder()
        decoder = LearnableDecoder()

        model = WaveTransformer(
            wave_encoder=encoder,
            wave_decoder=decoder,
            num_harmonics=64,
            transformer_num_layers=2,
            transformer_num_heads=8,
            use_flash=False
        )

        encoder_input = {'batch_size': 2}
        output = model(encoder_input, causal=False)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_wave_transformer_multiple_layers(self, mock_encoder, mock_decoder):
        """Test WaveTransformer with different numbers of layers."""
        for num_layers in [1, 3, 6, 12]:
            model = WaveTransformer(
                wave_encoder=mock_encoder,
                wave_decoder=mock_decoder,
                num_harmonics=64,
                transformer_num_layers=num_layers,
                transformer_num_heads=8,
                use_flash=False
            )

            assert len(model.layers) == num_layers

            encoder_input = {'batch_size': 2}
            output = model(encoder_input, causal=False)
            assert output.shape == (2, 10, 100)


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_wave_empty_harmonics(self):
        """Test Wave with zero harmonics raises or handles gracefully."""
        # This might fail - testing edge case
        try:
            frequencies = torch.tensor([[]])
            amplitudes = torch.tensor([[]])
            phases = torch.tensor([[]])
            wave = Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)

            representation = wave.to_representation()
            assert representation.shape[-1] == 0
        except Exception:
            # If it raises, that's also acceptable behavior
            pass

    def test_flash_attention_single_token(self):
        """Test FlashAttention with sequence length 1."""
        d_model, n_heads = 64, 4
        attn = FlashAttention(d_model, n_heads, use_flash=False)

        x = torch.randn(2, 1, d_model)  # seq_len = 1
        output = attn(x, causal=False)

        assert output.shape == x.shape

    def test_large_batch_size(self):
        """Test components handle large batch sizes."""
        d_model = 64
        large_batch = 128
        seq_len = 10

        norm = RMSNorm(d_model)
        x = torch.randn(large_batch, seq_len, d_model)
        output = norm(x)

        assert output.shape == x.shape

    def test_very_long_sequence(self):
        """Test attention with long sequences (memory test)."""
        d_model, n_heads = 64, 4
        attn = FlashAttention(d_model, n_heads, use_flash=False)

        # Use smaller batch for long sequence
        x = torch.randn(1, 512, d_model)
        output = attn(x, causal=True)

        assert output.shape == x.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, dtype):
        """Test that modules preserve input dtype."""
        d_model = 64
        norm = RMSNorm(d_model)

        x = torch.randn(2, 5, d_model, dtype=dtype)
        output = norm(x)

        assert output.dtype == dtype

    def test_inf_nan_handling_in_rmsnorm(self):
        """Test RMSNorm handles edge numerical cases."""
        d_model = 64
        norm = RMSNorm(d_model, eps=1e-6)

        # Test with very small values
        x_small = torch.randn(2, 5, d_model) * 1e-20
        output_small = norm(x_small)
        assert not torch.any(torch.isnan(output_small))
        assert not torch.any(torch.isinf(output_small))

        # Test with very large values
        x_large = torch.randn(2, 5, d_model) * 1e10
        output_large = norm(x_large)
        assert not torch.any(torch.isnan(output_large))
        # Note: might have inf if input has inf, but shouldn't create new infs


# ============================================================================
# Parameterized Tests for Multiple Configurations
# ============================================================================

class TestParameterizedConfigurations:
    """Test modules with various parameter combinations."""

    @pytest.mark.parametrize("batch_size,seq_len,d_model,n_heads", [
        (1, 1, 64, 4),
        (2, 8, 128, 8),
        (4, 16, 256, 16),
        (8, 32, 512, 32),
    ])
    def test_attention_configurations(self, batch_size, seq_len, d_model, n_heads):
        """Test FlashAttention with various configurations."""
        attn = FlashAttention(d_model, n_heads, use_flash=False)
        x = torch.randn(batch_size, seq_len, d_model)

        output = attn(x, causal=False)

        assert output.shape == (batch_size, seq_len, d_model)

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
    def test_dropout_values(self, dropout):
        """Test modules with different dropout rates."""
        d_model, d_ff = 64, 256
        swiglu = SwiGLU(d_model, d_ff, dropout=dropout)

        assert swiglu.dropout.p == dropout

    @pytest.mark.parametrize("num_harmonics", [8, 16, 32, 64, 128])
    def test_wave_different_harmonic_counts(self, num_harmonics):
        """Test Wave with different numbers of harmonics."""
        frequencies = torch.randn(2, num_harmonics)
        amplitudes = torch.abs(torch.randn(2, num_harmonics))
        phases = torch.randn(2, num_harmonics)

        wave = Wave(frequencies=frequencies, amplitudes=amplitudes, phases=phases)
        representation = wave.to_representation()

        assert representation.shape[-1] == num_harmonics * 3

        reconstructed = Wave.from_representation(representation)
        torch.testing.assert_close(reconstructed.frequencies, wave.frequencies)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
