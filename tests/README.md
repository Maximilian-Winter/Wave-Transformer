# WaveTransformer Test Suite

This directory contains comprehensive unit tests for the WaveTransformer PyTorch implementation.

## Test Structure

```
tests/
├── __init__.py
├── core/
│   ├── __init__.py
│   └── test_transformer.py  # Tests for core transformer components
└── README.md
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/core/test_transformer.py
```

### Run Specific Test Class or Function

```bash
# Run all tests in a class
pytest tests/core/test_transformer.py::TestWave

# Run a specific test function
pytest tests/core/test_transformer.py::TestWave::test_wave_initialization
```

### Run with Coverage Report

```bash
pytest tests/ --cov=src/wave_transformer --cov-report=html
```

### Run Tests in Parallel

```bash
pytest tests/ -n auto
```

## Test Coverage

The test suite covers the following components from `src/wave_transformer/core/transformer.py`:

### 1. Wave Dataclass (TestWave)
- Initialization and shape validation
- `to_representation()` and `from_representation()` methods
- `synthesize()` method with mathematical validation
- Gradient flow through Wave operations
- Device compatibility (CPU/GPU)
- Dtype preservation
- Edge cases (pure tones, superposition, phase offsets)

### 2. FlashAttention Module (TestFlashAttention)
- Initialization with various head configurations
- Forward pass shape transformations
- Causal vs non-causal attention masking
- Gradient flow verification
- Training vs evaluation mode (dropout behavior)
- Batch size variations
- QKV projection dimensions
- Batch independence property

### 3. SwiGLU Activation (TestSwiGLU)
- Initialization with different dimensions
- Forward pass shape preservation
- Non-linearity verification
- Gradient flow
- Zero input handling
- Dropout behavior in train/eval modes

### 4. RMSNorm (TestRMSNorm)
- Initialization with various dimensions
- Forward pass shape preservation
- Normalization correctness (RMS = 1 verification)
- Learnable weight parameters
- Gradient flow through normalization
- Numerical stability with extreme values
- Scale invariance property

### 5. ParallelBlock (TestParallelBlock)
- Component initialization (norm, attention, FFN)
- Forward pass shape preservation
- Residual connection implementation
- Causal vs non-causal modes
- Gradient flow through all components

### 6. DeepNormParallelBlock (TestDeepNormParallelBlock)
- Residual scaling factor computation
- Scaling effect with different layer counts
- Forward pass shape preservation
- Gradient flow verification

### 7. NonCausalParallelBlock (TestNonCausalParallelBlock)
- Non-causal attention enforcement
- Forward pass shape preservation
- Comparison with causal variants

### 8. PositionWiseFeedForward (TestPositionWiseFeedForward)
- Dimension initialization
- Forward pass shape preservation
- ReLU activation verification
- Gradient flow

### 9. WaveTransformer Integration (TestWaveTransformer)
- Full model initialization
- Forward pass with encoder/decoder
- Return encoder outputs option
- Causal vs non-causal modes
- Gradient flow through entire model
- Multiple layer configurations

### 10. Edge Cases (TestEdgeCases)
- Empty/zero harmonics
- Single token sequences
- Large batch sizes
- Very long sequences
- Dtype preservation (float32, float64)
- Inf/NaN handling

### 11. Parameterized Configurations (TestParameterizedConfigurations)
- Various batch sizes, sequence lengths, model dimensions
- Different dropout rates
- Multiple harmonic counts
- Attention head configurations

## Test Design Principles

1. **Mathematical Validation**: Tests verify mathematical correctness with known inputs/outputs (e.g., pure tone synthesis)

2. **Shape Invariance**: All modules are tested to ensure output shapes match expected dimensions

3. **Gradient Flow**: Every learnable module includes gradient flow tests to ensure backpropagation works correctly

4. **Device Compatibility**: Tests verify CPU/GPU compatibility where applicable

5. **Numerical Stability**: Tests check behavior with extreme values and edge cases

6. **Mode Consistency**: Tests verify correct behavior in training vs evaluation modes

7. **Batch Independence**: Tests ensure batch elements are processed independently

## Known Limitations

- **Flash Attention**: Tests use `use_flash=False` to test PyTorch fallback implementation, as flash-attn may require specific hardware/installation
- **GPU Tests**: GPU-specific tests are skipped if CUDA is not available
- **Plotting Tests**: Plotting methods are not extensively tested as they are primarily for visualization

## Adding New Tests

When adding new components to the transformer module, follow these guidelines:

1. Create a test class named `Test{ComponentName}`
2. Include tests for:
   - Initialization
   - Forward pass shape
   - Gradient flow
   - Edge cases
   - Any component-specific properties
3. Use fixtures for common test data
4. Add parametrized tests for multiple configurations
5. Include docstrings explaining what each test validates

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Recommended CI configuration:

```yaml
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    pytest tests/ --cov=src/wave_transformer --cov-report=xml
```

## Performance Notes

- Full test suite runs in approximately 30-60 seconds on CPU
- Use `-n auto` for parallel execution to reduce runtime
- GPU tests may be slower due to device initialization overhead
- Long sequence tests may require significant memory
