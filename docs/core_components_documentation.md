# Wave Transformer Core Components - Technical Documentation

## Overview

The `src/wave_transformer/core/` module implements a novel transformer architecture that operates directly on wave representations instead of traditional embeddings. This approach represents sequences as superpositions of sinusoidal waves characterized by frequencies, amplitudes, and phases. The core module provides:

1. **Wave Data Structure**: A representation of signals as frequency-domain components
2. **Flash Attention**: Efficient attention mechanism with Flash Attention support
3. **Modern Transformer Components**: SwiGLU activation, RMSNorm, and parallel block architectures
4. **Wave Transformer**: Complete transformer model operating on wave representations

---

## File Structure

```
src/wave_transformer/core/
├── __init__.py          # Empty module initializer
└── transformer.py       # Core transformer components and Wave dataclass
```

---

## Core Components

### 1. Wave (Dataclass)

**Location**: `E:\WaveML\Wave-Transformer\src\wave_transformer\core\transformer.py` (lines 16-148)

#### Purpose
A dataclass representing signals as superpositions of sinusoidal waves in the frequency domain. Each wave is decomposed into multiple harmonic components, each characterized by frequency, amplitude, and phase.

#### Mathematical Foundation
A wave is represented as a sum of sinusoids:
```
signal(t) = Σ A_i * sin(2π * f_i * t + φ_i)
```
where:
- `A_i` = amplitude of component i
- `f_i` = frequency of component i (Hz)
- `φ_i` = phase of component i (radians)
- `t` = time points

#### Attributes

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `frequencies` | `torch.Tensor` | `(..., num_harmonics)` | Frequency values for each harmonic component in Hz |
| `amplitudes` | `torch.Tensor` | `(..., num_harmonics)` | Amplitude coefficients for each component |
| `phases` | `torch.Tensor` | `(..., num_harmonics)` | Phase offsets for each component in radians |

**Shape Convention**: All three tensors must have the same shape. The last dimension represents the number of harmonic components. Preceding dimensions can represent batch, sequence length, etc.

#### Methods

##### `to_representation() -> torch.Tensor`
Converts wave components into a flat tensor representation by concatenating frequencies, amplitudes, and phases.

**Returns**:
- `torch.Tensor`: Shape `(..., num_harmonics * 3)` - concatenated representation

**Implementation Details**:
- Concatenates along the last dimension using `torch.cat`
- Order: [frequencies, amplitudes, phases]
- Differentiable operation for gradient flow

**Usage**:
```python
wave = Wave(
    frequencies=torch.randn(batch_size, seq_len, 64),
    amplitudes=torch.randn(batch_size, seq_len, 64),
    phases=torch.randn(batch_size, seq_len, 64)
)
representation = wave.to_representation()  # Shape: (batch_size, seq_len, 192)
```

##### `from_representation(x: torch.Tensor) -> Wave` (classmethod)
Reconstructs a Wave object from its flat tensor representation.

**Parameters**:
- `x` (`torch.Tensor`): Shape `(..., num_harmonics * 3)` - flat representation

**Returns**:
- `Wave`: Reconstructed wave object

**Implementation Details**:
- Uses `torch.Tensor.chunk(3, dim=-1)` to split into three equal parts
- Assumes input dimension is divisible by 3
- Inverse operation of `to_representation()`

**Usage**:
```python
representation = torch.randn(batch_size, seq_len, 192)
wave = Wave.from_representation(representation)
# wave.frequencies.shape: (batch_size, seq_len, 64)
```

##### `synthesize(t: torch.Tensor) -> torch.Tensor`
Generates time-domain signal at specified time points using the wave's frequency components.

**Parameters**:
- `t` (`torch.Tensor`): Shape `(num_time_points,)` - time values in seconds

**Returns**:
- `torch.Tensor`: Shape `(..., num_time_points)` - synthesized signal

**Mathematical Operation**:
```python
signal = Σ amplitudes * sin(2π * frequencies * t + phases)
```

**Implementation Details**:
- Broadcasting: `t` is unshaped to `(num_time_points, 1)` for proper broadcasting
- Each harmonic is computed independently and summed
- Fully differentiable for gradient-based optimization

**Usage**:
```python
t = torch.linspace(0, 1.0, 1000)  # 1 second at 1kHz
signal = wave.synthesize(t)  # Shape: (..., 1000)
```

##### Visualization Methods

The Wave class provides comprehensive visualization capabilities for signal analysis:

**`plot_waveform(duration: float = 1.0, sample_rate: int = 1000, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes`**
- Plots time-domain waveform
- Synthesizes signal over specified duration
- Returns matplotlib Axes object for further customization

**`plot_spectrum(ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes`**
- Stem plot of amplitude vs frequency
- Visualizes frequency domain representation directly

**`plot_phase_spectrum(ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes`**
- Stem plot of phase vs frequency
- Shows phase relationships between components

**`plot_spectrogram(duration: float = 1.0, sample_rate: int = 1000, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes`**
- Time-frequency representation using STFT
- Shows how frequency content evolves over time

**`plot_components(duration: float = 1.0, sample_rate: int = 1000, ax: Optional[plt.Axes] = None) -> plt.Axes`**
- Plots individual harmonic components separately
- Useful for understanding wave composition

**`plot_summary(duration: float = 1.0, sample_rate: int = 1000) -> Tuple[plt.Figure, np.ndarray]`**
- Creates 2×2 grid with waveform, spectrum, phase spectrum, and components
- Comprehensive visualization of all wave properties

**Usage Example**:
```python
wave = Wave(
    frequencies=torch.tensor([[5.0, 10.0, 15.0]]),
    amplitudes=torch.tensor([[1.0, 0.5, 0.3]]),
    phases=torch.tensor([[0.0, np.pi/4, np.pi/2]])
)

# Single plot
wave.plot_waveform(duration=2.0, sample_rate=2000)
plt.show()

# Comprehensive summary
fig, axes = wave.plot_summary(duration=1.0, sample_rate=1000)
plt.savefig('wave_analysis.png')
```

#### Helper Function: `plot_wave_series`

**Location**: Lines 150-208

Compares multiple Wave objects across different visualization modes.

**Parameters**:
- `waves` (`List[Wave]`): List of wave objects to compare
- `duration` (`float`): Duration in seconds for time-domain plots
- `sample_rate` (`int`): Sampling rate for synthesis
- `labels` (`Optional[List[str]]`): Custom labels for each wave

**Returns**:
- `Tuple[plt.Figure, np.ndarray]`: Figure and axes array

**Usage**:
```python
waves = [wave1, wave2, wave3]
labels = ['Original', 'Transformed', 'Decoded']
fig, axes = plot_wave_series(waves, duration=1.0, labels=labels)
```

---

### 2. FlashAttention (nn.Module)

**Location**: Lines 211-281

#### Purpose
Efficient multi-head self-attention module with Flash Attention support for memory-efficient computation. Provides both optimized Flash Attention and PyTorch fallback implementations.

#### Architecture Details

Flash Attention is a memory-efficient attention algorithm that:
- Reduces memory usage from O(N²) to O(N) for sequence length N
- Achieves 2-4× speedup on modern GPUs
- Provides exact attention computation (not an approximation)
- Supports causal masking for autoregressive models

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | Required | Model dimension (must be divisible by `n_heads`) |
| `n_heads` | `int` | Required | Number of attention heads |
| `dropout` | `float` | `0.1` | Dropout probability applied to attention weights |
| `use_flash` | `bool` | `True` | Whether to use Flash Attention (requires CUDA and flash-attn package) |

#### Attributes

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `d_head` | `int` | Scalar | Dimension per head (`d_model // n_heads`) |
| `scale` | `float` | Scalar | Attention scaling factor `1/√d_head` |
| `qkv` | `nn.Linear` | `(d_model, 3*d_model)` | Joint QKV projection without bias |
| `out_proj` | `nn.Linear` | `(d_model, d_model)` | Output projection with bias |

#### Forward Pass

**Signature**: `forward(x: torch.Tensor, causal: bool = True) -> torch.Tensor`

**Parameters**:
- `x` (`torch.Tensor`): Shape `(batch_size, seq_len, d_model)` - input embeddings
- `causal` (`bool`): Whether to apply causal masking (for autoregressive generation)

**Returns**:
- `torch.Tensor`: Shape `(batch_size, seq_len, d_model)` - attention output

**Computational Flow**:

1. **QKV Projection**:
   ```python
   qkv = self.qkv(x)  # Shape: (B, N, 3*d_model)
   qkv = qkv.reshape(B, N, 3, n_heads, d_head)
   ```

2. **Attention Computation** (two paths):
   - **Flash Attention Path** (`_flash_attention`):
     - Converts to FP16 for efficiency
     - Uses optimized Flash Attention kernel
     - Applies causal masking within kernel

   - **PyTorch Path** (`_pytorch_attention`):
     - Standard scaled dot-product attention
     - Manual causal masking using upper triangular mask
     - Higher memory usage but more compatible

3. **Output Projection**:
   ```python
   out = self.out_proj(out)
   ```

#### Implementation Details

**Flash Attention Path**:
- Requires CUDA-compatible GPU
- Input is converted to FP16 for kernel computation
- Output is converted back to original dtype
- Scaling factor applied within Flash Attention kernel
- Causal mask handled internally (more efficient)

**PyTorch Fallback Path**:
- Standard attention: `softmax(QK^T / √d_head) V`
- Causal mask applied manually using `torch.triu`
- Masked positions set to `-inf` before softmax
- Compatible with CPU and all GPU architectures

**Memory Complexity**:
- Flash Attention: O(N) memory
- PyTorch Attention: O(N²) memory due to attention matrix

**Usage Example**:
```python
# Initialize attention module
attn = FlashAttention(d_model=512, n_heads=8, dropout=0.1, use_flash=True)

# Forward pass
x = torch.randn(32, 128, 512)  # (batch, seq_len, d_model)
output = attn(x, causal=True)  # Shape: (32, 128, 512)

# Non-causal attention (for encoder)
output = attn(x, causal=False)
```

#### Device Considerations
- Flash Attention requires CUDA GPU with compute capability ≥ 7.0
- Automatically falls back to PyTorch implementation if Flash Attention fails
- Input tensors should be on CUDA device for Flash Attention path

---

### 3. SwiGLU (nn.Module)

**Location**: Lines 284-299

#### Purpose
Gated linear unit activation using the SiLU (Swish) activation function. Introduced in PaLM and used in LLaMA, SwiGLU consistently outperforms standard FFN with ReLU/GELU activations.

#### Mathematical Foundation
```
SwiGLU(x) = W2(SiLU(W1(x)) ⊙ W3(x))
```
where:
- `W1, W2, W3` are linear projections
- `SiLU(x) = x * σ(x)` where σ is sigmoid
- `⊙` denotes element-wise multiplication (gating)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | Required | Input/output dimension |
| `d_ff` | `int` | Required | Hidden (feedforward) dimension |
| `dropout` | `float` | `0.0` | Dropout probability applied after gating |

#### Architecture Details

**Projection Structure**:
- `w1`: `(d_model, d_ff)` - gate projection
- `w2`: `(d_ff, d_model)` - down projection
- `w3`: `(d_model, d_ff)` - value projection
- All projections have `bias=False` (following LLaMA design)

**Why Three Projections?**
- `w1`: Produces gating signal
- `w3`: Produces values to be gated
- `w2`: Projects back to model dimension
- Gating allows network to dynamically control information flow

#### Forward Pass

**Signature**: `forward(x: torch.Tensor) -> torch.Tensor`

**Parameters**:
- `x` (`torch.Tensor`): Shape `(..., d_model)` - input tensor

**Returns**:
- `torch.Tensor`: Shape `(..., d_model)` - output tensor

**Computational Steps**:
1. Gate: `g = SiLU(w1(x))`
2. Value: `v = w3(x)`
3. Gated value: `gv = dropout(g * v)`
4. Output: `y = w2(gv)`

#### Performance Characteristics

**Advantages over ReLU/GELU FFN**:
- Better gradient flow (no dead neurons like ReLU)
- Gating provides selectivity similar to attention
- Empirically superior across many benchmarks

**Computational Cost**:
- Parameters: `3 * d_model * d_ff` vs `2 * d_model * d_ff` for standard FFN
- FLOPs: ~1.5× standard FFN
- Worth the cost for improved performance

**Usage Example**:
```python
# Initialize SwiGLU (typically d_ff = 4 * d_model, but can vary)
ffn = SwiGLU(d_model=768, d_ff=3072, dropout=0.1)

# Forward pass
x = torch.randn(32, 128, 768)  # (batch, seq, d_model)
output = ffn(x)  # Shape: (32, 128, 768)
```

---

### 4. RMSNorm (nn.Module)

**Location**: Lines 301-314

#### Purpose
Root Mean Square Layer Normalization - a simplified normalization scheme used in T5, LLaMA, and other modern architectures. Normalizes using RMS statistic instead of mean and variance.

#### Mathematical Foundation
```
RMSNorm(x) = (x / RMS(x)) * γ

where RMS(x) = √(mean(x²) + ε)
```

Key differences from LayerNorm:
- No mean centering (no subtraction of mean)
- No learned bias parameter
- Only scale parameter γ (weight)

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | Required | Feature dimension to normalize |
| `eps` | `float` | `1e-6` | Small constant for numerical stability |

#### Attributes

| Attribute | Type | Shape | Description |
|-----------|------|------|-------------|
| `weight` | `nn.Parameter` | `(d_model,)` | Learned scale parameter γ, initialized to ones |

#### Forward Pass

**Signature**: `forward(x: torch.Tensor) -> torch.Tensor`

**Parameters**:
- `x` (`torch.Tensor`): Shape `(..., d_model)` - input tensor

**Returns**:
- `torch.Tensor`: Shape `(..., d_model)` - normalized tensor

**Computational Steps**:
1. Compute RMS: `rms = sqrt(mean(x², dim=-1, keepdim=True) + eps)`
2. Normalize: `x_norm = x / rms`
3. Scale: `output = weight * x_norm`

#### Advantages over LayerNorm

**Computational Efficiency**:
- ~10-15% faster than LayerNorm
- Fewer operations (no mean computation, no bias)
- Simpler gradient computation

**Performance**:
- Often matches or exceeds LayerNorm accuracy
- More stable training in some cases
- Better suited for very deep networks

**Memory**:
- Half the learned parameters (no bias)
- Slightly lower memory footprint

#### Implementation Details

**Normalization is applied over last dimension only**:
- Statistics computed per feature vector
- Independent normalization for each position in sequence
- Preserves batch and sequence dimensions

**Numerical Stability**:
- `eps` prevents division by zero
- Typically set to `1e-6` or `1e-5`

**Usage Example**:
```python
# Initialize RMSNorm
norm = RMSNorm(d_model=768, eps=1e-6)

# Forward pass
x = torch.randn(32, 128, 768)  # (batch, seq, d_model)
normalized = norm(x)  # Shape: (32, 128, 768)

# RMS is approximately 1.0 after normalization
print(torch.mean(normalized ** 2, dim=-1).sqrt())  # ~1.0
```

---

### 5. ParallelBlock (nn.Module)

**Location**: Lines 317-337

#### Purpose
Parallel attention and feedforward block inspired by GPT-J and PaLM. Computes attention and FFN in parallel rather than sequentially, reducing latency while maintaining model capacity.

#### Architecture Details

**Traditional Sequential Block**:
```python
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

**Parallel Block**:
```python
normed = Norm(x)
x = x + Attention(normed) + FFN(normed)
```

**Benefits**:
- Reduced latency (single normalization, parallel compute)
- Better hardware utilization (parallel operations)
- Often matches sequential performance with proper tuning

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | Required | Model dimension |
| `n_heads` | `int` | Required | Number of attention heads |
| `n_heads_kv` | `int` | Required | Number of key-value heads (currently unused, for future GQA support) |
| `d_ff` | `int` | Required | Feedforward hidden dimension |
| `dropout` | `float` | `0.0` | Dropout probability |
| `use_flash` | `bool` | `True` | Whether to use Flash Attention |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `norm` | `RMSNorm` | Pre-normalization layer |
| `attn` | `FlashAttention` | Multi-head attention |
| `ffn` | `SwiGLU` | Gated feedforward network |
| `dropout` | `nn.Dropout` | Dropout applied to combined output |

#### Forward Pass

**Signature**: `forward(x: torch.Tensor, causal: bool = True, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor`

**Parameters**:
- `x` (`torch.Tensor`): Shape `(batch_size, seq_len, d_model)` - input
- `causal` (`bool`): Whether to use causal masking
- `attention_mask` (`Optional[torch.Tensor]`): Currently unused (for future extension)

**Returns**:
- `torch.Tensor`: Shape `(batch_size, seq_len, d_model)` - output

**Computational Flow**:
```python
normed = self.norm(x)
attn_out = self.attn(normed, causal)
ffn_out = self.ffn(normed)
return x + self.dropout(attn_out + ffn_out)
```

#### Implementation Details

**Pre-Normalization**:
- Uses "Pre-LN" architecture (normalize before sublayers)
- More stable than "Post-LN" for deep networks
- Single normalization shared by both paths

**Residual Connection**:
- Single residual wraps both attention and FFN
- Dropout applied to combined sublayer outputs
- Helps gradient flow in deep networks

**Parallel Computation**:
- Attention and FFN can be computed simultaneously
- Implementation is sequential but could be parallelized
- Reduces depth of computational graph

**Usage Example**:
```python
# Initialize parallel block
block = ParallelBlock(
    d_model=768,
    n_heads=12,
    n_heads_kv=12,  # Not used yet
    d_ff=3072,
    dropout=0.1,
    use_flash=True
)

# Forward pass
x = torch.randn(32, 128, 768)
output = block(x, causal=True)  # Shape: (32, 128, 768)
```

---

### 6. DeepNormParallelBlock (nn.Module)

**Location**: Lines 339-358

#### Purpose
Parallel block with DeepNorm residual scaling for training very deep transformers (100+ layers). DeepNorm provides better gradient flow and stability than standard normalization schemes.

#### Mathematical Foundation

**DeepNorm Residual Scaling**:
```
x_{l+1} = x_l + α * Sublayer(Norm(x_l))

where α = 1 / √(2N)
```
- `N` is the total number of layers
- Scaling prevents gradient explosion/vanishing
- Enables stable training of 1000+ layer models

#### Parameters

Same as `ParallelBlock`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_layers` | `int` | `1` | Total number of layers in the model (for scaling factor) |

#### Attributes

Additional attribute:

| Attribute | Type | Description |
|-----------|------|-------------|
| `residual_scale` | `float` | DeepNorm scaling factor `1/√(2*num_layers)` |

#### Forward Pass

**Signature**: `forward(x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, causal: bool = True) -> torch.Tensor`

**Returns**:
- Same as `ParallelBlock` but with scaled residuals

**Computational Flow**:
```python
normed = self.norm(x)
attn_out = self.attn(normed, causal=causal)
ffn_out = self.ffn(normed)
return x + self.dropout((attn_out + ffn_out) * self.residual_scale)
```

#### Key Difference from ParallelBlock

The residual connection is scaled by `residual_scale`:
- Smaller scaling for deeper networks
- Prevents gradient explosion
- Maintains stable training dynamics

**Scaling Examples**:
- 6 layers: scale = 1/√12 ≈ 0.289
- 24 layers: scale = 1/√48 ≈ 0.144
- 100 layers: scale = 1/√200 ≈ 0.071

**Usage Example**:
```python
# For a 24-layer model
block = DeepNormParallelBlock(
    d_model=1024,
    n_heads=16,
    n_heads_kv=16,
    d_ff=4096,
    dropout=0.1,
    use_flash=True,
    num_layers=24  # Important: total layer count
)

x = torch.randn(16, 256, 1024)
output = block(x, causal=True)
```

#### When to Use
- Use `DeepNormParallelBlock` for models with >12 layers
- Use standard `ParallelBlock` for shallower models
- Critical for training 50+ layer transformers

---

### 7. PositionWiseFeedForward (nn.Module)

**Location**: Lines 360-370

#### Purpose
Standard two-layer feedforward network with ReLU activation. This is the classic transformer FFN implementation from "Attention Is All You Need".

#### Architecture
```
FFN(x) = W2(ReLU(W1(x)))
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | `int` | Required | Input/output dimension |
| `d_ff` | `int` | Required | Hidden dimension |
| `dropout` | `float` | `0.1` | Dropout after ReLU |

#### Forward Pass

**Signature**: `forward(x: torch.Tensor) -> torch.Tensor`

**Computational Steps**:
1. Linear expansion: `x = W1(x)` → shape `(..., d_ff)`
2. Activation: `x = ReLU(x)`
3. Dropout: `x = dropout(x)`
4. Linear projection: `x = W2(x)` → shape `(..., d_model)`

#### Comparison to SwiGLU

| Aspect | PositionWiseFeedForward | SwiGLU |
|--------|------------------------|--------|
| Activation | ReLU | SiLU + Gating |
| Parameters | `2 * d_model * d_ff` | `3 * d_model * d_ff` |
| Performance | Baseline | Superior |
| Speed | Faster | ~15% slower |
| Usage | Legacy/baseline | Modern architectures |

**Note**: This component is included for compatibility but `SwiGLU` is recommended for new implementations.

**Usage Example**:
```python
ffn = PositionWiseFeedForward(d_model=512, d_ff=2048, dropout=0.1)
x = torch.randn(32, 100, 512)
output = ffn(x)  # Shape: (32, 100, 512)
```

---

### 8. NonCausalParallelBlock (nn.Module)

**Location**: Lines 372-392

#### Purpose
Variant of `ParallelBlock` that always uses non-causal (bidirectional) attention. Intended for encoder architectures where full context is available.

#### Key Differences from ParallelBlock

1. **Attention is always non-causal**: `causal=False` hardcoded
2. **Different parameter signature**: requires `attention_mask` parameter
3. **Higher default dropout**: `0.1` vs `0.0`

#### Forward Pass

**Signature**: `forward(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor`

**Parameters**:
- `x` (`torch.Tensor`): Shape `(batch_size, seq_len, d_model)` - input
- `attention_mask` (`torch.Tensor`): Currently unused (for future custom masking)

**Implementation**:
```python
normalized = self.norm(x)
attn_out = self.attn(normalized, False)  # Causal=False
ffn_out = self.ffn(normalized)
return x + self.dropout(attn_out + ffn_out)
```

#### When to Use
- **Encoder models**: Where bidirectional context is needed
- **Discriminative tasks**: Classification, sequence tagging
- **Avoid for**: Autoregressive generation (use `ParallelBlock` instead)

**Usage Example**:
```python
# For encoder architecture
block = NonCausalParallelBlock(
    d_model=768,
    n_heads=12,
    n_heads_kv=12,
    d_ff=3072,
    dropout=0.1,
    use_flash=True
)

x = torch.randn(32, 128, 768)
# Attention mask not yet implemented
output = block(x, attention_mask=None)
```

---

### 9. WaveTransformer (nn.Module)

**Location**: Lines 394-435

#### Purpose
Complete transformer model that operates on wave representations. This is the main model class that integrates all components to transform input data through wave-based processing.

#### Architecture Overview

```
Input → WaveEncoder → Wave Representation → Transformer Layers → Final Norm → WaveDecoder → Output
```

**Novel Aspect**: Instead of traditional embeddings, sequences are represented as waves (superpositions of sinusoids). Each transformer layer operates on and transforms these wave representations.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wave_encoder` | `nn.Module` | Required | Encodes input to Wave representation (e.g., `AudioWaveEncoder`, `TokenEncoder`) |
| `wave_decoder` | `nn.Module` | Required | Decodes final wave representation to output (e.g., `AudioWaveDecoder`, `TokenDecoder`) |
| `num_harmonics` | `int` | `64` | Number of harmonic components per wave |
| `transformer_num_layers` | `int` | `6` | Number of transformer blocks |
| `transformer_num_heads` | `int` | `8` | Number of attention heads |
| `transformer_heads_kv` | `int` | `4` | Number of key-value heads (currently unused) |
| `transformer_d_ff_multi` | `int` | `4` | FFN dimension multiplier (`d_ff = input_dim * d_ff_multi`) |
| `dropout` | `float` | `0.1` | Dropout probability |
| `use_flash` | `bool` | `True` | Whether to use Flash Attention |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `num_harmonics` | `int` | Number of frequency components |
| `input_dim` | `int` | Transformer input dimension (`num_harmonics * 3`) |
| `wave_encoder` | `nn.Module` | Input encoder module |
| `layers` | `nn.ModuleList` | List of `ParallelBlock` layers |
| `norm_f` | `RMSNorm` | Final normalization layer |
| `wave_decoder` | `nn.Module` | Output decoder module |

#### Forward Pass

**Signature**:
```python
forward(
    encoder_input: dict[str, Any],
    causal: bool = True,
    return_encoder_outputs: bool = False,
    attention_mask: Optional[torch.Tensor] = None,
    plot_waves: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Wave]]
```

**Parameters**:
- `encoder_input` (`dict[str, Any]`): Dictionary of inputs for wave encoder (encoder-specific)
- `causal` (`bool`): Whether to use causal attention
- `return_encoder_outputs` (`bool`): If True, returns (output, encoder_wave) tuple
- `attention_mask` (`Optional[torch.Tensor]`): Attention mask (passed to encoder/decoder)
- `plot_waves` (`bool`): If True, saves wave visualizations at each layer (for debugging)

**Returns**:
- `torch.Tensor`: Model output (decoder-specific shape)
- OR `Tuple[torch.Tensor, Wave]`: If `return_encoder_outputs=True`

#### Computational Flow

1. **Encoding**:
   ```python
   wave = self.wave_encoder(attention_mask=attention_mask, **encoder_input)
   # wave is a Wave object with frequencies, amplitudes, phases
   ```

2. **Wave to Tensor**:
   ```python
   x = wave.to_representation()
   # Shape: (batch, seq_len, num_harmonics * 3)
   ```

3. **Transformer Layers**:
   ```python
   for block in self.layers:
       x = block(x, causal=causal, attention_mask=attention_mask)
   # Each block transforms the wave representation
   ```

4. **Final Normalization**:
   ```python
   x = self.norm_f(x)
   ```

5. **Decoding**:
   ```python
   output = self.wave_decoder(x, attention_mask=attention_mask)
   ```

#### Wave Representation Details

**Why Wave Representations?**
- Frequency domain is natural for many signals (audio, time series)
- Allows direct manipulation of spectral properties
- Potentially more interpretable than abstract embeddings
- Each transformer layer modifies frequency/amplitude/phase

**Tensor Shape Evolution**:
```
Input → Encoder → Wave(freq, amp, phase) each (B, N, H)
                → Concatenated: (B, N, 3H)
                → Through layers: (B, N, 3H)
                → Decoder → Output
```
where B=batch, N=seq_len, H=num_harmonics

#### Visualization Support

When `plot_waves=True`:
- Saves wave visualization before first layer: `wave_layer_0_input.png`
- Saves visualization after each layer: `wave_layer_1.png`, `wave_layer_2.png`, etc.
- Each plot shows waveform, spectrum, phase, and components
- Useful for understanding how transformer modifies wave properties

#### Integration with Encoders/Decoders

**Audio Domain**:
```python
from wave_transformer.audio import AudioWaveEncoder, AudioWaveDecoder

model = WaveTransformer(
    wave_encoder=AudioWaveEncoder(num_harmonics=64),
    wave_decoder=AudioWaveDecoder(num_harmonics=64),
    num_harmonics=64,
    transformer_num_layers=12
)

# Input: raw audio
encoder_input = {'audio': audio_tensor}
output_audio = model(encoder_input)
```

**Language Modeling**:
```python
from wave_transformer.language_modelling import TokenEncoder, TokenDecoder

model = WaveTransformer(
    wave_encoder=TokenEncoder(vocab_size=50000, num_harmonics=64),
    wave_decoder=TokenDecoder(vocab_size=50000, num_harmonics=64),
    num_harmonics=64,
    transformer_num_layers=6
)

# Input: token IDs
encoder_input = {'input_ids': token_ids}
logits = model(encoder_input, causal=True)
```

#### Training Considerations

**Memory Usage**:
- Wave representation adds overhead: `3 * num_harmonics` dimensions
- Flash Attention reduces memory for long sequences
- Typical `num_harmonics` values: 32-128

**Gradient Flow**:
- Wave operations are differentiable
- `synthesize()` method computes gradients w.r.t. frequencies/amplitudes/phases
- Pre-normalization architecture ensures stable gradients

**Computational Cost**:
- Dominated by attention (O(N²) for standard, O(N) for Flash)
- Wave operations are cheap (concatenation, chunking)
- FFN scales with `d_ff_multi` parameter

**Device Placement**:
- Model should be on CUDA for Flash Attention
- Automatically falls back to PyTorch attention on CPU

#### Usage Example

**Complete Training Setup**:
```python
import torch
from wave_transformer.core.transformer import WaveTransformer, Wave
from wave_transformer.language_modelling import TokenEncoder, TokenDecoder

# Model configuration
model = WaveTransformer(
    wave_encoder=TokenEncoder(vocab_size=10000, num_harmonics=64),
    wave_decoder=TokenDecoder(vocab_size=10000, num_harmonics=64),
    num_harmonics=64,
    transformer_num_layers=6,
    transformer_num_heads=8,
    transformer_heads_kv=8,
    transformer_d_ff_multi=4,
    dropout=0.1,
    use_flash=True
).cuda()

# Forward pass
batch_size, seq_len = 32, 128
input_ids = torch.randint(0, 10000, (batch_size, seq_len)).cuda()

encoder_input = {'input_ids': input_ids}
logits = model(encoder_input, causal=True)  # Shape: (32, 128, 10000)

# Loss computation
targets = torch.randint(0, 10000, (batch_size, seq_len)).cuda()
loss = F.cross_entropy(
    logits.reshape(-1, 10000),
    targets.reshape(-1)
)
loss.backward()

# Visualize intermediate waves (debugging)
with torch.no_grad():
    model(encoder_input, plot_waves=True)
    # Creates wave_layer_0_input.png, wave_layer_1.png, ..., wave_layer_6.png
```

**Inference with Encoder Outputs**:
```python
# Get both output and encoder wave
output, encoder_wave = model(
    encoder_input,
    causal=False,
    return_encoder_outputs=True
)

# Analyze encoder wave
encoder_wave.plot_summary()
plt.savefig('encoder_wave_analysis.png')

# Synthesize time-domain signal
t = torch.linspace(0, 1.0, 1000)
signal = encoder_wave.synthesize(t)
```

---

## Performance Optimization Guide

### Flash Attention Configuration

**Requirements**:
- CUDA GPU with compute capability ≥ 7.0
- `flash-attn` package installed
- FP16 or BF16 training for maximum benefit

**Performance Gains**:
- 2-4× speedup for long sequences (>512 tokens)
- 50-80% memory reduction
- Enables training with 2-4× longer sequences

**Fallback Behavior**:
```python
# Model automatically uses PyTorch attention if Flash Attention unavailable
model = WaveTransformer(..., use_flash=True)  # Will fallback if needed
```

### Memory Optimization

**Gradient Checkpointing** (for very deep models):
```python
# Apply to transformer layers
for layer in model.layers:
    layer = torch.utils.checkpoint.checkpoint_wrapper(layer)
```

**Mixed Precision Training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(encoder_input)
    loss = compute_loss(output, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Hyperparameter Tuning

**Model Scaling**:
- Small: `num_harmonics=32, layers=6, heads=8`
- Medium: `num_harmonics=64, layers=12, heads=12`
- Large: `num_harmonics=128, layers=24, heads=16`

**FFN Multiplier**:
- Standard: `d_ff_multi=4` (768 → 3072)
- Larger: `d_ff_multi=8` (more capacity, slower)
- Smaller: `d_ff_multi=2` (faster, less capacity)

---

## Common Issues and Solutions

### Issue 1: Out of Memory
**Symptoms**: CUDA out of memory error during training

**Solutions**:
1. Enable Flash Attention: `use_flash=True`
2. Reduce batch size
3. Reduce sequence length
4. Use gradient checkpointing
5. Reduce `num_harmonics` (64 → 32)

### Issue 2: Flash Attention Import Error
**Symptoms**: `ImportError: cannot import name 'flash_attn_func'`

**Solutions**:
1. Install flash-attn: `pip install flash-attn --no-build-isolation`
2. Model will automatically fallback to PyTorch attention
3. Or set `use_flash=False` explicitly

### Issue 3: Slow Training
**Symptoms**: Training slower than expected

**Solutions**:
1. Verify Flash Attention is active (check logs)
2. Use mixed precision training (AMP)
3. Ensure data loading isn't bottleneck
4. Profile with PyTorch profiler

### Issue 4: NaN Loss
**Symptoms**: Loss becomes NaN during training

**Solutions**:
1. Reduce learning rate
2. Use gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. Check for exploding frequencies/amplitudes in Wave
4. Use `DeepNormParallelBlock` for deep models
5. Verify input normalization

---

## Design Rationale

### Why Wave Representations?

1. **Domain Appropriateness**: Natural for signals with periodic structure (audio, time series)
2. **Interpretability**: Frequencies, amplitudes, and phases have clear physical meaning
3. **Manipulability**: Transformer can directly modify spectral properties
4. **Efficiency**: Compact representation for band-limited signals

### Why Parallel Blocks?

1. **Latency**: Single forward pass faster than sequential
2. **Hardware Utilization**: Better GPU utilization
3. **Performance**: Matches sequential with proper tuning
4. **Simplicity**: One normalization instead of two

### Why Flash Attention?

1. **Memory**: O(N) vs O(N²) - critical for long sequences
2. **Speed**: 2-4× faster through kernel optimization
3. **Exactness**: Not an approximation like some sparse attention methods
4. **Adoption**: Industry standard in modern LLMs

### Why SwiGLU over ReLU?

1. **Performance**: Consistently better across benchmarks
2. **Gradient Flow**: No dead neurons (unlike ReLU)
3. **Selectivity**: Gating provides learned information filtering
4. **Modern Standard**: Used in PaLM, LLaMA, Chinchilla

### Why RMSNorm over LayerNorm?

1. **Simplicity**: Fewer operations and parameters
2. **Speed**: 10-15% faster
3. **Performance**: Often equal or better than LayerNorm
4. **Stability**: Good gradient flow properties

---

## File Path Reference

All components documented here are located in:
```
E:\WaveML\Wave-Transformer\src\wave_transformer\core\transformer.py
```

Module structure:
- Lines 16-208: Wave dataclass and plotting utilities
- Lines 211-281: FlashAttention
- Lines 284-299: SwiGLU
- Lines 301-314: RMSNorm
- Lines 317-337: ParallelBlock
- Lines 339-358: DeepNormParallelBlock
- Lines 360-370: PositionWiseFeedForward
- Lines 372-392: NonCausalParallelBlock
- Lines 394-435: WaveTransformer

---

## Dependencies

**Required Packages**:
```
torch >= 2.0.0
numpy
matplotlib (for visualization only)
flash-attn (optional, for Flash Attention)
```

**Import Structure**:
```python
from wave_transformer.core.transformer import (
    Wave,
    FlashAttention,
    SwiGLU,
    RMSNorm,
    ParallelBlock,
    DeepNormParallelBlock,
    WaveTransformer,
    plot_wave_series
)
```

---

## Testing and Validation

**Basic Functionality Test**:
```python
import torch
from wave_transformer.core.transformer import (
    Wave, WaveTransformer, FlashAttention, SwiGLU, RMSNorm, ParallelBlock
)

# Test Wave dataclass
wave = Wave(
    frequencies=torch.randn(2, 10, 32),
    amplitudes=torch.randn(2, 10, 32),
    phases=torch.randn(2, 10, 32)
)
repr = wave.to_representation()
reconstructed = Wave.from_representation(repr)
assert torch.allclose(wave.frequencies, reconstructed.frequencies)

# Test FlashAttention
attn = FlashAttention(d_model=128, n_heads=4, use_flash=False)
x = torch.randn(2, 50, 128)
out = attn(x, causal=True)
assert out.shape == x.shape

# Test SwiGLU
ffn = SwiGLU(d_model=128, d_ff=512)
out = ffn(x)
assert out.shape == x.shape

# Test RMSNorm
norm = RMSNorm(d_model=128)
out = norm(x)
assert out.shape == x.shape
rms = torch.sqrt(torch.mean(out ** 2, dim=-1))
assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

# Test ParallelBlock
block = ParallelBlock(d_model=128, n_heads=4, n_heads_kv=4, d_ff=512, use_flash=False)
out = block(x, causal=True)
assert out.shape == x.shape

print("All tests passed!")
```

---

This documentation covers all PyTorch components in `src/wave_transformer/core/`. The module implements a novel wave-based transformer architecture with modern optimizations (Flash Attention, SwiGLU, RMSNorm, parallel blocks) for efficient and effective sequence modeling.
