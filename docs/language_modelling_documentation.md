# Wave Transformer Language Modelling - PyTorch Components Documentation

## Table of Contents
1. [Overview](#overview)
2. [Embedding Layers](#embedding-layers)
3. [Token Encoders](#token-encoders)
4. [Token Decoders](#token-decoders)
5. [Training Utilities](#training-utilities)
6. [Dataset Classes](#dataset-classes)

---

## Overview

The `wave_transformer.language_modelling` module provides PyTorch components for wave-based language modeling, including specialized embeddings, token-to-wave encoders, wave-to-token decoders, and comprehensive dataset utilities. The architecture converts discrete tokens into continuous wave representations (frequencies, amplitudes, phases) for processing.

**Key Design Principles:**
- Wave-based representations for language modeling
- Modular embedding strategies (sinusoidal, rotary, hash-based, character-level)
- Efficient streaming and prepared dataset implementations
- Comprehensive training utilities with advanced sampling strategies

---

## Embedding Layers

### **File:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\embeddings.py`

### 1. SinusoidalPositionEmbedding

**Purpose:** Provides fixed sinusoidal positional encodings for sequence position awareness, as introduced in "Attention Is All You Need".

**Architecture:**
- Precomputes sine/cosine position encodings up to `max_len`
- Even dimensions use sine, odd dimensions use cosine
- Stored as buffer (non-trainable parameters)

**Parameters:**
- `d_model` (int): Model embedding dimension
- `max_len` (int, default=5000): Maximum sequence length to precompute

**Attributes:**
- `pe` (torch.Tensor): Precomputed positional encodings of shape `(1, max_len, d_model)`

**Input/Output:**
- **Input:** `x` - torch.Tensor of shape `(batch_size, seq_len, d_model)`
- **Output:** torch.Tensor of shape `(batch_size, seq_len, d_model)` (input + positional encoding)

**Mathematical Foundation:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Usage Example:**

```python
import torch
from wave_transformer.core.embeddings import SinusoidalPositionEmbedding

pos_embed = SinusoidalPositionEmbedding(d_model=256, max_len=512)
x = torch.randn(32, 128, 256)  # (batch, seq_len, d_model)
x_with_pos = pos_embed(x)  # (32, 128, 256)
```

**Implementation Notes:**
- Positional encodings are added to input, not concatenated
- Supports sequences up to `max_len` (truncates longer sequences)
- No gradient computation for position encodings

---

### 2. RotaryPositionEmbedding

**Purpose:** Implements Rotary Position Embeddings (RoPE) that encode position information through rotation in feature space, preserving relative position information.

**Architecture:**
- Precomputes inverse frequencies and rotation matrices
- Splits features into pairs and applies 2D rotation
- Natural periodicity in position encoding

**Parameters:**
- `d_model` (int): Model embedding dimension (must be even)
- `max_len` (int, default=5000): Maximum sequence length

**Attributes:**
- `inv_freq` (torch.Tensor): Inverse frequencies of shape `(d_model//2,)`
- `cos` (torch.Tensor): Precomputed cosine values `(max_len, d_model//2)`
- `sin` (torch.Tensor): Precomputed sine values `(max_len, d_model//2)`

**Input/Output:**
- **Input:** `x` - torch.Tensor of shape `(batch_size, seq_len, d_model)`
- **Output:** Rotated tensor of shape `(batch_size, seq_len, d_model)`

**Mathematical Foundation:**
```
freq_i = 1 / (10000^(2i/d_model))
x_rotated = [x1*cos - x2*sin, x1*sin + x2*cos]
```

**Usage Example:**

```python
from wave_transformer.core.embeddings import RotaryPositionEmbedding

rope = RotaryPositionEmbedding(d_model=256, max_len=512)
x = torch.randn(32, 128, 256)
x_rotated = rope(x)  # (32, 128, 256)
```

**Implementation Notes:**
- Requires even `d_model` (features split into pairs)
- More effective at capturing relative positions than absolute
- Used in modern architectures like PaLM, LLaMA

---

### 3. HashEmbedding

**Purpose:** Memory-efficient embedding using multiple hash functions to reduce embedding table size while maintaining representation quality.

**Architecture:**
- Multiple smaller embedding tables instead of one large table
- Uses modulo-based hash functions
- Averages embeddings from all hash tables

**Parameters:**
- `num_embeddings` (int): Original vocabulary size
- `embedding_dim` (int): Embedding dimension
- `num_hashes` (int, default=2): Number of hash functions/tables

**Attributes:**
- `embeddings` (nn.ModuleList): List of `num_hashes` embedding tables, each of size `(num_embeddings // num_hashes, embedding_dim)`

**Input/Output:**
- **Input:** `input_ids` - torch.LongTensor of shape `(batch_size, seq_len)`
- **Output:** torch.Tensor of shape `(batch_size, seq_len, embedding_dim)`

**Memory Savings:**
- Standard embedding: `num_embeddings * embedding_dim` parameters
- Hash embedding: `(num_embeddings / num_hashes) * embedding_dim * num_hashes` parameters (same total with better distribution)

**Usage Example:**

```python
from wave_transformer.core.embeddings import HashEmbedding

vocab_size = 50000
embed_dim = 256
hash_embed = HashEmbedding(vocab_size, embed_dim, num_hashes=4)

input_ids = torch.randint(0, vocab_size, (32, 128))
embeddings = hash_embed(input_ids)  # (32, 128, 256)
```

**Implementation Notes:**
- Simple modulo hash: `hash_i(id) = id % table_size_i`
- Trade-off: reduced parameters vs. potential hash collisions
- Effective for large vocabularies with zipfian distributions

---

### 4. CharCNNEmbedding

**Purpose:** Character-level embeddings using CNN filters to capture morphological patterns and sub-word information.

**Architecture:**
- Character embedding layer
- Multiple 1D convolutional filters with different kernel sizes
- Max-pooling over character dimension
- Concatenation of filter outputs

**Parameters:**
- `vocab_size` (int): Character vocabulary size
- `char_embed_dim` (int, default=16): Character embedding dimension
- `filters` (List[int], default=[32, 64, 128]): Number of filters for each convolution
- `kernel_sizes` (List[int], default=[3, 4, 5]): Kernel sizes for each convolution

**Attributes:**
- `char_embed` (nn.Embedding): Character embeddings
- `convs` (nn.ModuleList): List of Conv1d layers
- `output_dim` (int): Sum of all filter sizes

**Input/Output:**
- **Input:** `char_ids` - torch.LongTensor of shape `(batch_size, seq_len, max_word_len)`
- **Output:** torch.Tensor of shape `(batch_size, seq_len, output_dim)`

**Usage Example:**

```python
from wave_transformer.core.embeddings import CharCNNEmbedding

char_vocab_size = 256  # ASCII characters
char_cnn = CharCNNEmbedding(
    vocab_size=char_vocab_size,
    char_embed_dim=16,
    filters=[32, 64, 128],
    kernel_sizes=[3, 4, 5]
)

# Each word represented by character IDs
char_ids = torch.randint(0, char_vocab_size, (32, 50, 15))  # (batch, seq, max_word_len)
word_embeddings = char_cnn(char_ids)  # (32, 50, 224) where 224 = 32+64+128
```

**Implementation Notes:**
- Captures morphological patterns (prefixes, suffixes)
- Handles OOV words better than word-level embeddings
- Output dimension is sum of filter counts
- ReLU activation applied after convolution

---

### 5. SubwordEmbedding

**Purpose:** Handles subword tokenization by averaging embeddings of subword units, useful for BPE or WordPiece tokenization schemes.

**Architecture:**
- Standard embedding layer for subword units
- Optional masking for padding subwords
- Average pooling over subword dimension

**Parameters:**
- `vocab_size` (int): Subword vocabulary size
- `embedding_dim` (int): Embedding dimension
- `max_subwords` (int, default=4): Maximum subwords per token

**Attributes:**
- `embedding` (nn.Embedding): Subword embedding table
- `max_subwords` (int): Maximum subwords per token

**Input/Output:**
- **Input:**
  - `subword_ids` - torch.LongTensor of shape `(batch_size, seq_len, max_subwords)`
  - `subword_mask` (optional) - torch.BoolTensor of shape `(batch_size, seq_len, max_subwords)`
- **Output:** torch.Tensor of shape `(batch_size, seq_len, embedding_dim)`

**Usage Example:**

```python
from wave_transformer.core.embeddings import SubwordEmbedding

subword_embed = SubwordEmbedding(vocab_size=30000, embedding_dim=256, max_subwords=4)

# Each token split into max 4 subwords
subword_ids = torch.randint(0, 30000, (32, 128, 4))
subword_mask = torch.randint(0, 2, (32, 128, 4)).bool()  # Valid subword mask

token_embeddings = subword_embed(subword_ids, subword_mask)  # (32, 128, 256)
```

**Implementation Notes:**
- Mask prevents padding subwords from affecting average
- Length normalization prevents bias toward tokens with more subwords
- Suitable for BPE, WordPiece, SentencePiece tokenization

---

### 6. HybridEmbedding

**Purpose:** Combines word-level and character-level embeddings to leverage both semantic (word) and morphological (character) information.

**Architecture:**
- Word-level embedding
- Character-level CNN embedding
- Concatenation followed by projection

**Parameters:**
- `word_vocab_size` (int): Word vocabulary size
- `char_vocab_size` (int): Character vocabulary size
- `word_embed_dim` (int, default=200): Word embedding dimension
- `char_embed_dim` (int, default=100): Character embedding dimension

**Attributes:**
- `word_embed` (nn.Embedding): Word embeddings
- `char_embed` (CharCNNEmbedding): Character CNN embeddings
- `projection` (nn.Linear): Projects concatenated embeddings to `word_embed_dim`

**Input/Output:**
- **Input:**
  - `word_ids` - torch.LongTensor of shape `(batch_size, seq_len)`
  - `char_ids` - torch.LongTensor of shape `(batch_size, seq_len, max_word_len)`
- **Output:** torch.Tensor of shape `(batch_size, seq_len, word_embed_dim)`

**Usage Example:**

```python
from wave_transformer.core.embeddings import HybridEmbedding

hybrid = HybridEmbedding(
    word_vocab_size=50000,
    char_vocab_size=256,
    word_embed_dim=200,
    char_embed_dim=100
)

word_ids = torch.randint(0, 50000, (32, 128))
char_ids = torch.randint(0, 256, (32, 128, 15))

embeddings = hybrid(word_ids, char_ids)  # (32, 128, 200)
```

**Implementation Notes:**
- Combines semantic and morphological features
- Character component helps with rare/OOV words
- Projection layer unifies dimensions
- More robust than either approach alone

---

## Token Encoders

### **File:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\token_encoder.py`

### 1. TokenToWaveEncoder

**Purpose:** Lightweight encoder that converts token sequences to wave representations (frequencies, amplitudes, phases) using attention and semantic layers.

**Architecture:**
- Token embedding layer
- Self-attention mechanism
- Input projection to hidden dimension
- Multiple semantic processing layers (Linear → RMSNorm → GELU → Dropout)
- Wave component projectors (frequencies, amplitudes, phases)

**Parameters:**
- `vocab_size` (int): Token vocabulary size
- `num_harmonics` (int, default=64): Number of wave harmonics per token
- `d_model` (int, default=256): Model embedding dimension
- `hidden_mult` (float, default=2.0): Hidden dimension multiplier
- `num_heads` (int, default=4): Number of attention heads
- `num_heads_kv` (int, default=4): Number of key-value heads
- `num_layers` (int, default=2): Number of semantic layers
- `shared_projector` (bool, default=False): Use shared vs separate projectors

**Attributes:**
- `embedding` (nn.Embedding): Token embeddings
- `self_attention` (FlashAttention): Self-attention layer
- `input_projection` (nn.Linear): Projects to hidden dimension
- `semantic_layers` (nn.ModuleList): Processing layers
- `freq_projector`, `amp_projector`, `phase_projector` (nn.Linear): Component projectors (if not shared)
- `projector` (nn.Linear): Shared projector (if shared_projector=True)

**Input/Output:**
- **Input:**
  - `token_ids` - torch.LongTensor of shape `(batch_size, seq_len)`
  - `attention_mask` - torch.BoolTensor of shape `(batch_size, seq_len)`
- **Output:** Wave object with:
  - `frequencies` - torch.Tensor of shape `(batch_size, seq_len, num_harmonics)` in range [0.1, 20.1]
  - `amplitudes` - torch.Tensor of shape `(batch_size, seq_len, num_harmonics)` in range [0, ∞)
  - `phases` - torch.Tensor of shape `(batch_size, seq_len, num_harmonics)` in range [-π, π]

**Wave Component Activations:**
```python
frequencies = sigmoid(f) * 20.0 + 0.1    # Range: [0.1, 20.1]
amplitudes = softplus(a)                  # Range: [0, ∞), unnormalized
phases = tanh(p) * π                      # Range: [-π, π]
```

**Usage Example:**
```python
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoder

encoder = TokenToWaveEncoder(
    vocab_size=50000,
    num_harmonics=64,
    d_model=256,
    hidden_mult=2.0,
    num_heads=4,
    num_layers=2,
    shared_projector=False
)

token_ids = torch.randint(0, 50000, (32, 128))
attention_mask = torch.ones(32, 128).bool()

wave = encoder(token_ids, attention_mask)
# wave.frequencies: (32, 128, 64)
# wave.amplitudes: (32, 128, 64)
# wave.phases: (32, 128, 64)
```

**Implementation Notes:**
- Residual connections in semantic layers
- Dropout for regularization (p=0.1)
- Shared projector reduces parameters but may limit expressiveness
- Attention mask supports variable-length sequences
- Wave components use appropriate activations for their physical constraints

---

### 2. WaveAwarePositionalEncoding

**Purpose:** Specialized positional encoding that applies different encoding strategies for frequency, amplitude, and phase components of wave representations.

**Architecture:**
- Frequency path: Sinusoidal embeddings (natural periodicity)
- Amplitude path: Learnable embeddings (flexible magnitude encoding)
- Phase path: Rotary embeddings (natural for phase information)

**Parameters:**
- `d_model` (int): Model embedding dimension
- `max_len` (int, default=512): Maximum sequence length

**Attributes:**
- `freq_pe` (SinusoidalPositionEmbedding): For frequency component
- `amp_pe` (nn.Embedding): For amplitude component
- `phase_pe` (RotaryPositionEmbedding): For phase component

**Input/Output:**
- **Input:**
  - `x` - torch.Tensor of shape `(batch_size, seq_len, d_model)`
  - `component` (str): One of ['freq', 'amp', 'phase']
- **Output:** torch.Tensor of shape `(batch_size, seq_len, d_model)`

**Usage Example:**
```python
from wave_transformer.language_modelling.token_encoder import WaveAwarePositionalEncoding

wave_pe = WaveAwarePositionalEncoding(d_model=256, max_len=512)

x = torch.randn(32, 128, 256)
x_freq = wave_pe(x, component='freq')   # Sinusoidal encoding
x_amp = wave_pe(x, component='amp')     # Learnable encoding
x_phase = wave_pe(x, component='phase') # Rotary encoding
```

**Implementation Notes:**
- Component-specific encodings match wave properties
- Frequency uses sinusoidal (periodic nature)
- Amplitude uses learnable (flexible magnitude)
- Phase uses rotary (natural rotation encoding)

---

### 3. FrequencyAwareEmbedding

**Purpose:** Token embeddings enhanced with Fourier-like frequency features, allowing tokens to have learnable frequency characteristics.

**Architecture:**
- Base token embedding (half dimension)
- Learnable frequency bands
- Token-to-frequency weight mapping
- Weighted combination of frequency bands

**Parameters:**
- `vocab_size` (int): Token vocabulary size
- `d_model` (int): Total embedding dimension
- `num_freq_bands` (int, default=8): Number of frequency bands

**Attributes:**
- `base_embedding` (nn.Embedding): Base token embeddings of dimension `d_model // 2`
- `freq_bands` (nn.Parameter): Learnable frequency bands of shape `(num_freq_bands, d_model // 2)`
- `token_to_freq` (nn.Embedding): Maps tokens to frequency band weights

**Input/Output:**
- **Input:** `token_ids` - torch.LongTensor of shape `(batch_size, seq_len)`
- **Output:** torch.Tensor of shape `(batch_size, seq_len, d_model)`

**Mathematical Operation:**
```python
freq_emb = Σ(softmax(token_weights) * freq_bands)
output = concat([base_emb, freq_emb], dim=-1)
```

**Usage Example:**
```python
from wave_transformer.language_modelling.token_encoder import FrequencyAwareEmbedding

freq_embed = FrequencyAwareEmbedding(
    vocab_size=50000,
    d_model=256,
    num_freq_bands=8
)

token_ids = torch.randint(0, 50000, (32, 128))
embeddings = freq_embed(token_ids)  # (32, 128, 256)
```

**Implementation Notes:**
- Half dimension for base, half for frequency features
- Softmax ensures valid frequency band weighting
- Learns which tokens should have which frequency characteristics
- Einsum operation: `'bsf,fd->bsd'` computes weighted frequency embeddings

---

### 4. WaveEncoderBlock

**Purpose:** Transformer-style encoder block for processing wave components with multi-head attention and feed-forward layers.

**Architecture:**
- Multiple non-causal parallel blocks (attention + FFN)
- Residual connections
- Final projection to harmonic dimension

**Parameters:**
- `d_model` (int): Model dimension
- `num_heads` (int): Number of query heads
- `num_heads_kv` (int): Number of key-value heads
- `d_ff` (int): Feed-forward dimension
- `dropout` (float): Dropout rate
- `num_harmonics` (int): Output harmonic dimension
- `num_layers` (int, default=2): Number of encoder layers
- `use_flash` (bool, default=False): Use flash attention

**Attributes:**
- `layers` (nn.ModuleList): List of NonCausalParallelBlock layers
- `proj` (nn.Linear): Projects to num_harmonics

**Input/Output:**
- **Input:**
  - `x` - torch.Tensor of shape `(batch_size, seq_len, d_model)`
  - `attention_mask` (optional) - torch.BoolTensor
- **Output:** torch.Tensor of shape `(batch_size, seq_len, num_harmonics)`

**Usage Example:**
```python
from wave_transformer.language_modelling.token_encoder import WaveEncoderBlock

block = WaveEncoderBlock(
    d_model=256,
    num_heads=8,
    num_heads_kv=8,
    d_ff=1024,
    dropout=0.1,
    num_harmonics=64,
    num_layers=4
)

x = torch.randn(32, 128, 256)
attention_mask = torch.ones(32, 128).bool()
harmonics = block(x, attention_mask)  # (32, 128, 64)
```

**Implementation Notes:**
- Residual connections around each layer
- Non-causal attention (bidirectional)
- Final projection to harmonic space
- Supports grouped-query attention (num_heads_kv)

---

### 5. TokenToWaveEncoderImproved

**Purpose:** Advanced encoder with component-specific processing paths for frequencies, amplitudes, and phases.

**Architecture:**
- Frequency-aware embeddings
- Wave-aware positional encodings
- Separate encoder blocks for each wave component
- Component normalization layers

**Parameters:**
- `vocab_size` (int): Token vocabulary size
- `d_model` (int, default=256): Model dimension
- `num_layers` (int, default=4): Layers per component encoder
- `d_ff` (int, default=1024): Feed-forward dimension
- `num_harmonics` (int, default=64): Number of harmonics
- `dropout` (float, default=0.1): Dropout rate

**Attributes:**
- `embedding` (FrequencyAwareEmbedding): Input embeddings
- `wave_pos_encoding` (WaveAwarePositionalEncoding): Positional encodings
- `freq_generator` (WaveEncoderBlock): Frequency encoder
- `amp_generator` (WaveEncoderBlock): Amplitude encoder
- `phase_generator` (WaveEncoderBlock): Phase encoder
- `freq_norm`, `amp_norm`, `phase_norm` (nn.LayerNorm): Normalization layers

**Input/Output:**
- **Input:**
  - `token_ids` - torch.LongTensor of shape `(batch_size, seq_len)`
  - `attention_mask` (optional) - torch.BoolTensor
- **Output:** Wave object with components of shape `(batch_size, seq_len, num_harmonics)`

**Wave Component Activations:**
```python
frequencies = exp(clamp(f, -3, 3)) * 2.0    # Exponential scaling
amplitudes = softplus(a) + 1e-6             # Non-zero guarantee
phases = atan2(sin(p), cos(p))              # Proper phase wrapping
```

**Usage Example:**
```python
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoderImproved

encoder = TokenToWaveEncoderImproved(
    vocab_size=50000,
    d_model=256,
    num_layers=4,
    d_ff=1024,
    num_harmonics=64,
    dropout=0.1
)

token_ids = torch.randint(0, 50000, (32, 128))
wave = encoder(token_ids)
```

**Implementation Notes:**
- Separate processing paths for each wave component
- Exponential activation for frequencies (multiplicative nature)
- Phase wrapping using atan2 ensures [-π, π] range
- Component-specific positional encodings
- More parameters than simple encoder but better wave representation

---

### 6. TokenToWaveEncoderSimple

**Purpose:** Streamlined encoder with shared positional encoding and optional flash attention for efficient training.

**Architecture:**
- Standard token embedding
- Sinusoidal positional encoding
- Three parallel encoder blocks (frequency, amplitude, phase)
- Simple activation functions

**Parameters:**
- `vocab_size` (int): Token vocabulary size
- `d_model` (int, default=256): Model dimension
- `num_layers` (int, default=4): Layers per encoder block
- `d_ff` (int, default=1024): Feed-forward dimension
- `num_harmonics` (int, default=64): Number of harmonics
- `dropout` (float, default=0.1): Dropout rate
- `use_flash` (bool, default=True): Enable flash attention

**Attributes:**
- `embedding` (nn.Embedding): Token embeddings
- `position_embedding` (SinusoidalPositionEmbedding): Shared positional encoding
- `freq_generator`, `amp_generator`, `phase_generator` (WaveEncoderBlock): Component encoders

**Input/Output:**
- **Input:**
  - `token_ids` - torch.LongTensor of shape `(batch_size, seq_len)`
  - `attention_mask` (optional) - torch.BoolTensor
- **Output:** Wave object with components of shape `(batch_size, seq_len, num_harmonics)`

**Usage Example:**
```python
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoderSimple

encoder = TokenToWaveEncoderSimple(
    vocab_size=50000,
    d_model=256,
    num_layers=4,
    d_ff=1024,
    num_harmonics=64,
    use_flash=True
)

token_ids = torch.randint(0, 50000, (32, 128))
attention_mask = torch.ones(32, 128).bool()
wave = encoder(token_ids, attention_mask)
```

**Implementation Notes:**
- Shared positional encoding across all components
- Flash attention for memory efficiency
- Standard activations (sigmoid, softplus, tanh)
- Simpler than improved version but faster
- Good baseline for wave-based language modeling

---

## Token Decoders

### **File:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\token_decoder.py`

### WaveToTokenDecoder

**Purpose:** Decodes wave representations (frequencies, amplitudes, phases) back to token logits for language modeling.

**Architecture:**
- Input projection from concatenated wave components
- Self-attention with causal masking
- Hidden dimension expansion
- Multiple decoder layers with residual connections
- Output projection to vocabulary (standard or low-rank)

**Parameters:**
- `vocab_size` (int): Token vocabulary size
- `num_harmonics` (int, default=64): Number of wave harmonics
- `d_model` (int, default=256): Model dimension
- `hidden_mult` (float, default=2.0): Hidden dimension multiplier
- `num_heads` (int, default=4): Number of attention heads
- `num_heads_kv` (int, default=4): Number of key-value heads
- `num_layers` (int, default=2): Number of decoder layers
- `low_rank_output` (int, optional): Low-rank bottleneck dimension (e.g., 256)
- `use_flash` (bool, default=False): Use flash attention

**Attributes:**
- `input_projection` (nn.Linear): Projects concatenated wave (num_harmonics * 3) to d_model
- `self_attention` (FlashAttention): Causal self-attention
- `hidden_projection` (nn.Linear): Expands to hidden_dim
- `decoder_layers` (nn.ModuleList): Processing layers (Linear → RMSNorm → GELU → Dropout)
- `output_projection` (nn.Linear or nn.Sequential): Projects to vocab_size

**Input/Output:**
- **Input:**
  - `representation` - torch.Tensor of shape `(batch_size, seq_len, num_harmonics * 3)` (concatenated wave components)
  - `attention_mask` (optional) - torch.BoolTensor
- **Output:** torch.Tensor of shape `(batch_size, seq_len, vocab_size)` - token logits

**Low-Rank Output:**
When `low_rank_output` is specified:
```python
output_projection = Sequential(
    Linear(hidden_dim, low_rank_output),
    GELU(),
    Linear(low_rank_output, vocab_size)
)
```

**Usage Example:**
```python
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder
from wave_transformer.core.transformer import Wave

decoder = WaveToTokenDecoder(
    vocab_size=50000,
    num_harmonics=64,
    d_model=256,
    hidden_mult=2.0,
    num_heads=4,
    num_layers=2,
    low_rank_output=256,  # Optional low-rank factorization
    use_flash=True
)

# Wave representation from encoder
wave = Wave(
    frequencies=torch.randn(32, 128, 64),
    amplitudes=torch.randn(32, 128, 64),
    phases=torch.randn(32, 128, 64)
)

representation = wave.to_representation()  # (32, 128, 192)
logits = decoder(representation)           # (32, 128, 50000)
```

**Usage with Full Pipeline:**
```python
from wave_transformer.language_modelling.token_encoder import TokenToWaveEncoderSimple
from wave_transformer.language_modelling.token_decoder import WaveToTokenDecoder

# Encoder
encoder = TokenToWaveEncoderSimple(vocab_size=50000, num_harmonics=64)
# Decoder
decoder = WaveToTokenDecoder(vocab_size=50000, num_harmonics=64)

# Forward pass
token_ids = torch.randint(0, 50000, (32, 128))
wave = encoder(token_ids)
representation = wave.to_representation()
logits = decoder(representation)
```

**Implementation Notes:**
- Causal attention for autoregressive generation
- Residual connections in decoder layers
- Low-rank output reduces parameters: `hidden_dim * vocab_size → (hidden_dim * low_rank + low_rank * vocab_size)`
- Dropout rate: 0.1 for regularization
- GELU activation in decoder layers
- RMSNorm for efficient normalization

**Performance Considerations:**
- Low-rank output saves memory for large vocabularies
- Flash attention reduces memory for long sequences
- Hidden multiplier controls model capacity
- Typical hidden_dim = d_model * hidden_mult = 256 * 2.0 = 512

---

## Training Utilities

### **File:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\train_utils.py`

### 1. compute_distillation_loss

**Purpose:** Computes combined loss for knowledge distillation from teacher to student model.

**Parameters:**
- `student_logits` - torch.Tensor of shape `(batch_size, seq_len, vocab_size)`: Student model predictions
- `teacher_logits` - torch.Tensor of shape `(batch_size, seq_len, vocab_size)`: Teacher model predictions
- `targets` - torch.LongTensor of shape `(batch_size, seq_len)`: Ground truth token IDs
- `pad_token_id` (int): Padding token ID to ignore in loss
- `alpha` (float, default=0.5): Weight for distillation vs. cross-entropy (0.0-1.0)
- `temperature` (float, default=2.0): Softening temperature for distributions

**Returns:**
- `loss` (torch.Tensor): Combined weighted loss
- `lm_loss` (float): Cross-entropy loss value
- `kl_loss` (float): KL divergence loss value

**Mathematical Foundation:**
```python
CE_loss = CrossEntropy(student_logits, targets)
KL_loss = KL_divergence(student_logits/T, teacher_logits/T) * T²
Total_loss = α * CE_loss + (1-α) * KL_loss
```

**Usage Example:**
```python
from wave_transformer.language_modelling.train_utils import compute_distillation_loss

student_logits = student_model(inputs)
with torch.no_grad():
    teacher_logits = teacher_model(inputs)

loss, ce_loss, kl_loss = compute_distillation_loss(
    student_logits=student_logits,
    teacher_logits=teacher_logits,
    targets=targets,
    pad_token_id=tokenizer.pad_token_id,
    alpha=0.5,
    temperature=2.0
)

loss.backward()
print(f"CE: {ce_loss:.4f}, KL: {kl_loss:.4f}")
```

**Implementation Notes:**
- Temperature scaling softens distributions for better knowledge transfer
- Temperature squared in KL loss compensates for gradient scaling
- Alpha balances task loss vs. distillation
- Ignores padding tokens in cross-entropy
- Uses batchmean reduction for KL divergence

---

### 2. generate_text

**Purpose:** Autoregressive text generation with advanced sampling strategies (temperature, top-k, top-p, min-p, repetition penalty).

**Parameters:**
- `model` (nn.Module): Language model
- `tokenizer` (Tokenizer): Tokenizer for encoding/decoding
- `prompt` (str or Encoding): Input prompt
- `device` (torch.device): Device for computation
- `max_tokens` (int, default=100): Maximum tokens to generate
- `temperature` (float, default=0.75): Sampling temperature (higher = more random)
- `top_k` (int, default=0): Top-k sampling (0 = disabled)
- `top_p` (float, default=0.9): Nucleus sampling threshold
- `min_p` (float, default=0.0): Minimum probability threshold
- `repetition_penalty` (float, default=1.2): Penalty for repeated tokens (>1.0 = discourage)

**Returns:**
- `str`: Generated text (prompt + completion)

**Sampling Pipeline:**
1. **Temperature scaling:** `logits = logits / temperature`
2. **Repetition penalty:** Penalize tokens in context
3. **Min-p filtering:** Remove tokens below `max_prob * min_p`
4. **Top-k filtering:** Keep only top k tokens
5. **Top-p (nucleus) filtering:** Keep tokens with cumulative probability ≤ top_p
6. **Sampling:** Multinomial sampling from filtered distribution

**Usage Example:**
```python
from wave_transformer.language_modelling.train_utils import generate_text

model.eval()
generated = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt="The meaning of life is",
    device=torch.device("cuda"),
    max_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    min_p=0.05,
    repetition_penalty=1.2
)
print(generated)
```

**Sampling Strategies Explained:**

- **Temperature (0.1-2.0):**
  - Low (0.1-0.7): Focused, deterministic
  - Medium (0.7-1.0): Balanced
  - High (1.0-2.0): Creative, random

- **Top-k (10-100):**
  - Restricts to k most probable tokens
  - Lower k = more focused

- **Top-p / Nucleus (0.8-0.95):**
  - Dynamic cutoff based on cumulative probability
  - 0.9 = keep tokens comprising top 90% probability

- **Min-p (0.01-0.1):**
  - Removes tokens below `threshold = max_prob * min_p`
  - Adapts to distribution sharpness

- **Repetition Penalty (1.0-1.5):**
  - Multiplies logits of already-seen tokens
  - >1.0 discourages repetition

**Implementation Notes:**
- Auto-detects EOS token (supports multiple formats)
- Uses bfloat16 autocast for efficiency
- Handles numerical stability (nan_to_num, epsilon addition)
- Stops generation on EOS token
- Model input format: `{"token_ids": tensor}`

---

### 3. test_generation

**Purpose:** Batch text generation for evaluation with predefined prompts.

**Parameters:**
- `model` (nn.Module): Language model
- `tokenizer` (Tokenizer): Tokenizer
- `max_tokens` (int): Maximum tokens per generation
- `device` (torch.device): Device
- `prompts` (List[str], optional): Custom prompts (uses defaults if None)

**Returns:**
- `List[str]`: Generated texts

**Default Prompts:**
```python
[
    "The tao that can be told",
    "Success is as dangerous as failure.",
    "Major Premise: All matter is composed of atoms,",
    "Claim: The most informative and foundational concept in science,",
    "Claim: A string with both ends fixed can only oscillate"
]
```

**Usage Example:**
```python
from wave_transformer.language_modelling.train_utils import test_generation

generations = test_generation(
    model=model,
    tokenizer=tokenizer,
    max_tokens=50,
    device=torch.device("cuda"),
    prompts=["Once upon a time", "In the beginning"]
)

for gen in generations:
    print(gen)
```

**Implementation Notes:**
- Uses greedy decoding (temperature=0.0, top_p=1.0, repetition_penalty=1.0)
- Prints input/output pairs
- Useful for qualitative evaluation

---

### 4. cosine_schedule_with_warmup

**Purpose:** Learning rate scheduler with linear warmup followed by cosine decay.

**Parameters:**
- `optimizer` (torch.optim.Optimizer): Optimizer
- `warmup_steps` (int): Number of warmup steps
- `total_steps` (int): Total training steps
- `base_lr` (float): Peak learning rate (after warmup)
- `final_lr` (float, default=0.0): Final learning rate

**Returns:**
- `LambdaLR`: Learning rate scheduler

**Schedule Formula:**
```python
# Warmup phase (steps 0 to warmup_steps)
lr = base_lr * (step / warmup_steps)

# Cosine decay phase (steps warmup_steps to total_steps)
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr = final_lr + (base_lr - final_lr) * 0.5 * (1 + cos(π * progress))
```

**Usage Example:**
```python
from wave_transformer.language_modelling.train_utils import cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = cosine_schedule_with_warmup(
    optimizer=optimizer,
    warmup_steps=1000,
    total_steps=10000,
    base_lr=1e-3,
    final_lr=1e-5
)

for epoch in range(epochs):
    for batch in dataloader:
        loss = train_step(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update LR every step
```

**Implementation Notes:**
- Step refers to optimizer steps (post-gradient accumulation)
- Warmup prevents early training instability
- Cosine decay smoothly reduces LR
- Compatible with gradient accumulation

---

### 5. prepare_autoregressive_batch

**Purpose:** Prepares input/target pairs for autoregressive language modeling by shifting sequences.

**Parameters:**
- `batch` (dict): Batch with 'input_ids' and 'attention_mask'
- `pad_token_id` (int): Padding token ID

**Returns:**
- `inputs` - torch.LongTensor of shape `(batch_size, seq_len-1)`: Input tokens
- `targets` - torch.LongTensor of shape `(batch_size, seq_len-1)`: Target tokens (shifted by 1)
- `input_mask` - torch.BoolTensor of shape `(batch_size, seq_len-1)`: Attention mask for inputs

**Shifting Logic:**
```python
inputs = token_ids[:, :-1]   # All tokens except last
targets = token_ids[:, 1:]   # All tokens except first
# Predicts: inputs[i] → targets[i] (next token)
```

**Usage Example:**
```python
from wave_transformer.language_modelling.train_utils import prepare_autoregressive_batch

batch = {
    'input_ids': torch.randint(0, 50000, (32, 512)),
    'attention_mask': torch.ones(32, 512).bool()
}

inputs, targets, mask = prepare_autoregressive_batch(batch, pad_token_id=0)
# inputs: (32, 511)
# targets: (32, 511)
# mask: (32, 511)

logits = model({"token_ids": inputs})
loss = compute_language_modeling_loss(logits, targets, pad_token_id=0)
```

**Implementation Notes:**
- Ensures attention_mask is boolean
- Removes last token from inputs (no target for it)
- Removes first token from targets (no input for it)
- Mask aligns with inputs

---

### 6. compute_language_modeling_loss

**Purpose:** Computes cross-entropy loss for language modeling with label smoothing.

**Parameters:**
- `logits` - torch.Tensor or tuple: Model output logits
- `targets` - torch.LongTensor of shape `(batch_size, seq_len)`: Target token IDs
- `pad_token_id` (int): Padding token ID to ignore

**Returns:**
- `torch.Tensor`: Scalar loss value

**Loss Formula:**
```python
CrossEntropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
    ignore_index=pad_token_id,
    label_smoothing=0.05
)
```

**Usage Example:**
```python
from wave_transformer.language_modelling.train_utils import compute_language_modeling_loss

logits = model(inputs)  # (32, 511, 50000)
targets = batch_targets  # (32, 511)

loss = compute_language_modeling_loss(logits, targets, pad_token_id=0)
loss.backward()
```

**Implementation Notes:**
- Handles various logit formats (tensor, tuple, object with .logits)
- Label smoothing (0.05) improves generalization
- Ignores padding tokens
- Flattens batch and sequence dimensions for CE loss

---

### 7. compute_diversity_metrics & diversity_report

**Purpose:** Evaluate text diversity using n-gram statistics.

**Parameters:**
- `texts` (List[str]): Generated texts
- `n` (int or tuple, default=3): N-gram size(s)

**Returns:**
- `dict`: Diversity metrics
  - `distinct_n`: Ratio of unique n-grams to total n-grams
  - `mean_repetition_n`: Average repetition within sequences

**Metrics:**
- **Distinct-n:** `unique_ngrams / total_ngrams` (higher = more diverse)
- **Repetition:** `(total_ngrams - unique_ngrams) / total_ngrams` (lower = less repetitive)

**Usage Example:**
```python
from wave_transformer.language_modelling.train_utils import diversity_report

texts = [
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps over a sleepy dog",
    "The quick brown fox jumps again"
]

metrics = diversity_report(texts, ns=(1, 2, 3))
print(metrics)
# {
#   'distinct_1': 0.85,
#   'mean_repetition_1': 0.15,
#   'distinct_2': 0.78,
#   'mean_repetition_2': 0.22,
#   ...
# }
```

**Implementation Notes:**
- Higher distinct-n indicates diverse vocabulary/phrasing
- Lower repetition indicates less redundant generation
- Unigrams (n=1): vocabulary diversity
- Bigrams/Trigrams (n=2,3): phrase diversity

---

### 8. Utility Functions

**save_training_chronicle**
```python
def save_training_chronicle(chronicle, experiment_name, timestamp):
    """Saves training logs to JSON file."""
    output_path = Path(f"{experiment_name}_{timestamp.replace(':', '-')}.json")
    with open(output_path, 'w') as f:
        json.dump(chronicle, f, indent=2, default=str)
    return output_path
```

**extract_architecture_details**
```python
def extract_architecture_details(model):
    """Extracts model architecture information."""
    return {
        'representation': str(model),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'layer_details': [...]  # Per-layer parameter counts
    }
```

**camel_to_snake**
```python
def camel_to_snake(camel_case):
    """Convert CamelCase to snake_case."""
    # "TokenEncoder" -> "token_encoder"
```

---

## Dataset Classes

### **File:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\text_datasets.py`

### 1. TextDatasetPadded

**Purpose:** Clean, line-aware dataset for language modeling that packs text into fixed-length sequences while respecting line boundaries.

**Architecture:**
- Packs text into chunks ≤ max_length
- Prefers cutting at line boundaries
- Falls back to word-level packing for long lines
- Optional remainder handling

**Parameters:**
- `texts` (List[str]): Input texts
- `tokenizer` (Tokenizer): Tokenizer instance
- `pad_token_id` (int): Padding token ID
- `max_length` (int, default=512): Maximum sequence length
- `device` (torch.device, default=cpu): Device for tensors
- `keep_remainder` (bool, default=True): Emit additional samples from overflow text

**Attributes:**
- `examples` (List[dict]): Preprocessed examples
  - Each example: `{"input_ids": LongTensor, "attention_mask": BoolTensor}`

**Input/Output:**
- **Input (init):** List of raw text strings
- **Output (__getitem__):** dict with:
  - `input_ids` - torch.LongTensor of shape `(max_length,)`
  - `attention_mask` - torch.BoolTensor of shape `(max_length,)` (True = valid, False = padding)

**Line-Aware Packing Algorithm:**
1. Split text into lines
2. Tokenize each line
3. Accumulate lines until max_length would be exceeded
4. If a single line exceeds max_length, fall back to word-level packing
5. Emit chunk when full
6. If keep_remainder=True, continue with remaining text

**Usage Example:**
```python
from wave_transformer.language_modelling.text_datasets import TextDatasetPadded
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
texts = [
    "First document with multiple lines.\nSecond line here.\nThird line.",
    "Another document.\nWith more content.\nAnd another line."
]

dataset = TextDatasetPadded(
    texts=texts,
    tokenizer=tokenizer,
    pad_token_id=0,
    max_length=128,
    keep_remainder=True
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    input_ids = batch['input_ids']        # (32, 128)
    attention_mask = batch['attention_mask']  # (32, 128) bool
```

**Implementation Notes:**
- Respects document structure (lines)
- More natural splits than arbitrary truncation
- keep_remainder=False mimics old behavior with cleaner cuts
- Handles edge cases (empty texts, single long tokens)
- Disables tokenizer truncation/padding (manual control)

---

### 2. BoundedStreamingDataset

**Purpose:** Memory-efficient streaming dataset that tokenizes text on-the-fly with sliding window and stride control.

**Architecture:**
- Iterates over streaming data source
- Maintains token buffer for window extraction
- Supports overlapping windows via stride
- Incremental processing (no full dataset in memory)

**Parameters:**
- `data_source` (str or HFIterableDataset): HuggingFace dataset name or dataset object
- `tokenizer` (Tokenizer): Tokenizer instance
- `pad_token_id` (int): Padding token ID
- `sequence_length` (int, default=512): Fixed sequence length
- `stride` (int, optional): Sliding window stride (default=sequence_length, no overlap)
- `text_column` (str, default="text"): Column name containing text
- `skip_first` (int, default=0): Number of entries to skip
- `max_entries` (int, optional): Maximum entries to yield
- `device` (torch.device, default=cpu): Device for tensors

**Sliding Window Behavior:**
```python
# stride = sequence_length: No overlap
# buffer = [1,2,3,4,5,6,7,8], sequence_length=4, stride=4
# → [1,2,3,4], [5,6,7,8]

# stride < sequence_length: Overlap
# buffer = [1,2,3,4,5,6,7,8], sequence_length=4, stride=2
# → [1,2,3,4], [3,4,5,6], [5,6,7,8]
```

**Input/Output:**
- **Input (init):** Dataset specification
- **Output (__iter__):** Iterator yielding dict:
  - `input_ids` - torch.LongTensor of shape `(sequence_length,)`
  - `attention_mask` - torch.BoolTensor of shape `(sequence_length,)`

**Usage Example:**
```python
from wave_transformer.language_modelling.text_datasets import BoundedStreamingDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

dataset = BoundedStreamingDataset(
    data_source="wikitext",  # HuggingFace dataset
    tokenizer=tokenizer,
    pad_token_id=0,
    sequence_length=512,
    stride=256,  # 50% overlap
    text_column="text",
    skip_first=0,
    max_entries=10000,
    device=torch.device("cuda")
)

# Streaming dataloader (no shuffle for iterable datasets)
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    # Process batch
    pass
```

**Implementation Notes:**
- Memory efficient (doesn't load full dataset)
- Stride enables context overlap for better learning
- Handles remaining buffer at end of stream
- Skips invalid/empty texts
- Proper error handling for tokenization failures

---

### 3. PreparedDataset

**Purpose:** Fast dataset loading from pre-prepared pickle files, ideal for repeated training runs.

**Architecture:**
- Loads preprocessed examples from disk
- Supports metadata storage
- Near-instant dataset loading

**Parameters:**
- `data_path` (str or Path): Path to pickle file
- `device` (torch.device, default=cpu): Device for tensors

**Attributes:**
- `examples` (List[dict]): Loaded examples
- `metadata` (dict): Dataset metadata (optional)

**Input/Output:**
- **Input (init):** Path to .pkl file
- **Output (__getitem__):** dict with:
  - `input_ids` - torch.LongTensor of shape `(sequence_length,)`
  - `attention_mask` - torch.BoolTensor of shape `(sequence_length,)`

**Usage Example:**
```python
from wave_transformer.language_modelling.text_datasets import PreparedDataset
from torch.utils.data import DataLoader

dataset = PreparedDataset(
    data_path="data/wikitext_512.pkl",
    device=torch.device("cuda")
)

print(f"Loaded {len(dataset)} examples")
print(f"Metadata: {dataset.metadata}")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**File Format:**
```python
{
    'examples': [
        {
            'input_ids': torch.LongTensor,
            'attention_mask': torch.BoolTensor
        },
        ...
    ],
    'metadata': {
        'data_source': 'wikitext',
        'sequence_length': 512,
        'num_examples': 10000,
        ...
    }
}
```

**Implementation Notes:**
- Orders of magnitude faster than streaming for repeated training
- Requires preprocessing step
- File size: ~4 bytes per token + overhead
- Supports shuffling (unlike streaming datasets)

---

### 4. prepare_and_save_dataset

**Purpose:** Processes streaming dataset and saves to disk for fast loading, with optional parallel processing.

**Parameters:**
- `data_source` (str or HFIterableDataset): Dataset name or object
- `tokenizer` (Tokenizer): Tokenizer instance
- `pad_token_id` (int): Padding token ID
- `save_path` (str or Path): Output file path
- `sequence_length` (int, default=512): Sequence length
- `stride` (int, optional): Sliding window stride
- `text_column` (str, default="text"): Text column name
- `skip_first` (int, default=0): Entries to skip
- `max_entries` (int, optional): Maximum entries to process
- `subset` (str, optional): Dataset subset/configuration
- `num_workers` (int, optional): Parallel workers (None=single process, 0=auto-detect)
- `batch_size` (int, default=1000): Texts per worker batch

**Returns:**
- `Path`: Path to saved dataset file

**Usage Example:**
```python
from wave_transformer.language_modelling.text_datasets import prepare_and_save_dataset
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

dataset_path = prepare_and_save_dataset(
    data_source="wikitext",
    tokenizer=tokenizer,
    pad_token_id=0,
    save_path="data/wikitext_train.pkl",
    sequence_length=512,
    stride=512,
    text_column="text",
    max_entries=50000,
    subset="wikitext-103-raw-v1",
    num_workers=8,  # Parallel processing
    batch_size=1000
)

print(f"Dataset saved to: {dataset_path}")
```

**Parallel Processing:**
```python
# Single process (num_workers=None)
# - Simple, no overhead
# - Good for small datasets

# Parallel (num_workers=8)
# - 8x faster tokenization
# - Uses multiprocessing.Pool
# - Splits into batches
# - Requires tokenizer identifier
```

**Implementation Notes:**
- Progress bars via tqdm
- Automatic file size reporting
- Saves with highest pickle protocol
- Handles tokenization errors gracefully
- Falls back to single process if parallel fails

---

### 5. prepare_and_save_multi_dataset

**Purpose:** Processes multiple datasets with weighted sampling and saves combined dataset to disk.

**Parameters:**
- `dataset_specs` (List[Dict]): Dataset specifications
  - Each dict: `{'name': str, 'weight': float, 'max_entries': int, 'skip': int, 'subset': str}`
- `tokenizer` (Tokenizer): Tokenizer
- `pad_token_id` (int): Padding token ID
- `save_path` (str or Path): Output path
- `sequence_length` (int, default=512): Sequence length
- `stride` (int, optional): Stride
- `text_column` (str, default="text"): Text column
- `global_max_entries` (int, optional): Total entries across all datasets
- `seed` (int, optional): Random seed
- `num_workers` (int, default=8): Parallel workers

**Returns:**
- `Path`: Path to combined dataset

**Usage Example:**
```python
from wave_transformer.language_modelling.text_datasets import prepare_and_save_multi_dataset
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

dataset_specs = [
    {
        'name': 'wikitext',
        'subset': 'wikitext-103-raw-v1',
        'weight': 0.5,
        'max_entries': 30000,
        'skip': 0
    },
    {
        'name': 'openwebtext',
        'weight': 0.3,
        'max_entries': 20000,
        'skip': 0
    },
    {
        'name': 'bookcorpus',
        'weight': 0.2,
        'max_entries': 10000,
        'skip': 0
    }
]

combined_path = prepare_and_save_multi_dataset(
    dataset_specs=dataset_specs,
    tokenizer=tokenizer,
    pad_token_id=0,
    save_path="data/multi_dataset.pkl",
    sequence_length=512,
    global_max_entries=50000,  # Total examples
    seed=42,
    num_workers=8
)
```

**Weighted Sampling:**
```python
# weights = [0.5, 0.3, 0.2]
# global_max_entries = 10000
# Samples per dataset:
# - Dataset 1: 5000 (50%)
# - Dataset 2: 3000 (30%)
# - Dataset 3: 2000 (20%)
```

**Implementation Notes:**
- Processes each dataset separately (temp files)
- Combines with weighted sampling
- Shuffles combined dataset
- Cleans up temporary files
- Stores dataset specs in metadata

---

### 6. MultiBoundedStreamingDataset

**Purpose:** Streams multiple datasets with weighted random sampling, proper buffering, and termination control.

**Architecture:**
- Separate iterator and buffer per dataset
- Weighted random selection
- Proper exhaustion handling
- Global entry limit

**Parameters:**
- `dataset_specs` (List[Dict]): Dataset specifications (see above)
- `tokenizer` (Tokenizer): Tokenizer
- `pad_token_id` (int): Padding token ID
- `sequence_length` (int, default=512): Sequence length
- `stride` (int, optional): Stride
- `text_column` (str, default="text"): Text column
- `device` (torch.device, default=cpu): Device
- `global_max_entries` (int, optional): Total entries to yield
- `seed` (int, optional): Random seed

**Input/Output:**
- **Input (init):** Dataset specifications
- **Output (__iter__):** Iterator yielding dict:
  - `input_ids` - torch.LongTensor of shape `(sequence_length,)`
  - `attention_mask` - torch.BoolTensor of shape `(sequence_length,)`

**Usage Example:**
```python
from wave_transformer.language_modelling.text_datasets import MultiBoundedStreamingDataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

dataset_specs = [
    {'name': 'wikitext', 'max_entries': 10000, 'weight': 0.6},
    {'name': 'openwebtext', 'max_entries': 5000, 'weight': 0.4}
]

dataset = MultiBoundedStreamingDataset(
    dataset_specs=dataset_specs,
    tokenizer=tokenizer,
    pad_token_id=0,
    sequence_length=512,
    stride=256,
    global_max_entries=12000,
    seed=42
)

dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    # Process mixed batch from both datasets
    pass
```

**Weighted Sampling Details:**
```python
# At each iteration:
# 1. Normalize weights: [0.6, 0.4] / 1.0 = [0.6, 0.4]
# 2. Random choice with weights
# 3. Yield example from selected dataset
# 4. Update if dataset exhausted
```

**Implementation Notes:**
- Each dataset maintains independent buffer
- Properly handles dataset exhaustion (removes from pool)
- Renormalizes weights when datasets are removed
- Stops when all datasets exhausted OR global_max_entries reached
- Requires max_entries in each spec (prevents infinite iteration)
- Validates all parameters at init

---

### Helper Functions

**apply_padding**
```python
def apply_padding(sequence: List[int], target_length: int, pad_token: int) -> Tuple[List[int], List[int]]:
    """
    Pads or truncates sequence to target_length.

    Returns:
        - padded_sequence: List[int] of length target_length
        - attention_mask: List[int] (1=valid, 0=padding)
    """
    if len(sequence) >= target_length:
        return sequence[:target_length], [1] * target_length

    padding_size = target_length - len(sequence)
    return (
        sequence + [pad_token] * padding_size,
        [1] * len(sequence) + [0] * padding_size
    )
```

---

## Best Practices and Usage Patterns

### 1. Choosing Dataset Strategy

**Use PreparedDataset when:**
- Repeated training runs on same data
- Dataset fits in disk space
- Training speed is critical
- Need shuffling support

**Use Streaming datasets when:**
- Very large datasets (TB scale)
- Limited disk space
- Exploratory training
- One-time training runs

### 2. Encoder Selection

**TokenToWaveEncoder (Slim):**
- Fast, lightweight
- Good for prototyping
- Shared processing path

**TokenToWaveEncoderImproved:**
- Best wave representation quality
- Component-specific processing
- Higher memory/compute cost

**TokenToWaveEncoderSimple:**
- Balanced performance
- Flash attention support
- Production-ready baseline

### 3. Training Pipeline Example

```python
from wave_transformer.language_modelling import *
from tokenizers import Tokenizer
import torch

# 1. Prepare dataset
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
dataset_path = prepare_and_save_dataset(
    data_source="wikitext",
    tokenizer=tokenizer,
    pad_token_id=0,
    save_path="data/train.pkl",
    sequence_length=512,
    max_entries=100000,
    num_workers=8
)

# 2. Load dataset
train_dataset = PreparedDataset(dataset_path, device=torch.device("cuda"))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Create model
encoder = TokenToWaveEncoderSimple(vocab_size=30522, num_harmonics=64)
decoder = WaveToTokenDecoder(vocab_size=30522, num_harmonics=64)
model = nn.Sequential(encoder, decoder).cuda()

# 4. Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = cosine_schedule_with_warmup(optimizer, 1000, 10000, 1e-3)

for epoch in range(epochs):
    for batch in train_loader:
        inputs, targets, mask = prepare_autoregressive_batch(batch, pad_token_id=0)

        logits = model(inputs)
        loss = compute_language_modeling_loss(logits, targets, pad_token_id=0)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# 5. Evaluation
texts = test_generation(model, tokenizer, max_tokens=100, device="cuda")
metrics = diversity_report(texts, ns=(1, 2, 3))
```

---

## Performance Optimization Tips

### Memory Optimization
- Use flash attention (`use_flash=True`)
- Enable gradient checkpointing for deep models
- Use low-rank output projection for large vocabularies
- Stream large datasets instead of loading to RAM
- Use bfloat16 autocast during inference

### Speed Optimization
- Prepare datasets ahead of time (PreparedDataset)
- Use parallel dataset preparation (`num_workers>0`)
- Batch generation for evaluation
- Enable cudnn benchmarking: `torch.backends.cudnn.benchmark = True`

### Quality Optimization
- Use HybridEmbedding for better OOV handling
- Component-specific encoders (TokenToWaveEncoderImproved)
- Label smoothing (0.05) in loss computation
- Cosine LR schedule with warmup
- Diversity metrics for generation quality

---

## Error Handling and Edge Cases

### Common Issues

**1. Tokenizer Compatibility**
```python
# Ensure tokenizer has required attributes
tokenizer.no_truncation()
tokenizer.no_padding()
```

**2. Device Mismatch**
```python
# Ensure all tensors on same device
dataset = PreparedDataset(path, device=model.device)
```

**3. Sequence Length**
```python
# Ensure sequence_length matches model expectations
# encoder max_len >= dataset sequence_length
```

**4. Attention Masks**
```python
# Always use boolean masks
attention_mask = attention_mask.bool()
```

**5. Streaming Dataset Length**
```python
# Streaming datasets require max_entries
# Cannot use len() without it
```

---

## File Paths Summary

- **Embeddings:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\embeddings.py`
- **Token Encoders:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\token_encoder.py`
- **Token Decoders:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\token_decoder.py`
- **Training Utils:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\train_utils.py`
- **Datasets:** `E:\WaveML\Wave-Transformer\src\wave_transformer\language_modelling\text_datasets.py`

---

## Dependencies

**Core PyTorch:**
- `torch.nn` - Neural network layers
- `torch.nn.functional` - Functional operations
- `torch.utils.data` - Dataset utilities

**External:**
- `tokenizers` - HuggingFace tokenizers
- `datasets` - HuggingFace datasets
- `flash_attn` - Flash attention (optional)
- `numpy` - Numerical operations
- `tqdm` - Progress bars

**Internal:**
- `wave_transformer.core.transformer` - Wave dataclass, attention, normalization layers

---

This documentation provides comprehensive coverage of all PyTorch components in the language_modelling module, including detailed parameter specifications, tensor shapes, usage examples, and implementation notes for each component.
