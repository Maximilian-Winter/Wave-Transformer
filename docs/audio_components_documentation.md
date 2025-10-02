# Wave Transformer Audio Components Documentation

## Overview

This documentation covers all PyTorch components in the `src/wave_transformer/audio/` module, which provides a complete pipeline for audio processing using Wave representations. The module implements novel approaches for converting between raw audio waveforms and semantic wave representations, suitable for audio generation, speech synthesis, and audio understanding tasks.

### Module Structure

```
audio/
├── audio_dataset.py          # Dataset implementations for FLAC and VCTK
├── audio_wave_encoder.py     # Audio → Wave transformation
├── audio_wave_decoder.py     # Wave → Audio transformation
└── speaker_conditioning.py   # Speaker-conditioned encoding
```

---

## 1. Audio Datasets (`audio_dataset.py`)

### 1.1 FLACDataset

A simple PyTorch Dataset for loading FLAC audio files with automatic resampling.

#### Class Definition

```python
class FLACDataset(Dataset):
    def __init__(self, root_dir, sample_rate=24000, transform=None)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dir` | str | Required | Root directory containing FLAC files (recursive search) |
| `sample_rate` | int | 24000 | Target sample rate in Hz for resampling |
| `transform` | callable | None | Optional transform to apply to waveforms |

#### Returns

Dictionary containing:
- `waveform`: `torch.Tensor` of shape `(channels, samples)` - Raw audio waveform
- `path`: `str` - Full path to the audio file

#### Usage Example

```python
from wave_transformer.audio import FLACDataset

dataset = FLACDataset(
    root_dir="/path/to/audio",
    sample_rate=24000,
    transform=None
)

sample = dataset[0]
print(sample["waveform"].shape)  # torch.Size([1, N])
print(sample["path"])            # "/path/to/audio/file.flac"
```

#### Implementation Notes

- Uses `glob.glob()` with `recursive=True` to find all `.flac` files
- Automatically resamples to target sample rate using `torchaudio.functional.resample()`
- Preserves original channel configuration (mono/stereo)

---

### 1.2 VCTKAudioDataset

Dataset for VCTK Corpus with fixed-length audio clips and normalization.

#### Class Definition

```python
class VCTKAudioDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sample_rate=24000,
        max_len_sec=4,
        file_format: str = "wav",
        wav_folder: str = "wav48"
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dir` | str | Required | Path to VCTK dataset root directory |
| `sample_rate` | int | 24000 | Target sample rate in Hz |
| `max_len_sec` | int | 4 | Maximum audio length in seconds |
| `file_format` | str | "wav" | Audio file extension (e.g., "wav", "flac") |
| `wav_folder` | str | "wav48" | Subfolder name containing audio files |

#### Returns

`torch.Tensor` of shape `(1, samples)` where `samples = sample_rate * max_len_sec`

#### Preprocessing Pipeline

1. **Resampling**: Converts audio to target sample rate
2. **Mono conversion**: Averages all channels: `wav.mean(dim=0, keepdim=True)`
3. **Normalization**: Divides by absolute maximum: `wav / wav.abs().max()`
4. **Length adjustment**:
   - Truncates if longer than `max_len`
   - Zero-pads if shorter: `F.pad(wav, (0, pad_len))`

#### Usage Example

```python
dataset = VCTKAudioDataset(
    root_dir="/path/to/VCTK-Corpus",
    sample_rate=24000,
    max_len_sec=4
)

waveform = dataset[0]
print(waveform.shape)  # torch.Size([1, 96000])  # 24000 * 4
```

---

### 1.3 VCTKDataset

Complete VCTK dataset with text transcripts and speaker information.

#### Class Definition

```python
class VCTKDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sample_rate=24000,
        max_len_sec=4,
        return_text=True,
        file_format: str = "flac",
        wav_folder: str = "wav48_silence_trimmed",
        txt_folder: str = "txt"
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_dir` | str | Required | VCTK dataset root directory |
| `sample_rate` | int | 24000 | Target sample rate in Hz |
| `max_len_sec` | int | 4 | Audio clip length in seconds |
| `return_text` | bool | True | Whether to return transcripts |
| `file_format` | str | "flac" | Audio file format ("flac" or "wav") |
| `wav_folder` | str | "wav48_silence_trimmed" | Audio subfolder name |
| `txt_folder` | str | "txt" | Transcript subfolder name |

#### Returns

Dictionary containing:
- `waveform`: `torch.Tensor` of shape `(1, sample_rate * max_len_sec)` - Preprocessed audio
- `text`: `str` - Transcript text (if `return_text=True`)
- `speaker`: `str` - Speaker ID (e.g., "p225")
- `path`: `str` - Full path to audio file

#### Data Matching Logic

- Automatically matches audio files with transcripts
- Handles microphone suffix removal (e.g., `"p225_001_mic1.flac"` → `"p225_001.txt"`)
- Skips samples without transcripts or with empty transcripts
- Prints dataset size on initialization

#### Preprocessing Pipeline

1. **Resampling**: Converts to target sample rate
2. **Mono conversion**: Averages channels
3. **Normalization**: `wav / (wav.abs().max() + 1e-8)` (with numerical stability)
4. **Length adjustment**: Truncate or zero-pad to exact length

#### Usage Example

```python
dataset = VCTKDataset(
    root_dir="/path/to/VCTK-Corpus",
    sample_rate=24000,
    max_len_sec=4,
    return_text=True,
    file_format="flac"
)

sample = dataset[0]
print(sample["waveform"].shape)  # torch.Size([1, 96000])
print(sample["text"])            # "Please call Stella."
print(sample["speaker"])         # "p225"
```

---

### 1.4 VCTKCollator

DataLoader collate function for batching VCTK samples with optional text tokenization.

#### Class Definition

```python
class VCTKCollator:
    def __init__(self, tokenizer=None, return_text=True, device="cpu")
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | transformers.PreTrainedTokenizer | None | HuggingFace tokenizer for text |
| `return_text` | bool | True | Whether to tokenize and return text |
| `device` | str | "cpu" | Target device for tensors |

#### Returns

Dictionary containing:
- `waveforms`: `torch.Tensor` of shape `(batch_size, 1, samples)` - Batched waveforms
- `speakers`: `List[str]` - List of speaker IDs
- `input_ids`: `torch.Tensor` of shape `(batch_size, seq_len)` - Tokenized text (if tokenizer provided)
- `attention_mask`: `torch.Tensor` of shape `(batch_size, seq_len)` - Attention masks (if tokenizer provided)
- `transcripts`: `List[str]` - Raw transcript strings (if `return_text=True`)

#### Usage Example

```python
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("gpt2")
collator = VCTKCollator(
    tokenizer=tokenizer,
    return_text=True,
    device="cuda"
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator
)

batch = next(iter(loader))
print(batch["waveforms"].shape)      # torch.Size([8, 1, 96000])
print(batch["input_ids"].shape)      # torch.Size([8, seq_len])
print(batch["attention_mask"].shape) # torch.Size([8, seq_len])
print(batch["speakers"])             # ['p225', 'p226', ...]
```

---

### 1.5 VCTKCollatorSpeakerEmbedding

Enhanced collator that converts speaker IDs to numeric indices for embedding layers.

#### Class Definition

```python
class VCTKCollatorSpeakerEmbedding:
    def __init__(
        self,
        tokenizer=None,
        return_text=True,
        device="cpu",
        speaker2id=None
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | transformers.PreTrainedTokenizer | None | HuggingFace tokenizer |
| `return_text` | bool | True | Whether to tokenize text |
| `device` | str | "cpu" | Target device |
| `speaker2id` | Dict[str, int] | None | Speaker name to ID mapping (auto-created if None) |

#### Attributes

- `speaker2id`: `Dict[str, int]` - Mapping from speaker names to integer IDs
- `next_speaker_id`: `int` - Counter for assigning new speaker IDs

#### Returns

Dictionary containing:
- `waveforms`: `torch.Tensor` of shape `(batch_size, 1, samples)`
- `speakers`: `torch.LongTensor` of shape `(batch_size,)` - Numeric speaker IDs
- `input_ids`: `torch.Tensor` of shape `(batch_size, seq_len)` (if tokenizer provided)
- `attention_mask`: `torch.Tensor` of shape `(batch_size, seq_len)` (if tokenizer provided)
- `transcripts`: `List[str]` - Raw text (if `return_text=True`)

#### Implementation Notes

- Automatically builds speaker vocabulary on-the-fly
- Assigns consistent IDs across batches
- Thread-safe ID assignment using internal counter
- Compatible with `nn.Embedding` layers

#### Usage Example

```python
from torch import nn

collator = VCTKCollatorSpeakerEmbedding(
    tokenizer=tokenizer,
    return_text=True,
    device="cuda"
)

loader = DataLoader(dataset, batch_size=8, collate_fn=collator)
batch = next(iter(loader))

# Use with embedding layer
num_speakers = len(collator.speaker2id)
speaker_emb = nn.Embedding(num_speakers, 128).to("cuda")
speaker_vectors = speaker_emb(batch["speakers"])
print(speaker_vectors.shape)  # torch.Size([8, 128])
```

---

## 2. Audio Wave Encoder (`audio_wave_encoder.py`)

### 2.1 AudioToWave

Main encoder that transforms raw audio waveforms into semantic Wave representations with learned harmonic decomposition.

#### Architecture Overview

The encoder uses three possible approaches:
1. **Learnable Filterbank** (default): Adaptive frequency decomposition
2. **Mel Spectrogram**: Traditional mel-scale features projected to harmonics
3. **Raw Waveform**: Direct multi-scale CNN processing

#### Class Definition

```python
class AudioToWave(nn.Module):
    def __init__(
        self,
        num_harmonics: int = 64,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        learnable_filterbank: bool = True,
        use_raw_waveform: bool = False,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        freq_range: Tuple[float, float] = (20.0, 8000.0)
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_harmonics` | int | 64 | Number of harmonic components in Wave representation |
| `sample_rate` | int | 16000 | Audio sample rate in Hz |
| `n_fft` | int | 1024 | FFT size for STFT |
| `hop_length` | int | 256 | Hop length between STFT frames |
| `n_mels` | int | 128 | Number of mel bands (if not using learnable filterbank) |
| `learnable_filterbank` | bool | True | Use learnable filterbank instead of fixed mel |
| `use_raw_waveform` | bool | False | Process raw waveform directly with CNNs |
| `d_model` | int | 512 | Hidden dimension for semantic encoder |
| `num_heads` | int | 8 | Number of attention heads |
| `num_layers` | int | 3 | Number of semantic transform blocks |
| `dropout` | float | 0.1 | Dropout probability |
| `freq_range` | Tuple[float, float] | (20.0, 8000.0) | Valid frequency range in Hz |

#### Forward Method

```python
def forward(
    self,
    audio: torch.Tensor,
    return_raw_spectrum: bool = False
) -> Wave
```

**Input Shapes:**
- `audio`: `(batch_size, samples)` or `(batch_size, channels, samples)`

**Output:**
- Returns `Wave` object with:
  - `frequencies`: `(batch_size, time_steps, num_harmonics)` - Range [0.1, 20.1]
  - `amplitudes`: `(batch_size, time_steps, num_harmonics)` - Positive values via Softplus
  - `phases`: `(batch_size, time_steps, num_harmonics)` - Range [-π, π]

**Optional Output:**
- If `return_raw_spectrum=True`, returns tuple `(Wave, raw_spectrum)`

#### Processing Pipeline

1. **Channel Handling**:
   - Stereo → Mono: Averages channels if input is stereo
   - Squeezes channel dim if already mono

2. **Feature Extraction** (3 approaches):
   - **Learnable Filterbank**:
     - Compute STFT: `(batch, freq_bins, time)`
     - Apply learnable triangular filters
     - Output: `(batch, time, num_harmonics)` complex values

   - **Mel Spectrogram**:
     - Compute mel spectrogram
     - Project mel bands to harmonics via linear layer

   - **Raw Waveform**:
     - Multi-scale 1D convolutions (kernels: 3, 5, 7, 11, 17, 31)
     - Concatenate and project to harmonics

3. **Magnitude/Phase Decomposition**:
   - Complex features: `|z|` for magnitude, `∠z` for phase
   - Real features: ReLU for magnitude, learned transform for phase

4. **Temporal Attention**:
   - FlashAttention over time dimension (non-causal)
   - RMSNorm for stabilization

5. **Semantic Transformation**:
   - Stack of `SemanticTransformBlock` layers
   - Self-attention + FFN for pattern learning

6. **Component-Specific Transforms**:
   - **Frequencies**: `sigmoid(x) * 20.0 + 0.1` → [0.1, 20.1]
   - **Amplitudes**: `softplus(x)` → positive values
   - **Phases**: `tanh(x) * π` → [-π, π]

#### Usage Example

```python
import torch
from wave_transformer.audio import AudioToWave

encoder = AudioToWave(
    num_harmonics=64,
    sample_rate=16000,
    learnable_filterbank=True,
    num_layers=3
).to("cuda")

# Process audio
audio = torch.randn(2, 16000).to("cuda")  # 1 second
semantic_wave = encoder(audio)

print(semantic_wave.frequencies.shape)  # torch.Size([2, T, 64])
print(semantic_wave.amplitudes.shape)   # torch.Size([2, T, 64])
print(semantic_wave.phases.shape)       # torch.Size([2, T, 64])

# Convert to flat representation
wave_repr = semantic_wave.to_representation()
print(wave_repr.shape)  # torch.Size([2, T, 192])  # 64 * 3
```

#### Key Design Decisions

1. **Learnable Filterbank**: Allows model to discover optimal frequency decomposition
2. **Separate Component Transforms**: Independent processing of freq/amp/phase
3. **Temporal Attention**: Captures long-range dependencies in audio
4. **Semantic Encoder**: Transforms acoustic features to semantic representations
5. **Multiple Approaches**: Flexibility to use different feature extraction methods

#### Computational Complexity

- **Time Complexity**: O(B × T × H²) for attention, where B=batch, T=time, H=harmonics
- **Space Complexity**: O(B × T × H) for activations
- **STFT Complexity**: O(B × L × log(N)) where L=samples, N=n_fft

---

### 2.2 LearnableFilterbank

Adaptive frequency decomposition with learnable filter shapes and positions.

#### Class Definition

```python
class LearnableFilterbank(nn.Module):
    def __init__(
        self,
        num_filters: int = 64,
        sample_rate: int = 16000,
        freq_range: Tuple[float, float] = (20.0, 8000.0),
        n_fft: int = 1024,
        init_mel: bool = True
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_filters` | int | 64 | Number of learnable filters |
| `sample_rate` | int | 16000 | Audio sample rate |
| `freq_range` | Tuple[float, float] | (20.0, 8000.0) | Valid frequency range |
| `n_fft` | int | 1024 | FFT size |
| `init_mel` | bool | True | Initialize with mel-scale spacing |

#### Learnable Parameters

1. **Center Frequencies** (`center_freqs`):
   - Shape: `(num_filters,)`
   - Initialization: Mel-scale or linear spacing converted to FFT bins
   - Represents peak response frequency for each filter

2. **Bandwidths** (`bandwidths`):
   - Shape: `(num_filters,)`
   - Initialization: `n_fft / num_filters / 4`
   - Controls filter width via `softplus()` activation

3. **Filter Shapes** (`filter_shapes`):
   - Shape: `(num_filters, 2)`
   - Initialization: `ones()`
   - Controls asymmetric slopes (left/right) via `softplus()`

#### Forward Method

```python
def forward(self, spectrum: torch.Tensor) -> torch.Tensor
```

**Input:**
- `spectrum`: `(batch_size, freq_bins, time_steps)` - Complex STFT

**Output:**
- `(batch_size, time_steps, num_filters)` - Complex filtered values

#### Filter Construction Algorithm

For each filter `i`:

1. **Extract parameters**:
   ```python
   center = center_freqs[i]
   bandwidth = softplus(bandwidths[i]) + 1e-3
   left_slope, right_slope = softplus(filter_shapes[i]) + 0.1
   ```

2. **Compute edges**:
   ```python
   left_edge = center - bandwidth * left_slope
   right_edge = center + bandwidth * right_slope
   ```

3. **Build triangular response**:
   - Rising edge: `(freq - left_edge) / (center - left_edge)` for `freq ∈ [left_edge, center]`
   - Falling edge: `1 - (freq - center) / (right_edge - center)` for `freq ∈ (center, right_edge]`
   - Elsewhere: 0

4. **Apply to spectrum**:
   ```python
   filtered = matmul(filterbank, spectrum)  # Preserves complex values
   ```

#### Key Features

- **Asymmetric Filters**: Left and right slopes learned independently
- **Complex-Valued**: Preserves phase information from STFT
- **Differentiable**: All operations support gradient flow
- **Mel Initialization**: Warm start with perceptually-motivated spacing

#### Usage Example

```python
filterbank = LearnableFilterbank(
    num_filters=64,
    sample_rate=16000,
    freq_range=(20.0, 8000.0),
    n_fft=1024,
    init_mel=True
)

# Apply to STFT
stft = torch.stft(audio, n_fft=1024, return_complex=True)
# stft shape: (batch, freq_bins, time)

filtered = filterbank(stft)
# filtered shape: (batch, time, 64) - complex values

magnitudes = torch.abs(filtered)
phases = torch.angle(filtered)
```

---

### 2.3 WaveformEncoder

Direct waveform processing using multi-scale convolutions (alternative to spectral methods).

#### Class Definition

```python
class WaveformEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_harmonics: int = 64,
        kernel_sizes: list = [3, 5, 7, 11, 17, 31]
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | 512 | Total feature dimension (distributed across scales) |
| `num_harmonics` | int | 64 | Number of harmonic components |
| `kernel_sizes` | list | [3,5,7,11,17,31] | Conv kernel sizes for multi-scale processing |

#### Architecture

1. **Multi-scale Convolutions**:
   - Each kernel size gets `d_model // len(kernel_sizes)` channels
   - Parallel 1D convolutions with different receptive fields
   - Captures features at multiple temporal scales

2. **Feature Combination**:
   - Concatenate all scales → `d_model` channels
   - 1×1 convolution → ReLU → 1×1 convolution
   - Final output: `num_harmonics * 3` channels (freq, amp, phase)

#### Forward Method

```python
def forward(self, waveform: torch.Tensor) -> torch.Tensor
```

**Input:**
- `waveform`: `(batch_size, samples)` - Raw audio

**Output:**
- `(batch_size, time_steps, num_harmonics * 3)` - Multi-scale features

#### Processing Steps

1. Add channel dimension: `(B, samples)` → `(B, 1, samples)`
2. Apply each conv in parallel
3. Concatenate: `(B, d_model, time)`
4. Combine and project: `(B, harmonics*3, time)`
5. Transpose: `(B, time, harmonics*3)`

#### Usage Example

```python
encoder = WaveformEncoder(
    d_model=512,
    num_harmonics=64,
    kernel_sizes=[3, 5, 7, 11, 17, 31]
)

waveform = torch.randn(4, 16000)  # 1 second
features = encoder(waveform)
print(features.shape)  # torch.Size([4, time, 192])  # 64*3
```

#### Design Rationale

- **Multi-scale**: Different kernels capture different temporal patterns
- **No FFT**: Learns features directly from raw waveform
- **Learnable**: All frequency decomposition is learned end-to-end
- **Efficient**: Single forward pass, no iterative algorithms

---

### 2.4 SemanticTransformBlock

Transformer block for converting acoustic patterns to semantic representations.

#### Class Definition

```python
class SemanticTransformBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | Required | Feature dimension |
| `num_heads` | int | 8 | Number of attention heads |
| `dropout` | float | 0.1 | Dropout probability |

#### Architecture

```
Input
  ↓
RMSNorm → FlashAttention (non-causal) → Residual
  ↓
RMSNorm → FFN (Linear → GELU → Dropout → Linear → Dropout) → Residual
  ↓
Output
```

#### Components

1. **RMSNorm**: Efficient normalization (faster than LayerNorm)
2. **FlashAttention**: Memory-efficient self-attention
3. **FFN**: 4× expansion ratio (d_model → 4×d_model → d_model)
4. **GELU**: Smooth activation function

#### Forward Method

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Input/Output:** `(batch_size, seq_len, d_model)`

#### Usage Example

```python
block = SemanticTransformBlock(
    d_model=192,  # num_harmonics * 3
    num_heads=8,
    dropout=0.1
)

features = torch.randn(2, 100, 192)
transformed = block(features)
print(transformed.shape)  # torch.Size([2, 100, 192])
```

---

## 3. Audio Wave Decoder (`audio_wave_decoder.py`)

### 3.1 WaveToAudio

Main decoder that transforms Wave representations back into audio waveforms.

#### Class Definition

```python
class WaveToAudio(nn.Module):
    def __init__(
        self,
        num_harmonics: int = 64,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        synthesis_method: str = "griffin_lim",
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_harmonics` | int | 64 | Number of harmonic components |
| `sample_rate` | int | 16000 | Audio sample rate in Hz |
| `n_fft` | int | 1024 | FFT size for synthesis |
| `hop_length` | int | 256 | Hop length for ISTFT |
| `synthesis_method` | str | "griffin_lim" | Synthesis method: "griffin_lim", "learned", or "neural_vocoder" |
| `d_model` | int | 512 | Hidden dimension |
| `num_heads` | int | 8 | Number of attention heads |
| `num_layers` | int | 3 | Number of acoustic transform blocks |
| `dropout` | float | 0.1 | Dropout probability |

#### Synthesis Methods

1. **"griffin_lim"** (Default):
   - Learned harmonic-to-frequency mapping
   - Learned phase prediction
   - Griffin-Lim iterations for phase reconstruction
   - Good balance of quality and speed

2. **"learned"**:
   - Fully learned synthesis network
   - Direct harmonic → spectrogram mapping
   - Single ISTFT (no iterations)
   - Faster inference

3. **"neural_vocoder"**:
   - WaveNet-style architecture
   - Harmonic synthesis → upsampling → waveform
   - Highest quality potential
   - More parameters

#### Forward Method

```python
def forward(
    self,
    semantic_wave: Wave,
    target_length: Optional[int] = None
) -> torch.Tensor
```

**Input:**
- `semantic_wave`: Wave object with frequencies, amplitudes, phases
- `target_length`: Optional desired output length in samples

**Output:**
- `torch.Tensor` of shape `(batch_size, samples)` - Audio waveform

#### Processing Pipeline

1. **Acoustic Transformation**:
   - Convert Wave to representation: `(batch, time, harmonics*3)`
   - Apply `AcousticTransformBlock` layers (semantic → acoustic)

2. **Synthesis** (method-dependent):
   - **Griffin-Lim**: Map harmonics → freq bins → iterative phase reconstruction
   - **Learned**: Direct network mapping → ISTFT
   - **Neural Vocoder**: Harmonic synthesis → upsampling layers → waveform

3. **Post-Processing**:
   - Audio enhancement (denoising)
   - Temporal smoothing (5-tap conv)
   - Length adjustment (pad/trim to target)

#### Usage Example

```python
from wave_transformer.audio import WaveToAudio
from wave_transformer.core.transformer import Wave

decoder = WaveToAudio(
    num_harmonics=64,
    sample_rate=16000,
    synthesis_method="griffin_lim",
    num_layers=3
).to("cuda")

# Create or get semantic wave
frequencies = torch.sigmoid(torch.randn(2, 63, 64)) * 20.0 + 0.1
amplitudes = F.softplus(torch.randn(2, 63, 64))
phases = torch.tanh(torch.randn(2, 63, 64)) * np.pi

semantic_wave = Wave(
    frequencies.cuda(),
    amplitudes.cuda(),
    phases.cuda()
)

# Generate audio
audio = decoder(semantic_wave, target_length=16000)
print(audio.shape)  # torch.Size([2, 16000])
print(audio.min(), audio.max())  # Check range
```

#### Synthesis Method Details

**Griffin-Lim Synthesis:**
```python
def _griffin_lim_synthesis(self, wave_repr, semantic_wave):
    # Map harmonics to frequency magnitude
    freq_magnitude = self.freq_mapping(semantic_wave)

    # Predict phase
    phase_input = torch.cat([
        semantic_wave.frequencies,
        semantic_wave.phases
    ], dim=-1)
    predicted_phase = self.harmonic_to_phase(phase_input) * π

    # Build complex spectrogram
    complex_spec = freq_magnitude * exp(1j * predicted_phase)

    # Griffin-Lim iterations (30 iterations)
    audio = self._griffin_lim_iterations(complex_spec, n_iters=30)
    return audio
```

**Learned Synthesis:**
```python
def _learned_synthesis(self, wave_repr, semantic_wave):
    # Direct network mapping
    freq_repr = self.harmonic_to_freq(wave_repr)
    magnitude, phase = freq_repr.chunk(2, dim=-1)

    # Ensure proper ranges
    magnitude = softplus(magnitude)
    phase = tanh(phase) * π

    # Build complex spec and ISTFT
    complex_spec = magnitude * exp(1j * phase)
    audio = torch.istft(complex_spec, ...)
    return audio
```

**Neural Vocoder Synthesis:**
```python
def _neural_vocoder_synthesis(self, wave_repr, semantic_wave):
    # Extract harmonic features
    features = self.harmonic_to_features(
        semantic_wave.frequencies,
        semantic_wave.amplitudes,
        semantic_wave.phases
    )

    # Decode to waveform with upsampling
    audio = self.waveform_decoder(features, wave_repr)
    return audio
```

#### Key Design Decisions

1. **Multiple Synthesis Options**: Flexibility for quality/speed trade-offs
2. **Acoustic Transform**: Reverse semantic → acoustic transformation
3. **Enhancement Networks**: Post-processing for quality improvement
4. **Length Control**: Exact output length via padding/trimming
5. **Learned Phase**: Better initialization for Griffin-Lim

#### Computational Complexity

- **Griffin-Lim**: O(I × B × F × T) where I=iterations, F=freq_bins, T=time
- **Learned**: O(B × T × H × F) for network forward
- **Neural Vocoder**: O(B × T × U × C) where U=upsampling_factor, C=channels

---

### 3.2 LearnableFrequencyMapping

Maps semantic harmonics to frequency bin magnitudes.

#### Class Definition

```python
class LearnableFrequencyMapping(nn.Module):
    def __init__(
        self,
        num_harmonics: int,
        freq_bins: int,
        sample_rate: int,
        n_fft: int
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_harmonics` | int | Required | Number of harmonic components |
| `freq_bins` | int | Required | Number of frequency bins (n_fft//2 + 1) |
| `sample_rate` | int | Required | Audio sample rate |
| `n_fft` | int | Required | FFT size |

#### Learnable Parameters

1. **Harmonic-to-Frequency Matrix** (`harmonic_to_freq_matrix`):
   - Shape: `(num_harmonics, freq_bins)`
   - Initialization: Sparse with peaks at evenly-spaced frequencies
   - Maps each harmonic to frequency bin contributions

2. **Spread** (`spread`):
   - Shape: `(num_harmonics,)`
   - Initialization: `2.0`
   - Controls how much each harmonic affects neighboring frequencies

#### Forward Method

```python
def forward(self, semantic_wave: Wave) -> torch.Tensor
```

**Input:** Wave object

**Output:** `(batch_size, time_steps, freq_bins)` - Magnitude spectrum

#### Mapping Algorithm

```python
# Weight matrix by amplitudes
weighted_matrix = harmonic_to_freq_matrix * amplitudes.unsqueeze(-1)

# Apply spread
spread_matrix = softplus(spread).unsqueeze(-1)
weighted_matrix *= spread_matrix

# Sum harmonic contributions
magnitude_spectrum = weighted_matrix.sum(dim=-2)

# Ensure positive
magnitude_spectrum = softplus(magnitude_spectrum)
```

#### Usage Example

```python
mapping = LearnableFrequencyMapping(
    num_harmonics=64,
    freq_bins=513,  # 1024//2 + 1
    sample_rate=16000,
    n_fft=1024
)

semantic_wave = Wave(freqs, amps, phases)
magnitude = mapping(semantic_wave)
print(magnitude.shape)  # torch.Size([batch, time, 513])
```

---

### 3.3 HarmonicSynthesizer

Neural synthesis from harmonic components (used in neural vocoder mode).

#### Class Definition

```python
class HarmonicSynthesizer(nn.Module):
    def __init__(
        self,
        num_harmonics: int,
        freq_bins: int,
        d_model: int
    )
```

#### Architecture

1. **Oscillator Network**:
   ```python
   Linear(harmonics*3, d_model) → ReLU → Linear(d_model, harmonics) → Tanh
   ```
   - Generates oscillation patterns from freq/amp/phase

2. **Mixer Network**:
   ```python
   Linear(harmonics*2, d_model) → ReLU → Linear(d_model, freq_bins)
   ```
   - Combines oscillations with phases to produce spectrum

#### Forward Method

```python
def forward(
    self,
    frequencies: torch.Tensor,
    amplitudes: torch.Tensor,
    phases: torch.Tensor
) -> torch.Tensor
```

**Inputs:** Each `(batch_size, time_steps, num_harmonics)`

**Output:** `(batch_size, time_steps, freq_bins)` - Spectrum magnitudes

#### Processing Flow

```python
# Combine inputs
harmonic_input = cat([frequencies, amplitudes, phases], dim=-1)

# Generate oscillations
oscillations = oscillator_net(harmonic_input)

# Mix with amplitudes and phases
mixed = cat([oscillations * amplitudes, phases], dim=-1)

# Generate spectrum
spectrum = mixer(mixed)
return softplus(spectrum)  # Ensure positive
```

---

### 3.4 WaveformDecoder

Direct waveform generation using transposed convolutions (neural vocoder approach).

#### Class Definition

```python
class WaveformDecoder(nn.Module):
    def __init__(
        self,
        freq_bins: int,
        d_model: int,
        sample_rate: int,
        hop_length: int,
        num_layers: int = 4
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `freq_bins` | int | Required | Input feature dimension |
| `d_model` | int | Required | Hidden dimension |
| `sample_rate` | int | Required | Target sample rate |
| `hop_length` | int | Required | Upsampling factor |
| `num_layers` | int | 4 | Number of upsampling layers |

#### Architecture

Hierarchical upsampling using ConvTranspose1d:

```
Layer 0: freq_bins → d_model, upsample by hop_length/2
Layer 1: d_model → d_model, upsample by hop_length/4
Layer 2: d_model → d_model, upsample by hop_length/8
Layer 3: d_model → d_model, upsample by hop_length/16
Final: d_model → 1 (waveform)
```

#### Forward Method

```python
def forward(
    self,
    features: torch.Tensor,
    context: torch.Tensor
) -> torch.Tensor
```

**Inputs:**
- `features`: `(batch_size, time_steps, freq_bins)` - Spectral features
- `context`: `(batch_size, time_steps, context_dim)` - Additional context

**Output:** `(batch_size, samples)` - Waveform in range [-1, 1]

#### Usage Example

```python
decoder = WaveformDecoder(
    freq_bins=513,
    d_model=512,
    sample_rate=16000,
    hop_length=256,
    num_layers=4
)

features = torch.randn(2, 63, 513)  # Frame-rate features
context = torch.randn(2, 63, 192)   # Context
waveform = decoder(features, context)
print(waveform.shape)  # torch.Size([2, ~16000])
```

---

### 3.5 AudioEnhancement

Post-processing enhancement network for denoising and quality improvement.

#### Class Definition

```python
class AudioEnhancement(nn.Module):
    def __init__(
        self,
        d_model: int,
        sample_rate: int
    )
```

#### Architecture

Denoising network using 1D convolutions:

```
Conv1d(1, d_model//4, kernel=15) → ReLU
  ↓
Conv1d(d_model//4, d_model//4, kernel=15) → ReLU
  ↓
Conv1d(d_model//4, 1, kernel=15)
  ↓
Residual connection with 0.1 weight
  ↓
Soft clipping: tanh(x/3) * 3
```

#### Forward Method

```python
def forward(self, audio: torch.Tensor) -> torch.Tensor
```

**Input/Output:** `(batch_size, samples)` - Audio waveform

#### Enhancement Process

```python
# Denoise
residual = denoise_network(audio.unsqueeze(1)).squeeze(1)

# Weak residual connection (avoid over-smoothing)
enhanced = audio + residual * 0.1

# Soft clipping (prevent clipping artifacts)
enhanced = tanh(enhanced / 3.0) * 3.0
```

#### Design Rationale

- **Residual Learning**: Predicts noise to subtract (easier than clean signal)
- **Weak Residual**: 0.1 weight preserves original signal characteristics
- **Soft Clipping**: Smooth limiting prevents harsh distortion
- **Large Kernels**: 15-tap filters capture longer-range artifacts

---

### 3.6 AcousticTransformBlock

Transformer block for semantic → acoustic transformation.

#### Class Definition

```python
class AcousticTransformBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    )
```

#### Architecture

Identical to `SemanticTransformBlock` but semantically represents the inverse transformation:

```
RMSNorm → FlashAttention (non-causal) → Residual
  ↓
RMSNorm → FFN (4× expansion) → Residual
```

#### Usage

Applied in decoder to transform semantic Wave representations back to acoustic features suitable for audio synthesis.

---

## 4. Speaker Conditioning (`speaker_conditioning.py`)

### 4.1 SpeakerConditionedAudioToWave

Speaker-aware audio encoder that produces speaker-specific Wave representations.

#### Class Definition

```python
class SpeakerConditionedAudioToWave(nn.Module):
    def __init__(
        self,
        base_audio_encoder: AudioToWave,
        num_speakers: int = 110,
        speaker_dim: int = 128,
        conditioning_type: str = "film"
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_audio_encoder` | AudioToWave | Required | Base encoder to condition |
| `num_speakers` | int | 110 | Number of unique speakers (VCTK has 110) |
| `speaker_dim` | int | 128 | Speaker embedding dimension |
| `conditioning_type` | str | "film" | Conditioning method: "film" or "concat" |

#### Conditioning Methods

1. **FiLM (Feature-wise Linear Modulation)** - Recommended:
   - Learns affine transformations for each Wave component
   - Separate γ (scale) and β (shift) for frequencies, amplitudes, phases
   - Formula: `output = γ * input + β`
   - More parameter efficient
   - Better generalization

2. **Concatenation**:
   - Concatenates speaker embedding to Wave representation
   - Projects combined features back to original dimension
   - Simpler but requires more parameters

#### Architecture (FiLM mode)

```
Speaker ID → Embedding(num_speakers, speaker_dim)
  ↓
Speaker Emb → Linear(speaker_dim, harmonics*2) → split → γ_freq, β_freq
Speaker Emb → Linear(speaker_dim, harmonics*2) → split → γ_amp, β_amp
Speaker Emb → Linear(speaker_dim, harmonics*2) → split → γ_phase, β_phase
```

#### Forward Method

```python
def forward(
    self,
    waveforms: torch.Tensor,
    speaker_ids: torch.LongTensor
) -> Wave
```

**Inputs:**
- `waveforms`: `(batch_size, samples)` or `(batch_size, 1, samples)` - Audio
- `speaker_ids`: `(batch_size,)` - Integer speaker IDs

**Output:** Wave object with speaker-conditioned harmonics

#### Processing Flow (FiLM)

```python
# Get base semantic waves (speaker-agnostic)
semantic_waves = base_audio_encoder(waveforms)

# Get speaker embeddings
speaker_emb = speaker_emb_layer(speaker_ids)  # (B, speaker_dim)

# Compute FiLM parameters for each component
freq_γ, freq_β = freq_film(speaker_emb).chunk(2, dim=-1)
amp_γ, amp_β = amp_film(speaker_emb).chunk(2, dim=-1)
phase_γ, phase_β = phase_film(speaker_emb).chunk(2, dim=-1)

# Modulate waves (broadcast over time dimension)
semantic_waves.frequencies = (
    semantic_waves.frequencies * freq_γ.unsqueeze(1) + freq_β.unsqueeze(1)
)
semantic_waves.amplitudes = (
    semantic_waves.amplitudes * amp_γ.unsqueeze(1) + amp_β.unsqueeze(1)
)
semantic_waves.phases = (
    semantic_waves.phases * phase_γ.unsqueeze(1) + phase_β.unsqueeze(1)
)

return semantic_waves
```

#### Usage Example

```python
from wave_transformer.audio import AudioToWave, SpeakerConditionedAudioToWave

# Base encoder
base_encoder = AudioToWave(
    num_harmonics=64,
    sample_rate=24000,
    learnable_filterbank=True
)

# Speaker-conditioned encoder
speaker_encoder = SpeakerConditionedAudioToWave(
    base_audio_encoder=base_encoder,
    num_speakers=110,  # VCTK speakers
    speaker_dim=128,
    conditioning_type="film"
).to("cuda")

# Process audio with speaker info
waveforms = torch.randn(8, 96000).cuda()  # 4 seconds at 24kHz
speaker_ids = torch.randint(0, 110, (8,)).cuda()

speaker_waves = speaker_encoder(waveforms, speaker_ids)
print(speaker_waves.frequencies.shape)  # torch.Size([8, T, 64])
```

#### Training Pipeline Example

```python
from torch.utils.data import DataLoader
from wave_transformer.audio import VCTKDataset, VCTKCollatorSpeakerEmbedding

# Dataset
dataset = VCTKDataset(
    "/path/to/VCTK-Corpus",
    sample_rate=24000,
    max_len_sec=4,
    return_text=True
)

# Collator with speaker ID conversion
collator = VCTKCollatorSpeakerEmbedding(
    tokenizer=tokenizer,
    return_text=True,
    device="cuda"
)

loader = DataLoader(dataset, batch_size=8, collate_fn=collator)

# Training loop
for batch in loader:
    waveforms = batch["waveforms"].squeeze(1)
    speaker_ids = batch["speakers"]

    # Get speaker-conditioned waves
    semantic_waves = speaker_encoder(waveforms, speaker_ids)

    # Use in downstream task (reconstruction, TTS, etc.)
    # ...
```

#### Key Benefits

1. **Speaker Disentanglement**: Separates content from speaker identity
2. **Few-shot Adaptation**: Can adapt to new speakers with few examples
3. **Voice Conversion**: Enables changing speaker while preserving content
4. **Multi-speaker TTS**: Single model for all speakers
5. **Speaker Recognition**: Embeddings encode speaker characteristics

#### Comparison: FiLM vs Concatenation

| Aspect | FiLM | Concatenation |
|--------|------|---------------|
| Parameters | O(S × H) | O(S × H + H²) |
| Expressiveness | High (multiplicative) | Medium (additive) |
| Generalization | Better | Good |
| Training Speed | Faster | Slower |
| Recommended | ✓ | For simple cases |

Where S = speaker_dim, H = num_harmonics

---

### 4.2 SpeakerConditionedEncoder

Alternative speaker conditioning approach (from original implementation).

#### Class Definition

```python
class SpeakerConditionedEncoder(nn.Module):
    def __init__(
        self,
        audio_encoder,
        num_speakers,
        speaker_dim=128
    )
```

#### Note

This is an older/simpler implementation that concatenates speaker embeddings with encoder features. The `SpeakerConditionedAudioToWave` class is recommended for new projects as it's specifically designed for Wave representations.

---

### 4.3 FiLMConditionedEncoder

Standalone FiLM conditioning module.

#### Class Definition

```python
class FiLMConditionedEncoder(nn.Module):
    def __init__(
        self,
        audio_encoder,
        num_speakers,
        speaker_dim=128
    )
```

#### Usage

Applies FiLM to generic encoder features. Can be used as a building block for custom architectures.

---

## 5. Core Dependencies

### 5.1 Wave Dataclass

Fundamental data structure for wave representations.

```python
from dataclasses import dataclass

@dataclass
class Wave:
    frequencies: torch.Tensor  # (batch, time, harmonics)
    amplitudes: torch.Tensor   # (batch, time, harmonics)
    phases: torch.Tensor       # (batch, time, harmonics)

    def to_representation(self) -> torch.Tensor:
        """Concatenate components: (batch, time, harmonics*3)"""
        return torch.cat([
            self.frequencies,
            self.amplitudes,
            self.phases
        ], dim=-1)

    @classmethod
    def from_representation(cls, x: torch.Tensor):
        """Split representation back to Wave"""
        chunks = x.chunk(3, dim=-1)
        return cls(
            frequencies=chunks[0],
            amplitudes=chunks[1],
            phases=chunks[2]
        )

    def synthesize(self, t: torch.Tensor) -> torch.Tensor:
        """Generate wave signal at time points t"""
        t = t.unsqueeze(-1)
        return (self.amplitudes * torch.sin(
            2 * np.pi * self.frequencies * t + self.phases
        )).sum(dim=-1)
```

---

### 5.2 RMSNorm

Root Mean Square Layer Normalization (used in LLaMA, T5).

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / norm
```

**Benefits:**
- Simpler than LayerNorm (no bias, no mean subtraction)
- Faster computation
- Often better performance
- More stable gradients

---

### 5.3 FlashAttention

Memory-efficient attention implementation.

```python
class FlashAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, causal=False):
        # Uses flash_attn_func for efficiency
        # ...
```

**Features:**
- O(N) memory complexity (vs O(N²) for standard attention)
- Kernel fusion for speed
- Causal masking support
- Dropout support

---

## 6. Complete Pipeline Example

### End-to-End Audio Processing

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from wave_transformer.audio import (
    VCTKDataset,
    VCTKCollatorSpeakerEmbedding,
    AudioToWave,
    WaveToAudio,
    SpeakerConditionedAudioToWave
)

# 1. Setup dataset
dataset = VCTKDataset(
    root_dir="/path/to/VCTK-Corpus",
    sample_rate=24000,
    max_len_sec=4,
    return_text=True,
    file_format="flac"
)

# 2. Setup collator
tokenizer = AutoTokenizer.from_pretrained("gpt2")
collator = VCTKCollatorSpeakerEmbedding(
    tokenizer=tokenizer,
    return_text=True,
    device="cuda"
)

# 3. Create data loader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collator,
    num_workers=4
)

# 4. Initialize encoder
base_encoder = AudioToWave(
    num_harmonics=64,
    sample_rate=24000,
    learnable_filterbank=True,
    num_layers=3,
    dropout=0.1
).cuda()

# 5. Add speaker conditioning
speaker_encoder = SpeakerConditionedAudioToWave(
    base_audio_encoder=base_encoder,
    num_speakers=110,
    speaker_dim=128,
    conditioning_type="film"
).cuda()

# 6. Initialize decoder
decoder = WaveToAudio(
    num_harmonics=64,
    sample_rate=24000,
    synthesis_method="griffin_lim",
    num_layers=3
).cuda()

# 7. Training loop
for batch in loader:
    waveforms = batch["waveforms"].squeeze(1)  # (B, samples)
    speaker_ids = batch["speakers"]             # (B,)
    input_ids = batch["input_ids"]             # (B, seq_len)

    # Encode audio to waves (speaker-conditioned)
    semantic_waves = speaker_encoder(waveforms, speaker_ids)

    # Process with transformer (not shown here)
    # processed_waves = wave_transformer(semantic_waves, input_ids)

    # Decode back to audio
    reconstructed = decoder(semantic_waves, target_length=96000)

    # Compute loss
    loss = F.mse_loss(reconstructed, waveforms)

    # Backward pass
    loss.backward()
    # optimizer.step() ...
```

---

## 7. Performance Considerations

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| AudioToWave | ~10-50 MB | Depends on num_harmonics, num_layers |
| WaveToAudio | ~10-80 MB | Varies by synthesis_method |
| Speaker Embedding | ~1 MB per 100 speakers | For speaker_dim=128 |
| FlashAttention | O(N) | vs O(N²) for standard attention |

### Speed Benchmarks (Approximate)

**Encoding (AudioToWave):**
- Learnable filterbank: ~50-100 ms/second of audio (GPU)
- Mel spectrogram: ~30-60 ms/second of audio (GPU)
- Raw waveform: ~40-80 ms/second of audio (GPU)

**Decoding (WaveToAudio):**
- Griffin-Lim: ~100-200 ms/second of audio (30 iterations)
- Learned synthesis: ~30-60 ms/second of audio
- Neural vocoder: ~50-120 ms/second of audio

### Optimization Tips

1. **Batch Processing**: Process multiple samples together
2. **Mixed Precision**: Use `torch.cuda.amp` for 2× speedup
3. **Gradient Checkpointing**: Reduce memory for long sequences
4. **Synthesis Method**: Choose based on quality/speed trade-off
5. **Reduce Harmonics**: Fewer harmonics = faster but less detail
6. **Reduce Layers**: Fewer transformer layers = faster but less capacity

---

## 8. Common Use Cases

### 8.1 Audio Reconstruction

```python
# Encode then decode
wave = encoder(audio)
reconstructed = decoder(wave)
loss = F.mse_loss(reconstructed, audio)
```

### 8.2 Voice Conversion

```python
# Change speaker while preserving content
source_wave = speaker_encoder(source_audio, source_speaker_id)
target_wave = modify_speaker(source_wave, target_speaker_id)
converted_audio = decoder(target_wave)
```

### 8.3 Text-to-Speech

```python
# Generate wave from text, then synthesize
text_features = text_encoder(input_ids)
predicted_wave = wave_predictor(text_features, speaker_id)
audio = decoder(predicted_wave)
```

### 8.4 Audio Enhancement

```python
# Encode noisy audio, clean in wave space, decode
noisy_wave = encoder(noisy_audio)
clean_wave = denoiser(noisy_wave)
clean_audio = decoder(clean_wave)
```

---

## 9. Troubleshooting

### Common Issues

**Issue: NaN losses during training**
- **Cause**: Unstable gradients in phase/frequency learning
- **Solution**: Lower learning rate, use gradient clipping, check normalization

**Issue: Poor audio quality**
- **Cause**: Insufficient harmonics or layers
- **Solution**: Increase `num_harmonics` to 128+, add more layers

**Issue: Slow inference**
- **Cause**: Griffin-Lim iterations or neural vocoder
- **Solution**: Switch to "learned" synthesis, reduce harmonics

**Issue: Speaker bleeding (multi-speaker)**
- **Cause**: Weak speaker conditioning
- **Solution**: Increase `speaker_dim`, use FiLM instead of concat

**Issue: Out of memory**
- **Cause**: Large batch size or long audio
- **Solution**: Reduce batch size, use gradient accumulation, shorter clips

---

## 10. Advanced Topics

### Custom Harmonic Initialization

```python
# Initialize with specific frequency bands
class CustomAudioToWave(AudioToWave):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize with perceptually-important frequencies
        with torch.no_grad():
            important_freqs = [100, 200, 400, 800, 1600, 3200, 6400]
            for i, freq in enumerate(important_freqs):
                if i < self.num_harmonics:
                    self.filterbank.center_freqs[i] = (
                        freq * self.n_fft / self.sample_rate
                    )
```

### Multi-Scale Synthesis

```python
# Combine multiple synthesis methods
class MultiScaleSynthesis(nn.Module):
    def __init__(self, num_harmonics, sample_rate):
        super().__init__()
        self.coarse = WaveToAudio(
            num_harmonics=num_harmonics//2,
            synthesis_method="griffin_lim"
        )
        self.fine = WaveToAudio(
            num_harmonics=num_harmonics,
            synthesis_method="learned"
        )

    def forward(self, wave):
        coarse_audio = self.coarse(wave)
        fine_audio = self.fine(wave)
        return coarse_audio + 0.3 * fine_audio
```

### Conditional Wave Generation

```python
# Condition on external features (emotion, style, etc.)
class ConditionalWaveGenerator(nn.Module):
    def __init__(self, base_encoder, condition_dim):
        super().__init__()
        self.encoder = base_encoder
        self.condition_proj = nn.Linear(
            condition_dim,
            base_encoder.num_harmonics * 3
        )

    def forward(self, audio, condition):
        wave = self.encoder(audio)
        condition_wave = self.condition_proj(condition)
        # Add condition as residual
        wave_repr = wave.to_representation() + condition_wave
        return Wave.from_representation(wave_repr)
```

---

## 11. File Paths Reference

All components are located in: `E:\WaveML\Wave-Transformer\src\wave_transformer\audio\`

- **Datasets**: `audio_dataset.py`
- **Encoder**: `audio_wave_encoder.py`
- **Decoder**: `audio_wave_decoder.py`
- **Speaker Conditioning**: `speaker_conditioning.py`

Core dependencies: `E:\WaveML\Wave-Transformer\src\wave_transformer\core\transformer.py`

---

## 12. API Summary

### Key Classes

| Class | Purpose | Key Parameters |
|-------|---------|----------------|
| `AudioToWave` | Audio → Wave encoding | `num_harmonics`, `learnable_filterbank`, `num_layers` |
| `WaveToAudio` | Wave → Audio decoding | `synthesis_method`, `num_layers` |
| `SpeakerConditionedAudioToWave` | Speaker-aware encoding | `num_speakers`, `conditioning_type` |
| `VCTKDataset` | VCTK data loading | `sample_rate`, `max_len_sec`, `return_text` |
| `VCTKCollatorSpeakerEmbedding` | Batching with speaker IDs | `tokenizer`, `speaker2id` |

### Key Methods

| Method | Input | Output |
|--------|-------|--------|
| `AudioToWave.forward()` | `(B, samples)` | `Wave(freqs, amps, phases)` |
| `WaveToAudio.forward()` | `Wave` | `(B, samples)` |
| `Wave.to_representation()` | - | `(B, T, harmonics*3)` |
| `Wave.from_representation()` | `(B, T, harmonics*3)` | `Wave` |
| `Wave.synthesize()` | `time_points` | `(B, T)` signal |

---

## Conclusion

This documentation covers the complete audio processing pipeline in the Wave Transformer project. The modular design allows for:

- **Flexible audio encoding** (spectral, learnable, or raw waveform)
- **Multiple synthesis methods** (Griffin-Lim, learned, neural vocoder)
- **Speaker conditioning** (FiLM or concatenation)
- **Easy integration** with transformer models

For questions or contributions, refer to the source code and inline comments for additional implementation details.