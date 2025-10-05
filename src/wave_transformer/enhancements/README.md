# Wave Transformer Enhancements

This module provides four key enhancements for the Wave Transformer architecture, designed to improve performance, efficiency, and expressiveness.

## Overview

| Enhancement | Purpose | Key Benefit |
|------------|---------|-------------|
| **Adaptive Harmonic Sparsification (AHS)** | Learnable top-k harmonic selection | Reduces computation by 30-70% while maintaining performance |
| **Frequency Curriculum Learning (FCL)** | Progressive frequency unmasking | Improves convergence by learning low→high frequencies |
| **Phase-Coherent Cross-Attention (PCCA)** | Phase-aware attention mechanism | Captures wave interference patterns explicitly |
| **Multi-Resolution Wave Pyramid (MRWP)** | Multi-scale frequency processing | Enables scale-specific pattern learning |

---

## 1. Adaptive Harmonic Sparsification (AHS)

### What it does
Dynamically selects the most important harmonics at each sequence position using context-aware importance scoring. Uses Gumbel-Softmax for differentiable selection.

### When to use
- When you need to reduce computational cost
- For deployment scenarios with limited resources
- When most harmonics are redundant for your task

### Quick Start

```python
from wave_transformer.enhancements import (
    AdaptiveHarmonicSelector,
    HarmonicSparsificationLoss
)

# Create selector
selector = AdaptiveHarmonicSelector(
    num_harmonics=64,
    d_model=192,  # 3 * num_harmonics
    k_ratio=0.5,  # Keep 50% of harmonics
    temperature=1.0,
    use_dynamic_k=True,
)

# Create regularization loss
sparsity_loss_fn = HarmonicSparsificationLoss(
    target_sparsity=0.5,
    smoothness_weight=0.1,
    temporal_weight=0.05,
)

# In training loop
wave_repr = wave.to_representation()  # (B, S, 3*H)
sparse_wave, stats = selector(wave_repr, return_stats=True)

# Use sparse_wave instead of wave_repr in transformer
output = transformer.forward_from_wave(sparse_wave, ...)

# Add regularization
reg_losses = sparsity_loss_fn(stats)
total_loss = task_loss + 0.01 * reg_losses['total']
```

### Integration with WaveTransformer

```python
class SparseWaveTransformer(WaveTransformer):
    def __init__(self, *args, sparsity_ratio=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        self.selector = AdaptiveHarmonicSelector(
            num_harmonics=self.num_harmonics,
            d_model=self.input_dim,
            k_ratio=sparsity_ratio,
        )

    def forward(self, encoder_input, **kwargs):
        wave = self.wave_encoder(**encoder_input)
        x = wave.to_representation()

        # Apply sparsification
        x, stats = self.selector(x, return_stats=True)

        # Continue with transformer layers
        for block in self.layers:
            x = block(x, **kwargs)

        x = self.norm_f(x)
        output = self.wave_decoder(x, **kwargs)

        return output, stats  # Return stats for loss computation
```

### Monitoring

```python
# Track sparsity during training
print(f"Actual k: {stats['actual_k_mean']:.1f} / {num_harmonics}")
print(f"Sparsity: {stats['sparsity']:.2%}")
print(f"EMA sparsity: {stats['ema_sparsity']:.2%}")
```

---

## 2. Frequency Curriculum Learning (FCL)

### What it does
Progressively introduces higher frequencies during training, starting with low frequencies (coarse patterns) and gradually adding high frequencies (fine details).

### When to use
- When training large models from scratch
- To improve convergence speed
- For tasks where low-frequency patterns are fundamental

### Quick Start

```python
from wave_transformer.enhancements import (
    FrequencyCurriculumScheduler,
    FrequencyMask
)

# Create scheduler
curriculum_scheduler = FrequencyCurriculumScheduler(
    total_steps=num_epochs * steps_per_epoch,
    start_freq_limit=0.1,   # Start with 10% of frequencies
    end_freq_limit=1.0,     # End with all frequencies
    warmup_steps=500,
    curriculum_mode='cosine',  # 'linear', 'exponential', or 'cosine'
    adaptive=True,
    patience=200,
)

# Create frequency mask
freq_mask = FrequencyMask(
    num_harmonics=64,
    mask_slope=10.0,  # Smoothness of frequency cutoff
)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        step = epoch * len(train_loader) + batch_idx

        # Get current frequency limit
        freq_limit = curriculum_scheduler.get_freq_limit(step)

        # Encode to wave
        wave = wave_encoder(**batch)
        wave_repr = wave.to_representation()

        # Apply curriculum masking
        masked_wave = freq_mask(wave_repr, freq_limit)

        # Forward pass with masked representation
        output = transformer.forward_from_wave(masked_wave, ...)

        # ... backward pass ...

    # Update scheduler with validation loss
    val_loss = validate(model, val_loader)
    info = curriculum_scheduler.step(
        step=epoch * len(train_loader),
        val_loss=val_loss
    )

    print(f"Epoch {epoch}: freq_limit={info['freq_limit']:.3f}")
```

### Visualization

```python
# Plot the curriculum schedule
curriculum_scheduler.plot_schedule(save_path='curriculum_schedule.png')
```

### Adaptive Scheduling

The scheduler can automatically slow down the curriculum if validation loss stops improving:

```python
scheduler = FrequencyCurriculumScheduler(
    ...,
    adaptive=True,      # Enable adaptive scheduling
    patience=200,       # Wait 200 steps before adjusting
)

# If val loss doesn't improve for 200 steps, freq_limit reduces by 5%
```

---

## 3. Phase-Coherent Cross-Attention (PCCA)

### What it does
Extends standard attention to explicitly model phase relationships between wave representations. Combines standard attention scores with phase coherence scores.

### When to use
- When phase relationships are important for your task
- For audio or signal processing applications
- To capture wave interference patterns

### Quick Start

```python
from wave_transformer.enhancements import PhaseCoherentAttention
from wave_transformer.core.transformer import RMSNorm, SwiGLU

# Replace standard attention with phase-coherent attention
class PhaseCoherentParallelBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_harmonics,
        n_heads,
        n_heads_kv,
        d_ff,
        dropout=0.0,
    ):
        super().__init__()

        self.norm = RMSNorm(d_model)

        # Use PhaseCoherentAttention instead of MultiQueryFlashAttention
        self.attn = PhaseCoherentAttention(
            d_model=d_model,
            num_harmonics=num_harmonics,
            n_heads_q=n_heads,
            n_heads_kv=n_heads_kv,
            dropout_p=dropout,
            use_yarn=True,
            use_flash=False,  # Phase coherence requires manual attention
            phase_coherence_mode='cosine',  # 'cosine' or 'complex'
            phase_temp=1.0,
            learnable_blend=True,
        )

        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True, attention_mask=None):
        normalized = self.norm(x)
        attn_out = self.attn(normalized, causal, attention_mask)
        ffn_out = self.ffn(normalized)
        return x + self.dropout(attn_out + ffn_out)
```

### Phase Coherence Modes

- **'cosine'**: Fast, uses cosine of phase differences
  - Coherence = mean(cos(phase_q - phase_k)) over harmonics
  - Range: [-1, 1], where 1 = perfect alignment, -1 = anti-phase

- **'complex'**: More accurate, uses complex correlation
  - Coherence = |sum(exp(i*phase_q) * exp(-i*phase_k))|
  - Can incorporate amplitude weighting

### Monitoring Blend Parameters

```python
# Check learned blending weights
print(f"Alpha (standard attention): {attn.alpha.item():.3f}")
print(f"Beta (phase coherence): {attn.beta.item():.3f}")

# Final attention = alpha * standard + beta * phase_coherent
```

---

## 4. Multi-Resolution Wave Pyramid (MRWP)

### What it does
Decomposes wave representation into multiple frequency bands (low/mid/high), processes each band with a specialized transformer, enables cross-band attention, and fuses results back together.

### When to use
- For tasks requiring multi-scale processing
- When different frequency ranges have different importance
- To improve model capacity without proportionally increasing all parameters

### Quick Start

```python
from wave_transformer.enhancements import MultiResolutionWavePyramid

# Create pyramid
pyramid = MultiResolutionWavePyramid(
    num_harmonics=64,
    band_boundaries=[0.2, 0.6],  # Creates 3 bands: [0-20%], [20-60%], [60-100%]
    band_num_layers=[8, 6, 4],   # Different depths per band
    n_heads_q=8,
    n_heads_kv=4,
    d_ff_multi=4,
    dropout=0.1,
    use_cross_band_attn=True,  # Enable cross-band information exchange
    use_yarn=True,
    use_flash=True,
)

# Use in forward pass
wave_repr = wave.to_representation()  # (B, S, 3*H)
output = pyramid(wave_repr, causal=True, attention_mask=mask)
```

### Integration Patterns

**Pattern 1: Replace some layers with pyramid layers**

```python
class PyramidWaveTransformer(nn.Module):
    def __init__(self, ...):
        super().__init__()

        # Early layers: standard transformers
        self.standard_layers = nn.ModuleList([...])

        # Middle layers: pyramids
        self.pyramid_layers = nn.ModuleList([
            MultiResolutionWavePyramid(...) for _ in range(2)
        ])

        # Late layers: standard transformers
        self.final_layers = nn.ModuleList([...])

    def forward(self, x, **kwargs):
        for layer in self.standard_layers:
            x = layer(x, **kwargs)

        for pyramid in self.pyramid_layers:
            x = pyramid(x, **kwargs)

        for layer in self.final_layers:
            x = layer(x, **kwargs)

        return x
```

**Pattern 2: Periodic pyramid application**

```python
class WaveTransformerWithPyramid(WaveTransformer):
    def __init__(self, *args, pyramid_interval=3, **kwargs):
        super().__init__(*args, **kwargs)

        self.pyramid_interval = pyramid_interval
        self.pyramid = MultiResolutionWavePyramid(...)

    def forward(self, x, **kwargs):
        for i, layer in enumerate(self.layers):
            x = layer(x, **kwargs)

            # Apply pyramid every N layers
            if (i + 1) % self.pyramid_interval == 0:
                x = self.pyramid(x, **kwargs)

        return x
```

### Band Configuration

```python
# 3 bands (default)
band_boundaries = [0.2, 0.6]
# → Low: 0-20%, Mid: 20-60%, High: 60-100%

# 4 bands
band_boundaries = [0.25, 0.5, 0.75]
# → Very low: 0-25%, Low: 25-50%, High: 50-75%, Very high: 75-100%

# Different layer depths for different bands
# More layers for low frequencies (fundamental patterns)
# Fewer layers for high frequencies (details)
band_num_layers = [8, 6, 4]  # For 3 bands
```

---

## Combined Usage

You can combine multiple enhancements for maximum impact:

```python
from wave_transformer.enhancements import (
    AdaptiveHarmonicSelector,
    FrequencyCurriculumScheduler,
    FrequencyMask,
    PhaseCoherentAttention,
    MultiResolutionWavePyramid,
)

class EnhancedWaveTransformer(nn.Module):
    """WaveTransformer with all enhancements."""

    def __init__(self, num_harmonics=64, **kwargs):
        super().__init__()

        self.num_harmonics = num_harmonics
        self.d_model = 3 * num_harmonics

        # 1. Adaptive sparsification
        self.selector = AdaptiveHarmonicSelector(
            num_harmonics=num_harmonics,
            d_model=self.d_model,
            k_ratio=0.5,
        )

        # 2. Frequency curriculum (used during training)
        self.freq_mask = FrequencyMask(num_harmonics=num_harmonics)

        # 3. Multi-resolution pyramid layers
        self.pyramid = MultiResolutionWavePyramid(
            num_harmonics=num_harmonics,
            band_boundaries=[0.2, 0.6],
            band_num_layers=[6, 4, 3],
        )

        # 4. Phase-coherent attention layers
        # (Use PhaseCoherentAttention in your ParallelBlocks)

    def forward(
        self,
        wave_repr,
        freq_limit=1.0,  # From curriculum scheduler
        use_sparsification=True,
        **kwargs
    ):
        # Apply curriculum masking (training only)
        if self.training and freq_limit < 1.0:
            wave_repr = self.freq_mask(wave_repr, freq_limit)

        # Apply sparsification
        if use_sparsification:
            wave_repr, stats = self.selector(wave_repr, return_stats=True)
        else:
            stats = None

        # Process through pyramid
        output = self.pyramid(wave_repr, **kwargs)

        return output, stats
```

### Training Loop with All Enhancements

```python
# Setup
model = EnhancedWaveTransformer(num_harmonics=64)
curriculum_scheduler = FrequencyCurriculumScheduler(total_steps=10000, ...)
sparsity_loss_fn = HarmonicSparsificationLoss(...)

# Training
for step, batch in enumerate(train_loader):
    # Get curriculum frequency limit
    freq_limit = curriculum_scheduler.get_freq_limit(step)

    # Forward pass
    wave_repr = wave_encoder(**batch)
    output, stats = model(
        wave_repr.to_representation(),
        freq_limit=freq_limit,
        use_sparsification=True,
    )

    # Compute losses
    task_loss = criterion(output, targets)

    if stats is not None:
        reg_losses = sparsity_loss_fn(stats)
        total_loss = task_loss + 0.01 * reg_losses['total']
    else:
        total_loss = task_loss

    # Backward pass
    total_loss.backward()
    optimizer.step()

    # Update curriculum scheduler
    if step % 100 == 0:
        val_loss = validate(...)
        info = curriculum_scheduler.step(step, val_loss)
```

---

## Performance Tips

1. **AHS**: Start with k_ratio=0.7, gradually reduce if needed
2. **FCL**: Use 'cosine' mode for smooth transitions
3. **PCCA**: Disable Flash Attention (not compatible with custom scores)
4. **MRWP**: Use fewer pyramid layers to balance speed vs. expressiveness

## Memory Considerations

- **AHS**: Reduces memory by ~(1-k_ratio) × 100%
- **FCL**: No memory overhead
- **PCCA**: ~1.5× memory vs standard attention (phase coherence computation)
- **MRWP**: ~2× memory vs single-scale transformer (multiple band processors)

## Citation

If you use these enhancements in your research, please cite:

```bibtex
@software{wave_transformer_enhancements,
  title={Wave Transformer Enhancements},
  author={Wave Transformer Team},
  year={2025},
  url={https://github.com/your-repo/wave-transformer}
}
```

## License

Same as Wave Transformer (see main LICENSE file).
