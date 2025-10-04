# Wave Transformer Generation Analysis

Comprehensive toolkit for analyzing autoregressive generation in Wave Transformer models. This module provides real-time visualization, trajectory tracking, confidence monitoring, and reconstruction analysis.

## Modules

### 1. Live Visualizer (`live_visualizer.py`)

Real-time visualization of wave evolution during token generation.

**Features:**
- Interactive matplotlib plots updating during generation
- Heatmaps of frequencies, amplitudes, and phases
- Energy and spectral centroid tracking
- Token probability monitoring
- Animation export (MP4/GIF)

**Example:**
```python
from wave_transformer.analysis.generation import LiveGenerationVisualizer

visualizer = LiveGenerationVisualizer(model, tokenizer)

# Generate with live visualization
output_ids, waves = visualizer.generate_with_visualization(
    prompt="Once upon a time",
    max_length=50,
    temperature=0.8,
    interactive=True
)

# Create animation from history
visualizer.create_animation("generation.mp4", fps=2)

# Get summary statistics
summary = visualizer.get_generation_summary()
print(f"Generated {summary['num_tokens_generated']} tokens")
print(f"Average energy: {summary['energy']['mean']:.4f}")
```

### 2. Trajectory Tracker (`trajectory_tracker.py`)

Track wave statistics evolution throughout generation to detect patterns and anomalies.

**Features:**
- Wave energy, spectral centroid, bandwidth tracking
- Phase coherence and harmonic entropy monitoring
- Mode collapse detection
- Trajectory visualization and comparison
- CSV export for external analysis

**Example:**
```python
from wave_transformer.analysis.generation import WaveTrajectoryTracker

tracker = WaveTrajectoryTracker(batch_idx=0)

# Track during generation
generated_ids, trajectory = tracker.track_generation(
    model=model,
    initial_ids=prompt_ids,
    max_length=100,
    temperature=0.9
)

# Visualize trajectory
tracker.plot_trajectory(save_path="trajectory.png")

# Detect mode collapse
collapse_info = tracker.detect_mode_collapse(window_size=10)
if collapse_info['mode_collapse_detected']:
    print("Warning: Mode collapse detected!")
    print(f"Energy variance: {collapse_info['energy_variance']:.4f}")

# Get summary statistics
stats = tracker.get_trajectory_statistics()
print(f"Energy trend: {stats['energy']['trend']:.4f}")

# Export trajectory data
tracker.export_trajectory("trajectory.csv")
```

### 3. Confidence Tracker (`confidence_tracker.py`)

Monitor model confidence and uncertainty during generation.

**Features:**
- Max probability (confidence) tracking
- Entropy and perplexity computation
- Top-K probability mass monitoring
- Uncertain region identification
- Wave-confidence correlation analysis

**Example:**
```python
from wave_transformer.analysis.generation import GenerationConfidenceTracker

tracker = GenerationConfidenceTracker(k=10, batch_idx=0)

# Track confidence during generation
generated_ids, confidence_stats = tracker.track_generation(
    model=model,
    initial_ids=prompt_ids,
    max_length=100,
    temperature=0.8
)

# Visualize confidence trajectory
tracker.plot_confidence_trajectory(save_path="confidence.png")

# Identify uncertain tokens
uncertain = tracker.identify_uncertain_regions(
    threshold=0.5,
    metric='max_probability'
)
print(f"Found {len(uncertain)} uncertain tokens")

# Analyze wave-confidence correlation
correlation = tracker.correlate_wave_confidence(wave_metric='energy')
print(f"Pearson correlation: {correlation['pearson']['confidence_vs_wave']['correlation']:.3f}")
print(f"P-value: {correlation['pearson']['confidence_vs_wave']['p_value']:.4f}")

# Visualize correlation
tracker.plot_wave_confidence_correlation(
    wave_metric='energy',
    save_path="wave_confidence_corr.png"
)
```

### 4. Round-trip Analyzer (`roundtrip_analyzer.py`)

Analyze token → wave → token reconstruction quality and wave distinguishability.

**Features:**
- Reconstruction accuracy measurement
- Per-position accuracy analysis
- Wave distinguishability metrics
- Token-wave property correlation
- Batch processing support

**Example:**
```python
from wave_transformer.analysis.generation import RoundTripAnalyzer

analyzer = RoundTripAnalyzer(model)

# Single sequence round-trip analysis
result = analyzer.analyze_roundtrip(token_ids, attention_mask)
print(f"Reconstruction accuracy: {result.reconstruction_accuracy:.2%}")

# Visualize results
analyzer.plot_roundtrip_analysis(
    result,
    tokenizer=tokenizer,
    save_path="roundtrip.png"
)

# Batch analysis with dataloader
batch_stats = analyzer.analyze_batch(
    dataloader=val_dataloader,
    max_batches=10
)
print(f"Overall accuracy: {batch_stats['overall_accuracy']:.2%}")
print(f"Total tokens: {batch_stats['total_tokens']}")

# Wave distinguishability analysis
sequences = [seq1_ids, seq2_ids, seq3_ids, seq4_ids]
distinguishability = analyzer.compute_wave_distinguishability(
    sequences,
    metric='cosine'
)
print(f"Mean distance: {distinguishability['mean_distance']:.4f}")

# Visualize distinguishability
analyzer.plot_wave_distinguishability(
    distinguishability,
    save_path="distinguishability.png"
)

# Token-wave correlation analysis
correlation = analyzer.analyze_wave_token_correlation(
    dataloader=train_dataloader,
    max_batches=20
)
print(f"Analyzed {correlation['num_unique_tokens']} unique tokens")
```

## Complete Generation Analysis Pipeline

Combine all tools for comprehensive generation analysis:

```python
from wave_transformer.analysis.generation import (
    LiveGenerationVisualizer,
    WaveTrajectoryTracker,
    GenerationConfidenceTracker,
    RoundTripAnalyzer
)

# Initialize all analyzers
visualizer = LiveGenerationVisualizer(model, tokenizer)
trajectory_tracker = WaveTrajectoryTracker()
confidence_tracker = GenerationConfidenceTracker(k=10)
roundtrip_analyzer = RoundTripAnalyzer(model)

# Generate with live visualization
output_ids, waves = visualizer.generate_with_visualization(
    prompt="The future of AI is",
    max_length=50,
    temperature=0.8,
    interactive=False  # Set to True for real-time display
)

# Track trajectory using stored waves
for step, wave in enumerate(waves):
    trajectory_tracker.track_step(step, wave)

    # Also track confidence (need logits - get from model)
    logits, _ = model(
        encoder_input={'token_ids': output_ids[:, :step+1]},
        return_encoder_outputs=True
    )
    next_token = output_ids[0, step+1] if step+1 < output_ids.size(1) else output_ids[0, -1]
    confidence_tracker.track_step(step, logits[:, -1, :], next_token.item(), wave)

# Generate all visualizations
visualizer.create_animation("generation_anim.mp4", fps=2)
trajectory_tracker.plot_trajectory(save_path="trajectory.png")
confidence_tracker.plot_confidence_trajectory(save_path="confidence.png")
confidence_tracker.plot_wave_confidence_correlation(save_path="correlation.png")

# Check for issues
collapse = trajectory_tracker.detect_mode_collapse()
if collapse['mode_collapse_detected']:
    print("⚠️ Mode collapse detected!")

uncertain = confidence_tracker.identify_uncertain_regions(threshold=0.5)
print(f"Found {len(uncertain)} uncertain tokens")

# Round-trip analysis on generated sequence
roundtrip_result = roundtrip_analyzer.analyze_roundtrip(output_ids)
print(f"Round-trip accuracy: {roundtrip_result.reconstruction_accuracy:.2%}")
roundtrip_analyzer.plot_roundtrip_analysis(roundtrip_result, tokenizer)

# Export all data
trajectory_tracker.export_trajectory("trajectory_data.csv")
gen_summary = visualizer.get_generation_summary()
conf_summary = confidence_tracker.get_confidence_summary()
traj_summary = trajectory_tracker.get_trajectory_statistics()

print("\n=== Generation Analysis Summary ===")
print(f"Tokens generated: {gen_summary['num_tokens_generated']}")
print(f"Average confidence: {conf_summary['max_probability']['mean']:.3f}")
print(f"Average entropy: {conf_summary['entropy']['mean']:.3f}")
print(f"Energy trend: {traj_summary['energy']['trend']:.6f}")
print(f"Round-trip accuracy: {roundtrip_result.reconstruction_accuracy:.2%}")
```

## Advanced Use Cases

### Comparing Different Temperatures

```python
temperatures = [0.5, 0.8, 1.0, 1.2]
results = {}

for temp in temperatures:
    tracker = WaveTrajectoryTracker()
    generated_ids, trajectory = tracker.track_generation(
        model, prompt_ids, max_length=50, temperature=temp
    )
    results[temp] = tracker.get_trajectory_statistics()

# Compare energy stability across temperatures
for temp, stats in results.items():
    print(f"Temp {temp}: Energy std = {stats['energy']['std']:.4f}")
```

### Identifying Low-Quality Generations

```python
confidence_tracker = GenerationConfidenceTracker()
generated_ids, _ = confidence_tracker.track_generation(
    model, prompt_ids, max_length=100
)

# Find uncertain regions
uncertain = confidence_tracker.identify_uncertain_regions(threshold=0.3)

if len(uncertain) > 10:
    print("⚠️ Low-quality generation detected!")
    print(f"  {len(uncertain)} uncertain tokens")

    # Get summary
    summary = confidence_tracker.get_confidence_summary()
    print(f"  Mean confidence: {summary['max_probability']['mean']:.3f}")
    print(f"  Mean perplexity: {summary['perplexity']['mean']:.2f}")
```

### Dataset-Wide Round-Trip Analysis

```python
analyzer = RoundTripAnalyzer(model)

# Analyze full validation set
batch_results = analyzer.analyze_batch(
    dataloader=val_dataloader,
    max_batches=None  # Process all batches
)

print(f"Dataset accuracy: {batch_results['overall_accuracy']:.2%}")

# Analyze per-position accuracy trends
position_accs = batch_results['position_accuracies']
for pos, acc in sorted(position_accs.items())[:10]:
    print(f"Position {pos}: {acc:.2%}")
```

## Tips and Best Practices

1. **Memory Management**: For long sequences, use `update_interval > 1` in LiveGenerationVisualizer to reduce overhead.

2. **Mode Collapse Detection**: Adjust `variance_threshold` based on your model and data. Lower values are more sensitive.

3. **Confidence Thresholds**: Typical good confidence is > 0.5. Below 0.3 often indicates issues.

4. **Wave Distinguishability**: Higher distances indicate more distinct representations. Cosine distance > 0.5 suggests good separation.

5. **Correlation Analysis**: Significant correlations (p < 0.05) between wave properties and confidence can reveal architectural insights.

6. **Batch Processing**: Use `max_batches` parameter to limit processing time during development.

7. **Animation Creation**: Requires ffmpeg installed for MP4 export. Use GIF format as fallback.

## Requirements

- PyTorch >= 1.12
- matplotlib >= 3.5
- numpy >= 1.21
- scipy >= 1.7 (for confidence tracker correlations)
- ffmpeg (optional, for MP4 animations)

## Integration with Training

These tools can be integrated into training loops for continuous monitoring:

```python
# During validation
if step % validation_interval == 0:
    tracker = GenerationConfidenceTracker()
    sample_gen, _ = tracker.track_generation(model, val_prompt, max_length=50)
    summary = tracker.get_confidence_summary()

    # Log to wandb/tensorboard
    wandb.log({
        'val/generation_confidence': summary['max_probability']['mean'],
        'val/generation_entropy': summary['entropy']['mean'],
    })
```
