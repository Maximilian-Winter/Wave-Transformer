# Wave Transformer Visualization & Reporting

This module provides comprehensive visualization and reporting tools for analyzing Wave Transformer models, suitable for research papers, experiments, and real-time monitoring.

## Components

### 1. TensorBoard Writer (`tensorboard_writer.py`)

Real-time logging to TensorBoard for monitoring training and analysis.

**Features:**
- Wave statistics logging (spectral centroid, energy, entropy)
- Component heatmaps (frequencies, amplitudes, phases)
- Spectrum visualizations
- Layer-wise comparisons
- Gradient flow analysis
- Harmonic importance tracking

**Example:**
```python
from wave_transformer.analysis.visualization import WaveTensorBoardWriter

writer = WaveTensorBoardWriter(log_dir='runs/experiment_1')

# Log wave statistics
writer.add_wave_statistics(wave, tag='encoder/wave', step=100)

# Log heatmaps
writer.add_wave_heatmaps(wave, tag='encoder/wave', step=100)

# Log spectrum
writer.add_wave_spectrum(wave, tag='encoder/wave', step=100)

writer.close()
```

View with: `tensorboard --logdir runs/experiment_1`

### 2. Weights & Biases Logger (`wandb_logger.py`)

Integration with Weights & Biases for experiment tracking and collaboration.

**Features:**
- Wave statistics and metrics
- Visualization logging (heatmaps, spectra)
- Generation examples with animations
- Layer-wise analysis
- Ablation study tables
- Graceful fallback when wandb is not installed

**Example:**
```python
from wave_transformer.analysis.visualization import WaveWandbLogger

logger = WaveWandbLogger(
    project='wave-transformer',
    name='experiment-1',
    tags=['wave-analysis']
)

# Log wave statistics
logger.log_wave_statistics(wave, prefix='encoder', step=100)

# Log visualizations
logger.log_wave_visualizations(wave, prefix='encoder', step=100)

# Log generation example
logger.log_generation_example(
    input_text="Once upon a time",
    generated_text="Once upon a time in a land far away...",
    confidence_scores=confidence_array
)

logger.finish()
```

### 3. Report Generator (`report_generator.py`)

Publication-quality figures and LaTeX tables for research papers.

**Features:**
- Publication style presets (IEEE, Nature, Science, arXiv)
- Training curve figures
- Layer evolution analysis
- Harmonic importance visualizations
- Generation analysis plots
- Model comparison figures
- LaTeX table generation
- Vector (PDF) and raster (PNG) output

**Example:**
```python
from wave_transformer.analysis.visualization import (
    PaperReportGenerator,
    PublicationStyle
)

generator = PaperReportGenerator(
    output_dir='paper_figures',
    style=PublicationStyle.IEEE
)

# Training curves
generator.create_training_curve_figure(
    train_losses=train_losses,
    val_losses=val_losses,
    save_name='training_curves'
)

# Layer analysis
generator.create_layer_analysis_figure(
    layer_metrics=layer_data,
    save_name='layer_evolution'
)

# Harmonic importance
generator.create_harmonic_importance_figure(
    importance_scores=scores,
    save_name='harmonic_importance'
)

# LaTeX table
generator.save_latex_table(
    data=ablation_results,
    filename='ablation_table',
    caption='Ablation Study Results'
)
```

## Publication Styles

The `PaperReportGenerator` supports multiple publication formats:

### IEEE (2-column)
```python
style = PublicationStyle.IEEE
# - Figure width: 3.5 inches
# - Font size: 10pt
# - DPI: 300
```

### Nature
```python
style = PublicationStyle.NATURE
# - Figure width: 3.5 inches
# - Font size: 8pt
# - DPI: 600
```

### Science
```python
style = PublicationStyle.SCIENCE
# - Figure width: 3.5 inches
# - Font size: 8pt
# - DPI: 600
```

### arXiv
```python
style = PublicationStyle.ARXIV
# - Figure width: 6.0 inches
# - Font size: 11pt
# - DPI: 150
```

## Configuration Management

The `config.py` module provides centralized configuration for all analysis components.

**Example:**
```python
from wave_transformer.analysis.utils import (
    AnalysisConfig,
    create_default_config
)

# Create default configuration
config = create_default_config(
    output_dir='analysis_results',
    experiment_name='experiment_1',
    enable_tensorboard=True,
    enable_wandb=False
)

# Save to YAML
config.to_yaml('config.yaml')

# Load from YAML
loaded_config = AnalysisConfig.from_yaml('config.yaml')

# Access component configs
viz_config = config.visualization
print(f"TensorBoard enabled: {viz_config.use_tensorboard}")
```

## Dependencies

### Required
- `torch`
- `numpy`
- `matplotlib`

### Optional
- `tensorboard` - for TensorBoard logging
- `wandb` - for Weights & Biases integration
- `pyyaml` - for YAML configuration support

Install optional dependencies:
```bash
pip install tensorboard wandb pyyaml
```

## Example Configuration YAML

```yaml
output_dir: "analysis_results"
experiment_name: "wave_transformer_analysis"
enable_wave_tracking: true
wave_sampling_rate: 0.1

visualization:
  use_tensorboard: true
  tensorboard_log_dir: "runs/wave_analysis"
  use_wandb: false
  wandb_project: "wave-transformer"
  plot_frequency: 1000
  figure_format: "png"
  dpi: 150

introspection:
  harmonic_importance_method: "energy"
  track_top_k_harmonics: 16
  enable_spectrum_tracking: true
  enable_layer_analysis: true
```

## Complete Example

See `examples/analysis/visualization_demo.py` for a complete demonstration of all features.

Run the demo:
```bash
python examples/analysis/visualization_demo.py
```

This will generate:
- TensorBoard logs in `runs/visualization_demo/`
- Publication figures in `demo_reports/`
- Example configuration in `demo_config.yaml`

## Tips

### For Real-Time Monitoring
- Use `WaveTensorBoardWriter` for live training visualization
- Set `plot_frequency` to balance performance vs. detail
- Use context managers for automatic cleanup:
  ```python
  with WaveTensorBoardWriter(log_dir='runs/exp') as writer:
      writer.add_wave_statistics(wave, tag='train', step=step)
  ```

### For Experiment Tracking
- Use `WaveWandbLogger` for cloud-based experiment tracking
- Log ablation results as tables for easy comparison
- Use tags to organize related experiments

### For Publications
- Use `PublicationStyle.IEEE` for most conferences
- Generate both PDF (vector) and PNG (raster) formats
- Use LaTeX table generation for consistent formatting
- Set `use_tex=True` in `FigureConfig` for LaTeX fonts (requires LaTeX installation)

### Memory Management
- Limit heatmap sequence length with `max_seq_len` parameter
- Use `batch_idx` parameter to visualize specific examples
- Set appropriate `flush_secs` for TensorBoard to manage memory
