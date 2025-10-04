# Comparative Analysis Tools

Comprehensive tools for comparing Wave Transformer models across checkpoints, inputs, and ablations.

## Components

### 1. CheckpointComparator

Compare multiple model checkpoints to analyze training evolution and identify critical checkpoints.

**Key Features:**
- Load and compare multiple checkpoints
- Compute divergence metrics (L2, cosine, KL, Wasserstein)
- Visualize metric evolution across training
- Identify critical checkpoints with large changes
- Layer-by-layer comparison

**Example Usage:**

```python
from wave_transformer.analysis import CheckpointComparator
from wave_transformer.language_modelling import TokenToWaveEncoder, WaveToTokenDecoder

# Load multiple checkpoints
comparator = CheckpointComparator(
    checkpoint_paths=[
        'checkpoints/step_1000',
        'checkpoints/step_2000',
        'checkpoints/step_3000',
        'checkpoints/step_4000'
    ],
    encoder_cls=TokenToWaveEncoder,
    decoder_cls=WaveToTokenDecoder,
    device='cuda'
)

# Prepare input
input_data = {'token_ids': torch.tensor([[1, 2, 3, 4, 5]])}

# Compute divergence between consecutive checkpoints
divergence = comparator.compute_checkpoint_divergence(
    encoder_input=input_data,
    metrics=['l2', 'cosine', 'kl', 'wasserstein']
)

# Visualize evolution
fig, axes = comparator.plot_checkpoint_evolution(divergence)
plt.show()

# Identify critical checkpoints
critical = comparator.identify_critical_checkpoints(
    divergence,
    threshold_percentile=75.0
)
print(f"Critical checkpoints: {critical}")

# Compare on dataset
dataset_comparison = comparator.compare_on_dataset(
    dataloader=val_loader,
    max_batches=50
)

# Layer-by-layer analysis
layer_comparison = comparator.compare_layer_by_layer(
    encoder_input=input_data,
    checkpoint_indices=[0, -1]  # First and last
)
```

**Divergence Metrics:**

- **L2 Distance**: Euclidean distance between wave representations
  - Measures overall magnitude of change
  - Higher = more different

- **Cosine Similarity**: Angular similarity between wave vectors
  - Measures directional similarity
  - Higher = more similar (range: [-1, 1])

- **KL Divergence**: Kullback-Leibler divergence of amplitude distributions
  - Measures information gain
  - Higher = more divergent

- **Wasserstein Distance**: Earth mover's distance for frequency distributions
  - Measures distribution shift
  - Higher = more different

---

### 2. InputComparator

Compare how different inputs are represented in wave space.

**Key Features:**
- Generate wave representations for multiple inputs
- Compute pairwise similarity matrices
- Cluster inputs (k-means, hierarchical)
- Visualize relationships (heatmaps, t-SNE, UMAP)
- Find nearest neighbors in wave space

**Example Usage:**

```python
from wave_transformer.analysis import InputComparator

# Initialize comparator
comparator = InputComparator(model, device='cuda')

# Prepare multiple inputs
inputs = [
    {'token_ids': torch.tensor([[1, 2, 3, 4]])},
    {'token_ids': torch.tensor([[5, 6, 7, 8]])},
    {'token_ids': torch.tensor([[1, 2, 5, 6]])},
    {'token_ids': torch.tensor([[3, 4, 7, 8]])},
]

# Generate wave representations
waves = comparator.compare_inputs(inputs, extract_layer='encoder')

# Compute similarity matrix
similarity = comparator.compute_input_similarity(
    waves,
    method='cosine'  # Options: 'cosine', 'correlation', 'spectral_overlap', 'l2'
)

# Visualize similarity
fig = comparator.plot_input_comparison(
    similarity,
    input_labels=['Input 1', 'Input 2', 'Input 3', 'Input 4']
)
plt.show()

# Cluster inputs
cluster_result = comparator.cluster_inputs(
    waves,
    method='kmeans',
    n_clusters=2
)
print(f"Cluster labels: {cluster_result['labels']}")

# Visualize clustering
fig = comparator.plot_clustering_results(
    cluster_result,
    waves,
    input_labels=['Input 1', 'Input 2', 'Input 3', 'Input 4']
)

# 2D projection with t-SNE
fig = comparator.plot_2d_projection(
    waves,
    method='tsne',
    labels=cluster_result['labels']
)

# Find nearest neighbors
neighbors = comparator.find_nearest_neighbors(
    waves,
    query_idx=0,
    k=3,
    metric='cosine'
)
print(f"Nearest neighbors to input 0: {neighbors['neighbor_indices']}")

# Compare statistics across inputs
fig, stats = comparator.compare_wave_statistics(
    waves,
    input_labels=['Input 1', 'Input 2', 'Input 3', 'Input 4']
)
```

**Similarity Methods:**

- **Cosine**: Cosine similarity of flattened wave tensors
  - Measures directional similarity
  - Range: [-1, 1]

- **Correlation**: Pearson correlation of harmonic patterns
  - Measures linear relationship
  - Range: [-1, 1]

- **Spectral Overlap**: Frequency domain overlap
  - Measures how similar frequency distributions are
  - Range: [0, 1]

- **L2**: Negative L2 distance
  - Measures Euclidean distance
  - Higher = more similar

---

### 3. AblationHelper

Systematic ablation studies to understand component importance.

**Key Features:**
- Ablate specific harmonics (zero, randomize, mean)
- Ablate transformer layers (identity, zero)
- Ablate entire wave components (freq/amp/phase)
- Run comprehensive ablation studies
- Automatic model restoration
- Visualize ablation impact

**Example Usage:**

```python
from wave_transformer.analysis import AblationHelper

# Initialize ablation helper
ablator = AblationHelper(model, device='cuda')

# Define ablation configurations
ablation_configs = [
    # Ablate first 3 harmonics
    {
        'type': 'harmonics',
        'indices': [0, 1, 2],
        'mode': 'zero',
        'component': 'all'
    },

    # Ablate middle harmonics with randomization
    {
        'type': 'harmonics',
        'indices': list(range(16, 32)),
        'mode': 'random',
        'component': 'amp'
    },

    # Make layer 2 an identity mapping
    {
        'type': 'layer',
        'layer_idx': 2,
        'mode': 'identity'
    },

    # Ablate multiple layers
    {
        'type': 'layer',
        'layer_idx': [3, 4],
        'mode': 'identity'
    },

    # Zero out all phase information
    {
        'type': 'wave_component',
        'component': 'phase',
        'mode': 'zero'
    },

    # Randomize all frequencies
    {
        'type': 'wave_component',
        'component': 'freq',
        'mode': 'random'
    },
]

# Run ablation study
results_df = ablator.run_ablation_study(
    ablation_configs=ablation_configs,
    dataloader=val_loader,
    include_baseline=True
)

# Display results
print(results_df)

# Visualize results
fig = ablator.plot_ablation_results(
    results_df,
    metrics=['avg_loss', 'perplexity', 'accuracy']
)
plt.show()

# Impact heatmap
fig = ablator.plot_impact_heatmap(
    results_df,
    metric='avg_loss'
)
plt.show()

# Model is automatically restored after the study
# But you can also manually restore:
ablator.restore_model()
```

**Ablation Modes:**

- **zero**: Set components to 0
- **random**: Randomize with same distribution
- **mean**: Replace with mean value
- **noise**: Add Gaussian noise
- **constant**: Set to constant value (for full components)
- **identity**: Make layer output = input (for layers)

**Ablation Types:**

1. **Harmonics**: Ablate specific harmonic indices
   - Can target individual components (freq/amp/phase) or all
   - Useful for understanding harmonic importance

2. **Layers**: Ablate entire transformer layers
   - Convert to identity (skip connection)
   - Useful for understanding layer contributions

3. **Wave Components**: Ablate entire wave components
   - Remove all frequency/amplitude/phase information
   - Useful for understanding component necessity

**Custom Evaluation Function:**

```python
def custom_eval_fn(model, dataloader):
    """Custom evaluation function"""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            # Your evaluation logic here
            loss = compute_loss(model, batch)
            total_loss += loss

    return {
        'custom_metric': total_loss,
        'another_metric': some_value
    }

# Use custom evaluation
results_df = ablator.run_ablation_study(
    ablation_configs=configs,
    eval_fn=custom_eval_fn,
    dataloader=val_loader
)
```

**Context Manager Usage:**

```python
# Automatic restoration on exit
with AblationHelper(model) as ablator:
    ablator.ablate_harmonics([0, 1, 2], mode='zero')

    # Evaluate ablated model
    loss = evaluate(model, dataloader)
    print(f"Loss with ablation: {loss}")

# Model automatically restored here
```

---

## Advanced Examples

### Comparing Training Dynamics

```python
from wave_transformer.analysis import CheckpointComparator

# Load checkpoints from different training runs
comparator = CheckpointComparator(
    checkpoint_paths=[
        'run1/step_5000',
        'run2/step_5000',
        'run3/step_5000'
    ],
    encoder_cls=TokenToWaveEncoder,
    decoder_cls=WaveToTokenDecoder
)

# Compare on validation set
comparison = comparator.compare_on_dataset(
    dataloader=val_loader,
    max_batches=100
)

# Plot pairwise divergence
for metric_name, matrix in comparison['divergences'].items():
    fig = comparator.plot_pairwise_divergence_matrix(
        matrix,
        metric_name=metric_name
    )
    plt.savefig(f'divergence_{metric_name}.png')
```

### Input Clustering for Data Analysis

```python
from wave_transformer.analysis import InputComparator

comparator = InputComparator(model)

# Generate waves for entire dataset
all_waves = []
for batch in dataloader:
    waves = comparator.compare_inputs([batch])
    all_waves.extend(waves)

# Hierarchical clustering
cluster_result = comparator.cluster_inputs(
    all_waves,
    method='hierarchical',
    n_clusters=10
)

# Visualize dendrogram
fig = comparator.plot_clustering_results(
    cluster_result,
    all_waves
)

# Analyze cluster characteristics
for cluster_id in range(10):
    cluster_waves = [w for i, w in enumerate(all_waves)
                     if cluster_result['labels'][i] == cluster_id]
    # Analyze cluster properties...
```

### Comprehensive Ablation Study

```python
from wave_transformer.analysis import AblationHelper
import numpy as np

ablator = AblationHelper(model)

# Test all layers individually
layer_configs = [
    {'type': 'layer', 'layer_idx': i, 'mode': 'identity'}
    for i in range(model.transformer_num_layers)
]

# Test harmonic ranges
harmonic_configs = []
for start in range(0, 64, 8):
    harmonic_configs.append({
        'type': 'harmonics',
        'indices': list(range(start, start + 8)),
        'mode': 'zero'
    })

# Test wave components
component_configs = [
    {'type': 'wave_component', 'component': c, 'mode': m}
    for c in ['freq', 'amp', 'phase']
    for m in ['zero', 'random', 'mean']
]

# Run comprehensive study
all_configs = layer_configs + harmonic_configs + component_configs
results = ablator.run_ablation_study(
    ablation_configs=all_configs,
    dataloader=val_loader
)

# Save results
results.to_csv('comprehensive_ablation.csv')

# Visualize
fig = ablator.plot_ablation_results(results)
plt.savefig('ablation_comprehensive.png', dpi=300)
```

---

## Tips and Best Practices

1. **Checkpoint Comparison:**
   - Use multiple divergence metrics for robust comparison
   - Consider both consecutive and pairwise comparisons
   - Identify critical checkpoints to save for later analysis

2. **Input Comparison:**
   - Use appropriate similarity metrics for your task
   - Try multiple clustering methods (k-means, hierarchical)
   - Visualize with both heatmaps and dimensionality reduction

3. **Ablation Studies:**
   - Always include baseline for reference
   - Test multiple ablation modes to understand robustness
   - Use custom evaluation functions for task-specific metrics
   - Save results to disk for later analysis

4. **Performance:**
   - Use `max_batches` parameter to limit dataset comparisons
   - Consider using smaller models for initial exploration
   - Cache wave representations when comparing many inputs

5. **Visualization:**
   - Save high-resolution figures (`dpi=300`) for publications
   - Use consistent color schemes across related figures
   - Add descriptive labels and titles

---

## Dependencies

**Required:**
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Pandas

**Optional:**
- scikit-learn (for t-SNE, k-means clustering)
- umap-learn (for UMAP dimensionality reduction)

Install optional dependencies:
```bash
pip install scikit-learn umap-learn
```

---

## API Reference

See individual module docstrings for detailed API documentation:

```python
from wave_transformer.analysis.comparative import (
    CheckpointComparator,
    InputComparator,
    AblationHelper
)

# View documentation
help(CheckpointComparator)
help(InputComparator)
help(AblationHelper)
```
