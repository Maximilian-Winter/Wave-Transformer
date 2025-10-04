"""
Comparative Analysis Tools for Wave Transformer

This module provides tools for comparing Wave Transformer models across:
- Different checkpoints (training evolution)
- Different inputs (representation similarity)
- Ablation studies (component importance)

## Components

### CheckpointComparator
Compare multiple model checkpoints to analyze training evolution and identify
critical checkpoints with significant changes.

### InputComparator
Compare how different inputs are represented in wave space, cluster similar
inputs, and find nearest neighbors.

### AblationHelper
Systematic ablation studies to understand component importance by zeroing,
randomizing, or perturbing specific model components.

## Quick Start

```python
from wave_transformer.analysis.comparative import (
    CheckpointComparator,
    InputComparator,
    AblationHelper
)

# Compare checkpoints
comparator = CheckpointComparator(checkpoint_paths)
divergence = comparator.compute_checkpoint_divergence(input_data)
comparator.plot_checkpoint_evolution(divergence)

# Compare inputs
input_comp = InputComparator(model)
similarity = input_comp.compute_input_similarity(inputs)
input_comp.plot_input_comparison(similarity)

# Ablation study
ablator = AblationHelper(model)
results = ablator.run_ablation_study(
    ablation_configs=[
        {'type': 'harmonics', 'indices': [0, 1, 2]},
        {'type': 'layer', 'layer_idx': 3}
    ],
    dataloader=val_loader
)
ablator.plot_ablation_results(results)
```
"""

from .checkpoint_comparator import CheckpointComparator
from .input_comparator import InputComparator
from .ablation_helper import AblationHelper

__all__ = [
    'CheckpointComparator',
    'InputComparator',
    'AblationHelper',
]
