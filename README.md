# WaveTransformer

We propose the **WaveTransformer**, a novel sequence model that replaces conventional token embeddings with structured **harmonic wave representations**. Instead of mapping each token to an unstructured vector lookup, our encoder produces a compact set of frequencies, amplitudes, and phases, which are concatenated into a dense representation. This formulation introduces an inductive bias: tokens become structured signals rather than arbitrary points, naturally encoding periodicity, hierarchy, and sequential order. Remarkably, phase information acts as a built-in positional encoding, allowing the WaveTransformer to train effectively without any explicit positional embeddings.

We evaluate the model on language modeling benchmarks and compare against standard Transformer architectures of similar parameter budgets. Results show that the WaveTransformer achieves **competitive or superior perplexity with significantly fewer parameters** (e.g., ~50M WaveTransformer matches the performance of a ~120M standard Transformer). We further observe faster early-stage convergence and robust training even when the number of Transformer layers is drastically reduced, indicating that the wave encoder and decoder contribute substantial representational capacity on their own.

# WaveTransformer for PyTorch and HuggingFace 🤗

This repository contains a PyTorch implementation of the WaveTransformer architecture, which uses wave-based semantic representations for language modeling.


## Installation

```bash
pip install transformers datasets accelerate tokenizers
pip install flash-attn --no-build-isolation  # For flash attention support
```

## Model Architecture

The Wave-Transformer uses a unique three-stage architecture:

1. **Wave Encoder**: Converts token embeddings into wave representations (frequency, amplitude, phase)
2. **Transformer Core**: Processes wave representations with parallel attention/FFN blocks
3. **Wave Decoder**: Reconstructs token probabilities from wave representations

### Key Components

- **TokenToWaveEncoder**: Maps tokens to semantic waves with harmonics
- **ParallelBlock**: Efficient parallel attention and feed-forward computation
- **WaveToTokenDecoder**: Converts wave representations back to vocabulary space


## Citation

If you use Wave-Transformer in your research, please cite:

```bibtex
@software{wave_transformer,
  title = {Wave-Transformer: Wave-based Semantic Representations for Transformer Models},
  author = {Maximilian Winter},
  year = {2025},
  url = {https://github.com/Maximilian-Winter/Wave-Transformer}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- HuggingFace Transformers library for the training infrastructure
- Flash Attention for efficient attention computation
- DeepSpeed for distributed training support
