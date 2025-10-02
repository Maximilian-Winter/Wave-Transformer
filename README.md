# WaveTransformer

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
