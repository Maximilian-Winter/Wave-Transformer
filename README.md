# 🌊 Wave-Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FlashAttention](https://img.shields.io/badge/FlashAttention-Enabled-blue)](https://github.com/Dao-AILab/flash-attention)

Wave-Transformer is a **PyTorch implementation** of a novel transformer architecture that leverages **wave-based semantic representations**.  
It introduces frequency, amplitude, and phase components into the transformer pipeline for more expressive sequence modeling.

---

## 🚀 Installation

```bash
pip install torch torchvision
pip install datasets tokenizers
pip install flash-attn --no-build-isolation   # Optional: FlashAttention support
````

---

## 🧪 Usage

Example: training on WikiText-103

```bash
python examples/language_modelling/py_torch/train_pytorch.py
```

Example: inference

```bash
python examples/language_modelling/py_torch/inference.py \
    --model_path ./wave_transformer_epoch_1.pt \
    --prompt "The tao that can be told"
```

---

## 📄 Citation

If you use Wave-Transformer in your research, please cite:

```bibtex
@software{Wave-Transformer,
  title = {Wave-Transformer: Wave-based Semantic Representations for Transformer Models},
  author = {Maximilian Winter},
  year = {2025},
  url = {https://github.com/Maximilian-Winter/Wave-Transformer}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).