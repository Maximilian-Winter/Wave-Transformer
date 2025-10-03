# ChatML Tokenizer Setup

A clean and simple ChatML-style tokenizer using HuggingFace's tokenizers library (Rust-backed for speed).

## Installation

```bash
pip install tokenizers
```

## Quick Start

### 1. Create and Train Your Tokenizer

```python
from chatml_tokenizer import create_chatml_tokenizer, train_tokenizer, setup_post_processor

# Create tokenizer
tokenizer, trainer, special_tokens = create_chatml_tokenizer(vocab_size=32000)

# Train on your corpus
tokenizer = train_tokenizer(tokenizer, trainer, ["your_corpus.txt"])

# Set up post-processor for ChatML format
setup_post_processor(tokenizer, special_tokens)

# Save it
tokenizer.save("my_tokenizer.json")
```

### 2. Use Your Tokenizer

```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("my_tokenizer.json")

# Format a ChatML conversation
conversation = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>"""

# Encode
encoding = tokenizer.encode(conversation)
print(f"Token IDs: {encoding.ids}")

# Decode
text = tokenizer.decode(encoding.ids)
print(f"Decoded: {text}")
```

## ChatML Format

The tokenizer uses these special tokens:

- `<|im_start|>` - Marks the start of a message
- `<|im_end|>` - Marks the end of a message
- `<|endoftext|>` - Marks the end of a complete text/document
- `<|pad|>` - Padding token

Message format:
```
<|im_start|>{role}
{content}<|im_end|>
```

Where `{role}` can be: system, user, assistant, etc.

## Files

- `chatml_tokenizer.py` - Main tokenizer creation and training code
- `usage_example.py` - Examples of how to use the tokenizer
- `requirements.txt` - Python dependencies

## Training on Your Data

Replace the example corpus with your actual training data:

```python
# Train on multiple files
files = [
    "corpus1.txt",
    "corpus2.txt",
    "corpus3.txt"
]
tokenizer = train_tokenizer(tokenizer, trainer, files)
```

## Integration with Your Model

Once trained, you can integrate this tokenizer with your transformer model:

```python
from tokenizers import Tokenizer

# Load your trained tokenizer
tokenizer = Tokenizer.from_file("my_tokenizer.json")

# Get vocabulary size for your model
vocab_size = tokenizer.get_vocab_size()

# Use in training/inference
input_ids = tokenizer.encode(text).ids
```

## Key Features

- **Fast**: Rust-backed implementation for high performance
- **ChatML Compatible**: Uses standard ChatML format with special tokens
- **BPE Algorithm**: Efficient byte-pair encoding for subword tokenization
- **Customizable**: Easy to adjust vocabulary size and special tokens
- **Simple API**: Clean and straightforward to use
