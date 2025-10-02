import json
import math
import random
from time import sleep
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch import optim, nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from wave_transformer.core.transformer import WaveTransformer

from wave_transformer.language_modelling.text_datasets import MultiBoundedStreamingDataset

def prepare_dataset():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model Parameters
    seq_len = 256
    d_model = 512
    num_layers = 4
    num_heads = 8
    dropout = 0.1
    num_harmonics = 64

    # Hyperparameters
    epochs = 5
    batch_size = 32
    eval_batch_size = 1
    accumulation_steps = 1
    base_lr = 3e-4
    final_lr = 5e-5
    warmup_pct = 0.25

    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Load tokenizer ...
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0
    tokenizer.save("SmolLM2-135M-Instruct-Tokenizer.json")
    print("Pad Token ID:", pad_token_id)
    print("Pad Token:", tokenizer.decode([pad_token_id], False))

    vocab_size = tokenizer.get_vocab_size()
    torch.set_float32_matmul_precision('high')
    dataset_specs = [
        {"name": "wikimedia/wikipedia", "subset": "20231101.en", "skip": 0, "max_entries": 1000_000, "weight": 0.4},
        {"name": "roneneldan/TinyStories", "skip": 0, "max_entries": 500_000, "weight": 0.1},
        {"name": "HuggingFaceFW/fineweb", "skip": 0, "max_entries": 1500_000, "weight": 0.5},
    ]
    train_dataset = MultiBoundedStreamingDataset(dataset_specs, tokenizer, pad_token_id, seq_len, device=device)
    train_dataset.prepare("prepared_datasets/train_dataset_prepared.json", 8)

    eval_dataset_specs = [
        {"name": "wikimedia/wikipedia", "subset": "20231101.en", "skip": 0, "max_entries": 4000, "weight": 0.4},
        {"name": "roneneldan/TinyStories", "skip": 0, "max_entries": 1000, "weight": 0.1},
        {"name": "HuggingFaceFW/fineweb", "skip": 0, "max_entries": 5000, "weight": 0.5},
    ]
    eval_dataset = MultiBoundedStreamingDataset(eval_dataset_specs, tokenizer, pad_token_id, seq_len, device=device)
    eval_dataset.prepare("prepared_datasets/eval_dataset_prepared.json", 8)

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prepare_dataset()