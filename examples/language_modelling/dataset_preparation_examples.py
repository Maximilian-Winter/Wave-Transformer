"""
Example usage of the prepare and load methods for BoundedStreamingDataset
and MultiBoundedStreamingDataset.
"""

import torch
from tokenizers import Tokenizer
from wave_transformer.language_modelling.text_datasets import BoundedStreamingDataset, MultiBoundedStreamingDataset


# Example 1: BoundedStreamingDataset
def example_bounded_dataset():
    """Example of preparing and loading a single bounded dataset."""

    # Initialize tokenizer (replace with your tokenizer)
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    pad_token_id = tokenizer.token_to_id("[PAD]")

    # Create dataset
    dataset = BoundedStreamingDataset(
        data_source="wikitext",  # HuggingFace dataset name
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        sequence_length=512,
        stride=512,
        text_column="text",
        max_entries=10000,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Prepare (tokenize and save) - use 8 workers for parallel processing
    print("Preparing dataset...")
    dataset.prepare(
        output_path="prepared_wikitext.json",
        num_workers=8,
        chunk_size=1000
    )

    # Load the prepared dataset
    print("\nLoading prepared dataset...")
    loaded_dataset = BoundedStreamingDataset.load(
        input_path="prepared_wikitext.json",
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        sequence_length=512,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Use the loaded dataset
    print("\nIterating over loaded dataset...")
    for i, batch in enumerate(loaded_dataset):
        print(f"Batch {i}: input_ids shape = {batch['input_ids'].shape}")
        if i >= 2:  # Just show first 3 batches
            break


# Example 2: MultiBoundedStreamingDataset
def example_multi_bounded_dataset():
    """Example of preparing and loading multiple datasets with weighted sampling."""

    # Initialize tokenizer
    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    pad_token_id = tokenizer.token_to_id("[PAD]")

    # Define multiple datasets with weights
    dataset_specs = [
        {
            "name": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "max_entries": 5000,
            "skip": 0,
            "weight": 2.0  # Sample this dataset twice as often
        },
        {
            "name": "bookcorpus",
            "max_entries": 3000,
            "skip": 100,
            "weight": 1.0
        }
    ]

    # Create multi-dataset
    multi_dataset = MultiBoundedStreamingDataset(
        dataset_specs=dataset_specs,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        sequence_length=512,
        stride=512,
        text_column="text",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        global_max_entries=8000,
        seed=42
    )

    # Prepare (tokenize and save) all datasets
    print("Preparing multiple datasets...")
    multi_dataset.prepare(
        output_path="prepared_multi_datasets.json",
        num_workers=8,
        chunk_size=1000
    )

    # Load the prepared datasets
    print("\nLoading prepared datasets...")
    loaded_multi_dataset = MultiBoundedStreamingDataset.load(
        input_path="prepared_multi_datasets.json",
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        dataset_specs=dataset_specs,  # Need specs for weights
        sequence_length=512,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        global_max_entries=8000,
        seed=42
    )

    # Use the loaded dataset with weighted sampling
    print("\nIterating over loaded multi-dataset...")
    for i, batch in enumerate(loaded_multi_dataset):
        print(f"Batch {i}: input_ids shape = {batch['input_ids'].shape}")
        if i >= 2:  # Just show first 3 batches
            break


# Example 3: Using with PyTorch DataLoader
def example_with_dataloader():
    """Example of using prepared dataset with PyTorch DataLoader."""
    from torch.utils.data import DataLoader

    tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
    pad_token_id = tokenizer.token_to_id("[PAD]")

    # Load prepared dataset
    dataset = BoundedStreamingDataset.load(
        input_path="prepared_wikitext.json",
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        sequence_length=512,
        device=torch.device("cpu")  # DataLoader will handle device transfer
    )

    # Create DataLoader for batching
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0  # IterableDataset doesn't support multi-process loading
    )

    print("Using DataLoader...")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: input_ids shape = {batch['input_ids'].shape}")
        if i >= 2:
            break


if __name__ == "__main__":
    print("=" * 50)
    print("Example 1: Single Bounded Dataset")
    print("=" * 50)
    example_bounded_dataset()

    print("\n" + "=" * 50)
    print("Example 2: Multi Bounded Dataset")
    print("=" * 50)
    #example_multi_bounded_dataset()

    print("\n" + "=" * 50)
    print("Example 3: With DataLoader")
    print("=" * 50)
    #example_with_dataloader()