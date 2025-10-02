"""
Example script showing how to prepare and save datasets for faster training.

This demonstrates:
1. Preparing and saving a single streaming dataset
2. Preparing and saving multiple datasets with weighted sampling
3. Loading prepared datasets for training
"""

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader


from wave_transformer.language_modelling.text_datasets import (
    prepare_and_save_dataset,
    prepare_and_save_multi_dataset,
    PreparedDataset,
    BoundedStreamingDataset,
)


def example_single_dataset_preparation():
    """Example: Prepare and save a single dataset"""
    
    # Load tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = Tokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0
    
    # Prepare and save dataset
    save_path = prepare_and_save_dataset(
        data_source="roneneldan/TinyStories",
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        save_path="prepared_datasets/tiny_stories.pkl",
        sequence_length=512,
        stride=512,  # No overlap
        max_entries=10000,  # Only process first 10k examples
    )
    
    print(f"\nDataset saved to: {save_path}")
    
    # Load the prepared dataset
    dataset = PreparedDataset(save_path, device=torch.device("cuda"))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Use in training
    for batch in loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        print(f"Batch shape: {input_ids.shape}")
        break  # Just show first batch


def example_multi_dataset_preparation():
    """Example: Prepare and save multiple datasets with weighted sampling"""
    
    # Load tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = Tokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0
    
    # Define dataset specifications
    dataset_specs = [
        {
            "name": "wikimedia/wikipedia",
            "subset": "20231101.en",
            "skip": 0,
            "max_entries": 1_500_000,
            "weight": 0.3,  # 30% of final dataset
        },
        {
            "name": "roneneldan/TinyStories",
            "skip": 0,
            "max_entries": 1_000_000,
            "weight": 0.2,  # 40% of final dataset
        },
        {
            "name": "HuggingFaceFW/fineweb",
            "skip": 0,
            "max_entries": 2_500_000,
            "weight": 0.5,  # 30% of final dataset
        },
    ]
    
    # Prepare and save combined dataset
    save_path = prepare_and_save_multi_dataset(
        dataset_specs=dataset_specs,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        save_path="prepared_datasets/multi_dataset.pkl",
        sequence_length=512,
        global_max_entries=5_000_000,  # Total examples in final dataset
        seed=42,  # For reproducibility
    )

    dataset_specs = [
        {
            "name": "wikimedia/wikipedia",
            "subset": "20231101.en",
            "skip": 1_500_000,
            "max_entries": 1500,
            "weight": 0.3,  # 30% of final dataset
        },
        {
            "name": "roneneldan/TinyStories",
            "skip": 1_000_000,
            "max_entries": 1000,
            "weight": 0.2,  # 40% of final dataset
        },
        {
            "name": "HuggingFaceFW/fineweb",
            "skip": 2_500_000,
            "max_entries": 2500,
            "weight": 0.5,  # 30% of final dataset
        },
    ]

    # Prepare and save combined dataset
    save_path = prepare_and_save_multi_dataset(
        dataset_specs=dataset_specs,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        save_path="prepared_datasets/multi_eval_dataset.pkl",
        sequence_length=512,
        global_max_entries=5000,  # Total examples in final dataset
        seed=42,  # For reproducibility
    )
    print(f"\nCombined dataset saved to: {save_path}")
    
    # Load the prepared dataset
    dataset = PreparedDataset(save_path, device=torch.device("cuda"))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} examples")


def example_streaming_vs_prepared():
    """Compare streaming vs prepared dataset usage"""
    
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = Tokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n=== Using Streaming Dataset ===")
    # Streaming dataset - processes on the fly
    streaming_dataset = BoundedStreamingDataset(
        data_source="roneneldan/TinyStories",
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        sequence_length=512,
        max_entries=1000,
        device=device,
    )
    
    # Can't shuffle streaming datasets easily
    streaming_loader = DataLoader(streaming_dataset, batch_size=16)
    
    print("\n=== Using Prepared Dataset ===")
    # Prepared dataset - loaded from disk (much faster)
    prepared_path = "prepared_datasets/tiny_stories.pkl"
    
    # Check if it exists, if not create it
    from pathlib import Path
    if not Path(prepared_path).exists():
        print("Preparing dataset (first time only)...")
        prepare_and_save_dataset(
            data_source="roneneldan/TinyStories",
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            save_path=prepared_path,
            sequence_length=512,
            max_entries=1000,
        )
    
    prepared_dataset = PreparedDataset(prepared_path, device=device)
    
    # Can shuffle, much faster iteration
    prepared_loader = DataLoader(prepared_dataset, batch_size=16, shuffle=True)
    
    print(f"\nPrepared dataset: {len(prepared_dataset)} examples")
    print("Benefits:")
    print("  ✓ Much faster iteration (no tokenization overhead)")
    print("  ✓ Can shuffle for better training")
    print("  ✓ Deterministic (same dataset every run)")
    print("  ✓ Easy to share/distribute")


def example_training_integration():
    """Example of integrating prepared datasets into training"""
    
    from torch import optim, nn
    
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = Tokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Option 1: Prepare datasets once, then use them for multiple training runs
    print("Preparing training dataset...")
    train_path = prepare_and_save_dataset(
        data_source="roneneldan/TinyStories",
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        save_path="prepared_datasets/train.pkl",
        sequence_length=512,
        max_entries=5000,
        skip_first=0,
    )
    
    print("\nPreparing eval dataset...")
    eval_path = prepare_and_save_dataset(
        data_source="roneneldan/TinyStories",
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        save_path="prepared_datasets/eval.pkl",
        sequence_length=512,
        max_entries=500,
        skip_first=5000,  # Skip the training examples
    )
    
    # Load for training
    train_dataset = PreparedDataset(train_path, device=device)
    eval_dataset = PreparedDataset(eval_path, device=device)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16)
    
    print(f"\nTraining: {len(train_dataset)} examples")
    print(f"Eval: {len(eval_dataset)} examples")
    print("\nNow you can train multiple times without re-processing!")


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Preparation Examples")
    print("=" * 60)
    
    # Choose which example to run
    print("\n1. Single dataset preparation")
    # example_single_dataset_preparation()
    
    print("\n2. Multi-dataset preparation")
    example_multi_dataset_preparation()
    
    print("\n3. Streaming vs Prepared comparison")
    # example_streaming_vs_prepared()
    
    print("\n4. Training integration")
    # example_training_integration()
    
    print("\nUncomment the examples you want to run!")
