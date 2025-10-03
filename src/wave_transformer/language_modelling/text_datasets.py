import dataclasses
import random
from time import sleep

import torch
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset, IterableDataset
from tokenizers import Tokenizer
from datasets import load_dataset, IterableDataset as HFIterableDataset
from typing import List, Iterator, Dict, Optional, Union


def _pad_to_length(ids, max_length, pad_id):
    if len(ids) >= max_length:
        return ids[:max_length], [1] * max_length
    pad_len = max_length - len(ids)
    return ids + [pad_id] * pad_len, [1] * len(ids) + [0] * pad_len


def apply_padding(sequence: list[int], target_length: int, pad_token: int) -> tuple[list[int], list[int]]:
    if len(sequence) >= target_length:
        return sequence[:target_length], [1] * target_length

    padding_size = target_length - len(sequence)
    padded_sequence = sequence + [pad_token] * padding_size
    attention_mask = [1] * len(sequence) + [0] * padding_size
    return padded_sequence, attention_mask


def _pack_by_lines_to_token_chunks(
        text: str,
        tokenizer: Tokenizer,
        max_length: int,
        add_special_tokens: bool = False,
        keep_remainder: bool = True,
) -> List[List[int]]:
    """
    Build token chunks up to max_length by appending whole *lines*.
    If adding the next line would exceed max_length, we stop (back to the start of that line).
    If a single line alone exceeds max_length, we fall back to word-wise packing for that line.
    If keep_remainder=True, we continue with the remaining lines to produce more chunks.
    """
    if not text or not isinstance(text, str):
        return []

    lines = text.splitlines(keepends=True)
    line_tokens = [tokenizer.encode(ln, add_special_tokens=add_special_tokens).ids for ln in lines]

    chunks: List[List[int]] = []
    curr: List[int] = []
    i = 0
    n = len(lines)

    def flush_curr():
        nonlocal curr
        if curr:
            chunks.append(curr)
            curr = []

    while i < n:
        lt = line_tokens[i]
        if len(lt) <= max_length:
            if len(curr) + len(lt) <= max_length:
                curr.extend(lt)
                i += 1
            else:
                if curr:
                    flush_curr()
                if not keep_remainder:
                    break
        else:
            words = lines[i].split()
            j = 0
            if not words:
                big_ids = lt[:max_length]
                if len(curr) == 0:
                    chunks.append(big_ids)
                else:
                    flush_curr()
                    chunks.append(big_ids)
                i += 1
                if not keep_remainder:
                    break
                continue

            while j < len(words):
                w_ids = tokenizer.encode(words[j] + " ", add_special_tokens=add_special_tokens).ids
                if len(w_ids) > max_length:
                    if not curr:
                        chunks.append(w_ids[:max_length])
                    else:
                        flush_curr()
                        chunks.append(w_ids[:max_length])
                    j += 1
                    continue

                if len(curr) + len(w_ids) <= max_length:
                    curr.extend(w_ids)
                    j += 1
                else:
                    flush_curr()

            i += 1
            if not keep_remainder and (i < n):
                break

    if curr:
        chunks.append(curr)

    return chunks


class TextDatasetPadded(Dataset):
    """
    Clean dataset for language modeling with line-aware packing.

    - Packs each input text into one or more examples of length <= max_length.
    - Prefer cutting at *line* boundaries. If a single line is too long, it packs by words for that line.
    - `keep_remainder=True` => do NOT discard overflow; emit additional samples.
      Set `keep_remainder=False` to mimic old behavior but cut at the last full line instead of mid-line.

    Returns dict with:
      - input_ids: LongTensor [max_length]
      - attention_mask: BoolTensor [max_length] (True = valid token, False = padding)
    """

    def __init__(
            self,
            texts,
            tokenizer: Tokenizer,
            pad_token_id: int,
            max_length: int = 512,
            device: torch.device = torch.device("cpu"),
            keep_remainder: bool = True,
    ):
        self.tokenizer = tokenizer
        self.pad_id = pad_token_id
        self.max_length = max_length
        self.device = device
        self.keep_remainder = keep_remainder

        self.examples = []
        texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]

        if hasattr(self.tokenizer, "no_truncation"):
            self.tokenizer.no_truncation()
        if hasattr(self.tokenizer, "no_padding"):
            self.tokenizer.no_padding()

        for text in texts:
            token_chunks = _pack_by_lines_to_token_chunks(
                text,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                add_special_tokens=False,
                keep_remainder=self.keep_remainder,
            )
            if not token_chunks:
                continue

            for ids in token_chunks:
                ids, attn = _pad_to_length(ids, self.max_length, self.pad_id)
                self.examples.append({
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.bool),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"].to(self.device),
            "attention_mask": self.examples[idx]["attention_mask"].to(self.device),
        }


@dataclasses.dataclass
class BoundedStreamingDataset:
    repo_id: str
    subset: Optional[str] = None
    split: str = "train"
    skip_first: int = 0
    max_entries: int = None
    text_column: str = "text"
    weight: float = 1.0  # Added for weighted sampling
    current_idx: int = 0  # Track current position in dataset


class MultiBoundedStreamingDataset(IterableDataset):
    """
    Simplified streaming dataset that tokenizes text directly without buffering.
    Supports batch processing for efficient data loading.
    """

    def __init__(
            self,
            dataset_specs: List[BoundedStreamingDataset],
            tokenizer: Tokenizer,
            pad_token_id: int,
            sequence_length: int = 512,
            batch_size: int = 32,
            prefetch_batches: int = 100,
            device: torch.device = torch.device("cpu"),
            global_max_entries: Optional[int] = None,
            seed: Optional[int] = None,
            weighted_sampling: bool = False
    ):
        super().__init__()

        self.dataset_specs = dataset_specs
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches  # Number of batches to fetch at once
        self.device = device
        self.global_max_entries = global_max_entries
        self.seed = seed
        self.weighted_sampling = weighted_sampling
        self.tokenizer.enable_padding(
            pad_id=pad_token_id,
            pad_token=tokenizer.decode([pad_token_id], False),  # This might need adjustment based on your tokenizer
            length=sequence_length
        )
        self.tokenizer.enable_truncation(max_length=sequence_length)


    def _create_dataset_iterator(self, spec: BoundedStreamingDataset, fetch_size: int):
        """Create an iterator for a chunk of entries from a dataset."""
        try:
            ds = load_dataset(
                spec.repo_id,
                spec.subset,
                split=spec.split,
                streaming=True
            )

            # Calculate how many entries to skip (initial skip + already processed)
            total_skip = spec.skip_first + spec.current_idx

            # Apply skip to get to current position
            if total_skip > 0:
                ds = ds.skip(total_skip)

            # Take only the requested chunk size
            # Make sure we don't exceed max_entries if specified
            if spec.max_entries is not None:
                remaining = spec.max_entries - spec.current_idx
                fetch_size = min(fetch_size, remaining)

            if fetch_size > 0:
                ds = ds.take(fetch_size)
                return ds
            else:
                return None

        except Exception as e:
            print(f"Warning: Failed to load dataset {spec.repo_id}: {e}")
            return None

    def _tokenize_sample(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a single text sample."""

        try:
            # Tokenize with padding and truncation
            encoding = self.tokenizer.encode(text, add_special_tokens=False)

            # Get the padded/truncated ids
            input_ids = encoding.ids

            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = encoding.attention_mask

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long).to(device=self.device, non_blocking=True),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.bool).to(device=self.device, non_blocking=True),
            }
        except Exception as e:
            print(f"Warning: Failed to tokenize text: {e}")
            # Return padded sequence on error
            return {
                "input_ids": torch.full((self.sequence_length,), self.pad_token_id, dtype=torch.long).to(device=self.device, non_blocking=True),
                "attention_mask": torch.zeros(self.sequence_length, dtype=torch.bool).to(device=self.device, non_blocking=True)
            }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.seed is not None:
            random.seed(self.seed)

        # Reset current indices
        for spec in self.dataset_specs:
            spec.current_idx = 0

        global_yielded = 0
        fetch_size = self.batch_size * self.prefetch_batches

        if self.weighted_sampling and len(self.dataset_specs) > 1:
            # Weighted sampling with chunked fetching
            while True:
                if self.global_max_entries and global_yielded >= self.global_max_entries:
                    break

                # Check which datasets still have data
                active_specs = []
                for spec in self.dataset_specs:
                    if spec.max_entries is None or spec.current_idx < spec.max_entries:
                        active_specs.append(spec)

                if not active_specs:
                    break

                # Collect samples from all active datasets based on weights
                batch_buffer = []
                samples_needed = min(
                    fetch_size,
                    (self.global_max_entries - global_yielded) if self.global_max_entries else fetch_size
                )

                # Calculate how many samples to take from each dataset based on weights
                total_weight = sum(spec.weight for spec in active_specs)
                samples_per_dataset = {}
                remaining_samples = samples_needed

                for spec in active_specs[:-1]:  # All but last
                    proportion = spec.weight / total_weight
                    samples = min(
                        int(samples_needed * proportion),
                        remaining_samples,
                        spec.max_entries - spec.current_idx if spec.max_entries else remaining_samples
                    )
                    samples_per_dataset[spec.repo_id] = samples
                    remaining_samples -= samples

                # Give remaining samples to last dataset
                if active_specs:
                    last_spec = active_specs[-1]
                    samples_per_dataset[last_spec.repo_id] = min(
                        remaining_samples,
                        last_spec.max_entries - last_spec.current_idx if last_spec.max_entries else remaining_samples
                    )

                # Fetch and process samples from each dataset
                all_samples = []
                for spec in active_specs:
                    samples_to_fetch = samples_per_dataset.get(spec.repo_id, 0)
                    if samples_to_fetch > 0:
                        ds = self._create_dataset_iterator(spec, samples_to_fetch)
                        if ds is not None:
                            for sample in ds:
                                text = sample.get(spec.text_column, "")
                                tokenized = self._tokenize_sample(text)
                                all_samples.append((spec.weight, tokenized))
                                spec.current_idx += 1

                # Shuffle samples based on weights
                if all_samples:
                    random.shuffle(all_samples)
                    batch_buffer = [sample for _, sample in all_samples]

                # Yield batches from buffer
                while len(batch_buffer) >= self.batch_size:
                    batch = self._create_batch(batch_buffer[:self.batch_size])
                    yield batch
                    batch_buffer = batch_buffer[self.batch_size:]
                    global_yielded += self.batch_size

                    if self.global_max_entries and global_yielded >= self.global_max_entries:
                        return

                # Handle remaining samples
                if batch_buffer and not active_specs:  # No more data available
                    batch = self._create_batch(batch_buffer)
                    yield batch
                    return

        else:
            # Sequential processing with chunked fetching
            for spec in self.dataset_specs:
                while spec.max_entries is None or spec.current_idx < spec.max_entries:
                    if self.global_max_entries and global_yielded >= self.global_max_entries:
                        return

                    # Calculate how many samples to fetch
                    samples_to_fetch = fetch_size
                    if spec.max_entries is not None:
                        remaining_in_dataset = spec.max_entries - spec.current_idx
                        samples_to_fetch = min(fetch_size, remaining_in_dataset)
                    if self.global_max_entries:
                        remaining_global = self.global_max_entries - global_yielded
                        samples_to_fetch = min(samples_to_fetch, remaining_global)

                    if samples_to_fetch <= 0:
                        break

                    # Fetch the next chunk
                    ds = self._create_dataset_iterator(spec, samples_to_fetch)
                    if ds is None:
                        break

                    batch_buffer = []
                    for sample in ds:
                        text = sample.get(spec.text_column, "")
                        tokenized = self._tokenize_sample(text)
                        batch_buffer.append(tokenized)
                        spec.current_idx += 1

                        # Yield complete batches
                        if len(batch_buffer) >= self.batch_size:
                            batch = self._create_batch(batch_buffer[:self.batch_size])
                            yield batch
                            batch_buffer = batch_buffer[self.batch_size:]
                            global_yielded += self.batch_size

                            if self.global_max_entries and global_yielded >= self.global_max_entries:
                                # Yield any remaining samples before returning
                                if batch_buffer:
                                    batch = self._create_batch(batch_buffer)
                                    yield batch
                                return

                    # Yield remaining samples from this chunk
                    if batch_buffer:
                        # Only yield if we're done with this dataset or global limit reached
                        if (spec.max_entries and spec.current_idx >= spec.max_entries) or \
                                (self.global_max_entries and global_yielded + len(batch_buffer) >= self.global_max_entries):
                            batch = self._create_batch(batch_buffer)
                            yield batch
                            global_yielded += len(batch_buffer)
                            if self.global_max_entries and global_yielded >= self.global_max_entries:
                                return

    def _create_batch(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack individual samples into a batch."""
        if not samples:
            return {
                "input_ids": torch.empty(0, self.sequence_length, dtype=torch.long).to(device=self.device, non_blocking=True),
                "attention_mask": torch.empty(0, self.sequence_length, dtype=torch.bool).to(device=self.device, non_blocking=True)
            }

        return {
            "input_ids": torch.stack([s["input_ids"] for s in samples]),
            "attention_mask": torch.stack([s["attention_mask"] for s in samples])
        }

    def __len__(self) -> int:
        total_samples = self.global_max_entries if self.global_max_entries else \
            sum(spec.max_entries for spec in self.dataset_specs if spec.max_entries is not None)
        return total_samples // self.batch_size  # Return number of batches, not samples


# Example usage
if __name__ == "__main__":
    from tokenizers import Tokenizer

    # Example setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset specifications
    dataset_specs = [
        BoundedStreamingDataset(
            repo_id="wikimedia/wikipedia",
            subset="20231101.en",
            skip_first=0,
            max_entries=10000,
            weight=0.4
        ),
        BoundedStreamingDataset(
            repo_id="roneneldan/TinyStories",
            skip_first=0,
            max_entries=5000,
            weight=0.1
        ),
        BoundedStreamingDataset(
            repo_id="HuggingFaceFW/fineweb",
            skip_first=0,
            max_entries=15000,
            weight=0.5
        ),
    ]

    # Load tokenizer (example - replace with your actual tokenizer)
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tokenizer = Tokenizer.from_pretrained(model_name)
    pad_token_id = tokenizer.token_to_id("<|im_end|>") or 0

    # Create dataset with sequential processing
    dataset_sequential = MultiBoundedStreamingDataset(
        dataset_specs=dataset_specs,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        sequence_length=512,
        device=device,
        global_max_entries=100,
        weighted_sampling=False  # Sequential processing
    )

    # Create dataset with weighted sampling
    dataset_weighted = MultiBoundedStreamingDataset(
        dataset_specs=dataset_specs,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
        sequence_length=512,
        device=device,
        global_max_entries=100,
        weighted_sampling=True,  # Weighted sampling
        seed=42
    )

    # Test iteration
    print("Testing sequential dataset:")
    for i, batch in enumerate(dataset_sequential):
        if i >= 5:
            break
        print(f"Batch {i}: input_ids shape = {batch['input_ids'].shape}, "
              f"attention_mask shape = {batch['attention_mask'].shape}")

    print("\nTesting weighted sampling dataset:")
    for i, batch in enumerate(dataset_weighted):
        if i >= 5:
            break
        print(f"Batch {i}: input_ids shape = {batch['input_ids'].shape}, "
              f"attention_mask shape = {batch['attention_mask'].shape}")
