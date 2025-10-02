import random
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


class BoundedStreamingDataset(IterableDataset):
    """
    Streaming dataset that tokenizes text on-the-fly and creates fixed-length sequences.

    Key improvements:
    - Uses a buffer to accumulate tokens and yield complete windows
    - Properly handles stride for overlapping sequences
    - Processes data incrementally without loading entire stream into memory
    - Can be prepared (tokenized and saved) and loaded for faster iteration
    """

    def __init__(
            self,
            data_source: Union[str, HFIterableDataset],
            tokenizer: Tokenizer,
            pad_token_id: int,
            sequence_length: int = 512,
            stride: Optional[int] = None,
            text_column: str = "text",
            skip_first: int = 0,
            max_entries: Optional[int] = None,
            device: torch.device = torch.device("cpu"),
            preloaded_data: Optional[List[Dict]] = None
    ):
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if stride is not None and stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if skip_first < 0:
            raise ValueError(f"skip_first must be non-negative, got {skip_first}")
        if max_entries is not None and max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")

        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.text_column = text_column
        self.skip_first = skip_first
        self.max_entries = max_entries
        self.device = device
        self.preloaded_data = preloaded_data

        if preloaded_data is None:
            if isinstance(data_source, str):
                try:
                    self.dataset = load_dataset(data_source, split="train", streaming=True)
                except Exception as e:
                    raise ValueError(f"Failed to load dataset '{data_source}': {e}")
            else:
                self.dataset = data_source
        else:
            self.dataset = None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.preloaded_data is not None:
            for item in self.preloaded_data:
                yield {
                    "input_ids": torch.tensor(item["input_ids"], dtype=torch.long, device=self.device),
                    "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.bool, device=self.device)
                }
            return

        buffer = []
        entries_skipped = 0
        entries_yielded = 0

        try:
            text_iterator = (sample[self.text_column] for sample in self.dataset)
        except KeyError:
            raise KeyError(f"Text column '{self.text_column}' not found in dataset")

        for text in text_iterator:
            if not isinstance(text, str) or not text.strip():
                continue

            try:
                encoding = self.tokenizer.encode(text, add_special_tokens=False)
                tokens = encoding.ids
            except Exception as e:
                print(f"Warning: Failed to tokenize text: {e}")
                continue

            buffer.extend(tokens)

            while len(buffer) >= self.sequence_length:
                sequence = buffer[:self.sequence_length]
                buffer = buffer[self.stride:]

                if entries_skipped < self.skip_first:
                    entries_skipped += 1
                    continue

                if self.max_entries and entries_yielded >= self.max_entries:
                    return

                padded_sequence, attention_mask = apply_padding(
                    sequence,
                    self.sequence_length,
                    self.pad_token_id
                )

                yield {
                    "input_ids": torch.tensor(padded_sequence, dtype=torch.long, device=self.device),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.bool, device=self.device)
                }

                entries_yielded += 1

        if buffer and (self.max_entries is None or entries_yielded < self.max_entries):
            if entries_skipped >= self.skip_first:
                padded_sequence, attention_mask = apply_padding(
                    buffer,
                    self.sequence_length,
                    self.pad_token_id
                )

                yield {
                    "input_ids": torch.tensor(padded_sequence, dtype=torch.long, device=self.device),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.bool, device=self.device)
                }

    @staticmethod
    def _process_entries(args):
        """Worker function: load tokenizer, process entries, return tokenized samples."""
        tokenizer_path, pad_token_id, sequence_length, stride, text_column, entries = args

        tokenizer = Tokenizer.from_file(tokenizer_path)
        results = []
        buffer = []

        for entry in entries:
            text = entry.get(text_column) if isinstance(entry, dict) else entry

            if not isinstance(text, str) or not text.strip():
                continue

            try:
                tokens = tokenizer.encode(text, add_special_tokens=False).ids
            except Exception:
                continue

            buffer.extend(tokens)

            while len(buffer) >= sequence_length:
                sequence = buffer[:sequence_length]
                buffer = buffer[stride:]

                padded_sequence, attention_mask = apply_padding(
                    sequence,
                    sequence_length,
                    pad_token_id
                )

                results.append({
                    "input_ids": padded_sequence,
                    "attention_mask": attention_mask
                })

        if buffer:
            padded_sequence, attention_mask = apply_padding(
                buffer,
                sequence_length,
                pad_token_id
            )
            results.append({
                "input_ids": padded_sequence,
                "attention_mask": attention_mask
            })

        return results

    def prepare(
            self,
            output_path: Union[str, Path],
            num_workers: Optional[int] = None
    ):
        """
        Tokenize and save the entire dataset to a JSON file using parallel workers.

        Args:
            output_path: Path to save the prepared dataset
            num_workers: Number of parallel workers (None = cpu_count(), 1 = no parallelization)
        """
        if self.preloaded_data is not None:
            raise ValueError("Cannot prepare an already loaded dataset")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tokenizer_path = output_path.parent / f"{output_path.stem}_tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))

        try:
            text_iterator = (sample[self.text_column] for sample in self.dataset)
        except KeyError:
            raise KeyError(f"Text column '{self.text_column}' not found in dataset")

        # Collect all entries
        print("Collecting entries...")
        entries = []
        for text in tqdm(text_iterator):
            if self.max_entries and len(entries) >= self.max_entries:
                break
            if isinstance(text, str) and text.strip():
                entries.append(text)

        print(f"Processing {len(entries)} entries with {num_workers or cpu_count()} workers...")

        if num_workers == 1:
            # Single-threaded processing
            args = (str(tokenizer_path), self.pad_token_id, self.sequence_length,
                    self.stride, self.text_column, entries)
            all_data = self._process_entries(args)
        else:
            # Parallel processing
            workers = num_workers or cpu_count()
            chunk_size = len(entries) // workers + 1
            chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]

            args = [
                (str(tokenizer_path), self.pad_token_id, self.sequence_length,
                 self.stride, self.text_column, chunk)
                for chunk in chunks
            ]

            all_data = []
            with Pool(workers) as pool:
                for chunk_results in tqdm(pool.imap(self._process_entries, args), total=len(chunks)):
                    all_data.extend(chunk_results)

        tokenizer_path.unlink()

        if self.skip_first > 0:
            all_data = all_data[self.skip_first:]

        print(f"Saving {len(all_data)} examples to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(all_data, f)

        print(f"Dataset prepared and saved to {output_path}")

    @classmethod
    def load(
            cls,
            input_path: Union[str, Path],
            tokenizer: Tokenizer,
            pad_token_id: int,
            sequence_length: int = 512,
            device: torch.device = torch.device("cpu")
    ):
        """
        Load a prepared dataset from JSON file.

        Args:
            input_path: Path to the prepared dataset JSON file
            tokenizer: Tokenizer instance (for consistency)
            pad_token_id: Padding token ID
            sequence_length: Sequence length (should match preparation)
            device: Device to place tensors on

        Returns:
            BoundedStreamingDataset instance with preloaded data
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Prepared dataset not found at {input_path}")

        print(f"Loading prepared dataset from {input_path}...")
        with open(input_path, 'r') as f:
            preloaded_data = json.load(f)

        print(f"Loaded {len(preloaded_data)} examples")

        return cls(
            data_source="",
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            sequence_length=sequence_length,
            device=device,
            preloaded_data=preloaded_data
        )


class MultiBoundedStreamingDataset(IterableDataset):
    """
    Streams multiple datasets with weighted random sampling.

    Key improvements:
    - Each dataset maintains its own buffer for proper tokenization
    - Weighted sampling works correctly
    - Properly terminates when all datasets exhausted or global limit reached
    - Can be prepared (tokenized and saved) and loaded for faster iteration
    """

    def __init__(
            self,
            dataset_specs: List[Dict],
            tokenizer: Tokenizer,
            pad_token_id: int,
            sequence_length: int = 512,
            stride: Optional[int] = None,
            text_column: str = "text",
            device: torch.device = torch.device("cpu"),
            global_max_entries: Optional[int] = None,
            seed: Optional[int] = None,
            preloaded_data: Optional[Dict[str, List[Dict]]] = None
    ):
        super().__init__()

        if not dataset_specs:
            raise ValueError("dataset_specs cannot be empty")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if stride is not None and stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if global_max_entries is not None and global_max_entries <= 0:
            raise ValueError(f"global_max_entries must be positive, got {global_max_entries}")

        self.dataset_specs = dataset_specs
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.text_column = text_column
        self.device = device
        self.global_max_entries = global_max_entries
        self.seed = seed
        self.preloaded_data = preloaded_data

        if preloaded_data is None:
            for i, spec in enumerate(dataset_specs):
                if "name" not in spec:
                    raise ValueError(f"Dataset spec {i} missing required 'name' field")
                if not spec.get("max_entries"):
                    raise ValueError(
                        f"Dataset {spec['name']} must define max_entries "
                        "to avoid infinite iteration in streaming mode."
                    )
                if spec["max_entries"] <= 0:
                    raise ValueError(f"Dataset {spec['name']} max_entries must be positive")
                if spec.get("skip", 0) < 0:
                    raise ValueError(f"Dataset {spec['name']} skip must be non-negative")
                if spec.get("weight", 1.0) <= 0:
                    raise ValueError(f"Dataset {spec['name']} weight must be positive")

    def _create_dataset_iterator(self, spec: Dict):
        """Create an iterator for a single dataset with its own buffer."""
        try:
            ds = load_dataset(
                spec["name"],
                spec.get("subset", None),
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"Warning: Failed to load dataset {spec['name']}: {e}")
            return None

        try:
            text_iterator = (sample[self.text_column] for sample in ds)
        except KeyError:
            print(f"Warning: Text column '{self.text_column}' not found in {spec['name']}")
            return None

        buffer = []
        entries_skipped = 0
        entries_yielded = 0
        skip_first = spec.get("skip", 0)
        max_entries = spec["max_entries"]

        def generate():
            nonlocal buffer, entries_skipped, entries_yielded

            for text in text_iterator:
                if not isinstance(text, str) or not text.strip():
                    continue

                try:
                    encoding = self.tokenizer.encode(text, add_special_tokens=False)
                    tokens = encoding.ids
                except Exception as e:
                    print(f"Warning: Failed to tokenize text in {spec['name']}: {e}")
                    continue

                buffer.extend(tokens)

                while len(buffer) >= self.sequence_length:
                    sequence = buffer[:self.sequence_length]
                    buffer = buffer[self.stride:]

                    if entries_skipped < skip_first:
                        entries_skipped += 1
                        continue

                    if entries_yielded >= max_entries:
                        return

                    padded_sequence, attention_mask = apply_padding(
                        sequence,
                        self.sequence_length,
                        self.pad_token_id
                    )

                    yield {
                        "input_ids": torch.tensor(padded_sequence, dtype=torch.long, device=self.device),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool, device=self.device)
                    }

                    entries_yielded += 1

            if buffer and entries_yielded < max_entries and entries_skipped >= skip_first:
                padded_sequence, attention_mask = apply_padding(
                    buffer,
                    self.sequence_length,
                    self.pad_token_id
                )

                yield {
                    "input_ids": torch.tensor(padded_sequence, dtype=torch.long, device=self.device),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.bool, device=self.device)
                }

        return generate()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.preloaded_data is not None:
            if self.seed is not None:
                random.seed(self.seed)

            # Create list of (dataset_name, example_index) pairs with weights
            weighted_indices = []
            weights_list = []

            for spec in self.dataset_specs:
                dataset_name = spec["name"]
                weight = spec.get("weight", 1.0)

                if dataset_name in self.preloaded_data:
                    for idx in range(len(self.preloaded_data[dataset_name])):
                        weighted_indices.append((dataset_name, idx))
                        weights_list.append(weight)

            # Normalize weights
            total_weight = sum(weights_list)
            weights_list = [w / total_weight for w in weights_list]

            # Randomly sample
            num_samples = self.global_max_entries or len(weighted_indices)
            sampled_indices = random.choices(weighted_indices, weights=weights_list, k=num_samples)

            for dataset_name, idx in sampled_indices:
                item = self.preloaded_data[dataset_name][idx]
                yield {
                    "input_ids": torch.tensor(item["input_ids"], dtype=torch.long, device=self.device),
                    "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.bool, device=self.device)
                }
            return

        if self.seed is not None:
            random.seed(self.seed)

        active_iterators = []
        weights = []

        for spec in self.dataset_specs:
            iterator = self._create_dataset_iterator(spec)
            if iterator is not None:
                active_iterators.append(iterator)
                weights.append(spec.get("weight", 1.0))

        if not active_iterators:
            raise RuntimeError("No datasets could be loaded successfully")

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        global_yielded = 0

        while active_iterators:
            if self.global_max_entries and global_yielded >= self.global_max_entries:
                break

            iterator = random.choices(active_iterators, weights=weights, k=1)[0]

            try:
                example = next(iterator)
                yield example
                global_yielded += 1
            except StopIteration:
                idx = active_iterators.index(iterator)
                active_iterators.pop(idx)
                weights.pop(idx)

                if weights:
                    total_weight = sum(weights)
                    weights = [w / total_weight for w in weights]

    def __len__(self) -> int:
        if self.global_max_entries:
            return self.global_max_entries
        return sum(spec["max_entries"] for spec in self.dataset_specs)

    @staticmethod
    def _process_entries(args):
        """Worker function: load tokenizer, process entries, return tokenized samples."""
        tokenizer_path, pad_token_id, sequence_length, stride, text_column, entries = args

        tokenizer = Tokenizer.from_file(tokenizer_path)
        results = []
        buffer = []

        for entry in entries:
            text = entry.get(text_column) if isinstance(entry, dict) else entry

            if not isinstance(text, str) or not text.strip():
                continue

            try:
                tokens = tokenizer.encode(text, add_special_tokens=False).ids
            except Exception:
                continue

            buffer.extend(tokens)

            while len(buffer) >= sequence_length:
                sequence = buffer[:sequence_length]
                buffer = buffer[stride:]

                padded_sequence, attention_mask = apply_padding(
                    sequence,
                    sequence_length,
                    pad_token_id
                )

                results.append({
                    "input_ids": padded_sequence,
                    "attention_mask": attention_mask
                })

        if buffer:
            padded_sequence, attention_mask = apply_padding(
                buffer,
                sequence_length,
                pad_token_id
            )
            results.append({
                "input_ids": padded_sequence,
                "attention_mask": attention_mask
            })

        return results

    def prepare(
            self,
            output_path: Union[str, Path],
            num_workers: Optional[int] = None
    ):
        """
        Tokenize and save all datasets to a JSON file using parallel workers.

        Args:
            output_path: Path to save the prepared datasets
            num_workers: Number of parallel workers (None = cpu_count(), 1 = no parallelization)
        """
        if self.preloaded_data is not None:
            raise ValueError("Cannot prepare an already loaded dataset")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tokenizer_path = output_path.parent / f"{output_path.stem}_tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))

        all_datasets = {}

        for spec in self.dataset_specs:
            dataset_name = spec["name"]
            print(f"\nProcessing dataset: {dataset_name}")

            try:
                ds = load_dataset(
                    spec["name"],
                    spec.get("subset", None),
                    split="train",
                    streaming=True,
                )
            except Exception as e:
                print(f"Warning: Failed to load dataset {dataset_name}: {e}")
                continue

            try:
                text_iterator = (sample[self.text_column] for sample in ds)
            except KeyError:
                print(f"Warning: Text column '{self.text_column}' not found in {dataset_name}")
                continue

            max_entries = spec["max_entries"]
            skip_first = spec.get("skip", 0)

            print(f"Collecting entries from {dataset_name}...")
            entries = []
            for text in tqdm(text_iterator):
                if len(entries) >= max_entries + skip_first:
                    break
                if isinstance(text, str) and text.strip():
                    entries.append(text)

            print(f"Processing {len(entries)} entries with {num_workers or cpu_count()} workers...")

            if num_workers == 1:
                # Single-threaded processing
                args = (str(tokenizer_path), self.pad_token_id, self.sequence_length,
                        self.stride, self.text_column, entries)
                dataset_data = self._process_entries(args)
            else:
                # Parallel processing
                workers = num_workers or cpu_count()
                chunk_size = len(entries) // workers + 1
                chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]

                args = [
                    (str(tokenizer_path), self.pad_token_id, self.sequence_length,
                     self.stride, self.text_column, chunk)
                    for chunk in chunks
                ]

                dataset_data = []
                with Pool(workers) as pool:
                    for chunk_results in tqdm(pool.imap(self._process_entries, args), total=len(chunks)):
                        dataset_data.extend(chunk_results)

            if skip_first > 0:
                dataset_data = dataset_data[skip_first:]

            if len(dataset_data) > max_entries:
                dataset_data = dataset_data[:max_entries]

            all_datasets[dataset_name] = dataset_data
            print(f"Dataset {dataset_name}: {len(dataset_data)} examples")

        tokenizer_path.unlink()

        print(f"\nSaving all datasets to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(all_datasets, f)

        print(f"All datasets prepared and saved to {output_path}")

    @classmethod
    def load(
            cls,
            input_path: Union[str, Path],
            tokenizer: Tokenizer,
            pad_token_id: int,
            dataset_specs: List[Dict],
            sequence_length: int = 512,
            device: torch.device = torch.device("cpu"),
            global_max_entries: Optional[int] = None,
            seed: Optional[int] = None
    ):
        """
        Load prepared datasets from JSON file.

        Args:
            input_path: Path to the prepared datasets JSON file
            tokenizer: Tokenizer instance (for consistency)
            pad_token_id: Padding token ID
            dataset_specs: List of dataset specifications (for weights)
            sequence_length: Sequence length (should match preparation)
            device: Device to place tensors on
            global_max_entries: Global max entries limit
            seed: Random seed for weighted sampling

        Returns:
            MultiBoundedStreamingDataset instance with preloaded data
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Prepared datasets not found at {input_path}")

        print(f"Loading prepared datasets from {input_path}...")
        with open(input_path, 'r') as f:
            preloaded_data = json.load(f)

        total_examples = sum(len(data) for data in preloaded_data.values())
        print(f"Loaded {total_examples} examples from {len(preloaded_data)} datasets")

        return cls(
            dataset_specs=dataset_specs,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            sequence_length=sequence_length,
            device=device,
            global_max_entries=global_max_entries,
            seed=seed,
            preloaded_data=preloaded_data
        )