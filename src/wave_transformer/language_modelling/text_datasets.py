import random
import torch
import pickle
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


def _tokenize_batch(args):
    """Helper function for parallel tokenization"""
    texts, tokenizer_name, sequence_length, stride, pad_token_id = args

    # Load tokenizer in worker process
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)

    examples = []
    buffer = []

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue

        try:
            encoding = tokenizer.encode(text, add_special_tokens=False)
            tokens = encoding.ids
        except Exception:
            continue

        buffer.extend(tokens)

        # Process complete windows
        while len(buffer) >= sequence_length:
            sequence = buffer[:sequence_length]
            buffer = buffer[stride:]

            padded_sequence, attention_mask = apply_padding(
                sequence,
                sequence_length,
                pad_token_id
            )

            examples.append({
                "input_ids": torch.tensor(padded_sequence, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)
            })

    # Handle remaining buffer
    if buffer:
        padded_sequence, attention_mask = apply_padding(
            buffer,
            sequence_length,
            pad_token_id
        )

        examples.append({
            "input_ids": torch.tensor(padded_sequence, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)
        })

    return examples


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
            device: torch.device = torch.device("cpu")
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

        if isinstance(data_source, str):
            try:
                self.dataset = load_dataset(data_source, split="train", streaming=True)
            except Exception as e:
                raise ValueError(f"Failed to load dataset '{data_source}': {e}")
        else:
            self.dataset = data_source

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
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

            # Process complete windows from buffer
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

        # Handle remaining tokens in buffer
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


class PreparedDataset(Dataset):
    """
    Dataset that loads pre-prepared examples from disk.
    Much faster than streaming datasets for repeated training runs.
    """

    def __init__(self, data_path: Union[str, Path], device: torch.device = torch.device("cpu")):
        self.data_path = Path(data_path)
        self.device = device

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        print(f"Loading dataset from {self.data_path}...")
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        self.examples = data['examples']
        self.metadata = data.get('metadata', {})

        print(f"Loaded {len(self.examples)} examples")
        if self.metadata:
            print(f"Metadata: {self.metadata}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"].to(self.device),
            "attention_mask": self.examples[idx]["attention_mask"].to(self.device),
        }


def prepare_and_save_dataset(
        data_source: Union[str, HFIterableDataset],
        tokenizer: Tokenizer,
        pad_token_id: int,
        save_path: Union[str, Path],
        sequence_length: int = 512,
        stride: Optional[int] = None,
        text_column: str = "text",
        skip_first: int = 0,
        max_entries: Optional[int] = None,
        subset: Optional[str] = None,
        num_workers: Optional[int] = None,
        batch_size: int = 1000,
):
    """
    Process a streaming dataset and save it to disk for fast loading.

    Args:
        data_source: HuggingFace dataset name or dataset object
        tokenizer: Tokenizer to use
        pad_token_id: ID for padding token
        save_path: Where to save the prepared dataset
        sequence_length: Length of each sequence
        stride: Stride for sliding window (default: sequence_length, no overlap)
        text_column: Name of text column in dataset
        skip_first: Number of entries to skip
        max_entries: Maximum number of entries to process
        subset: Dataset subset/configuration name
        num_workers: Number of parallel workers (None = single process, 0 = auto-detect CPUs)
        batch_size: Number of texts to process per worker batch (for parallel processing)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    stride = stride or sequence_length

    # Determine if we should use parallel processing
    use_parallel = num_workers is not None
    if use_parallel:
        if num_workers == 0:
            num_workers = max(1, cpu_count() - 1)
        print(f"Preparing dataset from {data_source} using {num_workers} workers...")
    else:
        print(f"Preparing dataset from {data_source} (single process)...")

    print(f"Sequence length: {sequence_length}, Stride: {stride}")

    # Load dataset
    if isinstance(data_source, str):
        try:
            dataset = load_dataset(data_source, subset, split="train", streaming=True)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{data_source}': {e}")
    else:
        dataset = data_source

    try:
        text_iterator = (sample[text_column] for sample in dataset)
    except KeyError:
        raise KeyError(f"Text column '{text_column}' not found in dataset")

    # Collect texts first
    print("Loading texts...")
    all_texts = []
    texts_skipped = 0

    with tqdm(desc="Loading texts", unit=" texts") as pbar:
        for text in text_iterator:
            if not isinstance(text, str) or not text.strip():
                continue

            if texts_skipped < skip_first:
                texts_skipped += 1
                continue

            all_texts.append(text)
            pbar.update(1)

            if max_entries and len(all_texts) >= max_entries * 2:  # Load more than needed
                break

    print(f"Loaded {len(all_texts)} texts")

    # Process texts
    if use_parallel:
        # Get tokenizer name for workers
        tokenizer_name = tokenizer.get_vocab()  # This won't work, need to pass the name
        # Workaround: assume tokenizer has a name or we pass it
        # For now, we'll use a different approach - pass the tokenizer directly

        # Split texts into batches
        text_batches = [all_texts[i:i + batch_size] for i in range(0, len(all_texts), batch_size)]

        print(f"Processing {len(text_batches)} batches with {num_workers} workers...")

        # Get tokenizer identifier (try common attributes)
        tokenizer_id = None
        for attr in ['name_or_path', '_name_or_path', 'name']:
            if hasattr(tokenizer, attr):
                tokenizer_id = getattr(tokenizer, attr)
                break

        if tokenizer_id is None:
            print("Warning: Could not identify tokenizer, falling back to single process")
            use_parallel = False
        else:
            # Prepare arguments for workers
            worker_args = [
                (batch, tokenizer_id, sequence_length, stride, pad_token_id)
                for batch in text_batches
            ]

            # Process in parallel
            all_examples = []
            with Pool(num_workers) as pool:
                for batch_examples in tqdm(
                        pool.imap(_tokenize_batch, worker_args),
                        total=len(worker_args),
                        desc="Tokenizing batches"
                ):
                    all_examples.extend(batch_examples)
                    if max_entries and len(all_examples) >= max_entries:
                        break

            examples = all_examples[:max_entries] if max_entries else all_examples

    # Fall back to single process if parallel didn't work
    if not use_parallel:
        examples = []
        buffer = []
        entries_yielded = 0

        with tqdm(total=len(all_texts), desc="Processing texts", unit=" texts") as pbar:
            for text in all_texts:
                try:
                    encoding = tokenizer.encode(text, add_special_tokens=False)
                    tokens = encoding.ids
                except Exception as e:
                    pbar.update(1)
                    continue

                buffer.extend(tokens)

                # Process complete windows
                while len(buffer) >= sequence_length:
                    sequence = buffer[:sequence_length]
                    buffer = buffer[stride:]

                    if max_entries and entries_yielded >= max_entries:
                        break

                    padded_sequence, attention_mask = apply_padding(
                        sequence,
                        sequence_length,
                        pad_token_id
                    )

                    examples.append({
                        "input_ids": torch.tensor(padded_sequence, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)
                    })

                    entries_yielded += 1

                pbar.update(1)

                if max_entries and entries_yielded >= max_entries:
                    break

        # Handle remaining buffer
        if buffer and (max_entries is None or entries_yielded < max_entries):
            padded_sequence, attention_mask = apply_padding(
                buffer,
                sequence_length,
                pad_token_id
            )

            examples.append({
                "input_ids": torch.tensor(padded_sequence, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)
            })

    # Save to disk
    print(f"\nSaving {len(examples)} examples to {save_path}...")
    data = {
        'examples': examples,
        'metadata': {
            'data_source': data_source if isinstance(data_source, str) else 'custom',
            'subset': subset,
            'sequence_length': sequence_length,
            'stride': stride,
            'text_column': text_column,
            'skip_first': skip_first,
            'num_examples': len(examples),
            'tokenizer_vocab_size': tokenizer.get_vocab_size(),
            'num_workers': num_workers if use_parallel else 1,
        }
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = save_path.stat().st_size / (1024 ** 2)
    print(f"✓ Dataset saved successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Examples: {len(examples)}")

    return save_path


def prepare_and_save_multi_dataset(
        dataset_specs: List[Dict],
        tokenizer: Tokenizer,
        pad_token_id: int,
        save_path: Union[str, Path],
        sequence_length: int = 512,
        stride: Optional[int] = None,
        text_column: str = "text",
        global_max_entries: Optional[int] = None,
        seed: Optional[int] = None,
):
    """
    Process multiple streaming datasets with weighted sampling and save to disk.

    Args:
        dataset_specs: List of dataset specifications with 'name', 'weight', 'max_entries', etc.
        tokenizer: Tokenizer to use
        pad_token_id: ID for padding token
        save_path: Where to save the prepared dataset
        sequence_length: Length of each sequence
        stride: Stride for sliding window
        text_column: Name of text column in datasets
        global_max_entries: Maximum total entries across all datasets
        seed: Random seed for reproducibility
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        random.seed(seed)

    print(f"Preparing multi-dataset from {len(dataset_specs)} sources...")

    # Create temporary files for each dataset
    temp_datasets = []
    for i, spec in enumerate(dataset_specs):
        temp_path = save_path.parent / f"temp_dataset_{i}.pkl"
        print(f"\nProcessing dataset {i + 1}/{len(dataset_specs)}: {spec['name']}")

        prepare_and_save_dataset(
            data_source=spec['name'],
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            save_path=temp_path,
            sequence_length=sequence_length,
            stride=stride,
            text_column=text_column,
            skip_first=spec.get('skip', 0),
            max_entries=spec['max_entries'],
            subset=spec.get('subset', None),
        )

        temp_datasets.append({
            'path': temp_path,
            'weight': spec.get('weight', 1.0),
            'name': spec['name']
        })

    # Load and combine datasets with weighted sampling
    print("\nCombining datasets with weighted sampling...")
    all_examples = []
    weights = [d['weight'] for d in temp_datasets]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Load all datasets
    loaded_datasets = []
    for temp_ds in temp_datasets:
        with open(temp_ds['path'], 'rb') as f:
            data = pickle.load(f)
            loaded_datasets.append(data['examples'])

    # Determine how many samples to take from each dataset
    if global_max_entries:
        target_samples = [int(global_max_entries * w) for w in normalized_weights]
        # Adjust last to hit exact target
        target_samples[-1] = global_max_entries - sum(target_samples[:-1])
    else:
        target_samples = [len(ds) for ds in loaded_datasets]

    # Sample from each dataset
    for i, (examples, target) in enumerate(zip(loaded_datasets, target_samples)):
        actual = min(target, len(examples))
        if actual < len(examples):
            sampled = random.sample(examples, actual)
        else:
            sampled = examples
        all_examples.extend(sampled)
        print(f"  {temp_datasets[i]['name']}: {actual} examples (weight: {weights[i]:.2f})")

    # Shuffle combined dataset
    random.shuffle(all_examples)

    # Save combined dataset
    print(f"\nSaving combined dataset to {save_path}...")
    data = {
        'examples': all_examples,
        'metadata': {
            'dataset_specs': dataset_specs,
            'sequence_length': sequence_length,
            'stride': stride,
            'text_column': text_column,
            'num_examples': len(all_examples),
            'seed': seed,
        }
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Clean up temp files
    for temp_ds in temp_datasets:
        temp_ds['path'].unlink()

    file_size_mb = save_path.stat().st_size / (1024 ** 2)
    print(f"✓ Combined dataset saved successfully!")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Total examples: {len(all_examples)}")

    return save_path


class MultiBoundedStreamingDataset(IterableDataset):
    """
    Streams multiple datasets with weighted random sampling.

    Key improvements:
    - Each dataset maintains its own buffer for proper tokenization
    - Weighted sampling works correctly
    - Properly terminates when all datasets exhausted or global limit reached
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

                # Yield complete windows
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

            # Handle remaining buffer
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
        if self.seed is not None:
            random.seed(self.seed)

        # Initialize all dataset iterators
        active_iterators = []
        weights = []

        for spec in self.dataset_specs:
            iterator = self._create_dataset_iterator(spec)
            if iterator is not None:
                active_iterators.append(iterator)
                weights.append(spec.get("weight", 1.0))

        if not active_iterators:
            raise RuntimeError("No datasets could be loaded successfully")

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        global_yielded = 0

        # Randomly sample until exhausted or limit reached
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