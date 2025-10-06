# instrumented MultiBoundedStreamingDataset with debug stats and sequence packing
from __future__ import annotations
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Dict, Iterator, Tuple, Callable
import threading
import queue
import random
import time
import itertools
from collections import deque

import torch
from torch.utils.data import IterableDataset
from tokenizers import Tokenizer
from datasets import load_dataset

import json
from pathlib import Path
from typing import Union


@dataclass
class BoundedStreamingDataset:
    repo_id: str
    subset: Optional[str] = None
    split: str = "train"
    skip_first: int = 0
    max_entries: Optional[int] = None
    text_column: str = "text"
    weight: float = 1.0
    current_idx: int = 0  # maintained per-iterator


class _Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.dt = time.perf_counter() - self.t0


class _Stats:
    """Thread-safe rolling stats for producer/consumer & queue."""

    def __init__(self, win_seconds: float = 60.0):
        self.lock = threading.Lock()
        self.start_time = time.time()
        # counters
        self.produced = 0
        self.consumed = 0
        self.refills = 0
        # packing stats
        self.packed_sequences = 0
        self.total_sequences = 0
        self.total_tokens_used = 0
        self.total_tokens_capacity = 0
        # time buckets
        self.time_fetch = 0.0
        self.time_tokenize = 0.0
        self.time_pack = 0.0
        self.time_put_block = 0.0
        self.time_get_block = 0.0
        # queue size (updated by prod/cons)
        self.qsize = 0
        self.qmax = 1
        # rolling rates (timestamps for produced/consumed increments)
        self.win_seconds = win_seconds
        self.prod_hist = deque()  # [(t, n)]
        self.cons_hist = deque()

    def on_produced(self, n: int, qsize: int, qmax: int):
        now = time.time()
        with self.lock:
            self.produced += n
            self.qsize = qsize
            self.qmax = qmax
            self.prod_hist.append((now, n))
            # drop old
            while self.prod_hist and now - self.prod_hist[0][0] > self.win_seconds:
                self.prod_hist.popleft()

    def on_consumed(self, n: int, qsize: int, qmax: int):
        now = time.time()
        with self.lock:
            self.consumed += n
            self.qsize = qsize
            self.qmax = qmax
            self.cons_hist.append((now, n))
            while self.cons_hist and now - self.cons_hist[0][0] > self.win_seconds:
                self.cons_hist.popleft()

    def add_fetch_time(self, dt: float):
        with self.lock:
            self.time_fetch += dt

    def add_tokenize_time(self, dt: float):
        with self.lock:
            self.time_tokenize += dt

    def add_pack_time(self, dt: float):
        with self.lock:
            self.time_pack += dt

    def add_put_block_time(self, dt: float):
        if dt <= 0:
            return
        with self.lock:
            self.time_put_block += dt

    def add_get_block_time(self, dt: float):
        if dt <= 0:
            return
        with self.lock:
            self.time_get_block += dt

    def inc_refill(self):
        with self.lock:
            self.refills += 1

    def record_packing(self, num_sequences: int, tokens_used: int, tokens_capacity: int):
        with self.lock:
            self.total_sequences += num_sequences
            if num_sequences > 1:
                self.packed_sequences += 1
            self.total_tokens_used += tokens_used
            self.total_tokens_capacity += tokens_capacity

    def snapshot(self):
        with self.lock:
            now = time.time()
            prod_rate = sum(n for _, n in self.prod_hist)
            cons_rate = sum(n for _, n in self.cons_hist)
            prod_span = (now - self.prod_hist[0][0]) if self.prod_hist else 1.0
            cons_span = (now - self.cons_hist[0][0]) if self.cons_hist else 1.0
            prod_sps = prod_rate / max(prod_span, 1e-6)
            cons_sps = cons_rate / max(cons_span, 1e-6)
            q_pct = (self.qsize / self.qmax * 100.0) if self.qmax else 0.0
            packing_rate = (self.packed_sequences / self.total_sequences * 100.0) if self.total_sequences else 0.0
            token_efficiency = (
                        self.total_tokens_used / self.total_tokens_capacity * 100.0) if self.total_tokens_capacity else 0.0
            return {
                "uptime_s": now - self.start_time,
                "produced": self.produced,
                "consumed": self.consumed,
                "refills": self.refills,
                "producer_sps": prod_sps,
                "consumer_sps": cons_sps,
                "queue_size": self.qsize,
                "queue_capacity": self.qmax,
                "queue_fill_pct": q_pct,
                "packing_rate_pct": packing_rate,
                "token_efficiency_pct": token_efficiency,
                "packed_sequences": self.packed_sequences,
                "total_sequences": self.total_sequences,
                "time_fetch_s": self.time_fetch,
                "time_tokenize_s": self.time_tokenize,
                "time_pack_s": self.time_pack,
                "time_put_block_s": self.time_put_block,
                "time_get_block_s": self.time_get_block,
            }


class MultiBoundedStreamingDataset(IterableDataset):
    """
    Streaming + batch tokenization + background prefetch with DEBUG STATS and SEQUENCE PACKING.

    - Produces CPU tensors (recommended). Move to CUDA in the train loop with non_blocking=True.
    - Debug: set debug=True to log every `log_interval_s` seconds, or pass a `stats_callback(dict)`.
    - Packing: set use_packing=True to combine multiple sequences when >packing_threshold space remains.
    """

    SENTINEL = ("__END_OF_STREAM__", None)

    def __init__(
            self,
            dataset_specs: List[BoundedStreamingDataset],
            tokenizer: Tokenizer,
            pad_token_id: int,
            sequence_length: int = 512,
            batch_size: int = 32,
            prefetch_batches: int = 128,
            prefetch_chunk_batches: int = 16,
            tokenizer_batch_size: int = 256,
            weighted_sampling: bool = False,
            global_max_entries: Optional[int] = None,
            seed: Optional[int] = None,
            move_to_device: bool = False,
            device: Optional[torch.device] = None,
            use_packing: bool = False,
            packing_threshold: float = 0.05,
            debug: bool = False,
            log_interval_s: float = 5.0,
            stats_window_s: float = 60.0,
            stats_callback: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__()
        assert prefetch_batches >= 1 and prefetch_chunk_batches >= 1
        assert 0.0 <= packing_threshold <= 1.0
        self.dataset_specs_template = dataset_specs
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.prefetch_chunk_batches = prefetch_chunk_batches
        self.tokenizer_batch_size = tokenizer_batch_size
        self.weighted_sampling = weighted_sampling
        self.global_max_entries = global_max_entries
        self.seed = seed
        self.move_to_device = move_to_device
        self.device = device
        self.use_packing = use_packing
        self.packing_threshold = packing_threshold
        self.debug = debug
        self.log_interval_s = log_interval_s
        self.stats_callback = stats_callback

        # stats
        self.stats = _Stats(win_seconds=stats_window_s)

        # tokenizer setup - when packing, don't enable padding/truncation here
        # (we'll handle it manually in the packing logic)
        if not use_packing:
            self.tokenizer.enable_padding(
                pad_id=pad_token_id,
                pad_token=self.tokenizer.decode([pad_token_id], skip_special_tokens=False),
                length=sequence_length,
            )
            self.tokenizer.enable_truncation(max_length=sequence_length)

    def _clone_specs(self) -> List[BoundedStreamingDataset]:
        return [dataclasses.replace(s, current_idx=0) for s in self.dataset_specs_template]

    def _ensure_stream(self, spec: BoundedStreamingDataset):
        if not hasattr(spec, "_stream"):
            ds = load_dataset(spec.repo_id, spec.subset, split=spec.split, streaming=True)
            if spec.skip_first > 0:
                ds = ds.skip(spec.skip_first)
            spec._stream = iter(ds)

    def _texts_to_tensors(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        if not texts:
            return []
        with _Timer() as t_tok:
            encs = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        self.stats.add_tokenize_time(t_tok.dt)

        out = []
        for e in encs:
            ids = torch.tensor(e.ids, dtype=torch.long)
            mask = torch.tensor(e.attention_mask, dtype=torch.bool)
            if self.move_to_device and self.device is not None:
                ids = ids.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)
            out.append({"input_ids": ids, "attention_mask": mask})
        return out

    def _pack_sequences(self, tokenized: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """Pack multiple sequences together when space permits."""
        if not self.use_packing or not tokenized:
            return tokenized

        with _Timer() as t_pack:
            packed = []
            current_ids = []
            current_mask = []

            for item in tokenized:
                ids = item["input_ids"]
                mask = item["attention_mask"]

                # Calculate current and new lengths
                current_len = len(current_ids)
                new_len = len(ids)
                combined_len = current_len + new_len

                # Check if we should start a new packed sequence
                if current_len == 0:
                    # First sequence in pack
                    current_ids = ids.tolist()
                    current_mask = mask.tolist()
                elif combined_len <= self.sequence_length:
                    # Can add to current pack
                    remaining_space = self.sequence_length - current_len
                    threshold_space = int(self.sequence_length * self.packing_threshold)

                    if remaining_space >= threshold_space:
                        # Enough space remaining, add this sequence
                        current_ids.extend(ids.tolist())
                        current_mask.extend(mask.tolist())
                    else:
                        # Not enough space, finalize current and start new
                        self._finalize_packed_sequence(current_ids, current_mask, packed)
                        current_ids = ids.tolist()
                        current_mask = mask.tolist()
                else:
                    # Would exceed max length, finalize current and start new
                    self._finalize_packed_sequence(current_ids, current_mask, packed)
                    current_ids = ids.tolist()
                    current_mask = mask.tolist()

            # Finalize last sequence
            if current_ids:
                self._finalize_packed_sequence(current_ids, current_mask, packed)

        self.stats.add_pack_time(t_pack.dt)
        return packed

    def _finalize_packed_sequence(self, ids: List[int], mask: List[bool],
                                  output: List[Dict[str, torch.Tensor]]):
        """Pad and convert a packed sequence to tensor."""
        current_len = len(ids)

        # Record packing stats
        num_sequences = sum(1 for i in range(len(ids) - 1) if ids[i] != self.pad_token_id and
                            (i == 0 or ids[i - 1] == self.pad_token_id))
        self.stats.record_packing(num_sequences, current_len, self.sequence_length)

        # Pad to sequence_length
        if current_len < self.sequence_length:
            padding_len = self.sequence_length - current_len
            ids.extend([self.pad_token_id] * padding_len)
            mask.extend([False] * padding_len)
        elif current_len > self.sequence_length:
            # Truncate if somehow exceeded
            ids = ids[:self.sequence_length]
            mask = mask[:self.sequence_length]

        # Convert to tensors
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        if self.move_to_device and self.device is not None:
            ids_tensor = ids_tensor.to(self.device, non_blocking=True)
            mask_tensor = mask_tensor.to(self.device, non_blocking=True)

        output.append({"input_ids": ids_tensor, "attention_mask": mask_tensor})

    def _fetch_tokenized_chunk(self, spec: BoundedStreamingDataset, samples_to_fetch: int) -> List[
        Dict[str, torch.Tensor]]:
        self._ensure_stream(spec)
        if spec.max_entries is not None:
            remaining = max(0, spec.max_entries - spec.current_idx)
            samples_to_fetch = min(samples_to_fetch, remaining)
        if samples_to_fetch <= 0:
            return []

        with _Timer() as t_fetch:
            it = itertools.islice(spec._stream, samples_to_fetch)
            buf, tokenized = [], []
            for row in it:
                text = row.get(spec.text_column, "")
                buf.append(text)
                spec.current_idx += 1
                if len(buf) >= self.tokenizer_batch_size:
                    tokenized.extend(self._texts_to_tensors(buf))
                    buf.clear()
            if buf:
                tokenized.extend(self._texts_to_tensors(buf))
        self.stats.add_fetch_time(t_fetch.dt)

        # Apply packing if enabled
        if self.use_packing:
            tokenized = self._pack_sequences(tokenized)

        return tokenized

    def _producer_base(self, q, specs, stop_event, weighted: bool):
        fetch_size = self.batch_size * self.prefetch_chunk_batches
        global_yielded = 0
        try:
            while not stop_event.is_set():
                # build a fetch plan
                if weighted and len(specs) > 1:
                    active = [s for s in specs if (s.max_entries is None or s.current_idx < s.max_entries)]
                    if not active:
                        break
                    remaining_global = (
                                self.global_max_entries - global_yielded) if self.global_max_entries is not None else fetch_size
                    if remaining_global <= 0:
                        break
                    total_w = sum(max(1e-8, s.weight) for s in active)
                    samples_left = min(fetch_size, remaining_global)
                    plan = []
                    for s in active[:-1]:
                        quota = int(samples_left * (max(1e-8, s.weight) / total_w))
                        plan.append((s, quota));
                        samples_left -= quota
                    plan.append((active[-1], samples_left))
                else:
                    # sequential through specs
                    plan = []
                    for s in specs:
                        if s.max_entries is None or s.current_idx < s.max_entries:
                            remaining_global = (
                                        self.global_max_entries - global_yielded) if self.global_max_entries is not None else fetch_size
                            if remaining_global <= 0:
                                break
                            plan = [(s, min(fetch_size, remaining_global))]
                            break
                    if not plan:
                        break

                self.stats.inc_refill()
                produced_this_refill = 0

                # execute plan
                for spec, n in plan:
                    if n <= 0:
                        continue
                    toks = self._fetch_tokenized_chunk(spec, n)
                    # enqueue
                    for s in toks:
                        t0 = time.perf_counter()
                        q.put(("ok", s))
                        self.stats.add_put_block_time(time.perf_counter() - t0)
                        produced_this_refill += 1
                        global_yielded += 1
                        self.stats.on_produced(1, q.qsize(), q.maxsize)
                        if self.global_max_entries is not None and global_yielded >= self.global_max_entries:
                            q.put(MultiBoundedStreamingDataset.SENTINEL)
                            return

                time.sleep(0)

            q.put(MultiBoundedStreamingDataset.SENTINEL)
        except Exception as e:
            q.put(("error", {"msg": str(e)}))
            q.put(MultiBoundedStreamingDataset.SENTINEL)

    def _logger_loop(self, stop_event: threading.Event):
        next_log = time.time() + self.log_interval_s
        while not stop_event.is_set():
            now = time.time()
            if now >= next_log:
                snap = self.stats.snapshot()
                if self.stats_callback:
                    try:
                        self.stats_callback(snap)
                    except Exception:
                        pass
                if self.debug:
                    msg = (f"[DatasetStats] up:{snap['uptime_s']:.1f}s | prod:{snap['producer_sps']:.2f}/s "
                           f"| cons:{snap['consumer_sps']:.2f}/s | q:{snap['queue_size']}/{snap['queue_capacity']} "
                           f"({snap['queue_fill_pct']:.0f}%)")
                    if self.use_packing:
                        msg += (f" | pack:{snap['packing_rate_pct']:.1f}% | eff:{snap['token_efficiency_pct']:.1f}%")
                    msg += (f" | fetch:{snap['time_fetch_s']:.1f}s | tok:{snap['time_tokenize_s']:.1f}s")
                    if self.use_packing:
                        msg += f" | pack:{snap['time_pack_s']:.1f}s"
                    msg += (f" | put_blk:{snap['time_put_block_s']:.1f}s | get_blk:{snap['time_get_block_s']:.1f}s "
                            f"| refills:{snap['refills']}")
                    print(msg)
                next_log = now + self.log_interval_s
            time.sleep(0.05)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.seed is not None:
            random.seed(self.seed)

        specs = self._clone_specs()

        qmax = max(1, self.prefetch_batches * self.batch_size)
        q: "queue.Queue[Tuple[str, Optional[Dict[str, torch.Tensor]]]]" = queue.Queue(maxsize=qmax)
        prod_stop = threading.Event()
        log_stop = threading.Event()

        producer = threading.Thread(
            target=self._producer_base,
            args=(q, specs, prod_stop, self.weighted_sampling and len(specs) > 1),
            daemon=True,
        )
        producer.start()

        logger = None
        if self.debug or self.stats_callback:
            logger = threading.Thread(target=self._logger_loop, args=(log_stop,), daemon=True)
            logger.start()

        try:
            while True:
                t0 = time.perf_counter()
                tag, payload = q.get()
                self.stats.add_get_block_time(time.perf_counter() - t0)

                if (tag, payload) == MultiBoundedStreamingDataset.SENTINEL:
                    break
                if tag == "error":
                    raise RuntimeError(f"Producer failed: {payload['msg']}")
                self.stats.on_consumed(1, q.qsize(), q.maxsize)
                yield payload
        finally:
            prod_stop.set()
            log_stop.set()
            time.sleep(0.05)

    def __len__(self) -> int:
        if self.global_max_entries is not None:
            return int(self.global_max_entries)
        total = 0
        for s in self.dataset_specs_template:
            if s.max_entries is not None:
                total += s.max_entries
        return total

    def save_config(self, filepath: Union[str, Path]) -> None:
        """Save dataset configuration to JSON."""
        filepath = Path(filepath)
        config = {
            "dataset_specs": [
                {
                    "repo_id": spec.repo_id,
                    "subset": spec.subset,
                    "split": spec.split,
                    "skip_first": spec.skip_first,
                    "max_entries": spec.max_entries,
                    "text_column": spec.text_column,
                    "weight": spec.weight,
                }
                for spec in self.dataset_specs_template
            ],
            "pad_token_id": self.pad_token_id,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "prefetch_batches": self.prefetch_batches,
            "prefetch_chunk_batches": self.prefetch_chunk_batches,
            "tokenizer_batch_size": self.tokenizer_batch_size,
            "weighted_sampling": self.weighted_sampling,
            "global_max_entries": self.global_max_entries,
            "seed": self.seed,
            "move_to_device": self.move_to_device,
            "device": str(self.device) if self.device else None,
            "use_packing": self.use_packing,
            "packing_threshold": self.packing_threshold,
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, filepath: Union[str, Path], tokenizer: Tokenizer, **overrides):
        """Load dataset configuration from JSON and create instance."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            config = json.load(f)

        dataset_specs = [
            BoundedStreamingDataset(**spec)
            for spec in config["dataset_specs"]
        ]

        device_str = config.pop("device", None)
        device = torch.device(device_str) if device_str else None

        config.update(overrides)

        return cls(
            dataset_specs=dataset_specs,
            tokenizer=tokenizer,
            device=device,
            **{k: v for k, v in config.items() if k != "dataset_specs"}
        )

    def save_data(self, filepath: Union[str, Path], max_samples: Optional[int] = None) -> int:
        """Save dataset entries to JSON file. Returns number of samples saved."""
        filepath = Path(filepath)
        saved = 0

        with open(filepath, 'w') as f:
            f.write('[\n')
            first = True

            for sample in self:
                if max_samples and saved >= max_samples:
                    break

                entry = {
                    "input_ids": sample["input_ids"].cpu().tolist(),
                    "attention_mask": sample["attention_mask"].cpu().tolist(),
                }

                if not first:
                    f.write(',\n')
                json.dump(entry, f)
                first = False
                saved += 1

            f.write('\n]')

        return saved

    @staticmethod
    def load_data(filepath: Union[str, Path], device: Optional[torch.device] = None) -> Iterator[
        Dict[str, torch.Tensor]]:
        """Load saved dataset entries from JSON file."""
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        for entry in data:
            sample = {
                "input_ids": torch.tensor(entry["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(entry["attention_mask"], dtype=torch.bool),
            }

            if device:
                sample["input_ids"] = sample["input_ids"].to(device)
                sample["attention_mask"] = sample["attention_mask"].to(device)

            yield sample
