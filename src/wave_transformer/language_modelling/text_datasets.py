import dataclasses
import queue
import random
import threading
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
            "input_ids": self.examples[idx]["input_ids"],
            "attention_mask": self.examples[idx]["attention_mask"],
        }


class TextDatasetPaddedSimple(Dataset):
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
            max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.pad_id = pad_token_id
        self.max_length = max_length

        self.examples = []
        texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
        encs = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        for enc in encs:
            self.examples.append({
                "input_ids": torch.tensor(enc.ids, dtype=torch.long),
                "attention_mask": torch.tensor(enc.attention_mask, dtype=torch.bool),
            })



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"],
            "attention_mask": self.examples[idx]["attention_mask"],
        }

