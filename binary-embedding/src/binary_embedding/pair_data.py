"""Pair sampling datasets and collators for contrastive + MLM training.

This module provides utilities to build batches consisting of pairs of
sequences from the same file, supporting two pair types:

- duplicate: two different masked views of the same chunk (SimCSE-style)
- same_file: two different chunks from the same file (non-overlapping)

The returned batches contain two views per pair and per-view MLM labels so the
trainer can compute both MLM and contrastive losses efficiently.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .tokenizer import BinaryTokenizer


class SameFilePairDataset(Dataset):
    """Dataset that yields pairs of chunks from the same file.

    Each item returns two tokenized sequences and metadata indicating whether
    the pair is a duplicate-view pair (same chunk, different masks) or a
    same-file pair (two distinct chunks from the same file).

    Design goals:
    - Keep I/O efficient by reusing a chunk-level index and simple LRU cache
    - Avoid global stratification across folders (pairs within a single file)
    - Support reproducible epochs (deterministic __len__), but randomized pairs
    """

    def __init__(
        self,
        directory_path: str | Path,
        tokenizer: BinaryTokenizer,
        max_length: int = 512,
        chunk_size: int = 4096,
        duplicate_prob: float = 0.5,
        min_chunk_separation: int | None = None,
        cache_size: int = 1024,
        max_files: int | None = None,
        file_extensions: list[str] | None = None,
        tokenize_in_dataset: bool = False,
    ) -> None:
        self.directory_path = Path(directory_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.duplicate_prob = float(max(0.0, min(1.0, duplicate_prob)))
        self.min_chunk_separation = (
            min_chunk_separation if min_chunk_separation is not None else chunk_size
        )
        self.cache_size = cache_size
        self.tokenize_in_dataset = tokenize_in_dataset

        # Collect all files
        self.file_paths = self._collect_files(file_extensions, max_files)

        # Build chunk index and mapping from file -> chunk indices
        self.index = self._build_index()
        self.file_to_chunks: dict[int, list[int]] = defaultdict(list)
        for idx, (file_idx, _offset) in enumerate(self.index):
            self.file_to_chunks[file_idx].append(idx)

        # LRU cache for raw chunk strings
        self.cache: dict[int, str] = {}
        self.cache_order: deque[int] = deque(maxlen=cache_size)

    # ---------- helpers ----------
    def _collect_files(
        self, file_extensions: list[str] | None, max_files: int | None
    ) -> list[Path]:
        paths: list[Path] = []
        for root, _, files in os.walk(self.directory_path):
            for name in files:
                p = Path(root) / name
                if not p.is_file() or p.is_symlink():
                    continue
                if file_extensions and p.suffix not in file_extensions:
                    continue
                paths.append(p)
                if max_files and len(paths) >= max_files:
                    return paths
        return paths

    def _get_file_size(self, file_path: Path) -> int:
        try:
            return file_path.stat().st_size
        except Exception:
            return 0

    def _build_index(self) -> list[tuple[int, int]]:
        index: list[tuple[int, int]] = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            sizes = list(ex.map(self._get_file_size, self.file_paths))
        for file_idx, size in enumerate(sizes):
            if size <= 0:
                continue
            for offset in range(0, size, self.chunk_size):
                index.append((file_idx, offset))
        return index

    def _load_chunk(self, idx: int) -> str:
        if idx in self.cache:
            # refresh LRU
            try:
                self.cache_order.remove(idx)
            except ValueError:
                pass
            self.cache_order.append(idx)
            return self.cache[idx]

        file_idx, offset = self.index[idx]
        file_path = self.file_paths[file_idx]
        try:
            with open(file_path, "rb") as f:
                f.seek(offset)
                chunk = f.read(self.chunk_size)
            if not chunk:
                return ""
            s = chunk.decode("latin-1", errors="replace")
            # insert into cache
            if len(self.cache) >= self.cache_size and self.cache_order:
                old_idx = self.cache_order.popleft()
                self.cache.pop(old_idx, None)
            self.cache[idx] = s
            self.cache_order.append(idx)
            return s
        except Exception:
            return ""

    # ---------- Dataset API ----------
    def __len__(self) -> int:
        # Tie epoch length to total chunks. Each item produces a pair.
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        # Pick a file based on the i-th chunk entry to get uniform coverage
        file_idx, offset_anchor = self.index[i]
        chunk_indices = self.file_to_chunks[file_idx]

        # Decide pair type
        pair_type = (
            "duplicate" if random.random() < self.duplicate_prob else "same_file"
        )

        if pair_type == "same_file" and len(chunk_indices) < 2:
            # Fallback if not enough chunks
            pair_type = "duplicate"

        if pair_type == "duplicate":
            # Use the anchor chunk twice
            idx1 = i
            idx2 = i
        else:
            # Sample a second chunk in the same file, far enough from the anchor
            # Filter candidates by separation
            candidates: list[int] = []
            for j in chunk_indices:
                if j == i:
                    continue
                _fj, off = self.index[j]
                if abs(off - offset_anchor) >= self.min_chunk_separation:
                    candidates.append(j)
            if not candidates:
                # Fallback: pick any other chunk in the file
                candidates = [j for j in chunk_indices if j != i]
                if not candidates:
                    # degenerate case
                    idx1 = i
                    idx2 = i
                    pair_type = "duplicate"
                else:
                    idx1 = i
                    idx2 = random.choice(candidates)
            else:
                idx1 = i
                idx2 = random.choice(candidates)

        # Load both views (masking applied later in collator)
        s1 = self._load_chunk(idx1)
        s2 = self._load_chunk(idx2)

        if self.tokenize_in_dataset:
            # Tokenize here (compat with older path)
            enc1 = self.tokenizer(
                s1,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            enc2 = self.tokenizer(
                s2,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            return {
                "input_ids_1": enc1["input_ids"],
                "attention_mask_1": enc1["attention_mask"],
                "input_ids_2": enc2["input_ids"],
                "attention_mask_2": enc2["attention_mask"],
                "pair_type": 0 if pair_type == "duplicate" else 1,
            }
        else:
            # Return raw strings for batched tokenization in the collator
            return {
                "text_1": s1,
                "text_2": s2,
                "pair_type": 0 if pair_type == "duplicate" else 1,
            }


class ContrastiveMLMCollator:
    """Collator that produces two masked views and MLM labels per pair.

    For each pair item, independently applies MLM masking to both views. This
    gives duplicated pairs distinct corruption patterns and preserves MLM for
    same-file pairs.
    """

    def __init__(
        self,
        tokenizer: BinaryTokenizer,
        mlm_probability: float = 0.20,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length

    def _mask(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Create labels as a copy
        labels = input_ids.clone()

        # Probability matrix
        prob = torch.full(labels.shape, self.mlm_probability, dtype=torch.float32)

        # Do not mask special tokens
        special_tokens_mask = (
            (labels == self.tokenizer.pad_token_id)
            | (labels == self.tokenizer.start_token_id)
            | (labels == self.tokenizer.end_token_id)
            | (labels == self.tokenizer.cls_token_id)
            | (labels == self.tokenizer.sep_token_id)
        )
        prob.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(prob).bool()

        # Non-masked tokens have label -100
        labels[~masked_indices] = -100

        # 80% -> mask token
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids = input_ids.clone()  # avoid in-place on shared tensors
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% -> random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.tokenizer.vocab_size, labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]
        # 10% -> unchanged masked tokens

        return input_ids, labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # If batch contains raw texts, batch-tokenize here for speed
        if "text_1" in batch[0]:
            texts1 = [b["text_1"] for b in batch]
            texts2 = [b["text_2"] for b in batch]
            enc1 = self.tokenizer(
                texts1,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            enc2 = self.tokenizer(
                texts2,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )
            input_ids_1 = enc1["input_ids"]
            attn_1 = enc1["attention_mask"]
            input_ids_2 = enc2["input_ids"]
            attn_2 = enc2["attention_mask"]
            pair_types = torch.tensor([b["pair_type"] for b in batch], dtype=torch.long)
        else:
            # Stack per-view inputs, handling potential extra batch dim from tokenizer
            def squeeze_if_needed(t: torch.Tensor) -> torch.Tensor:
                return t.squeeze(0) if t.dim() > 1 else t

            input_ids_1 = torch.stack(
                [squeeze_if_needed(b["input_ids_1"]) for b in batch]
            )
            attn_1 = torch.stack(
                [squeeze_if_needed(b["attention_mask_1"]) for b in batch]
            )
            input_ids_2 = torch.stack(
                [squeeze_if_needed(b["input_ids_2"]) for b in batch]
            )
            attn_2 = torch.stack(
                [squeeze_if_needed(b["attention_mask_2"]) for b in batch]
            )
            pair_types = torch.tensor([b["pair_type"] for b in batch], dtype=torch.long)

        # Apply masking independently per view
        input_ids_1, labels_1 = self._mask(input_ids_1)
        input_ids_2, labels_2 = self._mask(input_ids_2)

        # Shape as [batch, 2, seq_len]
        input_ids = torch.stack([input_ids_1, input_ids_2], dim=1)
        attention_mask = torch.stack([attn_1, attn_2], dim=1)
        labels = torch.stack([labels_1, labels_2], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pair_types": pair_types,  # 0=duplicate,1=same_file
        }


def create_pair_dataloader(
    directory_path: str | Path,
    tokenizer: BinaryTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    chunk_size: int = 4096,
    mlm_probability: float = 0.20,
    duplicate_prob: float = 0.5,
    min_chunk_separation: int | None = None,
    num_workers: int = 4,
    max_files: int | None = None,
    shuffle: bool = True,
    cache_size: int = 4096,
    prefetch_factor: int | None = 4,
    tokenize_in_dataset: bool = False,
    **kwargs: Any,
) -> DataLoader:
    """Create a DataLoader that yields same-file pairs with MLM labels.

    The effective number of sequences per batch is 2 x batch_size.
    """
    dataset = SameFilePairDataset(
        directory_path=directory_path,
        tokenizer=tokenizer,
        max_length=max_length,
        chunk_size=chunk_size,
        duplicate_prob=duplicate_prob,
        min_chunk_separation=min_chunk_separation,
        cache_size=cache_size,
        max_files=max_files,
        tokenize_in_dataset=tokenize_in_dataset,
        **kwargs,
    )

    collator = ContrastiveMLMCollator(
        tokenizer=tokenizer, mlm_probability=mlm_probability, max_length=max_length
    )

    # When using multiple workers, avoid persistent_workers if num_workers == 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor
        if (num_workers > 0 and prefetch_factor)
        else None,
    )


class StreamingSameFilePairDataset(IterableDataset):
    """Streaming iterable dataset that forms same-file pairs online.

    Background worker threads read chunks from files and push them to a shared
    queue with (file_idx, offset, text). The iterator consumes chunks,
    maintains a small per-file buffer of recent chunks, and emits pairs:
      - duplicate: same chunk twice (two masked views later)
      - same_file: current chunk paired with a buffered chunk from same file,
        respecting a minimum byte separation.
    """

    def __init__(
        self,
        directory_path: str | Path,
        tokenizer: BinaryTokenizer,
        max_length: int = 512,
        chunk_size: int = 4096,
        buffer_size: int = 10000,
        num_workers: int = 4,
        duplicate_prob: float = 0.5,
        min_chunk_separation: int | None = None,
        per_file_buffer: int = 32,
        shuffle: bool = True,
        cycle: bool = True,
    ) -> None:
        self.directory_path = Path(directory_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.duplicate_prob = float(max(0.0, min(1.0, duplicate_prob)))
        self.min_chunk_separation = (
            min_chunk_separation if min_chunk_separation is not None else chunk_size
        )
        self.per_file_buffer = per_file_buffer
        self.shuffle = shuffle
        self.cycle = cycle

        # Collect file paths
        self.file_paths: list[Path] = []
        for root, _, files in os.walk(self.directory_path):
            for name in files:
                p = Path(root) / name
                if p.is_file() and not p.is_symlink():
                    try:
                        if p.stat().st_size > 0:
                            self.file_paths.append(p)
                    except Exception:
                        continue

    def _file_reader(self, file_queue, chunk_queue) -> None:
        while True:
            try:
                item = file_queue.get(timeout=1)
            except Exception:
                continue
            if item is None:
                break
            file_idx, file_path = item
            try:
                with open(file_path, "rb") as f:
                    offset = 0
                    while True:
                        chunk = f.read(self.chunk_size)
                        if not chunk:
                            break
                        try:
                            text = chunk.decode("latin-1", errors="replace")
                        except Exception:
                            offset += self.chunk_size
                            continue
                        chunk_queue.put(
                            {"file_idx": file_idx, "offset": offset, "text": text}
                        )
                        offset += self.chunk_size
            except Exception:
                continue

    def __iter__(self):
        if not self.file_paths:
            return

        # Prepare list and optional shuffle
        files = list(enumerate(self.file_paths))
        if self.shuffle:
            random.shuffle(files)

        # Queues
        import queue as _queue

        file_queue: _queue.Queue = _queue.Queue(maxsize=100)
        chunk_queue: _queue.Queue = _queue.Queue(maxsize=self.buffer_size)

        # Start workers
        workers = []
        for _ in range(self.num_workers):
            t = threading.Thread(
                target=self._file_reader, args=(file_queue, chunk_queue), daemon=True
            )
            t.start()
            workers.append(t)

        # Producer thread to cycle through files
        def file_producer():
            while self.cycle:
                for item in files:
                    file_queue.put(item)
                if self.shuffle:
                    random.shuffle(files)
            if not self.cycle:
                for item in files:
                    file_queue.put(item)
            for _ in range(self.num_workers):
                file_queue.put(None)

        prod = threading.Thread(target=file_producer, daemon=True)
        prod.start()

        # Per-file recent chunk buffers
        buffers: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.per_file_buffer)
        )

        empty_count = 0
        while True:
            try:
                rec = chunk_queue.get(timeout=0.1)
            except _queue.Empty:
                empty_count += 1
                if not self.cycle and empty_count > 50:
                    break
                if self.cycle and empty_count > 100:
                    break
                continue

            empty_count = 0
            file_idx = rec["file_idx"]
            offset = rec["offset"]
            text = rec["text"]

            # Form pair
            do_dup = random.random() < self.duplicate_prob
            if not do_dup:
                # Try to find a buffered chunk far enough
                candidates = [
                    c
                    for c in buffers[file_idx]
                    if abs(c["offset"] - offset) >= self.min_chunk_separation
                ]
                if candidates:
                    other = random.choice(candidates)
                    yield {
                        "text_1": text,
                        "text_2": other["text"],
                        "pair_type": 1,  # same_file
                    }
                else:
                    # Fallback to duplicate until we have enough buffer
                    yield {"text_1": text, "text_2": text, "pair_type": 0}
            else:
                yield {"text_1": text, "text_2": text, "pair_type": 0}

            # Append current to buffer after pairing
            buffers[file_idx].append({"offset": offset, "text": text})


def create_streaming_pair_dataloader(
    directory_path: str | Path,
    tokenizer: BinaryTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    chunk_size: int = 4096,
    mlm_probability: float = 0.20,
    duplicate_prob: float = 0.5,
    min_chunk_separation: int | None = None,
    buffer_size: int = 10000,
    per_file_buffer: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    cycle: bool = True,
) -> DataLoader:
    """Create a streaming DataLoader that yields pairs continuously.

    Uses background threads for IO; DataLoader itself should use num_workers=0.
    """
    dataset = StreamingSameFilePairDataset(
        directory_path=directory_path,
        tokenizer=tokenizer,
        max_length=max_length,
        chunk_size=chunk_size,
        buffer_size=buffer_size,
        num_workers=num_workers,
        duplicate_prob=duplicate_prob,
        min_chunk_separation=min_chunk_separation,
        per_file_buffer=per_file_buffer,
        shuffle=shuffle,
        cycle=cycle,
    )

    collator = ContrastiveMLMCollator(
        tokenizer=tokenizer, mlm_probability=mlm_probability, max_length=max_length
    )

    # IterableDataset performs its own background IO; keep loader workers at 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
