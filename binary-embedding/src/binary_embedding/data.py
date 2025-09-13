"""Data loading and processing for binary files.

This module provides two dataset implementations:
1. BinaryDataset: Standard lazy-loading dataset with LRU cache (good for most cases)
2. StreamingBinaryDataset: Continuous streaming dataset (good for very large datasets)

Both datasets:
- Read binary files in chunks
- Convert to latin-1 encoding
- Tokenizer adds <|start|> and <|end|> tokens automatically
"""

from __future__ import annotations

import os
import queue
import random
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from binary_embedding.tokenizer import BinaryTokenizer


class BinaryDataset(Dataset):
    """Standard dataset with lazy loading and LRU cache.

    Best for:
    - Small to medium datasets (< 100GB)
    - Random access patterns
    - When you need reproducible epochs
    """

    def __init__(
        self,
        directory_path: str | Path,
        tokenizer: BinaryTokenizer,
        max_length: int = 512,
        chunk_size: int = 4096,
        max_files: int | None = None,
        file_extensions: list[str] | None = None,
        cache_size: int = 1000,
    ) -> None:
        """Initialize the dataset.

        Args:
            directory_path: Path to directory containing binary files.
            tokenizer: BinaryTokenizer instance.
            max_length: Maximum sequence length for tokenization.
            chunk_size: Size of chunks to read from binary files (bytes).
            max_files: Maximum number of files to load (None for all).
            file_extensions: List of file extensions to include (None for all).
            cache_size: Number of chunks to cache in memory.
        """
        self.directory_path = Path(directory_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.cache_size = cache_size

        # Collect all binary files
        self.file_paths = self._collect_files(file_extensions, max_files)

        # Build index for lazy loading
        self.index = self._build_index()

        # LRU cache for recently loaded chunks
        self.cache: dict[int, str] = {}
        self.cache_order: deque[int] = deque(maxlen=cache_size)

    def _collect_files(
        self, file_extensions: list[str] | None, max_files: int | None
    ) -> list[Path]:
        """Collect all valid files from directory."""
        file_paths = []

        for root, _, files in os.walk(self.directory_path):
            for file in files:
                file_path = Path(root) / file

                # Skip symbolic links and non-files
                if not file_path.is_file() or file_path.is_symlink():
                    continue

                # Filter by extension if specified
                if file_extensions and file_path.suffix not in file_extensions:
                    continue

                file_paths.append(file_path)

                # Stop if we have enough files
                if max_files and len(file_paths) >= max_files:
                    return file_paths

        return file_paths

    def _build_index(self) -> list[tuple[int, int]]:
        """Build index mapping chunk_idx to (file_idx, byte_offset)."""
        index = []

        # Get file sizes in parallel for efficiency
        with ThreadPoolExecutor(max_workers=8) as executor:
            file_sizes = list(executor.map(self._get_file_size, self.file_paths))

        # Build index
        for file_idx, file_size in enumerate(file_sizes):
            if file_size > 0:
                # Calculate chunks for this file
                for offset in range(0, file_size, self.chunk_size):
                    index.append((file_idx, offset))

        return index

    def _get_file_size(self, file_path: Path) -> int:
        """Get file size or 0 if unreadable."""
        try:
            return file_path.stat().st_size
        except:
            return 0

    def _load_chunk(self, idx: int) -> str:
        """Load a specific chunk by index with caching."""
        # Check cache
        if idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            return self.cache[idx]

        # Load from disk
        file_idx, offset = self.index[idx]
        file_path = self.file_paths[file_idx]

        try:
            with open(file_path, "rb") as f:
                f.seek(offset)
                chunk = f.read(self.chunk_size)
                if chunk:
                    latin1_string = chunk.decode("latin-1", errors="replace")

                    # Update cache
                    if len(self.cache) >= self.cache_size and self.cache_order:
                        # Remove least recently used
                        old_idx = self.cache_order[0]
                        del self.cache[old_idx]

                    self.cache[idx] = latin1_string
                    self.cache_order.append(idx)

                    return latin1_string
        except Exception:
            pass

        return ""  # Return empty string on error

    def __len__(self) -> int:
        """Return the number of chunks."""
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a tokenized chunk."""
        # Load chunk
        chunk = self._load_chunk(idx)

        if not chunk:
            # Return padding for failed chunks
            return {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            }

        # Tokenize (tokenizer adds <|start|> and <|end|> automatically)
        encoding = self.tokenizer(
            chunk,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
        )

        return encoding


class StreamingBinaryDataset(IterableDataset):
    """Streaming dataset with continuous background loading.

    Best for:
    - Very large datasets (> 100GB)
    - Continuous training without epochs
    - When memory is limited
    """

    def __init__(
        self,
        directory_path: str | Path,
        tokenizer: BinaryTokenizer,
        max_length: int = 512,
        chunk_size: int = 4096,
        buffer_size: int = 10000,
        num_workers: int = 4,
        shuffle: bool = True,
        cycle: bool = True,
    ) -> None:
        """Initialize streaming dataset.

        Args:
            directory_path: Path to directory containing binary files.
            tokenizer: BinaryTokenizer instance.
            max_length: Maximum sequence length for tokenization.
            chunk_size: Size of chunks to read from binary files (bytes).
            buffer_size: Size of chunk buffer.
            num_workers: Number of background workers for loading.
            shuffle: Whether to shuffle files.
            cycle: Whether to cycle through files infinitely.
        """
        self.directory_path = Path(directory_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.cycle = cycle

        # Collect file paths
        self.file_paths = []
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_file() and not file_path.is_symlink():
                    try:
                        # Quick check that file exists and is readable
                        if file_path.stat().st_size > 0:
                            self.file_paths.append(file_path)
                    except:
                        continue

    def _file_reader(self, file_queue: queue.Queue, chunk_queue: queue.Queue) -> None:
        """Worker that reads files and produces chunks."""
        while True:
            try:
                file_path = file_queue.get(timeout=1)
                if file_path is None:  # Poison pill
                    break

                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(self.chunk_size)
                        if not chunk:
                            break

                        try:
                            latin1_string = chunk.decode("latin-1", errors="replace")
                            # Tokenize here to avoid bottleneck in main thread
                            encoding = self.tokenizer(
                                latin1_string,
                                max_length=self.max_length,
                                truncation=True,
                                padding="max_length",
                                add_special_tokens=True,
                            )
                            chunk_queue.put(encoding)
                        except:
                            continue
            except queue.Empty:
                continue
            except Exception:
                continue

    def __iter__(self):
        """Iterate over chunks with background loading."""
        if not self.file_paths:
            return

        # Prepare file list
        files = self.file_paths.copy()
        if self.shuffle:
            random.shuffle(files)

        # Create queues
        file_queue = queue.Queue(maxsize=100)
        chunk_queue = queue.Queue(maxsize=self.buffer_size)

        # Start worker threads
        workers = []
        for _ in range(self.num_workers):
            worker = threading.Thread(
                target=self._file_reader, args=(file_queue, chunk_queue), daemon=True
            )
            worker.start()
            workers.append(worker)

        # Producer thread to feed files
        def file_producer():
            while self.cycle:
                for file_path in files:
                    file_queue.put(file_path)
                if self.shuffle:
                    random.shuffle(files)

            # Single pass if not cycling
            if not self.cycle:
                for file_path in files:
                    file_queue.put(file_path)

            # Send poison pills
            for _ in range(self.num_workers):
                file_queue.put(None)

        producer = threading.Thread(target=file_producer, daemon=True)
        producer.start()

        # Yield chunks as they become available
        empty_count = 0
        while True:
            try:
                chunk = chunk_queue.get(timeout=0.1)
                yield chunk
                empty_count = 0
            except queue.Empty:
                empty_count += 1
                # Stop if no data for 5 seconds and not cycling
                if not self.cycle and empty_count > 50:
                    break
                # For cycling mode, wait longer before giving up
                if self.cycle and empty_count > 100:
                    break


class MLMDataCollator:
    """Data collator for masked language modeling on binary data."""

    def __init__(
        self,
        tokenizer: BinaryTokenizer,
        mlm_probability: float = 0.20,
    ) -> None:
        """Initialize the collator.

        Args:
            tokenizer: BinaryTokenizer instance.
            mlm_probability: Probability of masking tokens.
        """
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(
        self,
        examples: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Collate examples for MLM.

        Args:
            examples: List of tokenized examples.

        Returns:
            Batch dictionary with input_ids, attention_mask, and labels.
        """
        # Stack all input_ids and attention_masks
        input_ids = torch.stack(
            [
                ex["input_ids"].squeeze(0)
                if ex["input_ids"].dim() > 1
                else ex["input_ids"]
                for ex in examples
            ]
        )
        attention_mask = torch.stack(
            [
                ex["attention_mask"].squeeze(0)
                if ex["attention_mask"].dim() > 1
                else ex["attention_mask"]
                for ex in examples
            ]
        )

        # Create labels (copy of input_ids for MLM)
        labels = input_ids.clone()

        # Create mask for MLM
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = (
            (labels == self.tokenizer.pad_token_id)
            | (labels == self.tokenizer.start_token_id)
            | (labels == self.tokenizer.end_token_id)
            | (labels == self.tokenizer.cls_token_id)
            | (labels == self.tokenizer.sep_token_id)
        )

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of the time, replace masked input tokens with mask token
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, replace masked input tokens with random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.tokenizer.vocab_size,
            labels.shape,
            dtype=torch.long,
        )
        input_ids[indices_random] = random_words[indices_random]

        # The rest 10% of the time, keep masked input tokens unchanged

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataloader(
    directory_path: str | Path,
    tokenizer: BinaryTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    chunk_size: int = 4096,
    mlm_probability: float = 0.20,
    num_workers: int = 4,
    max_files: int | None = None,
    shuffle: bool = True,
    streaming: bool = False,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for binary files.

    Args:
        directory_path: Path to directory containing binary files.
        tokenizer: BinaryTokenizer instance.
        batch_size: Batch size for training.
        max_length: Maximum sequence length.
        chunk_size: Size of chunks to read from files.
        mlm_probability: Probability of masking for MLM.
        num_workers: Number of workers for data loading.
        max_files: Maximum number of files to load (standard mode only).
        shuffle: Whether to shuffle the data.
        streaming: If True, use StreamingBinaryDataset; else use BinaryDataset.
        **kwargs: Additional arguments passed to dataset constructor.

    Returns:
        DataLoader instance configured for the selected dataset type.
    """
    if streaming:
        # Use streaming dataset for continuous loading
        dataset = StreamingBinaryDataset(
            directory_path=directory_path,
            tokenizer=tokenizer,
            max_length=max_length,
            chunk_size=chunk_size,
            shuffle=shuffle,
            **kwargs,  # Pass through buffer_size, cycle, etc.
        )
        # Streaming datasets handle their own parallelism
        dataloader_workers = 0
        dataloader_shuffle = False
    else:
        # Use standard dataset with lazy loading
        dataset = BinaryDataset(
            directory_path=directory_path,
            tokenizer=tokenizer,
            max_length=max_length,
            chunk_size=chunk_size,
            max_files=max_files,
            **kwargs,  # Pass through cache_size, file_extensions, etc.
        )
        dataloader_workers = num_workers
        dataloader_shuffle = shuffle

    # Create data collator for MLM
    collator = MLMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
    )

    # Create and return dataloader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataloader_shuffle,
        collate_fn=collator,
        num_workers=dataloader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=dataloader_workers > 0,  # Keep workers alive between epochs
    )
