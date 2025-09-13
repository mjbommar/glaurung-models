"""Data loading and processing for binary files."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from binary_embedding.tokenizer import BinaryTokenizer


class BinaryDataset(Dataset):
    """Dataset for loading and tokenizing binary files."""

    def __init__(
        self,
        directory_path: str | Path,
        tokenizer: BinaryTokenizer,
        max_length: int = 512,
        chunk_size: int = 4096,
        max_files: int | None = None,
        file_extensions: list[str] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            directory_path: Path to directory containing binary files.
            tokenizer: BinaryTokenizer instance.
            max_length: Maximum sequence length for tokenization.
            chunk_size: Size of chunks to read from binary files.
            max_files: Maximum number of files to load (None for all).
            file_extensions: List of file extensions to include (None for all).
        """
        self.directory_path = Path(directory_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size

        # Collect all binary files
        self.file_paths: list[Path] = []
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                file_path = Path(root) / file

                # Skip symbolic links and non-files
                if not file_path.is_file() or file_path.is_symlink():
                    continue

                # Filter by extension if specified
                if file_extensions and file_path.suffix not in file_extensions:
                    continue

                # Check if file is readable
                try:
                    with open(file_path, "rb") as f:
                        _ = f.read(1)
                    self.file_paths.append(file_path)
                except (PermissionError, OSError):
                    continue

                # Limit number of files if specified
                if max_files and len(self.file_paths) >= max_files:
                    break

            if max_files and len(self.file_paths) >= max_files:
                break

        # Pre-compute chunks for all files
        self.chunks: list[str] = []
        self.use_file_boundaries = True  # Flag to add <|start|> and <|end|> tokens
        for file_path in self.file_paths:
            self._load_file_chunks(file_path)

    def _load_file_chunks(self, file_path: Path) -> None:
        """Load chunks from a binary file.

        Args:
            file_path: Path to the binary file.
        """
        try:
            with open(file_path, "rb") as f:
                # Read entire file content
                file_content = f.read()
                if not file_content:
                    return
                    
                # Convert entire file to latin-1 string
                latin1_string = file_content.decode("latin-1")
                
                # The tokenizer automatically adds <|start|> and <|end|> tokens
                # when add_special_tokens=True is used, so we just need to
                # ensure each chunk represents a complete file
                
                # If file is small enough, keep it as one chunk
                if len(latin1_string) <= self.chunk_size:
                    self.chunks.append(latin1_string)
                else:
                    # For larger files, split into chunks
                    # Each chunk will get start/end tokens from the tokenizer
                    # This treats each chunk as a separate "file"
                    remaining = latin1_string
                    
                    while remaining:
                        chunk_content = remaining[:self.chunk_size]
                        remaining = remaining[self.chunk_size:]
                        self.chunks.append(chunk_content)

        except Exception:
            pass  # Skip files that can't be read

    def __len__(self) -> int:
        """Return the number of chunks."""
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a tokenized chunk.

        Args:
            idx: Index of the chunk.

        Returns:
            Dictionary with input_ids and attention_mask.
        """
        chunk = self.chunks[idx]

        # Tokenize the chunk
        encoding = self.tokenizer(
            chunk,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
        )

        return encoding


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
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.vocab_size = tokenizer.vocab_size

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
        # Handle potential extra dimension from tokenizer
        input_ids = torch.stack(
            [
                (
                    ex["input_ids"].squeeze(0)
                    if ex["input_ids"].dim() > 1
                    else ex["input_ids"]
                )
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

        # Don't mask special tokens or padding
        # Include all special tokens: pad, start, end, cls, sep, mask
        special_tokens_mask = (
            (labels == self.pad_token_id) |
            (labels == self.tokenizer.start_token_id) |
            (labels == self.tokenizer.end_token_id) |
            (labels == self.tokenizer.cls_token_id) |
            (labels == self.tokenizer.sep_token_id)
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
        input_ids[indices_replaced] = self.mask_token_id

        # 10% of the time, replace masked input tokens with random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.vocab_size,
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
        max_files: Maximum number of files to load.
        shuffle: Whether to shuffle the data.

    Returns:
        DataLoader instance.
    """
    # Create dataset
    dataset = BinaryDataset(
        directory_path=directory_path,
        tokenizer=tokenizer,
        max_length=max_length,
        chunk_size=chunk_size,
        max_files=max_files,
    )

    # Create data collator
    collator = MLMDataCollator(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader
