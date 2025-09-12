"""Binary tokenizer module for processing executable files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer


class BinaryTokenizer:
    """Tokenizer for binary executable files with transformers compatibility."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        """Initialize the binary tokenizer.

        Args:
            tokenizer_path: Path to the tokenizer.json file.
        """
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size: int = self.tokenizer.get_vocab_size()

        # Get special token IDs
        self.pad_token_id: int = self.tokenizer.token_to_id("<|pad|>") or 0
        self.unk_token_id: int = self.tokenizer.token_to_id("<|unk|>") or 1
        self.cls_token_id: int = self.tokenizer.token_to_id("<|cls|>") or 2
        self.sep_token_id: int = self.tokenizer.token_to_id("<|sep|>") or 3
        self.mask_token_id: int = self.tokenizer.token_to_id("<|mask|>") or 4

        # Store special tokens
        self.special_tokens: dict[str, str] = {
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "cls_token": "<|cls|>",
            "sep_token": "<|sep|>",
            "mask_token": "<|mask|>",
        }

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to encode.
            add_special_tokens: Whether to add special tokens.

        Returns:
            List of token IDs.
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        """Encode batch of texts to token IDs.

        Args:
            texts: List of input texts.
            add_special_tokens: Whether to add special tokens.

        Returns:
            List of token ID lists.
        """
        encodings = self.tokenizer.encode_batch(
            texts,
            add_special_tokens=add_special_tokens,
        )
        return [e.ids for e in encodings]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            Decoded text.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self,
        text: str | list[str],
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
        return_tensors: str | None = "pt",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Make tokenizer callable for compatibility with transformers.

        Args:
            text: Single text or list of texts.
            max_length: Maximum sequence length.
            padding: Padding strategy.
            truncation: Whether to truncate sequences.
            add_special_tokens: Whether to add special tokens.
            return_tensors: Return type ("pt" for PyTorch tensors).
            **kwargs: Additional arguments.

        Returns:
            Dictionary with input_ids and attention_mask.
        """
        if isinstance(text, str):
            text = [text]

        # Encode all texts
        encodings = self.encode_batch(text, add_special_tokens=add_special_tokens)

        # Process each encoding
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for encoding in encodings:
            # Truncate if needed
            if truncation and len(encoding) > max_length:
                encoding = encoding[:max_length]

            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1] * len(encoding)

            # Pad if needed
            if padding == "max_length" and len(encoding) < max_length:
                padding_length = max_length - len(encoding)
                encoding = encoding + [self.pad_token_id] * padding_length
                mask = mask + [0] * padding_length

            input_ids.append(encoding)
            attention_mask.append(mask)

        # Convert to tensors if requested
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def load_tokenizer(
    tokenizer_path: str | Path | None = None,
) -> BinaryTokenizer:
    """Load the binary tokenizer.

    Args:
        tokenizer_path: Path to tokenizer.json file. If None, uses default path.

    Returns:
        BinaryTokenizer instance.
    """
    if tokenizer_path is None:
        tokenizer_path = Path(
            "/home/mjbommar/src/glaurung-models/tokenizers/"
            "tokenizer-001/tokenizers/binary-tokenizer-01/iteration-005/tokenizer.json"
        )

    return BinaryTokenizer(tokenizer_path)
