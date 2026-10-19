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

        # Get special token IDs - support multiple formats
        # binary-tokenizer-005: <|start|>, <|end|>, etc. (at IDs 0-134)
        # glaurung-tokenizer-002: <s>, </s>, etc. (at IDs 65529-65535)
        def get_special_token_id(alternatives: list[str], default: int | None = None) -> int:
            """Try multiple token formats, return first match or default."""
            for token_str in alternatives:
                token_id = self.tokenizer.token_to_id(token_str)
                if token_id is not None:
                    return token_id
            if default is None:
                raise ValueError(f"None of {alternatives} found in tokenizer and no default provided")
            return default

        self.start_token_id: int = get_special_token_id(["<|start|>", "<s>", "[BOS]"])
        self.end_token_id: int = get_special_token_id(["<|end|>", "</s>", "[EOS]"])
        self.sep_token_id: int = get_special_token_id(["<|sep|>", "<sep>", "[SEP]"])
        self.cls_token_id: int = get_special_token_id(["<|cls|>", "<cls>", "[CLS]"])
        self.pad_token_id: int = get_special_token_id(["<|pad|>", "<pad>", "[PAD]"])
        self.mask_token_id: int = get_special_token_id(["<|mask|>", "<mask>", "[MASK]"])
        self.unk_token_id: int = get_special_token_id(["<|unk|>", "<unk>", "[UNK]"])

        # Store actual special token strings found in tokenizer
        vocab_reversed = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.special_tokens: dict[str, str] = {
            "pad_token": vocab_reversed[self.pad_token_id],
            "unk_token": vocab_reversed[self.unk_token_id],
            "cls_token": vocab_reversed[self.cls_token_id],
            "sep_token": vocab_reversed[self.sep_token_id],
            "mask_token": vocab_reversed[self.mask_token_id],
            "start_token": vocab_reversed[self.start_token_id],
            "end_token": vocab_reversed[self.end_token_id],
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
        tokenizer_path: Path to tokenizer.json file, or HuggingFace Hub repo (e.g., 'username/repo').
                       If None, uses default path.

    Returns:
        BinaryTokenizer instance.
    """
    if tokenizer_path is None:
        tokenizer_path = Path("binary-tokenizer-005/tokenizer.json")

    # Check if it's a HuggingFace Hub repo (format: username/repo)
    tokenizer_str = str(tokenizer_path)
    if "/" in tokenizer_str and not tokenizer_str.startswith(("./", "../", "/")):
        # Looks like a HuggingFace Hub repo
        try:
            from huggingface_hub import hf_hub_download
            # Download tokenizer.json from the repo
            local_path = hf_hub_download(
                repo_id=tokenizer_str,
                filename="tokenizer.json",
            )
            return BinaryTokenizer(local_path)
        except Exception as e:
            # If HF Hub loading fails, fall back to treating it as a local path
            print(f"Warning: Failed to load from HuggingFace Hub: {e}")
            print(f"Attempting to load as local path: {tokenizer_path}")

    # Load from local path
    return BinaryTokenizer(tokenizer_path)
