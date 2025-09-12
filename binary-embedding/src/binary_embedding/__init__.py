"""Binary Embedding Models for Executable Analysis.

This package provides tools for training BERT/RoBERTa-style embedding models
on binary executable files using custom tokenizers.
"""

__version__ = "0.1.0"

from binary_embedding.data import BinaryDataset, create_dataloader
from binary_embedding.models import (
    BinaryEmbeddingConfig,
    ModelSize,
    create_model,
)
from binary_embedding.tokenizer import BinaryTokenizer, load_tokenizer
from binary_embedding.training import Trainer, TrainerConfig

__all__ = [
    "BinaryTokenizer",
    "load_tokenizer",
    "BinaryDataset",
    "create_dataloader",
    "BinaryEmbeddingConfig",
    "create_model",
    "ModelSize",
    "Trainer",
    "TrainerConfig",
]
