"""Model architectures for binary embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedModel,
    RobertaConfig,
    RobertaForMaskedLM,
)


class ModelSize(Enum):
    """Predefined model sizes."""

    SMALL = "small"
    BASE = "base"
    LARGE = "large"


@dataclass
class BinaryEmbeddingConfig:
    """Configuration for binary embedding models."""

    model_type: str = "roberta"
    vocab_size: int = 65536
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 520  # Increased for RoBERTa padding offset
    type_vocab_size: int = 1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 4
    position_embedding_type: str = "absolute"
    use_cache: bool = True
    classifier_dropout: float | None = None

    # Training hyperparameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    mlm_probability: float = 0.20

    def to_transformers_config(
        self,
    ) -> BertConfig | RobertaConfig:
        """Convert to transformers library config.

        Returns:
            Transformers config object.

        Raises:
            ValueError: If model_type is not supported.
        """
        config_kwargs = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id,
            "position_embedding_type": self.position_embedding_type,
            "use_cache": self.use_cache,
            "classifier_dropout": self.classifier_dropout,
        }

        if self.model_type == "bert":
            return BertConfig(**config_kwargs)
        elif self.model_type == "roberta":
            return RobertaConfig(**config_kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


def get_model_config(size: ModelSize) -> BinaryEmbeddingConfig:
    """Get predefined model configuration.

    Args:
        size: Model size enum.

    Returns:
        Model configuration.
    """
    if size == ModelSize.SMALL:
        return BinaryEmbeddingConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
        )
    elif size == ModelSize.BASE:
        return BinaryEmbeddingConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
    else:  # LARGE
        return BinaryEmbeddingConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            mlm_probability=0.40,  # Higher masking for large models
        )


def create_model(
    config: BinaryEmbeddingConfig | None = None,
    size: ModelSize | None = None,
    pretrained_path: str | None = None,
) -> tuple[PreTrainedModel, BinaryEmbeddingConfig]:
    """Create a binary embedding model.

    Args:
        config: Model configuration. If None, uses size parameter.
        size: Model size enum. Used if config is None.
        pretrained_path: Optional path to pretrained weights.

    Returns:
        Tuple of (model, config).

    Raises:
        ValueError: If neither config nor size is provided.
    """
    if config is None:
        if size is None:
            size = ModelSize.BASE
        config = get_model_config(size)

    transformers_config = config.to_transformers_config()

    if pretrained_path:
        if config.model_type == "bert":
            model = BertForMaskedLM.from_pretrained(
                pretrained_path,
                config=transformers_config,
            )
        else:
            model = RobertaForMaskedLM.from_pretrained(
                pretrained_path,
                config=transformers_config,
            )
    else:
        if config.model_type == "bert":
            model = BertForMaskedLM(transformers_config)
        else:
            model = RobertaForMaskedLM(transformers_config)

    return model, config


class BinaryEmbeddingModelWithPooler(nn.Module):
    """Binary embedding model with additional pooler for downstream tasks."""

    def __init__(
        self,
        base_model: PreTrainedModel,
        hidden_size: int = 768,
    ) -> None:
        """Initialize the model with pooler.

        Args:
            base_model: Base masked LM model.
            hidden_size: Hidden size for pooler.
        """
        super().__init__()
        self.base_model = base_model
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> Any:
        """Forward pass.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            labels: Labels for MLM loss.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary.

        Returns:
            Model outputs with optional pooler output.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if output_hidden_states and hasattr(outputs, "hidden_states"):
            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]

            # Pool using the CLS token (first token)
            cls_embedding = last_hidden_state[:, 0, :]
            pooled_output = self.pooler(cls_embedding)

            # Add pooled output to the outputs
            if return_dict:
                outputs.pooler_output = pooled_output

        return outputs

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get embeddings for input sequences.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            Pooled embeddings.
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            return outputs.pooler_output
