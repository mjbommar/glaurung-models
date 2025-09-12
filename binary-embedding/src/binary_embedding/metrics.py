"""Metrics tracking and early stopping for binary embedding training.

This module provides comprehensive metrics tracking to monitor model
performance and prevent overfitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MetricHistory:
    """Track history of a single metric."""

    name: str
    values: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)

    def add(self, value: float, step: int):
        """Add a new value."""
        self.values.append(value)
        self.steps.append(step)

        # Limit history size to prevent memory issues
        if len(self.values) > 10000:
            # Keep most recent 5000 entries
            self.values = self.values[-5000:]
            self.steps = self.steps[-5000:]

    def get_best(self, mode: str = "min") -> tuple[float, int]:
        """Get best value and its step.

        Args:
            mode: "min" for loss-like metrics, "max" for accuracy-like metrics.

        Returns:
            Tuple of (best_value, step).
        """
        if not self.values:
            return (float("inf") if mode == "min" else float("-inf"), -1)

        if mode == "min":
            idx = np.argmin(self.values)
        else:
            idx = np.argmax(self.values)

        return self.values[idx], self.steps[idx]

    def get_recent_average(self, n: int = 10) -> float:
        """Get average of last n values."""
        if not self.values:
            return 0.0
        return np.mean(self.values[-n:])

    def is_improving(self, patience: int = 5, mode: str = "min") -> bool:
        """Check if metric is improving.

        Args:
            patience: Number of steps to look back.
            mode: "min" or "max".

        Returns:
            True if improving, False otherwise.
        """
        if len(self.values) < patience + 1:
            return True  # Not enough history

        recent = self.values[-1]
        past = self.values[-(patience + 1)]

        if mode == "min":
            return recent < past
        else:
            return recent > past


@dataclass
class TrainingMetrics:
    """Collection of all training metrics."""

    mlm_loss: MetricHistory = field(default_factory=lambda: MetricHistory("mlm_loss"))
    learning_rate: MetricHistory = field(
        default_factory=lambda: MetricHistory("learning_rate")
    )
    gradient_norm: MetricHistory = field(
        default_factory=lambda: MetricHistory("gradient_norm")
    )

    # Optional metrics
    embedding_quality: MetricHistory | None = None
    perplexity: MetricHistory | None = None
    validation_loss: MetricHistory | None = None

    def __post_init__(self):
        """Initialize optional metrics."""
        if self.embedding_quality is None:
            self.embedding_quality = MetricHistory("embedding_quality")
        if self.perplexity is None:
            self.perplexity = MetricHistory("perplexity")
        if self.validation_loss is None:
            self.validation_loss = MetricHistory("validation_loss")

    def log(self, metrics: dict[str, float], step: int):
        """Log a batch of metrics."""
        for name, value in metrics.items():
            if hasattr(self, name):
                metric = getattr(self, name)
                if isinstance(metric, MetricHistory):
                    metric.add(value, step)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}
        for name in ["mlm_loss", "embedding_quality", "perplexity", "validation_loss"]:
            metric = getattr(self, name)
            if metric and metric.values:
                best_val, best_step = metric.get_best(
                    mode="min" if "loss" in name else "max"
                )
                summary[name] = {
                    "current": metric.values[-1] if metric.values else None,
                    "best": best_val,
                    "best_step": best_step,
                    "recent_avg": metric.get_recent_average(),
                }
        return summary

    def save(self, path: Path):
        """Save metrics to JSON file."""
        data = {}
        for name in [
            "mlm_loss",
            "learning_rate",
            "gradient_norm",
            "embedding_quality",
            "perplexity",
            "validation_loss",
        ]:
            metric = getattr(self, name)
            if metric and metric.values:
                data[name] = {
                    "values": metric.values,
                    "steps": metric.steps,
                }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        """Load metrics from JSON file."""
        with open(path) as f:
            data = json.load(f)

        for name, values in data.items():
            if hasattr(self, name):
                metric = MetricHistory(name)
                metric.values = values["values"]
                metric.steps = values["steps"]
                setattr(self, name, metric)


class EmbeddingQualityMonitor:
    """Monitor embedding quality during training."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """Initialize the monitor.

        Args:
            model: The model to monitor.
            device: Device to use for computations.
        """
        self.model = model
        self.device = device

    def compute_embedding_separation(
        self,
        similar_pairs: list[tuple[str, str]],
        dissimilar_pairs: list[tuple[str, str]],
        tokenizer,
        max_length: int = 128,
    ) -> float:
        """Compute separation between similar and dissimilar embeddings.

        Args:
            similar_pairs: List of similar text pairs.
            dissimilar_pairs: List of dissimilar text pairs.
            tokenizer: Tokenizer to use.
            max_length: Maximum sequence length.

        Returns:
            Separation score (higher is better).
        """
        self.model.eval()

        def get_embedding(text: str) -> torch.Tensor:
            inputs = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Mean pooling of last hidden states
                hidden_states = outputs.hidden_states[-1]
                mask = (
                    inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size())
                )
                masked_hidden = hidden_states * mask
                summed = torch.sum(masked_hidden, dim=1)
                count = torch.clamp(mask.sum(dim=1), min=1e-9)
                embedding = summed / count

            return embedding[0]

        # Compute similarities
        similar_sims = []
        for text1, text2 in similar_pairs:
            emb1 = get_embedding(text1)
            emb2 = get_embedding(text2)
            sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            similar_sims.append(sim)

        dissimilar_sims = []
        for text1, text2 in dissimilar_pairs:
            emb1 = get_embedding(text1)
            emb2 = get_embedding(text2)
            sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            dissimilar_sims.append(sim)

        # Compute separation (difference in average similarities)
        avg_similar = np.mean(similar_sims) if similar_sims else 0
        avg_dissimilar = np.mean(dissimilar_sims) if dissimilar_sims else 0

        return avg_similar - avg_dissimilar

    def compute_representation_collapse(
        self,
        sample_texts: list[str],
        tokenizer,
        max_length: int = 128,
    ) -> float:
        """Compute representation collapse metric.

        Lower values indicate more collapse (bad).

        Args:
            sample_texts: Sample texts to compute embeddings for.
            tokenizer: Tokenizer to use.
            max_length: Maximum sequence length.

        Returns:
            Variance of embeddings (higher is better).
        """
        self.model.eval()
        embeddings = []

        for text in sample_texts:
            inputs = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                mask = (
                    inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size())
                )
                masked_hidden = hidden_states * mask
                summed = torch.sum(masked_hidden, dim=1)
                count = torch.clamp(mask.sum(dim=1), min=1e-9)
                embedding = summed / count
                embeddings.append(embedding[0])

        # Stack embeddings and compute variance
        embeddings = torch.stack(embeddings)

        # Compute average pairwise distance (diversity metric)
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = torch.norm(embeddings[i] - embeddings[j]).item()
                distances.append(dist)

        return np.mean(distances) if distances else 0.0


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = "min",
        baseline: float | None = None,
        restore_best_weights: bool = True,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of steps with no improvement to wait.
            min_delta: Minimum change to qualify as improvement.
            mode: "min" or "max".
            baseline: Baseline value to compare against.
            restore_best_weights: Whether to restore best weights.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

        self.best_score = None
        self.best_weights = None
        self.patience_counter = 0
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module | None = None) -> bool:
        """Check if should stop training.

        Args:
            score: Current score to check.
            model: Model to save weights from.

        Returns:
            True if should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            if model and self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            return False

        # Check if improved
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.patience_counter = 0
            if model and self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                if model and self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)

        return self.should_stop

    def reset(self):
        """Reset the early stopping state."""
        self.best_score = None
        self.best_weights = None
        self.patience_counter = 0
        self.should_stop = False


class MultiMetricEarlyStopping:
    """Early stopping based on multiple metrics."""

    def __init__(
        self,
        metrics_config: dict[str, dict[str, Any]],
        patience: int = 10,
        restore_best_weights: bool = True,
    ):
        """Initialize multi-metric early stopping.

        Args:
            metrics_config: Dict mapping metric names to their config:
                {
                    "mlm_loss": {"mode": "min", "weight": 1.0},
                    "embedding_quality": {"mode": "max", "weight": 0.5},
                }
            patience: Number of steps with no improvement.
            restore_best_weights: Whether to restore best weights.
        """
        self.metrics_config = metrics_config
        self.patience = patience
        self.restore_best_weights = restore_best_weights

        self.best_scores = {}
        self.best_combined_score = None
        self.best_weights = None
        self.patience_counter = 0
        self.should_stop = False

    def compute_combined_score(self, metrics: dict[str, float]) -> float:
        """Compute combined score from multiple metrics."""
        combined = 0.0
        total_weight = 0.0

        for name, config in self.metrics_config.items():
            if name in metrics:
                value = metrics[name]
                weight = config.get("weight", 1.0)
                mode = config.get("mode", "min")

                # Normalize based on mode
                if mode == "min":
                    # Lower is better, so invert
                    normalized = -value
                else:
                    # Higher is better
                    normalized = value

                combined += normalized * weight
                total_weight += weight

        return combined / total_weight if total_weight > 0 else 0.0

    def __call__(
        self, metrics: dict[str, float], model: nn.Module | None = None
    ) -> bool:
        """Check if should stop training.

        Args:
            metrics: Current metrics to check.
            model: Model to save weights from.

        Returns:
            True if should stop, False otherwise.
        """
        combined_score = self.compute_combined_score(metrics)

        if self.best_combined_score is None:
            self.best_combined_score = combined_score
            self.best_scores = metrics.copy()
            if model and self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            return False

        # Check if improved
        if combined_score > self.best_combined_score:
            self.best_combined_score = combined_score
            self.best_scores = metrics.copy()
            self.patience_counter = 0
            if model and self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                if model and self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)

        return self.should_stop
