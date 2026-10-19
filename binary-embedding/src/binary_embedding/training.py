"""Training module for binary embedding models with advanced features."""

from __future__ import annotations

import atexit
import json
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_linear_schedule_with_warmup

from binary_embedding.models import BinaryEmbeddingConfig
from binary_embedding.tokenizer import BinaryTokenizer

from .losses import info_nce_pairwise, pool_embeddings, split_by_pair_type

# Import WandB if available
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import advanced features if available
try:
    from binary_embedding.metrics import (
        EarlyStopping,
        EmbeddingQualityMonitor,
        MultiMetricEarlyStopping,
        TrainingMetrics,
    )
    from binary_embedding.schedulers import (
        AdaptiveScheduler,
        SchedulerConfig,
        SchedulerType,
        create_scheduler,
    )

    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False

console = Console()


@dataclass
class TrainingMetricSpec:
    key: str
    field: str
    label: str
    wandb_name: str


@dataclass
class StepResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    extras: dict[str, float] | None = None


class JSONLLogger:
    """Simple JSONL writer for detailed local diagnostics."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        # Line-buffered text file to keep entries durable without large flushes.
        self._file = open(self.path, "a", buffering=1, encoding="utf-8")
        atexit.register(self.close)

    def log(self, record: dict[str, Any]) -> None:
        line = json.dumps(record)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


@dataclass
class TrainerConfig:
    """Configuration for training with optional advanced features."""

    # Basic training parameters
    learning_rate: float = 5e-5  # Better for diverse binary datasets
    weight_decay: float = 0.0001  # Reduced for better memorization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 2.0

    # Adaptive gradient clipping
    use_adaptive_grad_clip: bool = False
    grad_clip_percentile: float = 95.0  # Use 95th percentile of recent gradients
    grad_clip_history_size: int = 100  # Number of steps to track
    grad_clip_initial_threshold: float = 10.0  # Initial threshold for first 100 steps

    # Training duration
    num_epochs: int = 3
    max_steps: int | None = None  # If set, overrides num_epochs

    # Batch size and accumulation
    batch_size: int = 16  # Per-device batch size
    gradient_accumulation_steps: int = 2  # Effective batch = batch_size * accumulation

    # Learning rate schedule
    scheduler_type: str = (
        "linear"  # linear, cosine, cosine_with_restarts, two_phase, polynomial
    )
    warmup_steps: int | None = None
    warmup_ratio: float = 0.1  # Used if warmup_steps is None
    min_lr_ratio: float = 0.01

    # Advanced scheduler options (for specific scheduler types)
    phase1_steps: int = 50000  # For two_phase scheduler
    phase1_lr_factor: float = 1.0
    phase2_lr_factor: float = 0.3
    restart_interval: int = 10000  # For cosine_with_restarts
    restart_mult: int = 2
    polynomial_power: float = 1.0  # For polynomial scheduler
    plateau_ratio: float = 0.6  # For trapezoidal scheduler (fraction of steps at peak LR)

    # Early stopping
    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    monitor_embedding_quality: bool = False

    # Checkpointing
    output_dir: str = "./output"
    save_steps: int = 1000
    save_total_limit: int | None = 5
    save_best_only: bool = False

    # Evaluation
    eval_steps: int = 500
    logging_steps: int = 10
    assessment_steps: int = 1000  # Run full assessment every N steps
    run_assessment: bool = True  # Whether to run periodic assessments

    # WandB configuration
    use_wandb: bool = True
    wandb_project: str = "binary-embedding"
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    wandb_notes: str | None = None

    # Performance
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

    # Local logging
    enable_jsonl_logging: bool = True
    jsonl_log_path: str | None = None
    jsonl_log_frequency: int = 1

    # Contrastive training options
    contrastive_enabled: bool = False  # Enable dual-loss training
    contrastive_temperature: float = 0.07
    # Loss weights (can be ramped in during early steps)
    mlm_weight: float = 1.0
    view_contrastive_weight: float = 0.0  # duplicate-view pairs
    same_file_contrastive_weight: float = 0.0  # different chunks in same file
    contrastive_ramp_steps: int = (
        0  # if >0, linearly scale contrastive weights up to step
    )
    # Pooling
    pooling: str = "mean"  # 'mean' supported
    # Pair sampling knobs (mirrors pair_data defaults; used for CLI display only)
    duplicate_pair_probability: float = 0.5
    min_chunk_separation: int | None = None

    @classmethod
    def from_model_size(cls, model_size: str, **kwargs):
        """Create config optimized for model size.

        Args:
            model_size: "nano", "micro", "small", "base", or "large".
            **kwargs: Override any default settings.
        """
        if model_size == "nano":
            config = cls(
                learning_rate=1e-4,  # Higher LR for tiny models
                batch_size=32,
                gradient_accumulation_steps=1,
                scheduler_type="trapezoidal",  # Better than cosine for small models
                warmup_ratio=0.15,  # 15% warmup
                weight_decay=5e-5,  # Lower regularization for tiny models
            )
        elif model_size == "micro":
            config = cls(
                learning_rate=7e-5,  # Higher LR for small models
                batch_size=24,
                gradient_accumulation_steps=2,
                scheduler_type="cosine",
                warmup_ratio=0.12,
            )
        elif model_size == "small":
            config = cls(
                learning_rate=5e-5,  # Increased for better learning
                batch_size=16,
                gradient_accumulation_steps=2,
                scheduler_type="cosine",
                warmup_ratio=0.1,
            )
        elif model_size == "base":
            config = cls(
                learning_rate=5e-5,  # Increased for 50GB diverse dataset
                batch_size=8,
                gradient_accumulation_steps=4,
                scheduler_type="two_phase",
                warmup_ratio=0.06,
            )
        else:  # large
            config = cls(
                learning_rate=3e-5,  # Increased but still conservative for large
                batch_size=4,
                gradient_accumulation_steps=8,
                scheduler_type="cosine_with_restarts",
                warmup_ratio=0.03,
                gradient_checkpointing=True,
            )

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


class BaseTrainer(ABC):
    """Shared trainer base class for MLM and contrastive stages."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BinaryTokenizer,
        train_dataloader: DataLoader,
        config: TrainerConfig,
        stage: str,
        metric_specs: tuple[TrainingMetricSpec, ...],
        model_config: BinaryEmbeddingConfig | None = None,
        val_dataloader: DataLoader | None = None,
        resume_from_checkpoint: Path | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model to train.
            tokenizer: Tokenizer instance.
            train_dataloader: Training data loader.
            config: Training configuration.
            model_config: Model configuration.
            val_dataloader: Optional validation data loader.
            resume_from_checkpoint: Path to checkpoint to resume from.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.model_config = model_config
        self.stage = stage
        self.metric_specs = metric_specs
        self.metric_map = {spec.key: spec for spec in metric_specs}
        self.metric_keys = tuple(spec.key for spec in metric_specs)

        # Update config to reflect stage for downstream logging/checkpointing
        self.config.contrastive_enabled = stage == "contrastive"
        if self.config.jsonl_log_frequency < 1:
            self.config.jsonl_log_frequency = 1

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Enable memory-efficient attention kernels if available
        try:
            # PyTorch 2.0+
            if hasattr(torch.backends, "cuda"):
                if hasattr(torch.backends.cuda, "sdp_kernel"):
                    # Newer API
                    torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_mem_efficient=True, enable_math=False
                    )
                else:
                    # Older API
                    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                        torch.backends.cuda.enable_flash_sdp(True)
                    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                        torch.backends.cuda.enable_mem_efficient_sdp(True)
                    if hasattr(torch.backends.cuda, "enable_math_sdp"):
                        torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass

        # Respect gradient checkpointing early (before Accelerator wrapping)
        if config.gradient_checkpointing:
            try:
                if hasattr(self.model, "config"):
                    # Disable cache for checkpointing compat
                    try:
                        self.model.config.use_cache = False
                    except Exception:
                        pass
                if hasattr(self.model, "gradient_checkpointing_enable"):
                    self.model.gradient_checkpointing_enable()
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()
            except Exception:
                pass

        # Initialize accelerator (default to BF16 when mixed precision is enabled)
        self.accelerator = Accelerator(
            mixed_precision="bf16" if config.mixed_precision else "no",
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Calculate total steps and handle streaming dataloaders without __len__
        self.is_streaming = False
        try:
            dl_len = len(train_dataloader)
            steps_per_epoch = max(
                1,
                math.ceil(dl_len / config.gradient_accumulation_steps),
            )
        except TypeError:
            self.is_streaming = True
            steps_per_epoch = None

        if config.max_steps:
            self.total_steps = config.max_steps
            # Virtual steps_per_epoch for progress display
            if steps_per_epoch is None:
                self.steps_per_epoch = max(
                    1, self.total_steps // max(config.num_epochs, 1)
                )
            else:
                self.steps_per_epoch = steps_per_epoch
        else:
            if steps_per_epoch is None:
                raise ValueError(
                    "Streaming dataloader requires --max-steps to be set (no __len__)."
                )
            self.total_steps = steps_per_epoch * config.num_epochs
            self.steps_per_epoch = steps_per_epoch

        self.jsonl_logger: JSONLLogger | None = None
        self._jsonl_log_path: Path | None = None
        self._last_jsonl_log_time: float | None = None

        if self.config.enable_jsonl_logging and self.accelerator.is_main_process:
            log_path = (
                Path(self.config.jsonl_log_path)
                if self.config.jsonl_log_path
                else self.output_dir / f"{self.stage}_training_log.jsonl"
            )
            if not log_path.is_absolute():
                log_path = Path.cwd() / log_path
            self.jsonl_logger = JSONLLogger(log_path)
            self._jsonl_log_path = log_path

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Setup metrics and monitoring
        self.metrics = None
        self.early_stopping = None
        self.embedding_monitor = None

        if ADVANCED_FEATURES:
            self.metrics = TrainingMetrics()

            if config.use_early_stopping:
                if config.monitor_embedding_quality:
                    # Multi-metric early stopping
                    metrics_config = {
                        "mlm_loss": {"mode": "min", "weight": 1.0},
                        "embedding_quality": {"mode": "max", "weight": 0.5},
                    }
                    self.early_stopping = MultiMetricEarlyStopping(
                        metrics_config=metrics_config,
                        patience=config.early_stopping_patience,
                        restore_best_weights=True,
                    )
                else:
                    # Single metric early stopping
                    self.early_stopping = EarlyStopping(
                        patience=config.early_stopping_patience,
                        min_delta=config.early_stopping_min_delta,
                        mode="min",
                        restore_best_weights=True,
                    )

            if config.monitor_embedding_quality:
                self.embedding_monitor = EmbeddingQualityMonitor(
                    model=model,
                    device=self.accelerator.device,
                )

        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler,
        )

        if self.val_dataloader:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        # Post-prepare safety: ensure use_cache remains disabled if requested
        if config.gradient_checkpointing and hasattr(self.model, "config"):
            try:
                self.model.config.use_cache = False
            except Exception:
                pass

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history: list[dict[str, Any]] = []
        self.assessment_history: list[dict[str, Any]] = []

        # Gradient norm tracking for adaptive clipping
        self.grad_norm_history: deque[float] = deque(
            maxlen=config.grad_clip_history_size
        )
        self.starting_epoch = 0

        # Initialize WandB if enabled
        self.wandb_run = None
        if WANDB_AVAILABLE and config.use_wandb and self.accelerator.is_main_process:
            init_wandb = getattr(self, "_init_wandb", None)
            if callable(init_wandb):
                init_wandb()

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

        if self.jsonl_logger and self.accelerator.is_main_process:
            config_snapshot = asdict(self.config)
            config_snapshot["output_dir"] = str(self.output_dir)
            event_payload = {
                "stage": self.stage,
                "total_steps": int(self.total_steps),
                "steps_per_epoch": int(self.steps_per_epoch),
                "is_streaming": bool(self.is_streaming),
                "resume_from_checkpoint": str(resume_from_checkpoint)
                if resume_from_checkpoint
                else None,
                "jsonl_log_path": str(self._jsonl_log_path)
                if self._jsonl_log_path
                else None,
                "config": config_snapshot,
            }
            self._log_jsonl_event("session_start", event_payload, include_step=False)

    def _setup_optimizer(self) -> AdamW:
        """Setup AdamW optimizer with weight decay.

        Returns:
            Configured optimizer.
        """
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler based on configuration."""
        # Calculate warmup steps
        if self.config.warmup_steps is not None:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(self.total_steps * self.config.warmup_ratio)

        # Use advanced schedulers if available
        if ADVANCED_FEATURES and self.config.scheduler_type != "linear":
            try:
                scheduler_type = SchedulerType(self.config.scheduler_type)
                scheduler_config = SchedulerConfig(
                    scheduler_type=scheduler_type,
                    warmup_steps=warmup_steps,
                    warmup_ratio=self.config.warmup_ratio,
                    min_lr_ratio=self.config.min_lr_ratio,
                    phase1_steps=self.config.phase1_steps,
                    phase1_lr_factor=self.config.phase1_lr_factor,
                    phase2_lr_factor=self.config.phase2_lr_factor,
                    restart_interval=self.config.restart_interval,
                    restart_mult=self.config.restart_mult,
                    power=self.config.polynomial_power,
                    plateau_ratio=self.config.plateau_ratio,
                )
                return create_scheduler(
                    self.optimizer,
                    scheduler_config,
                    self.total_steps,
                )
            except (ValueError, NameError):
                pass

        # Fallback to linear scheduler
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    @abstractmethod
    def _forward_step(self, batch: Any) -> StepResult:
        """Compute the loss and metrics for a single optimizer step."""
        raise NotImplementedError

    def train(self) -> None:
        """Run the training loop."""
        self.model.train()
        self._last_jsonl_log_time = None
        console.print(
            f"[bold green]Starting training on {self.accelerator.device}[/bold green]"
        )
        console.print(f"[cyan]Total steps: {self.total_steps}[/cyan]")
        console.print(
            f"[cyan]Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}[/cyan]"
        )

        if self.jsonl_logger and self.accelerator.is_main_process:
            self._log_jsonl_event(
                "training_start",
                {
                    "effective_batch_size": self.config.batch_size
                    * self.config.gradient_accumulation_steps,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "total_steps": int(self.total_steps),
                },
                include_step=False,
            )

        should_stop = False

        try:
            dataloader_len = len(self.train_dataloader)
            steps_per_epoch = max(
                1,
                math.ceil(dataloader_len / self.config.gradient_accumulation_steps),
            )
            console.print(f"[dim]Dataloader length: {dataloader_len} batches[/dim]")
        except TypeError:
            dataloader_len = None
            steps_per_epoch = self.steps_per_epoch
            console.print("[dim]Dataloader: streaming (no fixed length)[/dim]")

        console.print(
            f"[dim]Gradient accumulation steps: {self.config.gradient_accumulation_steps}[/dim]"
        )

        if self.config.max_steps:
            num_epochs = max(1, math.ceil(self.total_steps / steps_per_epoch))
        else:
            num_epochs = self.config.num_epochs

        console.print(f"[cyan]Steps per epoch: {steps_per_epoch}[/cyan]")
        console.print(f"[cyan]Number of epochs: {num_epochs}[/cyan]")

        start_epoch = self.starting_epoch + 1 if hasattr(self, "starting_epoch") else 1
        if start_epoch > 1:
            console.print(f"[yellow]Resuming from epoch {start_epoch}[/yellow]")

        for epoch in range(start_epoch, num_epochs + 1):
            if should_stop or self.global_step >= self.total_steps:
                break

            remaining_steps = self.total_steps - self.global_step
            steps_this_epoch = min(steps_per_epoch, remaining_steps)
            if steps_this_epoch <= 0:
                break

            if epoch == start_epoch and self.global_step > 0 and self.training_history:
                last_entry = self.training_history[-1]
                epoch_loss_sum = last_entry.get("loss", 0.0)
                epoch_steps = 1
                metric_sums = {
                    spec.key: last_entry.get(spec.key, 0.0)
                    for spec in self.metric_specs
                }
            else:
                epoch_loss_sum = 0.0
                epoch_steps = 0
                metric_sums = {spec.key: 0.0 for spec in self.metric_specs}

                metric_columns = [
                    TextColumn(
                        f"• {spec.label}: " + f"{{task.fields[{spec.field}]:.4f}}"
                    )
                    for spec in self.metric_specs
                ]

            with Progress(
                TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("• Loss: {task.fields[loss]:.4f}"),
                *metric_columns,
                TextColumn("• LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                disable=not self.accelerator.is_local_main_process,
            ) as progress:
                current_lr = (
                    self.scheduler.get_last_lr()[0]
                    if hasattr(self.scheduler, "get_last_lr")
                    else self.config.learning_rate
                )

                task_fields = {
                    "epoch": epoch,
                    "loss": float(epoch_loss_sum / epoch_steps) if epoch_steps else 0.0,
                    "lr": float(current_lr),
                }
                for spec in self.metric_specs:
                    avg_val = (
                        metric_sums[spec.key] / epoch_steps if epoch_steps else 0.0
                    )
                    task_fields[spec.field] = float(avg_val)

                task = progress.add_task(
                    "Training",
                    total=steps_this_epoch,
                    **task_fields,
                )

                batches_to_skip = 0
                steps_to_skip = 0
                if epoch == start_epoch and self.global_step > 0 and steps_per_epoch:
                    steps_in_epoch = self.global_step % steps_per_epoch
                    if steps_in_epoch > 0:
                        batches_to_skip = (
                            steps_in_epoch * self.config.gradient_accumulation_steps
                        )
                        steps_to_skip = steps_in_epoch
                        console.print(
                            f"[dim]Resuming epoch {epoch} from step {steps_in_epoch}/{steps_per_epoch}[/dim]"
                        )
                        console.print(
                            f"[dim]Skipping {batches_to_skip} batches already processed[/dim]"
                        )
                        progress.update(task, completed=steps_to_skip)

                for batch_idx, batch in enumerate(self.train_dataloader):
                    if batch_idx < batches_to_skip:
                        continue

                    if self.global_step >= self.total_steps:
                        break

                    next_step = self.global_step + 1
                    jsonl_candidate = (
                        self.jsonl_logger is not None
                        and self.accelerator.is_main_process
                        and (next_step % self.config.jsonl_log_frequency == 0)
                    )
                    batch_stats: dict[str, Any] | None = None
                    grad_stats: dict[str, Any] | None = None
                    log_this_step = False

                    with self.accelerator.accumulate(self.model):
                        step_result = self._forward_step(batch)
                        loss = step_result.loss

                        self.accelerator.backward(loss)

                        grad_norm = 0.0
                        adaptive_clip_threshold = self.config.max_grad_norm

                        if self.accelerator.sync_gradients:
                            if self.config.use_adaptive_grad_clip:
                                total_norm = 0.0
                                for p in self.model.parameters():
                                    if p.grad is not None:
                                        param_norm = p.grad.data.norm(2)
                                        total_norm += param_norm.item() ** 2
                                current_grad_norm = total_norm**0.5

                                if (
                                    self.global_step
                                    < self.config.grad_clip_history_size
                                ):
                                    clip_threshold = (
                                        self.config.grad_clip_initial_threshold
                                    )
                                else:
                                    if self.grad_norm_history:
                                        clip_threshold = np.percentile(
                                            list(self.grad_norm_history),
                                            self.config.grad_clip_percentile,
                                        )
                                    else:
                                        clip_threshold = (
                                            self.config.grad_clip_initial_threshold
                                        )

                                clip_threshold = max(clip_threshold, 1.0)

                                # Enforce the configured maximum even when adaptive clipping is enabled
                                if self.config.max_grad_norm > 0:
                                    clip_threshold = min(
                                        clip_threshold,
                                        self.config.max_grad_norm,
                                    )

                                grad_norm_tensor = self.accelerator.clip_grad_norm_(
                                    self.model.parameters(),
                                    clip_threshold,
                                )
                                grad_norm = float(grad_norm_tensor)
                                self.grad_norm_history.append(current_grad_norm)
                                adaptive_clip_threshold = clip_threshold
                            else:
                                grad_norm_tensor = self.accelerator.clip_grad_norm_(
                                    self.model.parameters(),
                                    self.config.max_grad_norm,
                                )
                                grad_norm = float(grad_norm_tensor)
                                adaptive_clip_threshold = self.config.max_grad_norm

                            if jsonl_candidate:
                                batch_stats = self._collect_batch_stats(batch)
                                grad_stats = self._collect_gradient_stats()
                                log_this_step = True

                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad(set_to_none=True)

                        if not self.accelerator.sync_gradients:
                            continue

                        step_loss = float(loss.detach().float().cpu())
                        epoch_loss_sum += step_loss
                        epoch_steps += 1
                        self.global_step += 1

                        step_metrics: dict[str, float] = {}
                        for spec in self.metric_specs:
                            value = step_result.metrics.get(spec.key)
                            numeric = (
                                float(value.detach().float().cpu())
                                if value is not None
                                else 0.0
                            )
                            metric_sums[spec.key] += numeric
                            step_metrics[spec.key] = numeric

                        avg_loss = epoch_loss_sum / epoch_steps
                        avg_metrics = {
                            spec.key: metric_sums[spec.key] / epoch_steps
                            for spec in self.metric_specs
                        }

                        current_lr = (
                            self.scheduler.get_last_lr()[0]
                            if hasattr(self.scheduler, "get_last_lr")
                            else self.config.learning_rate
                        )

                        update_fields = {
                            "loss": float(avg_loss),
                            "lr": float(current_lr),
                        }
                        for spec in self.metric_specs:
                            update_fields[spec.field] = float(
                                avg_metrics.get(spec.key, 0.0)
                            )

                        progress.update(task, advance=1, **update_fields)

                        if log_this_step:
                            extras_payload = (
                                {k: float(v) for k, v in step_result.extras.items()}
                                if step_result.extras
                                else None
                            )
                            batch_payload = batch_stats if batch_stats else None
                            grad_payload = grad_stats if grad_stats else None
                            self._log_jsonl_step(
                                epoch=epoch,
                                epoch_step=epoch_steps,
                                steps_this_epoch=steps_this_epoch,
                                step_loss=step_loss,
                                avg_loss=avg_loss,
                                grad_norm=grad_norm,
                                adaptive_clip_threshold=adaptive_clip_threshold,
                                current_lr=current_lr,
                                step_metrics=step_metrics,
                                batch_stats=batch_payload,
                                grad_stats=grad_payload,
                                extras=extras_payload,
                            )

                        if self.global_step % self.config.logging_steps == 0:
                            log_entry = {
                                "step": self.global_step,
                                "epoch": epoch,
                                "loss": float(avg_loss),
                                "grad_norm": float(grad_norm),
                                "grad_clip_threshold": float(adaptive_clip_threshold),
                                "learning_rate": float(current_lr),
                            }
                            for spec in self.metric_specs:
                                log_entry[spec.key] = float(
                                    avg_metrics.get(spec.key, 0.0)
                                )

                            self.training_history.append(log_entry)

                            if len(self.training_history) > 10000:
                                self.training_history = self.training_history[-5000:]

                            if self.metrics:
                                metrics_payload = {
                                    "learning_rate": float(current_lr),
                                    "gradient_norm": float(grad_norm),
                                }
                                if "mlm_loss" in step_metrics:
                                    metrics_payload["mlm_loss"] = float(
                                        step_metrics["mlm_loss"]
                                    )
                                self.metrics.log(metrics_payload, self.global_step)

                            if self.wandb_run and self.accelerator.is_main_process:
                                wandb_log = {
                                    "train/loss": float(avg_loss),
                                    "train/learning_rate": float(current_lr),
                                    "train/epoch": epoch,
                                    "train/global_step": self.global_step,
                                    "train/gradient_norm": float(grad_norm),
                                }
                                for spec in self.metric_specs:
                                    wandb_log[spec.wandb_name] = float(
                                        step_metrics.get(spec.key, 0.0)
                                    )

                                    if step_result.extras:
                                        for key, value in step_result.extras.items():
                                            wandb_log[f"train/{key}"] = float(value)

                                wandb.log(wandb_log, step=self.global_step)

                        if (
                            self.val_dataloader
                            and self.global_step % self.config.eval_steps == 0
                        ):
                            val_loss = self.evaluate()
                            self.model.train()
                            improved = val_loss < self.best_val_loss
                            if improved:
                                self.best_val_loss = val_loss
                                if self.config.save_best_only:
                                    self.save_checkpoint(is_best=True)

                            console.print(
                                f"[yellow]Step {self.global_step} - "
                                f"Val Loss: {val_loss:.4f} "
                                f"{'(improved)' if improved else ''}[/yellow]"
                            )

                            if self.jsonl_logger and self.accelerator.is_main_process:
                                self._log_jsonl_event(
                                    "evaluation",
                                    {
                                        "val_loss": float(val_loss),
                                        "improved": bool(improved),
                                    },
                                )

                            if self.wandb_run and self.accelerator.is_main_process:
                                wandb.log(
                                    {"eval/loss": float(val_loss)},
                                    step=self.global_step,
                                )

                            if self.early_stopping:
                                metrics_dict = {"mlm_loss": float(val_loss)}
                                if (
                                    self.embedding_monitor
                                    and self.config.monitor_embedding_quality
                                ):
                                    quality = self._compute_embedding_quality()
                                    metrics_dict["embedding_quality"] = quality
                                    console.print(
                                        f"[yellow]Embedding quality: {quality:.4f}[/yellow]"
                                    )

                                should_stop = self.early_stopping(
                                    metrics_dict, self.model
                                )
                                if should_stop:
                                    console.print(
                                        "[red]Early stopping triggered![/red]"
                                    )
                                    break

                        if (
                            self.config.run_assessment
                            and self.global_step % self.config.assessment_steps == 0
                            and self.global_step > 0
                        ):
                            assessment_results = self.run_assessment()
                            assessment_log = {
                                key: float(value)
                                for key, value in assessment_results.items()
                            }
                            self.assessment_history.append(
                                {"step": self.global_step, **assessment_log}
                            )
                            if self.jsonl_logger and self.accelerator.is_main_process:
                                self._log_jsonl_event(
                                    "assessment",
                                    {
                                        "assessment": assessment_log,
                                    },
                                )
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if self.wandb_run and self.accelerator.is_main_process:
                                wandb_payload = {
                                    **assessment_log,
                                    "assessment/step": float(self.global_step),
                                }
                                wandb.log(wandb_payload, step=self.global_step)
                                self.wandb_run.summary.update(assessment_log)
                                self.wandb_run.summary["assessment/last_step"] = float(
                                    self.global_step
                                )

                        if (
                            not self.config.save_best_only
                            and self.global_step % self.config.save_steps == 0
                            and self.global_step > 0
                        ):
                            self.save_checkpoint()

                        if self.global_step % 100 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                avg_epoch_loss = (
                    epoch_loss_sum / epoch_steps if epoch_steps > 0 else 0.0
                )
                console.print(
                    f"[green]Epoch {epoch} complete - avg loss: {avg_epoch_loss:.4f}[/green]"
                )

            if should_stop:
                break

        self.save_checkpoint(is_final=True)
        self.save_training_history()

        if self.jsonl_logger and self.accelerator.is_main_process:
            self._log_jsonl_event(
                "training_complete",
                {
                    "final_step": int(self.global_step),
                    "best_val_loss": float(self.best_val_loss),
                },
            )
            self.jsonl_logger.close()
            self.jsonl_logger = None

        if self.wandb_run:
            wandb.finish()

        console.print("[bold green]Training complete![/bold green]")

    def evaluate(self) -> float:
        """Evaluate the model on validation data.

        Returns:
            Average validation loss.
        """
        if not self.val_dataloader:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                total_loss += outputs.loss.detach().float()
                total_steps += 1

        avg_loss = total_loss / total_steps
        return float(avg_loss)

    def _compute_embedding_quality(self) -> float:
        """Compute embedding quality metric.

        Returns:
            Quality score (higher is better).
        """
        if not self.embedding_monitor:
            return 0.0

        # Generate some sample texts for quality assessment
        # In practice, you'd want to use a held-out set
        sample_texts = [
            b"\x7fELF\x01\x01\x01".decode("latin-1"),
            b"MZ\x90\x00\x03".decode("latin-1"),
            b"\xca\xfe\xba\xbe".decode("latin-1"),
        ]

        # Compute representation diversity
        diversity = self.embedding_monitor.compute_representation_collapse(
            sample_texts, self.tokenizer, max_length=128
        )

        return diversity

    def save_checkpoint(
        self,
        is_best: bool = False,
        is_final: bool = False,
    ) -> None:
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far.
            is_final: Whether this is the final checkpoint.
        """
        if not self.accelerator.is_local_main_process:
            return

        # Determine checkpoint name
        if is_final:
            checkpoint_dir = self.output_dir / "final_model"
        elif is_best:
            checkpoint_dir = self.output_dir / "best_model"
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Handle different model types
        if isinstance(unwrapped_model, PreTrainedModel):
            unwrapped_model.save_pretrained(checkpoint_dir)
        elif hasattr(unwrapped_model, "base_model"):
            unwrapped_model.base_model.save_pretrained(checkpoint_dir)
        else:
            # Save state dict as fallback
            torch.save(
                unwrapped_model.state_dict(),
                checkpoint_dir / "pytorch_model.bin",
            )

        # Always save tokenizer
        if hasattr(self.tokenizer, "tokenizer"):
            # BinaryTokenizer wraps the actual tokenizer
            self.tokenizer.tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
        else:
            # In case it's a different tokenizer type
            self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        torch.save(
            {
                "global_step": self.global_step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config.__dict__,
            },
            checkpoint_dir / "training_state.pt",
        )

        console.print(f"[dim]Saved checkpoint to {checkpoint_dir}[/dim]")

        if self.jsonl_logger and self.accelerator.is_main_process:
            self._log_jsonl_event(
                "checkpoint_saved",
                {
                    "path": str(checkpoint_dir),
                    "is_best": bool(is_best),
                    "is_final": bool(is_final),
                },
            )

        # Clean up old checkpoints if limit is set
        if self.config.save_total_limit and not is_best and not is_final:
            self._cleanup_checkpoints()

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.
        """
        checkpoint_path = Path(checkpoint_path)

        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(
                training_state_path, map_location=self.accelerator.device
            )

            # Restore training state
            self.global_step = state["global_step"]
            self.best_val_loss = state.get("best_val_loss", float("inf"))

            # Restore optimizer and scheduler states
            try:
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
                self.scheduler.load_state_dict(state["scheduler_state_dict"])

                # Verify the learning rate is correctly restored
                current_lr = self.scheduler.get_last_lr()[0]
                console.print(f"[dim]  Learning rate restored: {current_lr:.2e}[/dim]")

                # Debug: print scheduler state for verification
                if hasattr(self.scheduler, "state_dict"):
                    scheduler_state = self.scheduler.state_dict()
                    console.print(
                        f"[dim]  Scheduler last_epoch: {scheduler_state.get('last_epoch', 'N/A')}[/dim]"
                    )
                    console.print(
                        f"[dim]  Scheduler _step_count: {scheduler_state.get('_step_count', 'N/A')}[/dim]"
                    )
            except Exception as e:
                console.print(
                    f"[yellow]⚠ Warning: Could not fully restore optimizer/scheduler state: {e}[/yellow]"
                )
                console.print(
                    "[yellow]  Training will continue but learning rate schedule may be reset[/yellow]"
                )

            # Calculate starting epoch based on global_step
            # For streaming datasets, we can't get the length, so estimate or skip
            try:
                steps_per_epoch = (
                    len(self.train_dataloader)
                    // self.config.gradient_accumulation_steps
                )
                self.starting_epoch = self.global_step // steps_per_epoch
            except TypeError:
                # Streaming dataset doesn't have length, estimate based on global_step
                # Assume a reasonable default or use config if available
                self.starting_epoch = 0  # Will be updated during training
                console.print(
                    "[yellow]  Note: Using streaming dataset, epoch tracking approximate[/yellow]"
                )

            console.print(
                f"[green]✓ Resumed from checkpoint at step {self.global_step}[/green]"
            )
            console.print(f"[dim]  Starting from epoch {self.starting_epoch + 1}[/dim]")
            console.print(
                f"[dim]  Best validation loss: {self.best_val_loss:.4f}[/dim]"
            )

            if self.jsonl_logger and self.accelerator.is_main_process:
                self._log_jsonl_event(
                    "checkpoint_loaded",
                    {
                        "path": str(checkpoint_path),
                        "restored_step": int(self.global_step),
                        "best_val_loss": float(self.best_val_loss),
                    },
                )

            # Load training history if it exists
            history_path = checkpoint_path.parent / "training_history.json"
            if history_path.exists():
                import json

                with open(history_path) as f:
                    self.training_history = json.load(f)
                console.print(
                    f"[dim]  Training history loaded: {len(self.training_history)} entries[/dim]"
                )

            assessment_history_path = checkpoint_path.parent / "assessment_history.json"
            if assessment_history_path.exists():
                with open(assessment_history_path) as f:
                    self.assessment_history = json.load(f)
                console.print(
                    f"[dim]  Assessment history loaded: {len(self.assessment_history)} entries[/dim]"
                )
        else:
            console.print(
                f"[yellow]⚠ No training state found at {checkpoint_path}, starting fresh[/yellow]"
            )
            if self.jsonl_logger and self.accelerator.is_main_process:
                self._log_jsonl_event(
                    "checkpoint_missing",
                    {"path": str(checkpoint_path)},
                )

    def _init_wandb(self) -> None:
        """Initialize WandB run with configuration."""
        if not WANDB_AVAILABLE:
            return

        # Prepare config for logging
        # Try to fetch dataloader length if available
        try:
            dl_length = len(self.train_dataloader)
        except TypeError:
            dl_length = None

        wandb_config = {
            # Model configuration
            "model_type": self.model_config.model_type
            if self.model_config
            else "unknown",
            "vocab_size": self.model_config.vocab_size if self.model_config else None,
            "hidden_size": self.model_config.hidden_size if self.model_config else None,
            "num_hidden_layers": self.model_config.num_hidden_layers
            if self.model_config
            else None,
            "num_attention_heads": self.model_config.num_attention_heads
            if self.model_config
            else None,
            "intermediate_size": self.model_config.intermediate_size
            if self.model_config
            else None,
            "max_position_embeddings": self.model_config.max_position_embeddings
            if self.model_config
            else None,
            "mlm_probability": self.model_config.mlm_probability
            if self.model_config
            else None,
            # Training configuration
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "effective_batch_size": self.config.batch_size
            * self.config.gradient_accumulation_steps,
            "num_epochs": self.config.num_epochs,
            "max_steps": self.config.max_steps,
            "warmup_steps": self.config.warmup_steps,
            "warmup_ratio": self.config.warmup_ratio,
            "scheduler_type": self.config.scheduler_type,
            "weight_decay": self.config.weight_decay,
            "adam_beta1": self.config.adam_beta1,
            "adam_beta2": self.config.adam_beta2,
            "adam_epsilon": self.config.adam_epsilon,
            "max_grad_norm": self.config.max_grad_norm,
            "mixed_precision": self.config.mixed_precision,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            # Dataset info
            "dataloader_length": dl_length,
            "dataloader_streaming": self.is_streaming,
            "total_steps": self.total_steps,
            # Assessment configuration
            "run_assessment": self.config.run_assessment,
            "assessment_steps": self.config.assessment_steps,
        }

        # Initialize WandB run
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            notes=self.config.wandb_notes,
            config=wandb_config,
            resume="allow",
        )

        # Watch the model for gradient tracking (optional)
        wandb.watch(self.model, log="all", log_freq=100)

        console.print(f"[green]✓ WandB initialized: {self.wandb_run.url}[/green]")

    def run_assessment(self) -> dict[str, float]:
        """Run full assessment and return metrics."""
        from binary_embedding.assessment import BinaryAssessment

        console.print("[yellow]Running assessment...[/yellow]")

        # Set model to eval mode and disable gradients
        self.model.eval()

        # Create assessment instance
        assessment = BinaryAssessment(
            model=self.accelerator.unwrap_model(self.model),
            tokenizer=self.tokenizer,
            device=self.accelerator.device,
        )

        # Run all assessments with no gradients
        results = {}

        try:
            with torch.no_grad():
                # File header recognition
                header_result = assessment.assess_file_header_recognition()
                results["assessment/file_header_score"] = header_result.score
                results["assessment/file_header_passed"] = float(header_result.passed)

                # Binary pattern learning
                pattern_result = assessment.assess_binary_pattern_learning()
                results["assessment/pattern_learning_score"] = pattern_result.score
                results["assessment/pattern_learning_passed"] = float(
                    pattern_result.passed
                )

                # Context understanding
                context_result = assessment.assess_context_understanding()
                results["assessment/context_understanding_score"] = context_result.score
                results["assessment/context_understanding_passed"] = float(
                    context_result.passed
                )

                # Embedding quality
                embedding_result = assessment.assess_embedding_quality()
                results["assessment/embedding_quality_score"] = embedding_result.score
                results["assessment/embedding_quality_passed"] = float(
                    embedding_result.passed
                )

                # Overall metrics
                all_scores = [
                    header_result.score,
                    pattern_result.score,
                    context_result.score,
                    embedding_result.score,
                ]
                results["assessment/average_score"] = sum(all_scores) / len(all_scores)
                results["assessment/pass_rate"] = (
                    sum(
                        [
                            header_result.passed,
                            pattern_result.passed,
                            context_result.passed,
                            embedding_result.passed,
                        ]
                    )
                    / 4.0
                )

                console.print(
                    f"[green]Assessment complete - Average score: {results['assessment/average_score']:.2%}[/green]"
                )

        except Exception as e:
            console.print(f"[red]Assessment failed: {e}[/red]")
            results["assessment/error"] = 1.0

        finally:
            # Always restore training mode
            self.model.train()

        return results

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints to maintain save_total_limit."""
        if not self.config.save_total_limit:
            return

        # Find all checkpoint directories
        checkpoint_dirs = sorted(
            [d for d in self.output_dir.glob("checkpoint-*") if d.is_dir()],
            key=lambda x: int(x.name.split("-")[1]),
        )

        # Keep only the most recent ones
        while len(checkpoint_dirs) > self.config.save_total_limit:
            old_checkpoint = checkpoint_dirs.pop(0)
            console.print(f"[dim]Removing old checkpoint: {old_checkpoint}[/dim]")
            for file in old_checkpoint.iterdir():
                file.unlink()
            old_checkpoint.rmdir()

    def save_training_history(self) -> None:
        """Save training history to JSON."""
        if not self.accelerator.is_local_main_process:
            return

        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        console.print(f"[dim]Saved training history to {history_path}[/dim]")

        assessment_path = self.output_dir / "assessment_history.json"
        with open(assessment_path, "w") as f:
            json.dump(self.assessment_history, f, indent=2)

        console.print(f"[dim]Saved assessment history to {assessment_path}[/dim]")

    # ------------------------------------------------------------------
    # JSONL diagnostics helpers

    def _log_jsonl_event(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        include_step: bool = True,
    ) -> None:
        if not self.jsonl_logger or not self.accelerator.is_main_process:
            return

        record: dict[str, Any] = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds"),
        }
        if include_step:
            record["step"] = int(self.global_step)
        record.update(payload)
        self.jsonl_logger.log(record)

    def _infer_optimizer_step(self) -> int | None:
        try:
            for group in self.optimizer.param_groups:
                for param in group.get("params", []):
                    state = self.optimizer.state.get(param)
                    if state and "step" in state:
                        return int(state["step"])
        except Exception:
            return None
        return None

    def _collect_optimizer_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "type": self.optimizer.__class__.__name__,
            "param_groups": [],
        }
        for idx, group in enumerate(self.optimizer.param_groups):
            group_info: dict[str, Any] = {
                "index": idx,
                "lr": float(group.get("lr", 0.0)),
                "weight_decay": float(group.get("weight_decay", 0.0)),
            }
            betas = group.get("betas")
            if betas:
                group_info["beta1"] = float(betas[0])
                group_info["beta2"] = float(betas[1])
            if "eps" in group:
                group_info["epsilon"] = float(group["eps"])
            if "initial_lr" in group:
                group_info["initial_lr"] = float(group["initial_lr"])
            snapshot["param_groups"].append(group_info)
        optimizer_step = self._infer_optimizer_step()
        if optimizer_step is not None:
            snapshot["step"] = optimizer_step
        return snapshot

    def _collect_batch_stats(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        try:
            input_ids = batch["input_ids"].detach()
            attention_mask = batch["attention_mask"].detach()
        except KeyError:
            return stats

        if input_ids.device.type != "cpu":
            input_ids = input_ids.to("cpu", non_blocking=True)
        if attention_mask.device.type != "cpu":
            attention_mask = attention_mask.to("cpu", non_blocking=True)

        # Normalize shapes for MLM vs contrastive batches.
        if input_ids.dim() == 3:
            batch_size, views, seq_len = input_ids.shape
            reshaped_inputs = input_ids.view(batch_size * views, seq_len)
            reshaped_mask = attention_mask.view(batch_size * views, seq_len)
        else:
            reshaped_inputs = input_ids
            reshaped_mask = attention_mask
            batch_size, seq_len = input_ids.shape
            views = 1

        active_tokens = int(reshaped_mask.sum().item())
        total_tokens = reshaped_mask.numel()
        pad_tokens = total_tokens - active_tokens
        empty_sequences = int((reshaped_mask.sum(dim=-1) == 0).sum().item())

        stats.update(
            {
                "batch_size": int(batch_size),
                "views": int(views),
                "sequence_length": int(seq_len),
                "active_tokens": active_tokens,
                "pad_tokens": int(pad_tokens),
                "pad_ratio": float(pad_tokens / total_tokens) if total_tokens else 0.0,
                "empty_sequences": empty_sequences,
            }
        )

        if "labels" in batch:
            labels = batch["labels"].detach()
            if labels.device.type != "cpu":
                labels = labels.to("cpu", non_blocking=True)
            effective_labels = labels.view(reshaped_inputs.shape)
            mask_tokens = int((effective_labels != -100).sum().item())
            stats["masked_tokens"] = mask_tokens
            stats["mask_ratio"] = (
                float(mask_tokens / active_tokens) if active_tokens else 0.0
            )

        token_selector = reshaped_mask.bool()
        valid_tokens = reshaped_inputs[token_selector]
        if valid_tokens.numel() > 0:
            valid_tokens_float = valid_tokens.float()
            stats["token_mean"] = float(valid_tokens_float.mean().item())
            stats["token_std"] = float(valid_tokens_float.std(unbiased=False).item())
            stats["token_min"] = int(valid_tokens.min().item())
            stats["token_max"] = int(valid_tokens.max().item())
            stats["unique_tokens"] = int(torch.unique(valid_tokens).numel())
        else:
            stats["token_mean"] = 0.0
            stats["token_std"] = 0.0
            stats["token_min"] = 0
            stats["token_max"] = 0
            stats["unique_tokens"] = 0

        return stats

    def _collect_gradient_stats(self) -> dict[str, Any]:
        if not self.accelerator.sync_gradients:
            return {}

        stats: dict[str, Any] = {"groups": []}
        total_abs = 0.0
        total_sq = 0.0
        total_params = 0
        max_abs = 0.0

        with torch.no_grad():
            for idx, group in enumerate(self.optimizer.param_groups):
                group_abs = 0.0
                group_sq = 0.0
                group_params = 0
                group_max = 0.0
                for param in group.get("params", []):
                    grad = param.grad
                    if grad is None:
                        continue
                    grad_data = grad.detach()
                    grad_float = grad_data.float()
                    group_abs += float(grad_float.abs().sum().item())
                    group_sq += float(torch.sum(grad_float * grad_float).item())
                    group_params += grad_data.numel()
                    param_max = float(grad_float.abs().max().item())
                    if param_max > group_max:
                        group_max = param_max
                if group_params > 0:
                    stats["groups"].append(
                        {
                            "index": idx,
                            "mean_abs": group_abs / group_params,
                            "rms": math.sqrt(group_sq / group_params),
                            "max_abs": group_max,
                            "param_count": int(group_params),
                        }
                    )
                    total_abs += group_abs
                    total_sq += group_sq
                    total_params += group_params
                    if group_max > max_abs:
                        max_abs = group_max

        if total_params > 0:
            stats["global"] = {
                "mean_abs": total_abs / total_params,
                "rms": math.sqrt(total_sq / total_params),
                "max_abs": max_abs,
                "param_count": int(total_params),
            }

        return stats if stats.get("groups") else {}

    def _log_jsonl_step(
        self,
        *,
        epoch: int,
        epoch_step: int,
        steps_this_epoch: int,
        step_loss: float,
        avg_loss: float,
        grad_norm: float,
        adaptive_clip_threshold: float,
        current_lr: float,
        step_metrics: dict[str, float],
        batch_stats: dict[str, Any] | None,
        grad_stats: dict[str, Any] | None,
        extras: dict[str, float] | None,
    ) -> None:
        if not self.jsonl_logger or not self.accelerator.is_main_process:
            return

        now = time.time()
        step_time = None
        samples_per_sec = None
        if self._last_jsonl_log_time is not None:
            step_time = now - self._last_jsonl_log_time
            effective_batch = (
                self.config.batch_size * self.config.gradient_accumulation_steps
            )
            if step_time and step_time > 0:
                samples_per_sec = effective_batch / step_time
        self._last_jsonl_log_time = now

        optimizer_snapshot = self._collect_optimizer_snapshot()
        payload: dict[str, Any] = {
            "epoch": int(epoch),
            "epoch_step": int(epoch_step),
            "steps_this_epoch": int(steps_this_epoch),
            "progress": float(self.global_step / self.total_steps)
            if self.total_steps
            else None,
            "loss": {
                "current": float(step_loss),
                "running_avg": float(avg_loss),
            },
            "metrics": step_metrics,
            "optimizer": {
                **optimizer_snapshot,
                "learning_rate": float(current_lr),
                "grad_norm": float(grad_norm),
                "adaptive_clip_threshold": float(adaptive_clip_threshold),
                "max_grad_norm": float(self.config.max_grad_norm),
                "use_adaptive_grad_clip": bool(self.config.use_adaptive_grad_clip),
            },
        }

        if grad_stats:
            payload["gradients"] = grad_stats
        if batch_stats:
            payload["batch"] = batch_stats
        if extras:
            payload["extras"] = extras
        if step_time is not None:
            payload["timing"] = {
                "step_time_sec": step_time,
                "samples_per_sec": samples_per_sec,
            }

        self._log_jsonl_event("train_step", payload)



class MLMTrainer(BaseTrainer):
    """Trainer for the MLM pretraining stage."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BinaryTokenizer,
        train_dataloader: DataLoader,
        config: TrainerConfig,
        model_config: BinaryEmbeddingConfig | None = None,
        val_dataloader: DataLoader | None = None,
        resume_from_checkpoint: Path | None = None,
    ) -> None:
        metric_specs = (
            TrainingMetricSpec(
                key="mlm_loss",
                field="mlm",
                label="MLM",
                wandb_name="train/mlm_loss",
            ),
        )
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            config=config,
            stage="mlm",
            metric_specs=metric_specs,
            model_config=model_config,
            val_dataloader=val_dataloader,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    def _forward_step(self, batch: dict[str, torch.Tensor]) -> StepResult:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        with self.accelerator.autocast():
            if hasattr(self.model, "roberta"):
                backbone = self.model.roberta
            elif hasattr(self.model, "bert"):
                backbone = self.model.bert
            else:
                backbone = None

            if hasattr(self.model, "lm_head"):
                mlm_head = self.model.lm_head
            elif hasattr(self.model, "cls"):
                mlm_head = self.model.cls
            else:
                mlm_head = None

            if backbone is not None and mlm_head is not None:
                base_out = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                last_hidden = base_out.last_hidden_state
                prediction_scores = mlm_head(last_hidden)
                vocab_size = prediction_scores.size(-1)
                loss = F.cross_entropy(
                    prediction_scores.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss

        metrics = {"mlm_loss": loss.detach().float()}
        extras = {"mlm_weight": float(self.config.mlm_weight)}
        return StepResult(loss=loss, metrics=metrics, extras=extras)


class ContrastiveTrainer(BaseTrainer):
    """Trainer for the contrastive fine-tuning stage."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BinaryTokenizer,
        train_dataloader: DataLoader,
        config: TrainerConfig,
        model_config: BinaryEmbeddingConfig | None = None,
        val_dataloader: DataLoader | None = None,
        resume_from_checkpoint: Path | None = None,
    ) -> None:
        metric_specs = (
            TrainingMetricSpec(
                key="mlm_loss",
                field="mlm",
                label="MLM",
                wandb_name="train/mlm_loss",
            ),
            TrainingMetricSpec(
                key="view_loss",
                field="vctr",
                label="VCtr",
                wandb_name="train/view_contrastive_loss",
            ),
            TrainingMetricSpec(
                key="same_file_loss",
                field="sfctr",
                label="SFCtr",
                wandb_name="train/same_file_contrastive_loss",
            ),
        )
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            config=config,
            stage="contrastive",
            metric_specs=metric_specs,
            model_config=model_config,
            val_dataloader=val_dataloader,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    def _forward_step(self, batch: dict[str, torch.Tensor]) -> StepResult:
        if "input_ids" not in batch or batch["input_ids"].dim() != 3:
            raise ValueError("ContrastiveTrainer expects batches with shape [B, 2, L]")

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        pair_types = batch.get("pair_types")
        if pair_types is None:
            raise ValueError("Contrastive batches must include 'pair_types'")

        bsz, views, seqlen = input_ids.shape
        flat_input_ids = input_ids.view(bsz * views, seqlen)
        flat_attention = attention_mask.view(bsz * views, seqlen)
        flat_labels = labels.view(bsz * views, seqlen)

        with self.accelerator.autocast():
            if hasattr(self.model, "roberta"):
                backbone = self.model.roberta
            elif hasattr(self.model, "bert"):
                backbone = self.model.bert
            else:
                backbone = None

            if hasattr(self.model, "lm_head"):
                mlm_head = self.model.lm_head
            elif hasattr(self.model, "cls"):
                mlm_head = self.model.cls
            else:
                mlm_head = None

            if backbone is not None and mlm_head is not None:
                base_out = backbone(
                    input_ids=flat_input_ids,
                    attention_mask=flat_attention,
                    return_dict=True,
                )
                last_hidden = base_out.last_hidden_state
                prediction_scores = mlm_head(last_hidden)
                vocab_size = prediction_scores.size(-1)
                mlm_loss = F.cross_entropy(
                    prediction_scores.view(-1, vocab_size),
                    flat_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                outputs = self.model(
                    input_ids=flat_input_ids,
                    attention_mask=flat_attention,
                    labels=flat_labels,
                    return_dict=True,
                )
                mlm_loss = outputs.loss
                if hasattr(outputs, "last_hidden_state"):
                    last_hidden = outputs.last_hidden_state
                else:
                    outputs_h = self.model(
                        input_ids=flat_input_ids,
                        attention_mask=flat_attention,
                        return_dict=True,
                        output_hidden_states=True,
                    )
                    last_hidden = outputs_h.hidden_states[-1]

        embeddings = pool_embeddings(
            last_hidden,
            flat_attention,
            method=self.config.pooling,
        )
        embeddings = embeddings.view(bsz, views, -1)

        pair_types = pair_types.to(embeddings.device)
        dup_a, dup_b, sf_a, sf_b = split_by_pair_type(embeddings, pair_types)

        view_loss = (
            info_nce_pairwise(dup_a, dup_b, self.config.contrastive_temperature)
            if dup_a.size(0) > 0
            else mlm_loss.new_tensor(0.0)
        )
        same_file_loss = (
            info_nce_pairwise(sf_a, sf_b, self.config.contrastive_temperature)
            if sf_a.size(0) > 0
            else mlm_loss.new_tensor(0.0)
        )

        if (
            self.config.contrastive_ramp_steps
            and self.config.contrastive_ramp_steps > 0
        ):
            ramp = min(
                1.0,
                self.global_step / float(max(1, self.config.contrastive_ramp_steps)),
            )
        else:
            ramp = 1.0

        mlm_weight = float(self.config.mlm_weight)
        view_weight = float(self.config.view_contrastive_weight * ramp)
        same_file_weight = float(self.config.same_file_contrastive_weight * ramp)

        total_loss = (
            mlm_weight * mlm_loss
            + view_weight * view_loss
            + same_file_weight * same_file_loss
        )

        metrics = {
            "mlm_loss": mlm_loss.detach().float(),
            "view_loss": view_loss.detach().float(),
            "same_file_loss": same_file_loss.detach().float(),
        }
        extras = {
            "mlm_weight": mlm_weight,
            "view_weight": view_weight,
            "same_file_weight": same_file_weight,
        }

        return StepResult(loss=total_loss, metrics=metrics, extras=extras)
