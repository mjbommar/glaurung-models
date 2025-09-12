"""Training module for binary embedding models with advanced features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
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
class TrainerConfig:
    """Configuration for training with optional advanced features."""

    # Basic training parameters
    learning_rate: float = 2e-5  # Optimal for BERT models
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

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

    # Early stopping
    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    monitor_embedding_quality: bool = False

    # Checkpointing
    output_dir: str = "./output"
    save_steps: int = 1000
    save_total_limit: int | None = None
    save_best_only: bool = False

    # Evaluation
    eval_steps: int = 500
    logging_steps: int = 10

    # Performance
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

    @classmethod
    def from_model_size(cls, model_size: str, **kwargs):
        """Create config optimized for model size.

        Args:
            model_size: "small", "base", or "large".
            **kwargs: Override any default settings.
        """
        if model_size == "small":
            config = cls(
                learning_rate=2e-5,
                batch_size=16,
                gradient_accumulation_steps=2,
                scheduler_type="cosine",
                warmup_ratio=0.1,
            )
        elif model_size == "base":
            config = cls(
                learning_rate=3e-5,
                batch_size=8,
                gradient_accumulation_steps=4,
                scheduler_type="two_phase",
                warmup_ratio=0.06,
            )
        else:  # large
            config = cls(
                learning_rate=1e-5,
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


class Trainer:
    """Trainer for binary embedding models with optional advanced features."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BinaryTokenizer,
        train_dataloader: DataLoader,
        config: TrainerConfig,
        model_config: BinaryEmbeddingConfig | None = None,
        val_dataloader: DataLoader | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model to train.
            tokenizer: Tokenizer instance.
            train_dataloader: Training data loader.
            config: Training configuration.
            model_config: Model configuration.
            val_dataloader: Optional validation data loader.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.model_config = model_config

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision="fp16" if config.mixed_precision else "no",
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Calculate total steps
        steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
        if config.max_steps:
            self.total_steps = config.max_steps
        else:
            self.total_steps = steps_per_epoch * config.num_epochs

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

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing and hasattr(
            self.model, "gradient_checkpointing_enable"
        ):
            self.model.gradient_checkpointing_enable()

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history: list[dict[str, Any]] = []

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

    def train(self) -> None:
        """Run the training loop."""
        self.model.train()
        console.print(
            f"[bold green]Starting training on {self.accelerator.device}[/bold green]"
        )
        console.print(f"[cyan]Total steps: {self.total_steps}[/cyan]")
        console.print(
            f"[cyan]Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}[/cyan]"
        )

        epoch = 0
        should_stop = False

        while self.global_step < self.total_steps and not should_stop:
            epoch += 1
            epoch_loss = 0.0
            epoch_steps = 0

            with Progress(
                TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("• Loss: {task.fields[loss]:.4f}"),
                TextColumn("• LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                disable=not self.accelerator.is_local_main_process,
            ) as progress:
                task = progress.add_task(
                    "Training",
                    total=len(self.train_dataloader),
                    epoch=epoch,
                    loss=0.0,
                    lr=self.config.learning_rate,
                )

                for _batch_idx, batch in enumerate(self.train_dataloader):
                    if self.global_step >= self.total_steps:
                        break

                    with self.accelerator.accumulate(self.model):
                        # Forward pass
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )

                        loss = outputs.loss

                        # Backward pass
                        self.accelerator.backward(loss)

                        # Gradient clipping
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm,
                            )
                        else:
                            grad_norm = 0.0

                        # Optimizer step
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    # Update metrics
                    epoch_loss += loss.detach().float()
                    epoch_steps += 1
                    self.global_step += 1

                    # Update progress bar
                    current_loss = epoch_loss / epoch_steps
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress.update(
                        task,
                        advance=1,
                        loss=current_loss,
                        lr=current_lr,
                    )

                    # Log metrics
                    if self.global_step % self.config.logging_steps == 0:
                        log_dict = {
                            "step": self.global_step,
                            "epoch": epoch,
                            "loss": float(current_loss),
                            "learning_rate": float(current_lr),
                        }

                        if self.accelerator.sync_gradients:
                            log_dict["gradient_norm"] = float(grad_norm)

                        self.training_history.append(log_dict)

                        # Limit history size to prevent memory issues
                        if len(self.training_history) > 10000:
                            self.training_history = self.training_history[-5000:]

                        # Track in metrics object if available
                        if self.metrics:
                            self.metrics.log(
                                {
                                    "mlm_loss": float(current_loss),
                                    "learning_rate": float(current_lr),
                                    "gradient_norm": float(grad_norm)
                                    if self.accelerator.sync_gradients
                                    else 0.0,
                                },
                                self.global_step,
                            )

                    # Evaluation
                    if (
                        self.val_dataloader
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        val_loss = self.evaluate()
                        self.model.train()

                        # Check for improvement
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

                        # Early stopping check
                        if self.early_stopping:
                            metrics_dict = {"mlm_loss": val_loss}

                            # Add embedding quality if monitored
                            if (
                                self.embedding_monitor
                                and self.config.monitor_embedding_quality
                            ):
                                quality = self._compute_embedding_quality()
                                metrics_dict["embedding_quality"] = quality
                                console.print(
                                    f"[yellow]Embedding quality: {quality:.4f}[/yellow]"
                                )

                            should_stop = self.early_stopping(metrics_dict, self.model)
                            if should_stop:
                                console.print("[red]Early stopping triggered![/red]")
                                break

                    # Save checkpoint
                    if (
                        not self.config.save_best_only
                        and self.global_step % self.config.save_steps == 0
                    ):
                        self.save_checkpoint()

                    # Periodically clear GPU cache to prevent memory fragmentation
                    if self.global_step % 100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            console.print(
                f"[green]Epoch {epoch} completed - "
                f"Average Loss: {avg_epoch_loss:.4f}[/green]"
            )

        # Save final model
        self.save_checkpoint(is_final=True)
        self.save_training_history()

        # Save metrics if available
        if self.metrics:
            metrics_path = self.output_dir / "training_metrics.json"
            self.metrics.save(metrics_path)
            console.print(f"[dim]Saved metrics to {metrics_path}[/dim]")

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
            self.tokenizer.tokenizer.save_pretrained(checkpoint_dir)
            torch.save(
                unwrapped_model.state_dict(),
                checkpoint_dir / "pytorch_model.bin",
            )

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

        # Clean up old checkpoints if limit is set
        if self.config.save_total_limit and not is_best and not is_final:
            self._cleanup_checkpoints()

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
