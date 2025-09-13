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
    assessment_steps: int = 10000  # Run full assessment every N steps
    run_assessment: bool = False  # Whether to run periodic assessments

    # WandB configuration
    use_wandb: bool = True
    wandb_project: str = "binary-embedding"
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    wandb_notes: str | None = None

    # Performance
    mixed_precision: bool = True
    gradient_checkpointing: bool = False

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

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision="fp16" if config.mixed_precision else "no",
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Calculate total steps and handle streaming dataloaders without __len__
        self.is_streaming = False
        try:
            dl_len = len(train_dataloader)
            steps_per_epoch = max(1, dl_len // config.gradient_accumulation_steps)
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
        self.starting_epoch = 0

        # Initialize WandB if enabled
        self.wandb_run = None
        if WANDB_AVAILABLE and config.use_wandb and self.accelerator.is_main_process:
            self._init_wandb()

        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)

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

        should_stop = False

        # Calculate number of epochs needed
        # Note: steps_per_epoch should match the calculation used for total_steps
        try:
            dataloader_len = len(self.train_dataloader)
            steps_per_epoch = max(
                1, dataloader_len // self.config.gradient_accumulation_steps
            )
            console.print(f"[dim]Dataloader length: {dataloader_len} batches[/dim]")
        except TypeError:
            # Streaming: use precomputed virtual steps_per_epoch for progress display
            dataloader_len = None
            steps_per_epoch = self.steps_per_epoch
            console.print("[dim]Dataloader: streaming (no fixed length)[/dim]")
        console.print(
            f"[dim]Gradient accumulation steps: {self.config.gradient_accumulation_steps}[/dim]"
        )

        # If num_epochs was specified in config, use that; otherwise calculate from total_steps
        if self.config.max_steps:
            num_epochs = (
                (self.total_steps + steps_per_epoch - 1) // steps_per_epoch
            )  # Ceiling division (works for streaming via virtual steps)
        else:
            num_epochs = self.config.num_epochs

        console.print(f"[cyan]Steps per epoch: {steps_per_epoch}[/cyan]")
        console.print(f"[cyan]Number of epochs: {num_epochs}[/cyan]")

        # Start from the correct epoch if resuming
        start_epoch = self.starting_epoch + 1 if hasattr(self, "starting_epoch") else 1
        if start_epoch > 1:
            console.print(f"[yellow]Resuming from epoch {start_epoch}[/yellow]")

        for epoch in range(start_epoch, num_epochs + 1):
            if should_stop or self.global_step >= self.total_steps:
                break

            # Initialize epoch metrics
            # When resuming mid-epoch, start with last known loss to avoid 0.0000 display
            if (
                epoch == start_epoch
                and self.global_step > 0
                and len(self.training_history) > 0
            ):
                # Get the last logged loss
                last_entry = self.training_history[-1]
                epoch_loss = last_entry.get("loss", 0.0)
                epoch_steps = 1  # Avoid division by zero
            else:
                epoch_loss = 0.0
                epoch_steps = 0
            # Component losses (running sums for averaged display)
            epoch_mlm_loss = 0.0
            epoch_view_ctr_loss = 0.0
            epoch_samefile_ctr_loss = 0.0

            # Calculate how many steps to show for this epoch
            remaining_steps = self.total_steps - self.global_step
            steps_this_epoch = min(steps_per_epoch, remaining_steps)

            with Progress(
                TextColumn("[bold blue]Epoch {task.fields[epoch]}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("• Loss: {task.fields[loss]:.4f}"),
                TextColumn("• MLM: {task.fields[mlm]:.4f}"),
                TextColumn("• VCtr: {task.fields[vctr]:.4f}"),
                TextColumn("• SFCtr: {task.fields[sfctr]:.4f}"),
                TextColumn("• LR: {task.fields[lr]:.2e}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                disable=not self.accelerator.is_local_main_process,
            ) as progress:
                # Get current learning rate (especially important when resuming)
                current_lr = (
                    self.scheduler.get_last_lr()[0]
                    if hasattr(self.scheduler, "get_last_lr")
                    else self.config.learning_rate
                )

                # Progress bar tracks STEPS, not batches
                # Initialize with current loss if resuming
                initial_loss = (
                    float(epoch_loss / epoch_steps) if epoch_steps > 0 else 0.0
                )
                task = progress.add_task(
                    "Training",
                    total=steps_this_epoch,
                    epoch=epoch,
                    loss=initial_loss,
                    mlm=0.0,
                    vctr=0.0,
                    sfctr=0.0,
                    lr=current_lr,
                )

                # Calculate batches to skip if resuming mid-epoch
                batches_to_skip = 0
                steps_to_skip = 0
                if epoch == start_epoch and self.global_step > 0:
                    # Check if we're actually mid-epoch (not at an epoch boundary)
                    steps_in_epoch = self.global_step % steps_per_epoch
                    if steps_in_epoch > 0:
                        # We're resuming mid-epoch, need to skip already processed batches
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
                        # Update progress bar to show we're starting from the middle
                        progress.update(task, completed=steps_to_skip)

                for batch_idx, batch in enumerate(self.train_dataloader):
                    # Skip batches if resuming mid-epoch
                    if batch_idx < batches_to_skip:
                        continue

                    if self.global_step >= self.total_steps:
                        break

                    with self.accelerator.accumulate(self.model):
                        # Defaults for component logging (torch scalars)
                        log_mlm_t = torch.tensor(0.0, device=self.accelerator.device)
                        log_view_t = torch.tensor(0.0, device=self.accelerator.device)
                        log_sf_t = torch.tensor(0.0, device=self.accelerator.device)
                        # Determine if batch contains contrastive pairs
                        is_pair_batch = (
                            self.config.contrastive_enabled
                            and isinstance(batch, dict)
                            and "pair_types" in batch
                            and batch["input_ids"].dim() == 3
                        )

                        if is_pair_batch:
                            # Flatten [B, 2, L] -> [2B, L]
                            bsz, views, seqlen = batch["input_ids"].shape
                            input_ids = batch["input_ids"].view(bsz * views, seqlen)
                            attention_mask = batch["attention_mask"].view(
                                bsz * views, seqlen
                            )
                            labels = batch["labels"].view(bsz * views, seqlen)

                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                output_hidden_states=True,
                                return_dict=True,
                            )

                            mlm_loss = outputs.loss

                            # Embeddings via mean pooling
                            last_hidden = outputs.hidden_states[-1]
                            emb = pool_embeddings(
                                last_hidden,
                                attention_mask,
                                method=self.config.pooling,
                            )
                            emb = emb.view(bsz, views, -1)

                            # Contrastive subsets
                            pair_types = batch["pair_types"].to(emb.device)
                            dup_a, dup_b, sf_a, sf_b = split_by_pair_type(
                                emb, pair_types
                            )

                            view_loss = (
                                info_nce_pairwise(
                                    dup_a, dup_b, self.config.contrastive_temperature
                                )
                                if dup_a.size(0) > 0
                                else mlm_loss.new_tensor(0.0)
                            )
                            same_file_loss = (
                                info_nce_pairwise(
                                    sf_a, sf_b, self.config.contrastive_temperature
                                )
                                if sf_a.size(0) > 0
                                else mlm_loss.new_tensor(0.0)
                            )

                            # Linear ramp for contrastive weights
                            if (
                                self.config.contrastive_ramp_steps
                                and self.config.contrastive_ramp_steps > 0
                            ):
                                ramp = min(
                                    1.0,
                                    self.global_step
                                    / float(self.config.contrastive_ramp_steps),
                                )
                            else:
                                ramp = 1.0

                            mlm_w = self.config.mlm_weight
                            view_w = self.config.view_contrastive_weight * ramp
                            sf_w = self.config.same_file_contrastive_weight * ramp

                            total_loss = (
                                mlm_w * mlm_loss
                                + view_w * view_loss
                                + sf_w * same_file_loss
                            )
                            loss = total_loss
                            # Record components for display
                            log_mlm_t = mlm_loss.detach().float()
                            log_view_t = view_loss.detach().float()
                            log_sf_t = same_file_loss.detach().float()
                        else:
                            # Standard MLM batch
                            outputs = self.model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"],
                            )
                            loss = outputs.loss
                            log_mlm_t = loss.detach().float()

                        # Backward pass
                        self.accelerator.backward(loss)

                        # Only do optimizer step when gradients are synchronized
                        if self.accelerator.sync_gradients:
                            # Gradient clipping
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm,
                            )

                            # Optimizer step
                            self.optimizer.step()
                            self.scheduler.step()
                            # Use set_to_none to reduce VRAM and improve perf
                            self.optimizer.zero_grad(set_to_none=True)
                        else:
                            grad_norm = 0.0

                    # Only update metrics and progress on actual optimizer steps
                    # (not on every batch during gradient accumulation)
                    if self.accelerator.sync_gradients:
                        # This is an actual optimizer step
                        epoch_loss += loss.detach().float()
                        epoch_steps += 1
                        self.global_step += 1

                        # Update progress bar (once per step, not per batch)
                        current_loss = epoch_loss / epoch_steps
                        # Update component running averages
                        epoch_mlm_loss += log_mlm_t
                        epoch_view_ctr_loss += log_view_t
                        epoch_samefile_ctr_loss += log_sf_t
                        current_mlm = epoch_mlm_loss / epoch_steps
                        current_vctr = epoch_view_ctr_loss / epoch_steps
                        current_sfctr = epoch_samefile_ctr_loss / epoch_steps
                        current_lr = self.scheduler.get_last_lr()[0]
                        progress.update(
                            task,
                            advance=1,
                            loss=float(current_loss),
                            mlm=float(current_mlm),
                            vctr=float(current_vctr),
                            sfctr=float(current_sfctr),
                            lr=current_lr,
                        )

                        # Log metrics (only after we have computed current_loss)
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

                            # Log to WandB
                            if self.wandb_run and self.accelerator.is_main_process:
                                wandb_log = {
                                    "train/loss": float(current_loss),
                                    "train/learning_rate": float(current_lr),
                                    "train/epoch": epoch,
                                    "train/global_step": self.global_step,
                                }
                                if self.accelerator.sync_gradients:
                                    wandb_log["train/gradient_norm"] = float(grad_norm)

                                # Add per-loss components if available
                                # Defaults are zero when running pure-MLM batches
                                try:
                                    mlm_step = float(log_mlm_t)
                                except Exception:
                                    mlm_step = 0.0
                                try:
                                    view_step = float(log_view_t)
                                except Exception:
                                    view_step = 0.0
                                try:
                                    sf_step = float(log_sf_t)
                                except Exception:
                                    sf_step = 0.0

                                wandb_log.update(
                                    {
                                        "train/mlm_loss": mlm_step,
                                        "train/view_contrastive_loss": view_step,
                                        "train/same_file_contrastive_loss": sf_step,
                                    }
                                )

                                # Effective weights (with ramp applied) for transparency
                                if (
                                    self.config.contrastive_ramp_steps
                                    and self.config.contrastive_ramp_steps > 0
                                ):
                                    ramp = min(
                                        1.0,
                                        self.global_step
                                        / float(self.config.contrastive_ramp_steps),
                                    )
                                else:
                                    ramp = 1.0

                                eff_view_w = self.config.view_contrastive_weight * ramp
                                eff_sf_w = (
                                    self.config.same_file_contrastive_weight * ramp
                                )

                                wandb_log.update(
                                    {
                                        "train/mlm_weight": float(
                                            self.config.mlm_weight
                                        ),
                                        "train/view_contrastive_weight": float(
                                            eff_view_w
                                        ),
                                        "train/same_file_contrastive_weight": float(
                                            eff_sf_w
                                        ),
                                    }
                                )

                                wandb.log(wandb_log, step=self.global_step)

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

                            # Log validation loss to WandB
                            if self.wandb_run and self.accelerator.is_main_process:
                                wandb.log(
                                    {"eval/loss": val_loss}, step=self.global_step
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

                                should_stop = self.early_stopping(
                                    metrics_dict, self.model
                                )
                                if should_stop:
                                    console.print(
                                        "[red]Early stopping triggered![/red]"
                                    )
                                    break

                        # Run periodic assessment
                        if (
                            self.config.run_assessment
                            and self.global_step % self.config.assessment_steps == 0
                            and self.global_step > 0
                        ):
                            assessment_results = self.run_assessment()

                            # Log assessment results to WandB
                            if self.wandb_run and self.accelerator.is_main_process:
                                wandb.log(assessment_results, step=self.global_step)

                        # Save checkpoint
                        if (
                            not self.config.save_best_only
                            and self.global_step % self.config.save_steps == 0
                            and self.global_step > 0
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

        # Run final assessment
        if self.config.run_assessment:
            console.print("[yellow]Running final assessment...[/yellow]")
            final_assessment = self.run_assessment()

            # Save assessment results
            assessment_path = self.output_dir / "final_assessment.json"
            with open(assessment_path, "w") as f:
                json.dump(final_assessment, f, indent=2)
            console.print(f"[dim]Saved final assessment to {assessment_path}[/dim]")

            # Log to WandB
            if self.wandb_run and self.accelerator.is_main_process:
                # Add "final/" prefix to distinguish from periodic assessments
                final_metrics = {f"final/{k}": v for k, v in final_assessment.items()}
                wandb.log(final_metrics, step=self.global_step)

        # Finish WandB run
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
            steps_per_epoch = (
                len(self.train_dataloader) // self.config.gradient_accumulation_steps
            )
            self.starting_epoch = self.global_step // steps_per_epoch

            console.print(
                f"[green]✓ Resumed from checkpoint at step {self.global_step}[/green]"
            )
            console.print(f"[dim]  Starting from epoch {self.starting_epoch + 1}[/dim]")
            console.print(
                f"[dim]  Best validation loss: {self.best_val_loss:.4f}[/dim]"
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
        else:
            console.print(
                f"[yellow]⚠ No training state found at {checkpoint_path}, starting fresh[/yellow]"
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

        # Create assessment instance
        assessment = BinaryAssessment(
            model=self.accelerator.unwrap_model(self.model),
            tokenizer=self.tokenizer,
            device=self.accelerator.device,
        )

        # Run all assessments
        results = {}

        try:
            # File header recognition
            header_result = assessment.assess_file_header_recognition()
            results["assessment/file_header_score"] = header_result.score
            results["assessment/file_header_passed"] = float(header_result.passed)

            # Binary pattern learning
            pattern_result = assessment.assess_binary_pattern_learning()
            results["assessment/pattern_learning_score"] = pattern_result.score
            results["assessment/pattern_learning_passed"] = float(pattern_result.passed)

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
