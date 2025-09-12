"""Advanced learning rate schedulers for binary embedding training.

This module provides sophisticated scheduling strategies to preserve
embedding quality while improving task performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    SequentialLR,
    _LRScheduler,
)
from transformers import get_linear_schedule_with_warmup


class SchedulerType(Enum):
    """Available scheduler types."""

    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    TWO_PHASE = "two_phase"
    POLYNOMIAL = "polynomial"


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduling."""

    scheduler_type: SchedulerType = SchedulerType.LINEAR
    warmup_steps: int | None = None  # If None, use warmup_ratio
    warmup_ratio: float = 0.1  # Use 10% of total steps for warmup

    # For cosine schedulers
    num_cycles: float = 0.5  # For cosine scheduler

    # For cosine with restarts
    restart_interval: int = 10000  # T_0 for CosineAnnealingWarmRestarts
    restart_mult: int = 2  # T_mult for CosineAnnealingWarmRestarts

    # For two-phase training
    phase1_steps: int = 50000
    phase1_lr_factor: float = 1.0  # Multiplier for base LR in phase 1
    phase2_lr_factor: float = 0.2  # Multiplier for base LR in phase 2

    # For polynomial decay
    power: float = 1.0  # 1.0 = linear, >1.0 = slower decay

    # General settings
    min_lr_ratio: float = 0.01  # Minimum LR as ratio of initial LR


class WarmupScheduler(_LRScheduler):
    """Base class for schedulers with warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_scheduler: _LRScheduler,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler after warmup
            return self.base_scheduler.get_lr()

    def step(self, epoch=None):
        super().step(epoch)
        if self.last_epoch >= self.warmup_steps:
            self.base_scheduler.step()


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    num_training_steps: int,
    num_epochs: int | None = None,
) -> _LRScheduler:
    """Create a learning rate scheduler based on configuration.

    Args:
        optimizer: The optimizer to schedule.
        config: Scheduler configuration.
        num_training_steps: Total number of training steps.
        num_epochs: Number of epochs (optional, for epoch-based schedulers).

    Returns:
        Configured learning rate scheduler.
    """
    # Calculate warmup steps
    if config.warmup_steps is not None:
        warmup_steps = config.warmup_steps
    else:
        warmup_steps = int(num_training_steps * config.warmup_ratio)

    # Get minimum learning rate
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr = base_lr * config.min_lr_ratio

    if config.scheduler_type == SchedulerType.LINEAR:
        # Standard linear schedule with warmup (like original BERT)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

    elif config.scheduler_type == SchedulerType.COSINE:
        # Cosine annealing with warmup
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=min_lr,
        )

        if warmup_steps > 0:
            warmup_scheduler = LambdaLR(
                optimizer,
                lambda step: min(1.0, step / warmup_steps),
            )

            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )
        return main_scheduler

    elif config.scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
        # Cosine annealing with warm restarts
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.restart_interval,
            T_mult=config.restart_mult,
            eta_min=min_lr,
        )

        if warmup_steps > 0:
            warmup_scheduler = LambdaLR(
                optimizer,
                lambda step: min(1.0, step / warmup_steps),
            )

            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )
        return main_scheduler

    elif config.scheduler_type == SchedulerType.TWO_PHASE:
        # Two-phase training with different learning rates
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Warmup phase
                return (step / warmup_steps) * config.phase1_lr_factor
            elif step < config.phase1_steps:
                # Phase 1: Normal learning rate
                return config.phase1_lr_factor
            else:
                # Phase 2: Reduced learning rate
                return config.phase2_lr_factor

        return LambdaLR(optimizer, lr_lambda)

    elif config.scheduler_type == SchedulerType.POLYNOMIAL:
        # Polynomial decay (generalization of linear)
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps

            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return max(config.min_lr_ratio, (1 - progress) ** config.power)

        return LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")


class AdaptiveScheduler:
    """Adaptive scheduler that adjusts learning rate based on metrics."""

    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: _LRScheduler,
        patience: int = 5,
        factor: float = 0.5,
        threshold: float = 0.01,
        min_lr: float = 1e-7,
    ):
        """Initialize adaptive scheduler.

        Args:
            optimizer: The optimizer.
            base_scheduler: Base scheduler to use.
            patience: Number of steps to wait before reducing LR.
            factor: Factor to reduce LR by.
            threshold: Threshold for significant improvement.
            min_lr: Minimum learning rate.
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.patience = patience
        self.factor = factor
        self.threshold = threshold
        self.min_lr = min_lr

        self.best_metric = None
        self.patience_counter = 0

    def step(self, metric: float | None = None):
        """Step the scheduler, optionally with a metric."""
        # Always step the base scheduler
        self.base_scheduler.step()

        if metric is not None:
            if self.best_metric is None:
                self.best_metric = metric
            elif metric < self.best_metric * (1 - self.threshold):
                # Metric improved significantly
                self.best_metric = metric
                self.patience_counter = 0
            else:
                # No significant improvement
                self.patience_counter += 1

                if self.patience_counter >= self.patience:
                    # Reduce learning rate
                    for param_group in self.optimizer.param_groups:
                        old_lr = param_group["lr"]
                        new_lr = max(self.min_lr, old_lr * self.factor)
                        param_group["lr"] = new_lr

                    self.patience_counter = 0

    def get_last_lr(self):
        """Get the last learning rate."""
        return [group["lr"] for group in self.optimizer.param_groups]


def create_optimal_scheduler_for_model_size(
    optimizer: Optimizer,
    model_size: str,
    num_training_steps: int,
    batch_size: int = 8,
) -> tuple[_LRScheduler, SchedulerConfig]:
    """Create an optimal scheduler based on model size and training setup.

    Args:
        optimizer: The optimizer.
        model_size: "small", "base", or "large".
        num_training_steps: Total training steps.
        batch_size: Training batch size.

    Returns:
        Tuple of (scheduler, config).
    """
    if model_size == "small":
        # Small model: Conservative approach
        config = SchedulerConfig(
            scheduler_type=SchedulerType.COSINE,
            warmup_ratio=0.1,
            min_lr_ratio=0.01,
        )
    elif model_size == "base":
        # Base model: Two-phase training
        config = SchedulerConfig(
            scheduler_type=SchedulerType.TWO_PHASE,
            warmup_ratio=0.06,  # 6% warmup
            phase1_steps=int(num_training_steps * 0.6),
            phase1_lr_factor=1.0,
            phase2_lr_factor=0.3,
            min_lr_ratio=0.001,
        )
    else:  # large
        # Large model: Cosine with restarts
        config = SchedulerConfig(
            scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
            warmup_ratio=0.03,  # Shorter warmup for large models
            restart_interval=num_training_steps // 10,
            restart_mult=2,
            min_lr_ratio=0.001,
        )

    # Adjust for batch size (square root scaling)
    if batch_size > 32:
        scale_factor = (batch_size / 32) ** 0.5
        for param_group in optimizer.param_groups:
            param_group["lr"] *= scale_factor

    scheduler = create_scheduler(optimizer, config, num_training_steps)
    return scheduler, config
