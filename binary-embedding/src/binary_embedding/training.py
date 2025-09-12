"""Training module for binary embedding models."""

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

console = Console()


@dataclass
class TrainerConfig:
    """Configuration for training."""

    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    num_epochs: int = 3
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    output_dir: str = "./output"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int | None = 3


class Trainer:
    """Trainer for binary embedding models."""

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

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        total_steps = (
            len(train_dataloader)
            * config.num_epochs
            // config.gradient_accumulation_steps
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
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

    def train(self) -> None:
        """Run the training loop."""
        self.model.train()
        console.print(
            f"[bold green]Starting training on {self.accelerator.device}[/bold green]"
        )

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            with Progress(
                TextColumn(
                    "[bold blue]Epoch {task.fields[epoch]}/{task.fields[total_epochs]}"
                ),
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
                    epoch=epoch + 1,
                    total_epochs=self.config.num_epochs,
                    loss=0.0,
                    lr=self.config.learning_rate,
                )

                for _, batch in enumerate(self.train_dataloader):
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
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm,
                            )

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

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        self.training_history.append(
                            {
                                "step": self.global_step,
                                "epoch": epoch,
                                "loss": float(current_loss),
                                "learning_rate": float(current_lr),
                            }
                        )

                    # Evaluation
                    if (
                        self.val_dataloader
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        val_loss = self.evaluate()
                        self.model.train()

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(is_best=True)

                        console.print(
                            f"[yellow]Step {self.global_step} - "
                            f"Val Loss: {val_loss:.4f}[/yellow]"
                        )

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps
            console.print(
                f"[green]Epoch {epoch + 1} completed - "
                f"Average Loss: {avg_epoch_loss:.4f}[/green]"
            )

        # Save final model
        self.save_checkpoint(is_final=True)
        self.save_training_history()
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

    def save_training_history(self) -> None:
        """Save training history to JSON."""
        if not self.accelerator.is_local_main_process:
            return

        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        console.print(f"[dim]Saved training history to {history_path}[/dim]")

