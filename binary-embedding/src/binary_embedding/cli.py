"""Command-line interface for binary embedding training."""

from __future__ import annotations

from pathlib import Path

import click
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from binary_embedding.assessment import load_and_assess_model
from binary_embedding.data import create_dataloader
from binary_embedding.models import ModelSize, create_model
from binary_embedding.pair_data import (
    create_pair_dataloader,
    create_streaming_pair_dataloader,
)
from binary_embedding.tokenizer import load_tokenizer
from binary_embedding.training import Trainer, TrainerConfig

console = Console()


@click.group()
def cli() -> None:
    """Binary Embedding Models - Train BERT/RoBERTa models on binary executables."""
    pass


@cli.command()
@click.option(
    "--model-size",
    type=click.Choice(["small", "base", "large"]),
    default="small",
    help="Model size to train",
)
@click.option(
    "--model-type",
    type=click.Choice(["bert", "roberta"]),
    default="roberta",
    help="Model architecture type",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default="/usr/bin",
    help="Directory containing binary files",
)
@click.option(
    "--tokenizer-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to tokenizer.json file (uses default if not provided)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="./output",
    help="Output directory for checkpoints",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Per-device batch size (auto-selected based on model size if not specified)",
)
@click.option(
    "--num-epochs",
    type=int,
    default=3,
    help="Number of training epochs",
)
@click.option(
    "--max-steps",
    type=int,
    default=None,
    help="Maximum training steps (overrides num-epochs if set)",
)
@click.option(
    "--learning-rate",
    type=float,
    default=None,
    help="Learning rate (auto-selected based on model size if not specified)",
)
@click.option(
    "--scheduler-type",
    type=click.Choice(
        ["linear", "cosine", "cosine_with_restarts", "two_phase", "polynomial"]
    ),
    default=None,
    help="Learning rate scheduler type (auto-selected based on model size if not specified)",
)
@click.option(
    "--warmup-steps",
    type=int,
    default=None,
    help="Number of warmup steps (uses warmup-ratio if not specified)",
)
@click.option(
    "--warmup-ratio",
    type=float,
    default=0.1,
    help="Warmup ratio (fraction of total steps)",
)
@click.option(
    "--max-length",
    type=int,
    default=512,
    help="Maximum sequence length",
)
@click.option(
    "--mlm-probability",
    type=float,
    default=0.20,
    help="Masked language modeling probability",
)
@click.option(
    "--max-files",
    type=int,
    default=None,
    help="Maximum number of files to load",
)
@click.option(
    "--chunk-size",
    type=int,
    default=4096,
    help="Size of chunks to read from binary files",
)
@click.option(
    "--num-workers",
    type=int,
    default=4,
    help="Number of data loading workers",
)
@click.option(
    "--mixed-precision/--no-mixed-precision",
    default=True,
    help="Use mixed precision training",
)
@click.option(
    "--gradient-checkpointing",
    is_flag=True,
    default=False,
    help="Enable gradient checkpointing (reduces VRAM at some speed cost)",
)
@click.option(
    "--gradient-accumulation-steps",
    type=int,
    default=None,
    help="Gradient accumulation steps (auto-selected based on model size if not specified)",
)
@click.option(
    "--early-stopping/--no-early-stopping",
    default=False,
    help="Use early stopping",
)
@click.option(
    "--monitor-embedding/--no-monitor-embedding",
    default=False,
    help="Monitor embedding quality during training",
)
@click.option(
    "--save-steps",
    type=int,
    default=1000,
    help="Save checkpoint every N steps",
)
@click.option(
    "--save-total-limit",
    type=int,
    default=None,
    help="Maximum number of checkpoints to keep",
)
@click.option(
    "--save-best-only/--save-all",
    default=False,
    help="Only save best checkpoint (requires validation)",
)
@click.option(
    "--eval-steps",
    type=int,
    default=500,
    help="Evaluate every N steps",
)
@click.option(
    "--logging-steps",
    type=int,
    default=10,
    help="Log metrics every N steps",
)
@click.option(
    "--resume-from-checkpoint",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Resume training from a checkpoint directory",
)
@click.option(
    "--run-assessment/--no-assessment",
    default=False,
    help="Run periodic assessments during training",
)
@click.option(
    "--assessment-steps",
    type=int,
    default=10000,
    help="Run assessment every N steps",
)
@click.option(
    "--use-wandb/--no-wandb",
    default=True,
    help="Use Weights & Biases for logging",
)
@click.option(
    "--wandb-project",
    type=str,
    default="binary-embedding",
    help="WandB project name",
)
@click.option(
    "--wandb-run-name",
    type=str,
    default=None,
    help="WandB run name (auto-generated if not specified)",
)
@click.option(
    "--wandb-tags",
    type=str,
    multiple=True,
    default=None,
    help="WandB tags (can be specified multiple times)",
)
@click.option(
    "--wandb-notes",
    type=str,
    default=None,
    help="WandB run notes",
)
@click.option(
    "--streaming/--no-streaming",
    default=False,
    help="Use streaming dataset (background IO); recommended for network storage",
)
@click.option(
    "--contrastive/--no-contrastive",
    default=False,
    help="Enable contrastive training with pair sampling",
)
@click.option(
    "--pooling",
    type=click.Choice(["mean", "cls"]),
    default="mean",
    help="Embedding pooling method for contrastive loss",
)
@click.option(
    "--dup-prob",
    type=float,
    default=0.5,
    help="Probability of duplicate-view pairs within a file",
)
@click.option(
    "--pair-cache-size",
    type=int,
    default=4096,
    help="LRU cache size (number of chunks) for pair dataset",
)
@click.option(
    "--prefetch-factor",
    type=int,
    default=4,
    help="DataLoader prefetch_factor (per worker) for pair dataset",
)
@click.option(
    "--min-chunk-separation",
    type=int,
    default=None,
    help="Minimum byte separation between chunks for same-file pairs (default: chunk-size)",
)
@click.option(
    "--contrastive-temp",
    type=float,
    default=0.07,
    help="Contrastive temperature",
)
@click.option(
    "--mlm-weight",
    type=float,
    default=1.0,
    help="Weight for MLM loss",
)
@click.option(
    "--view-weight",
    type=float,
    default=0.5,
    help="Weight for duplicate-view contrastive loss",
)
@click.option(
    "--samefile-weight",
    type=float,
    default=0.5,
    help="Weight for same-file contrastive loss",
)
@click.option(
    "--contrastive-ramp-steps",
    type=int,
    default=0,
    help="Linear ramp-in steps for contrastive loss weights",
)
def train(
    model_size: str,
    model_type: str,
    data_dir: Path,
    tokenizer_path: Path | None,
    output_dir: Path,
    batch_size: int | None,
    num_epochs: int,
    max_steps: int | None,
    learning_rate: float | None,
    scheduler_type: str | None,
    warmup_steps: int | None,
    warmup_ratio: float,
    max_length: int,
    mlm_probability: float,
    max_files: int | None,
    chunk_size: int,
    num_workers: int,
    mixed_precision: bool,
    gradient_checkpointing: bool,
    gradient_accumulation_steps: int | None,
    early_stopping: bool,
    monitor_embedding: bool,
    save_steps: int,
    save_total_limit: int | None,
    save_best_only: bool,
    eval_steps: int,
    logging_steps: int,
    resume_from_checkpoint: Path | None,
    run_assessment: bool,
    assessment_steps: int,
    use_wandb: bool,
    wandb_project: str,
    wandb_run_name: str | None,
    wandb_tags: tuple[str, ...] | None,
    wandb_notes: str | None,
    streaming: bool,
    contrastive: bool,
    pooling: str,
    dup_prob: float,
    pair_cache_size: int,
    prefetch_factor: int,
    min_chunk_separation: int | None,
    contrastive_temp: float,
    mlm_weight: float,
    view_weight: float,
    samefile_weight: float,
    contrastive_ramp_steps: int,
) -> None:
    """Train a binary embedding model with optional advanced features."""
    # Display configuration
    console.print(
        Panel.fit(
            "[bold cyan]Binary Embedding Model Training[/bold cyan]\n"
            "[dim]Using optimized hyperparameters for best results[/dim]",
            border_style="cyan",
        )
    )

    # Create config based on model size with user overrides
    base_config = TrainerConfig.from_model_size(
        model_size,
        output_dir=str(output_dir),
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        mixed_precision=mixed_precision,
        use_early_stopping=early_stopping,
        monitor_embedding_quality=monitor_embedding,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_best_only=save_best_only,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        run_assessment=run_assessment,
        assessment_steps=assessment_steps,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_tags=list(wandb_tags) if wandb_tags else None,
        wandb_notes=wandb_notes,
    )

    # Apply user overrides
    if batch_size is not None:
        base_config.batch_size = batch_size
    if learning_rate is not None:
        base_config.learning_rate = learning_rate
    if max_steps is not None:
        base_config.max_steps = max_steps
    if gradient_accumulation_steps is not None:
        base_config.gradient_accumulation_steps = gradient_accumulation_steps
    if scheduler_type is not None:
        base_config.scheduler_type = scheduler_type
    if warmup_steps is not None:
        base_config.warmup_steps = warmup_steps
    if gradient_checkpointing:
        base_config.gradient_checkpointing = True

    # Display configuration table
    config_table = Table(title="Training Configuration", show_header=False)
    config_table.add_column("Parameter", style="dim")
    config_table.add_column("Value", style="green")

    config_table.add_row("Model Size", model_size)
    config_table.add_row("Model Type", model_type)
    config_table.add_row("Data Directory", str(data_dir))
    config_table.add_row("Batch Size", str(base_config.batch_size))
    config_table.add_row(
        "Gradient Accumulation", str(base_config.gradient_accumulation_steps)
    )
    config_table.add_row(
        "Effective Batch Size",
        str(base_config.batch_size * base_config.gradient_accumulation_steps),
    )
    config_table.add_row("Learning Rate", f"{base_config.learning_rate:.2e}")
    config_table.add_row("Scheduler", base_config.scheduler_type)
    config_table.add_row(
        "Warmup",
        f"{warmup_ratio:.1%}" if warmup_steps is None else f"{warmup_steps} steps",
    )
    config_table.add_row("Epochs", str(num_epochs))
    if max_steps:
        config_table.add_row("Max Steps", str(max_steps))
    config_table.add_row("Max Length", str(max_length))
    config_table.add_row("MLM Probability", f"{mlm_probability:.2%}")
    config_table.add_row("Mixed Precision", "✓" if mixed_precision else "✗")
    if gradient_checkpointing:
        config_table.add_row("Gradient Checkpointing", "✓")
    if early_stopping:
        config_table.add_row("Early Stopping", "✓")
    if monitor_embedding:
        config_table.add_row("Embedding Monitoring", "✓")
    if run_assessment:
        config_table.add_row("Periodic Assessment", f"Every {assessment_steps:,} steps")
    if use_wandb:
        config_table.add_row("WandB Logging", f"Project: {wandb_project}")
    config_table.add_row("Output Directory", str(output_dir))
    if contrastive:
        config_table.add_row("Contrastive", "✓")
        config_table.add_row("Pooling", pooling)
        config_table.add_row("Dup Pair Prob", f"{dup_prob:.2f}")
        config_table.add_row("Pair Cache Size", str(pair_cache_size))
        config_table.add_row("Prefetch Factor", str(prefetch_factor))
        if min_chunk_separation is not None:
            config_table.add_row("Min Chunk Sep", str(min_chunk_separation))
        config_table.add_row("Ctr Temp", f"{contrastive_temp:.2f}")
        config_table.add_row(
            "Loss Weights",
            f"MLM={mlm_weight}, View={view_weight}, SameFile={samefile_weight}",
        )

    console.print(config_table)
    console.print()

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[yellow]Using device: {device}[/yellow]\n")

    # Load tokenizer
    with console.status("[bold green]Loading tokenizer..."):
        tokenizer = load_tokenizer(tokenizer_path)
        console.print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")

    # Create or load model
    if resume_from_checkpoint:
        with console.status(
            f"[bold green]Loading model from {resume_from_checkpoint}..."
        ):
            from transformers import AutoConfig, AutoModelForMaskedLM

            # Load model config
            config = AutoConfig.from_pretrained(resume_from_checkpoint)
            model = AutoModelForMaskedLM.from_pretrained(resume_from_checkpoint)

            # Create model_config for compatibility
            from binary_embedding.models import BinaryEmbeddingConfig

            model_config = BinaryEmbeddingConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                max_position_embeddings=config.max_position_embeddings,
                model_type=model_type,
                learning_rate=base_config.learning_rate,
                warmup_steps=base_config.warmup_steps or int(warmup_ratio * 10000),
                mlm_probability=mlm_probability,
            )

            total_params = sum(p.numel() for p in model.parameters())
            console.print(
                f"✓ Model loaded from checkpoint ({total_params:,} parameters)"
            )
    else:
        with console.status(f"[bold green]Creating {model_size} model..."):
            size_enum = ModelSize(model_size)
            model, model_config = create_model(size=size_enum)
            model_config.model_type = model_type
            model_config.learning_rate = base_config.learning_rate
            model_config.warmup_steps = base_config.warmup_steps or int(
                warmup_ratio * 10000
            )
            model_config.mlm_probability = mlm_probability

            total_params = sum(p.numel() for p in model.parameters())
            console.print(f"✓ Model created ({total_params:,} parameters)")

    # Create data loader
    with console.status(f"[bold green]Loading data from {data_dir}..."):
        if contrastive:
            if streaming:
                train_dataloader = create_streaming_pair_dataloader(
                    directory_path=data_dir,
                    tokenizer=tokenizer,
                    batch_size=base_config.batch_size,
                    max_length=max_length,
                    chunk_size=chunk_size,
                    mlm_probability=mlm_probability,
                    duplicate_prob=dup_prob,
                    min_chunk_separation=min_chunk_separation
                    if min_chunk_separation is not None
                    else chunk_size,
                    buffer_size=pair_cache_size,
                    per_file_buffer=32,
                    num_workers=num_workers,
                    shuffle=True,
                    cycle=True,
                )
            else:
                train_dataloader = create_pair_dataloader(
                    directory_path=data_dir,
                    tokenizer=tokenizer,
                    batch_size=base_config.batch_size,
                    max_length=max_length,
                    chunk_size=chunk_size,
                    mlm_probability=mlm_probability,
                    duplicate_prob=dup_prob,
                    min_chunk_separation=min_chunk_separation
                    if min_chunk_separation is not None
                    else chunk_size,
                    num_workers=num_workers,
                    max_files=max_files,
                    cache_size=pair_cache_size,
                    prefetch_factor=prefetch_factor,
                    tokenize_in_dataset=False,
                )
        else:
            train_dataloader = create_dataloader(
                directory_path=data_dir,
                tokenizer=tokenizer,
                batch_size=base_config.batch_size,
                max_length=max_length,
                chunk_size=chunk_size,
                mlm_probability=mlm_probability,
                num_workers=num_workers,
                max_files=max_files,
                streaming=streaming,
            )
        # len() not defined for streaming IterableDatasets
        try:
            num_batches = len(train_dataloader)
            console.print(f"✓ Data loaded ({num_batches} batches)")
        except TypeError:
            console.print("✓ Data loaded (streaming)")

    # Create trainer
    # Configure contrastive options on the config
    if contrastive:
        base_config.contrastive_enabled = True
        base_config.pooling = pooling
        base_config.contrastive_temperature = contrastive_temp
        base_config.mlm_weight = mlm_weight
        base_config.view_contrastive_weight = view_weight
        base_config.same_file_contrastive_weight = samefile_weight
        base_config.contrastive_ramp_steps = contrastive_ramp_steps
        base_config.duplicate_pair_probability = dup_prob
        base_config.min_chunk_separation = (
            min_chunk_separation if min_chunk_separation is not None else chunk_size
        )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        config=base_config,
        model_config=model_config,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    # Train
    console.print("\n[bold green]Starting training...[/bold green]\n")
    trainer.train()

    console.print(
        f"\n[bold green]✓ Training complete! Model saved to {output_dir}[/bold green]"
    )


@cli.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint",
)
@click.option(
    "--tokenizer-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to tokenizer.json file",
)
@click.option(
    "--text",
    type=str,
    default="48 65 6c 6c 6f",
    help="Hex string to test",
)
def test(
    checkpoint_path: Path,
    tokenizer_path: Path | None,
    text: str,
) -> None:
    """Test a trained model on a hex string."""
    import torch
    from transformers import AutoModelForMaskedLM

    console.print(
        Panel.fit(
            "[bold magenta]Model Testing[/bold magenta]",
            border_style="magenta",
        )
    )

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Load model
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
    model.eval()

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Show results
    console.print(f"\n[cyan]Input text:[/cyan] {text}")
    console.print(f"[cyan]Token IDs:[/cyan] {inputs['input_ids'][0].tolist()}")
    console.print(f"[cyan]Logits shape:[/cyan] {logits.shape}")


@cli.command()
@click.option(
    "--checkpoint-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model checkpoint",
)
@click.option(
    "--tokenizer-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to tokenizer.json file",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Path to save assessment results JSON",
)
def assess(
    checkpoint_path: Path,
    tokenizer_path: Path | None,
    output: Path | None,
) -> None:
    """Run comprehensive assessment on a trained model."""
    console.print(
        Panel.fit(
            "[bold cyan]Binary Embedding Model Assessment[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print(f"Checkpoint: {checkpoint_path}")
    if output:
        console.print(f"Output: {output}")

    console.print("\nRunning assessment tests...")

    # Run assessment
    suite = load_and_assess_model(checkpoint_path, tokenizer_path, output)

    # Display results
    results_table = Table(title="Assessment Results")
    results_table.add_column("Test", style="cyan")
    results_table.add_column("Status", justify="center")
    results_table.add_column("Score", justify="right")

    for result in suite.results:
        status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"
        score = f"{result.score:.1%}"
        results_table.add_row(result.test_name, status, score)

    console.print()
    console.print(results_table)

    # Summary
    summary = suite.get_summary()
    console.print()
    console.print(
        Panel(
            f"[bold]Overall Pass Rate: {summary['pass_rate']:.1%}[/bold]\n"
            f"Average Score: {summary['average_score']:.1%}",
            title="Summary",
            border_style="green" if summary["pass_rate"] > 0.5 else "yellow",
        )
    )


@cli.command()
def info() -> None:
    """Display information about the package."""
    info_panel = Panel(
        "[bold cyan]Binary Embedding Models[/bold cyan]\n\n"
        "Train BERT/RoBERTa-style embedding models on binary executable files.\n\n"
        "[dim]Features:[/dim]\n"
        "• Custom BPE tokenizer (65536 vocab)\n"
        "• RoBERTa-style models with latest optimizations\n"
        "• Multiple model sizes (small/base/large)\n"
        "• Mixed precision training\n"
        "• Advanced learning rate scheduling\n"
        "• Early stopping and embedding quality monitoring\n\n"
        "[dim]Usage:[/dim]\n"
        "  binary-embedding train --help\n"
        "  binary-embedding assess --help",
        border_style="cyan",
    )
    console.print(info_panel)


if __name__ == "__main__":
    cli()
