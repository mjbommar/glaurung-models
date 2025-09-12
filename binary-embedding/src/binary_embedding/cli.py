"""Command-line interface for binary embedding training."""

from __future__ import annotations

from pathlib import Path

import click
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from binary_embedding.data import create_dataloader
from binary_embedding.models import ModelSize, create_model
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
    help="Path to tokenizer.json file",
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
    default=8,
    help="Training batch size",
)
@click.option(
    "--num-epochs",
    type=int,
    default=3,
    help="Number of training epochs",
)
@click.option(
    "--learning-rate",
    type=float,
    default=5e-5,
    help="Learning rate",
)
@click.option(
    "--warmup-steps",
    type=int,
    default=1000,
    help="Number of warmup steps",
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
    "--gradient-accumulation-steps",
    type=int,
    default=1,
    help="Number of gradient accumulation steps",
)
@click.option(
    "--save-steps",
    type=int,
    default=500,
    help="Save checkpoint every N steps",
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
def train(
    model_size: str,
    model_type: str,
    data_dir: Path,
    tokenizer_path: Path | None,
    output_dir: Path,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    warmup_steps: int,
    max_length: int,
    mlm_probability: float,
    max_files: int | None,
    chunk_size: int,
    num_workers: int,
    mixed_precision: bool,
    gradient_accumulation_steps: int,
    save_steps: int,
    eval_steps: int,
    logging_steps: int,
) -> None:
    """Train a binary embedding model."""
    # Display configuration
    console.print(
        Panel.fit(
            "[bold cyan]Binary Embedding Model Training[/bold cyan]",
            border_style="cyan",
        )
    )

    config_table = Table(title="Configuration", show_header=False)
    config_table.add_column("Parameter", style="dim")
    config_table.add_column("Value", style="green")

    config_table.add_row("Model Size", model_size)
    config_table.add_row("Model Type", model_type)
    config_table.add_row("Data Directory", str(data_dir))
    config_table.add_row("Batch Size", str(batch_size))
    config_table.add_row("Epochs", str(num_epochs))
    config_table.add_row("Learning Rate", f"{learning_rate:.2e}")
    config_table.add_row("Max Length", str(max_length))
    config_table.add_row("MLM Probability", f"{mlm_probability:.2%}")
    config_table.add_row("Mixed Precision", "✓" if mixed_precision else "✗")
    config_table.add_row("Output Directory", str(output_dir))

    console.print(config_table)
    console.print()

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[yellow]Using device: {device}[/yellow]\n")

    # Load tokenizer
    with console.status("[bold green]Loading tokenizer..."):
        tokenizer = load_tokenizer(tokenizer_path)
        console.print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")

    # Create model
    with console.status(f"[bold green]Creating {model_size} model..."):
        size_enum = ModelSize(model_size)
        model, model_config = create_model(size=size_enum)
        model_config.model_type = model_type
        model_config.learning_rate = learning_rate
        model_config.warmup_steps = warmup_steps
        model_config.mlm_probability = mlm_probability

        total_params = sum(p.numel() for p in model.parameters())
        console.print(f"✓ Model created ({total_params:,} parameters)")

    # Create data loader
    with console.status(f"[bold green]Loading data from {data_dir}..."):
        train_dataloader = create_dataloader(
            directory_path=data_dir,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            chunk_size=chunk_size,
            mlm_probability=mlm_probability,
            num_workers=num_workers,
            max_files=max_files,
        )
        console.print(f"✓ Data loaded ({len(train_dataloader)} batches)")

    # Setup training
    trainer_config = TrainerConfig(
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        output_dir=str(output_dir),
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        config=trainer_config,
        model_config=model_config,
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
    """Test a trained model."""
    console.print("[bold cyan]Testing Binary Embedding Model[/bold cyan]\n")

    # Load tokenizer
    with console.status("[bold green]Loading tokenizer..."):
        tokenizer = load_tokenizer(tokenizer_path)
        console.print("✓ Tokenizer loaded")

    # Load model
    with console.status("[bold green]Loading model..."):
        from transformers import AutoModelForMaskedLM

        model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
        model.eval()
        console.print(f"✓ Model loaded from {checkpoint_path}")

    # Test encoding
    console.print(f"\n[yellow]Input text:[/yellow] {text}")

    inputs = tokenizer(text, return_tensors="pt")
    token_ids = inputs['input_ids'][0].tolist()[:20]
    console.print(f"[yellow]Token IDs:[/yellow] {token_ids}...")

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    console.print(f"[yellow]Output shape:[/yellow] {logits.shape}")
    console.print("\n[green]Model is working correctly![/green]")


@cli.command()
def info() -> None:
    """Display information about the package."""
    info_panel = Panel(
        """[bold]Binary Embedding Models[/bold]

Train BERT/RoBERTa-style embedding models on binary executable files.

[yellow]Features:[/yellow]
• Custom binary tokenizer with 65536 vocabulary
• Modern training optimizations (mixed precision, gradient accumulation)
• Multiple model sizes (small, base, large)
• Rich progress tracking and logging

[yellow]Quick Start:[/yellow]
binary-embedding train --model-size small --data-dir /usr/bin --num-epochs 1

[yellow]Repository:[/yellow]
https://github.com/yourusername/binary-embedding
        """,
        title="Binary Embedding v0.1.0",
        border_style="blue",
    )
    console.print(info_panel)


if __name__ == "__main__":
    cli()

