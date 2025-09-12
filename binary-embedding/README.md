# Binary Embedding Models

Train BERT/RoBERTa-style embedding models on binary executable files using custom tokenizers.

## Installation

```bash
uv sync
```

## Quick Start

### Command Line Interface

```bash
# View help
uv run python -m binary_embedding.cli --help

# Train a small model for testing
uv run python -m binary_embedding.cli train \
  --model-size small \
  --data-dir /usr/bin \
  --max-files 10 \
  --num-epochs 1

# Train a base model with more data
uv run python -m binary_embedding.cli train \
  --model-size base \
  --data-dir /usr/bin \
  --max-files 100 \
  --num-epochs 3 \
  --batch-size 8 \
  --learning-rate 5e-5

# Test a trained model
uv run python -m binary_embedding.cli test \
  --checkpoint-path output/final_model \
  --text "48 65 6c 6c 6f"
```

### Python API

```python
from binary_embedding import (
    load_tokenizer,
    create_model,
    create_dataloader,
    ModelSize,
    Trainer,
    TrainerConfig,
)

# Load tokenizer
tokenizer = load_tokenizer()

# Create model
model, config = create_model(size=ModelSize.BASE)

# Create data loader
dataloader = create_dataloader(
    directory_path="/usr/bin",
    tokenizer=tokenizer,
    batch_size=8,
    max_files=100,
)

# Setup training
trainer_config = TrainerConfig(
    num_epochs=3,
    learning_rate=5e-5,
    output_dir="./output",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataloader=dataloader,
    config=trainer_config,
)

# Train
trainer.train()
```

## Features

- **Custom Binary Tokenizer**: Pre-trained on x86_64, ARM64, RISC binaries (65536 vocab)
- **Modern Architecture**: RoBERTa-style models with latest optimizations
- **Efficient Training**:
  - 20% masking for base models (40% for large)
  - Mixed precision training (FP16)
  - Gradient accumulation
  - AdamW optimizer with linear warmup
- **Rich CLI**: Beautiful progress bars and status tracking
- **Multiple Model Sizes**:
  - Small: 256 hidden, 6 layers (~22M params)
  - Base: 768 hidden, 12 layers (~125M params)
  - Large: 1024 hidden, 24 layers (~355M params)

## CLI Options

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-size` | small | Model size (small/base/large) |
| `--model-type` | roberta | Architecture (bert/roberta) |
| `--data-dir` | /usr/bin | Directory with binary files |
| `--output-dir` | ./output | Output directory |
| `--batch-size` | 8 | Training batch size |
| `--num-epochs` | 3 | Number of epochs |
| `--learning-rate` | 5e-5 | Learning rate |
| `--warmup-steps` | 1000 | Warmup steps |
| `--max-length` | 512 | Max sequence length |
| `--mlm-probability` | 0.20 | Masking probability |
| `--max-files` | None | Max files to load |
| `--mixed-precision` | True | Use FP16 training |

## Architecture Details

The implementation follows modern best practices:

1. **RoBERTa Improvements**:
   - No NSP task
   - Dynamic masking
   - Larger training batches
   - Position embeddings adjusted for padding offset

2. **Training Optimizations**:
   - Mixed precision (FP16) via Accelerate
   - Gradient clipping
   - Linear warmup scheduler
   - Proper weight decay

3. **Data Processing**:
   - Binary files → hex representation
   - Byte-level spacing for tokenization
   - Dynamic chunk loading
   - Efficient batch collation

## Using Trained Models

```python
from transformers import AutoModelForMaskedLM
from binary_embedding import load_tokenizer

# Load tokenizer
tokenizer = load_tokenizer()

# Load trained model
model = AutoModelForMaskedLM.from_pretrained("./output/final_model")

# Process binary data
hex_data = "48 65 6c 6c 6f"  # Example hex bytes
inputs = tokenizer(hex_data, return_tensors="pt")

# Get predictions
outputs = model(**inputs)
logits = outputs.logits
```

## Development

### Code Quality

```bash
# Lint with ruff
uvx ruff check src/ --fix

# Type check
uvx ty check src/
```

### Project Structure

```
src/binary_embedding/
├── __init__.py       # Package exports
├── tokenizer.py      # Binary tokenizer wrapper
├── data.py          # Dataset and data loading
├── models.py        # Model architectures
├── training.py      # Training loop with Accelerate
└── cli.py           # Click CLI with Rich UI
```

## License

MIT