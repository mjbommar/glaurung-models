# Binary Embedding Models

Train BERT/RoBERTa-style embedding models on binary executable files using custom tokenizers.

## ⚠️ CRITICAL: Data Encoding Format

**The tokenizer expects binary data encoded as latin-1 strings, NOT hex strings!**

```python
# CORRECT - How the tokenizer was trained
raw_bytes = b'\x7fELF\x01\x01'
text = raw_bytes.decode('latin-1')  # → '\x7fELF\x01\x01'
tokens = tokenizer(text)

# WRONG - Do not use hex strings
raw_bytes = b'\x7fELF\x01\x01'
hex_str = "7f 45 4c 46 01 01"  # ❌ WRONG!
tokens = tokenizer(hex_str)  # This will not work correctly
```

The tokenizer uses BPE on latin-1 encoded bytes where each byte (0-255) maps to a single character. This allows it to learn patterns directly from binary data.

## Installation

```bash
uv sync
```

## Quick Start

### Command Line Interface

```bash
# View help
uv run python -m binary_embedding.cli --help

# Basic training with automatic hyperparameter selection
uv run python -m binary_embedding.cli train \
  --model-size small \
  --data-dir /usr/bin \
  --max-files 10 \
  --num-epochs 1

# Training with custom configuration
uv run python -m binary_embedding.cli train \
  --model-size base \
  --data-dir /usr/bin \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --learning-rate 2e-5 \
  --scheduler-type cosine \
  --num-epochs 3

# Advanced training with monitoring and early stopping
uv run python -m binary_embedding.cli train \
  --model-size base \
  --data-dir /nas4/data/glaurung-data/binaries \
  --scheduler-type cosine_with_restarts \
  --learning-rate 2e-5 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --monitor-embedding \
  --early-stopping \
  --save-best-only

# Assess a trained model
uv run python -m binary_embedding.cli assess \
  --checkpoint-path output/final_model \
  --output assessment_results.json
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

# IMPORTANT: When processing your own binary data, use latin-1 encoding:
with open("binary_file", "rb") as f:
    raw_bytes = f.read()
    text = raw_bytes.decode('latin-1')  # Convert to latin-1 string
    tokens = tokenizer(text)  # Tokenize

# Create model
model, config = create_model(size=ModelSize.BASE)

# Create data loader (automatically handles latin-1 encoding)
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

- **Custom Binary Tokenizer**: Pre-trained BPE tokenizer on latin-1 encoded binaries (65536 vocab)
- **Modern Architecture**: RoBERTa-style models with latest optimizations
- **Advanced Training Features**:
  - Automatic hyperparameter selection based on model size
  - Multiple learning rate schedulers (linear, cosine, two-phase, etc.)
  - Gradient accumulation for larger effective batch sizes
  - Mixed precision training (FP16)
  - Early stopping with multi-metric monitoring
  - Embedding quality preservation
- **Efficient Training**:
  - Optimized learning rates (2e-5 for small, 3e-5 for base, 1e-5 for large)
  - Smart batch sizing and accumulation
  - AdamW optimizer with configurable warmup
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
| `--batch-size` | Auto | Per-device batch size (auto-selected by model size) |
| `--num-epochs` | 3 | Number of epochs |
| `--max-steps` | None | Max training steps (overrides epochs) |
| `--learning-rate` | Auto | Learning rate (auto-selected by model size) |
| `--scheduler-type` | Auto | Scheduler (linear/cosine/cosine_with_restarts/two_phase) |
| `--warmup-steps` | None | Warmup steps (uses warmup-ratio if not set) |
| `--warmup-ratio` | 0.1 | Warmup as fraction of total steps |
| `--gradient-accumulation-steps` | Auto | Gradient accumulation (auto-selected by model size) |
| `--max-length` | 512 | Max sequence length |
| `--mlm-probability` | 0.20 | Masking probability |
| `--max-files` | None | Max files to load |
| `--mixed-precision` | True | Use FP16 training |
| `--early-stopping` | False | Enable early stopping |
| `--monitor-embedding` | False | Monitor embedding quality |
| `--save-best-only` | False | Only save best checkpoint |

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
   - Binary files → latin-1 encoded strings
   - BPE tokenization on byte-level characters
   - Dynamic chunk loading (4KB default)
   - Efficient batch collation with MLM

## Using Trained Models

```python
from transformers import AutoModelForMaskedLM
from binary_embedding import load_tokenizer

# Load tokenizer
tokenizer = load_tokenizer()

# Load trained model
model = AutoModelForMaskedLM.from_pretrained("./output/final_model")

# Process binary data (MUST use latin-1 encoding!)
with open("binary_file", "rb") as f:
    raw_bytes = f.read(512)  # Read first 512 bytes
    text = raw_bytes.decode('latin-1')  # Convert to latin-1
    
# Tokenize and get predictions
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# For embeddings, use the hidden states
outputs = model(**inputs, output_hidden_states=True)
embeddings = outputs.hidden_states[-1].mean(dim=1)  # Mean pooling
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
├── tokenizer.py      # Binary tokenizer wrapper (expects latin-1 input)
├── data.py          # Dataset and data loading (converts bytes to latin-1)
├── models.py        # Model architectures
├── training.py      # Training loop with Accelerate
├── assessment.py    # Model assessment framework (uses latin-1)
└── cli.py           # Click CLI with Rich UI
```

## License

MIT