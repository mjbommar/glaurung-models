# IOC-NER Model Project Structure

## Directory Layout
```
glaurung-models/
├── data/
│   ├── raw/                 # Raw IOC datasets
│   ├── processed/            # Processed training data
│   └── synthetic/            # Generated synthetic data
├── src/
│   ├── data_generation/
│   │   ├── __init__.py
│   │   ├── ioc_patterns.py   # IOC regex patterns
│   │   ├── synthetic_gen.py  # Synthetic data generator
│   │   └── augmentation.py   # Data augmentation
│   ├── model/
│   │   ├── __init__.py
│   │   ├── architecture.py   # Model architecture
│   │   ├── layers.py         # Custom layers
│   │   └── heads.py          # Task-specific heads
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Training loop
│   │   ├── loss.py           # Loss functions
│   │   └── metrics.py        # Evaluation metrics
│   └── optimization/
│       ├── __init__.py
│       ├── quantization.py   # Quantization code
│       ├── onnx_export.py    # ONNX conversion
│       └── benchmarks.py     # Performance testing
├── configs/
│   ├── model_config.yaml     # Model hyperparameters
│   └── training_config.yaml  # Training settings
├── scripts/
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   └── generate_data.py      # Data generation script
├── tests/
│   └── test_*.py             # Unit tests
├── notebooks/
│   └── eda.ipynb             # Data exploration
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## Key Components

### 1. IOC Pattern Library (`src/data_generation/ioc_patterns.py`)
- Comprehensive regex patterns for all IOC types
- Validation functions for each IOC type
- Pattern variations for adversarial examples

### 2. Synthetic Generator (`src/data_generation/synthetic_gen.py`)
- Pattern-based generation (fast, deterministic)
- LLM-based generation (realistic contexts)
- Adversarial example generation
- Benign lookalike generation

### 3. Model Architecture (`src/model/architecture.py`)
- Character-level CNN for pattern detection
- Lightweight transformer (2 layers, 4 heads)
- Multi-task learning (NER + risk scoring)
- Position-aware embeddings for short sequences

### 4. Training Pipeline (`src/training/trainer.py`)
- Multi-task loss balancing
- Gradient accumulation
- Mixed precision training (FP16)
- Knowledge distillation from teacher model

### 5. CPU Optimization (`src/optimization/`)
- Dynamic quantization to INT8
- ONNX export and optimization
- Graph fusion and operator optimization
- Benchmark suite for latency/throughput

## Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Model Size | <5MB (INT8) | CPU cache efficiency |
| Latency | <5ms/sample | Real-time processing |
| Throughput | 10K samples/sec | High-volume analysis |
| F1 Score | >95% | Production reliability |
| False Positive Rate | <1% | Minimize alert fatigue |

## Training Phases

1. **Bootstrap Phase** (Week 1)
   - Collect 100K real IOCs from threat feeds
   - Generate 1M synthetic samples
   - Create validation/test splits

2. **Pre-training Phase** (Week 2)
   - Train on synthetic data
   - Pattern recognition focus
   - Character-level feature extraction

3. **Fine-tuning Phase** (Week 3)
   - Real IOC datasets
   - Few-shot learning for rare IOCs
   - Multi-task optimization

4. **Distillation Phase** (Week 4)
   - Teacher model: RoBERTa-base
   - Student model: Our lightweight architecture
   - Soft label training

5. **Optimization Phase** (Week 5)
   - Quantization-aware training
   - ONNX conversion
   - CPU-specific optimizations

6. **Deployment Phase** (Week 6)
   - Performance benchmarking
   - API development
   - Integration testing