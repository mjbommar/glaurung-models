#!/bin/bash
# Quick training demo with all improvements

echo "==================================="
echo "IOC-NER Model Training Demo"
echo "==================================="

# Create directories
mkdir -p data/processed
mkdir -p models

# Run training with optimal settings
python scripts/train.py \
    --generate_synthetic \
    --train_samples 5000 \
    --val_samples 1000 \
    --hidden_dim 512 \
    --num_layers 4 \
    --num_heads 8 \
    --batch_size 64 \
    --num_epochs 5 \
    --learning_rate 2e-4 \
    --fp16 \
    --device cuda \
    --num_workers 4

echo ""
echo "Training demo complete!"
echo "For full training, use: python scripts/train.py --generate_synthetic"