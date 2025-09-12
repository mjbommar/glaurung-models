#!/bin/bash
# Find the latest checkpoint in the output directory

# Find all checkpoint directories and sort by step number
latest=$(ls -d output/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)

if [ -n "$latest" ]; then
    echo "Latest checkpoint: output/checkpoint-$latest"
    echo "Step: $latest"
    
    # Check if training state exists
    if [ -f "output/checkpoint-$latest/training_state.pt" ]; then
        echo "Training state: ✓ Found"
    else
        echo "Training state: ✗ Not found"
    fi
    
    # Check if model exists
    if [ -f "output/checkpoint-$latest/model.safetensors" ] || [ -f "output/checkpoint-$latest/pytorch_model.bin" ]; then
        echo "Model weights: ✓ Found"
    else
        echo "Model weights: ✗ Not found"
    fi
    
    echo ""
    echo "To resume training, use:"
    echo "  --resume-from-checkpoint output/checkpoint-$latest"
else
    echo "No checkpoints found in output/"
fi