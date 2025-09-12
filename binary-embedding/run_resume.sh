#!/bin/bash
# Resume training from the latest checkpoint

# Generate timestamp for run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Find the latest checkpoint
latest=$(ls -d output/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)

if [ -n "$latest" ]; then
    echo "Resuming from checkpoint: output/checkpoint-$latest"
    
    PYTHONPATH=src uv run python -m binary_embedding.cli train \
               --model-size base \
               --model-type roberta \
               --data-dir /home/ubuntu/data/binaries/ \
               --max-files 10000000 \
               --max-steps 1000000 \
               --save-steps 1000 \
               --batch-size 104 \
               --gradient-accumulation-steps 32 \
               --monitor-embedding \
               --save-total-limit 10 \
               --run-assessment \
               --assessment-steps 1000 \
               --use-wandb \
               --wandb-project "glaurung-binary-001" \
               --wandb-run-name "resume_${TIMESTAMP}" \
               --wandb-tags "base" \
               --wandb-tags "roberta" \
               --wandb-tags "resumed" \
               --wandb-notes "Resumed training from checkpoint-$latest" \
               --resume-from-checkpoint output/checkpoint-$latest
else
    echo "No checkpoints found. Use run.sh to start fresh."
fi