#!/bin/bash

# Generate timestamp for run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src

uv run python -m binary_embedding.cli train \
       --model-size base \
       --model-type roberta \
       --data-dir /home/ubuntu/data/binaries/ \
       --max-steps 100000 \
       --warmup-ratio 0.05 \
       --save-steps 100 \
       --gradient-checkpointing \
       --learning-rate 0.0001 \
       --batch-size 96 \
       --gradient-accumulation-steps 8 \
       --save-total-limit 10 \
       --run-assessment \
       --assessment-steps 100 \
       --contrastive \
       --streaming \
       --pooling mean \
       --dup-prob 0.5 \
       --min-chunk-separation 4096 \
       --contrastive-temp 0.07 \
       --mlm-weight 1.0 \
       --view-weight 0.25 \
       --samefile-weight 0.25 \
       --contrastive-ramp-steps 5000 \
       --pair-cache-size 4096 \
       --prefetch-factor 4 \
       --num-workers 4 \
       --use-wandb \
       --wandb-project "glaurung-binary-002" \
       --wandb-run-name "run_${TIMESTAMP}" \
       --wandb-tags "base" \
       --wandb-tags "roberta" \
       --wandb-notes "Binary embedding training with MLM + 2x contrastive loss"
