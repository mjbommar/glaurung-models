#!/bin/bash

# Generate timestamp for run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# nas4/data/glaurung-data/binaries/ 

PYTHONPATH=src uv run python -m binary_embedding.cli train \
       --model-size small \
       --model-type roberta \
       --data-dir /usr/bin \
       --max-steps 100000 \
       --warmup-ratio 0.02 \
       --save-steps 100 \
       --gradient-checkpointing \
       --batch-size 10 \
       --gradient-accumulation-steps 10 \
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
       --view-weight 0.5 \
       --samefile-weight 0.5 \
       --contrastive-ramp-steps 2000 \
       --pair-cache-size 4096 \
       --prefetch-factor 4 \
       --num-workers 4 \
       --use-wandb \
       --wandb-project "glaurung-binary-002" \
       --wandb-run-name "run_${TIMESTAMP}" \
       --wandb-tags "base" \
       --wandb-tags "roberta" \
       --wandb-notes "Binary embedding training with MLM + 2x contrastive loss"
