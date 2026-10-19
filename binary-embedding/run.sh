#!/usr/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

# Run 3: Optimal NANO training based on empirical results
# Run 1 (cosine, 50k): Peaked at 26k (24.72%), LR decayed too fast
# Run 2 (trapezoid, 80k): Peaked at 37k (29.06%), then overfitted
# Run 3: 50k steps captures peak window without overfitting
uv run python -m binary_embedding.cli mlm-pretrain \
   --model-size nano \
   --model-type roberta \
   --data-dir /nas4/data/glaurung-data/binaries-small/ \
   --streaming \
   --tokenizer-path ./tokenizer.json \
   --max-steps 50000 \
   --warmup-steps 7500 \
   --scheduler-type trapezoidal \
   --learning-rate 0.0001 \
   --weight-decay 0.00005 \
   --max-length 512 \
   --mixed-precision \
   --gradient-accumulation-steps 1 \
   --max-grad-norm 2.0 \
   --batch-size 32 \
   --run-assessment \
   --assessment-steps 1000 \
   --save-steps 1000 \
   --use-wandb \
   --wandb-project "glaurung-binary-nano" \
   --wandb-run-name "nano_optimal_${TIMESTAMP}" \
   --wandb-tags "nano" \
   --wandb-tags "roberta" \
   --wandb-tags "mlm" \
   --wandb-tags "trapezoidal" \
   --wandb-tags "optimal" \
   --wandb-notes "NANO optimal: 50k steps, trap schedule, stops before overfitting"
