#!/bin/bash

# Generate timestamp for run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

PYTHONPATH=src uv run python -m binary_embedding.cli train \
	       --model-size base \
	       --model-type roberta \
	       --data-dir /home/ubuntu/data/binaries/ \
	       --max-files 10000000 \
	       --max-steps 100000 \
	       --warmup-ratio 0.01 \
	       --save-steps 1000 \
	       --batch-size 104 \
	       --gradient-accumulation-steps 32 \
	       --monitor-embedding \
	       --save-total-limit 10 \
	       --run-assessment \
	       --assessment-steps 1000 \
	       --use-wandb \
	       --wandb-project "glaurung-binary-001" \
	       --wandb-run-name "run_${TIMESTAMP}" \
	       --wandb-tags "base" \
	       --wandb-tags "roberta" \
	       --wandb-notes "Binary embedding training on GH200"
