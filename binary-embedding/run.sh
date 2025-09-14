#!/bin/bash

# Generate timestamp for run name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src

#        
#        --contrastive \
#       --pooling mean \
#       --dup-prob 0.5 \
#       --contrastive-temp 0.07 \
#       --mlm-weight 1.0 \
#       --view-weight 0.25 \
#       --samefile-weight 0.25 \
#       --contrastive-ramp-steps 5000 \
#       --pair-cache-size 4096 \
#       --prefetch-factor 4 \
#       --num-workers 4 \


uv run python -m binary_embedding.cli train \
       --model-size base \
       --model-type roberta \
       --data-dir /nas4/data/glaurung-data/binaries/ \
       --max-steps 100000 \
       --warmup-ratio 0.1 \
       --save-steps 1000 \
       --gradient-checkpointing \
       --learning-rate 0.00002 \
       --batch-size 8 \
       --gradient-accumulation-steps 1 \
       --use-adaptive-grad-clip \
       --grad-clip-percentile 95.0 \
       --grad-clip-history-size 100 \
       --grad-clip-initial-threshold 10.0 \
       --save-total-limit 10 \
       --run-assessment \
       --assessment-steps 1000 \
       --streaming \
       --enable-entropy-filtering \
       --entropy-bins "0,1.0,3.0,6.0,7.5,8.0" \
       --entropy-weights "0.3,0.7,1.5,1.0,0.5" \
       --use-wandb \
       --wandb-project "glaurung-binary-002" \
       --wandb-run-name "run_${TIMESTAMP}" \
       --wandb-tags "base" \
       --wandb-tags "roberta" \
       --wandb-tags "entropy_filter" \
       --wandb-notes "Binary embedding training with MLM + 2x contrastive loss + entropy filtering"
