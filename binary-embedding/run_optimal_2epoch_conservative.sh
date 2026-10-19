#!/usr/bin/bash

# CONSERVATIVE OPTIMAL TRAINING - 2 Full Epochs
# Alternative to 3.0e-4 run with more conservative LR 2.5e-4
# LR 2.5e-4 achieved 32.89% peak in Round 2 with good stability

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="optimal_2epoch_conservative_${TIMESTAMP}"

MAX_STEPS=229376
WARMUP_STEPS=15000
LR=0.00025              # 2.5e-4 (conservative alternative)
WD=0.00005
BATCH_SIZE=32

echo "========================================="
echo "CONSERVATIVE OPTIMAL - 2 FULL EPOCHS"
echo "========================================="
echo "Total steps:    ${MAX_STEPS} (2 epochs)"
echo "Learning rate:  ${LR} (2.5e-4 - conservative)"
echo "Warmup:         ${WARMUP_STEPS} steps"
echo "Output:         ${OUTPUT_DIR}"
echo "========================================="
echo ""

uv run python -m binary_embedding.cli mlm-pretrain \
   --model-size nano \
   --model-type roberta \
   --data-dir /nas4/data/glaurung-data/binaries-small/ \
   --streaming \
   --tokenizer-path ./tokenizer.json \
   --max-steps $MAX_STEPS \
   --warmup-steps $WARMUP_STEPS \
   --scheduler-type trapezoidal \
   --learning-rate $LR \
   --weight-decay $WD \
   --max-length 512 \
   --mixed-precision \
   --gradient-accumulation-steps 1 \
   --max-grad-norm 2.0 \
   --batch-size $BATCH_SIZE \
   --run-assessment \
   --assessment-steps 2000 \
   --save-steps 20000 \
   --save-total-limit 6 \
   --use-wandb \
   --wandb-project "glaurung-nano-optimal" \
   --wandb-run-name "nano_2epoch_conservative_${TIMESTAMP}" \
   --wandb-tags "nano" \
   --wandb-tags "2-epoch" \
   --wandb-tags "conservative" \
   --wandb-tags "lr-2.5e-4" \
   --wandb-notes "Conservative 2-epoch run with LR 2.5e-4 for comparison with 3.0e-4 run. Testing if lower LR provides better stability over full data exposure." \
   --output-dir "$OUTPUT_DIR"

echo "Training Complete! Model saved to: $OUTPUT_DIR"
