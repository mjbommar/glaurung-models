#!/usr/bin/bash

# OPTIMAL TRAINING RUN - 2 Full Epochs
# Based on Round 1 & 2 experimental results
#
# Key learnings:
# - 14GB data = 114,688 steps/epoch → 229,376 steps for 2 epochs
# - Best LR: 3.0e-4 (35.85% peak, 32.04% final in Round 2)
# - Best scheduler: Trapezoidal (beat two-phase consistently)
# - Chunk-level shuffle: CRITICAL (implemented in data.py)
# - Previous 50k runs only saw 43.6% of unique data!

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="optimal_2epoch_${TIMESTAMP}"

# Training parameters (optimized for 2 full epochs)
MAX_STEPS=400000        # 2 full passes through 14GB data
WARMUP_STEPS=50000      # 6.5% warmup (conservative for high LR)
PLATEAU_RATIO=0.8       # 60% plateau proven optimal
LR=0.0003               # 3.0e-4 (best from Round 2 exp06)
WD=0.00005              # 5e-5 (proven optimal)
BATCH_SIZE=34           # Stable batch size

echo "========================================="
echo "OPTIMAL NANO TRAINING - 2 FULL EPOCHS"
echo "========================================="
echo "Total steps:    ${MAX_STEPS} (2 epochs × 114,688 steps)"
echo "Learning rate:  ${LR} (3.0e-4)"
echo "Warmup:         ${WARMUP_STEPS} steps (6.5%)"
echo "Plateau:        60% of training"
echo "Weight decay:   ${WD}"
echo "Batch size:     ${BATCH_SIZE}"
echo "Output:         ${OUTPUT_DIR}"
echo "========================================="
echo ""

# Calculate expected training time
# Round 2 avg: ~25 min per 1k steps = 0.42 min/step
HOURS_ESTIMATE=$(python3 -c "print(f'{(229376 * 0.42 / 60):.1f}')")
echo "Estimated duration: ${HOURS_ESTIMATE} hours"
echo ""
read -p "Press Enter to start training (or Ctrl+C to cancel)..."

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
   --gradient-accumulation-steps 16 \
   --max-grad-norm 4.0 \
   --batch-size $BATCH_SIZE \
   --run-assessment \
   --assessment-steps 1000 \
   --save-steps 10000 \
   --save-total-limit 5 \
   --use-wandb \
   --wandb-project "glaurung-nano-optimal" \
   --wandb-run-name "nano_2epoch_optimal_${TIMESTAMP}" \
   --wandb-tags "nano" \
   --wandb-tags "2-epoch" \
   --wandb-tags "optimal" \
   --wandb-tags "lr-3e-4" \
   --wandb-notes "Optimal NANO training: 2 full epochs (229k steps), LR 3e-4, trapezoidal scheduler with 60% plateau. Based on Round 1-2 experiments showing 3e-4 as optimal LR and need for full data exposure. Includes chunk-level shuffle fix." \
   --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================="
