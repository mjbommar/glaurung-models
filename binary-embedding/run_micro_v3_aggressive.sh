#!/usr/bin/bash

# MICRO ARCHITECTURE TRAINING v3 AGGRESSIVE - Maximum Stability
#
# This is the more aggressive version with:
# - Full 1M token batches (vs 393K in standard v3)
# - Slightly lower LR (2.5e-4 vs 3e-4) for extra stability
# - Cosine schedule (same as standard v3)
#
# Use this if standard v3 still shows instability.
# Tradeoff: 4x training time (25+ days) but maximum stability.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="micro_v3_aggressive_${TIMESTAMP}"

MODEL_SIZE="micro"

# Training parameters - v3 AGGRESSIVE
MAX_STEPS=400000
WARMUP_STEPS=1000
SCHEDULER="cosine"
LR=0.00025              # LOWER: 2.5e-4 (vs 3e-4) for extra stability
WD=0.00005
BATCH_SIZE=32
GRAD_ACCUM=64           # LARGE: 64 (effective batch 2048 seqs = 1M tokens)

echo "==========================================="
echo "MICRO v3 AGGRESSIVE - Maximum Stability"
echo "==========================================="
echo "Architecture: MICRO (192 hidden / 3 layers / 14M params)"
echo ""
echo "Configuration:"
echo "  Scheduler:          ${SCHEDULER} (continuous decay)"
echo "  Learning rate:      ${LR} peak (LOWER for stability)"
echo "  Effective batch:    $((BATCH_SIZE * GRAD_ACCUM)) sequences ($((BATCH_SIZE * GRAD_ACCUM * 512)) tokens)"
echo "  Note: 1M token batch is at high end of 500K-2M recommendation"
echo ""
echo "⚠️  WARNING: 4x slower than v2, ~25 days training time"
echo ""
read -p "Press Enter to start training (or Ctrl+C to cancel)..."

uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size $MODEL_SIZE \
  --model-type roberta \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --streaming \
  --tokenizer-path ./tokenizer.json \
  --max-steps $MAX_STEPS \
  --warmup-steps $WARMUP_STEPS \
  --scheduler-type $SCHEDULER \
  --learning-rate $LR \
  --weight-decay $WD \
  --max-length 512 \
  --mixed-precision \
  --gradient-accumulation-steps $GRAD_ACCUM \
  --max-grad-norm 4.0 \
  --batch-size $BATCH_SIZE \
  --run-assessment \
  --assessment-steps 1000 \
  --save-steps 10000 \
  --save-total-limit 5 \
  --use-wandb \
  --wandb-project "glaurung-micro-v3-aggressive" \
  --wandb-run-name "micro_v3_aggressive_${TIMESTAMP}" \
  --wandb-tags "micro" \
  --wandb-tags "v3-aggressive" \
  --wandb-tags "cosine-schedule" \
  --wandb-tags "large-batch" \
  --wandb-tags "lower-lr" \
  --wandb-notes "MICRO v3 AGGRESSIVE: Cosine scheduler, 1M token batch (2048 seqs), lower LR (2.5e-4). Maximum stability configuration. 4x training time vs v2." \
  --output-dir "$OUTPUT_DIR"

echo "Training Complete! Model saved to: $OUTPUT_DIR"
