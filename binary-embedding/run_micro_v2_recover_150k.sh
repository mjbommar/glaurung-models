#!/usr/bin/bash

# MICRO v2 RECOVERY - Resume from Checkpoint-150000
#
# Strategy: Resume from healthy checkpoint-150000 (before collapse)
#           with cosine scheduler to prevent catastrophic acceleration
#
# Checkpoint-150000 health:
#   L2 Query: 55.7% (SAFE, above 50% critical threshold)
#   L2 Key:   59.7% (healthy)
#   L2 Value: 80.8% (excellent)
#   Pattern:  ~77% (good)
#   Assessment: ~45% (near peak)
#
# Expected outcome: Stable training to 400k with 45-50% final performance

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="micro_v2_recovered_${TIMESTAMP}"
CHECKPOINT_PATH="micro_20251024_105559/checkpoint-150000"

echo "=========================================="
echo "MICRO v2 RECOVERY from Checkpoint-150000"
echo "=========================================="
echo "Resuming from: ${CHECKPOINT_PATH}"
echo ""
echo "Original v2 Status:"
echo "  @ 150k: L2 query 55.7%, pattern ~77%, assessment ~45% ✅ HEALTHY"
echo "  @ 190k: L2 query 39.9%, pattern 67.8%, assessment 40.2% 💀 COLLAPSED"
echo ""
echo "Recovery Strategy:"
echo "  Scheduler: Cosine (vs original trapezoidal 60% plateau)"
echo "  Batch: 768 seqs = 393K tokens (vs original 544 seqs = 278K)"
echo "  LR: 3e-4 peak (same, worked well)"
echo "  Steps remaining: 250k (will complete @ 400k)"
echo ""
echo "Expected Final Performance:"
echo "  L2 Query: >55% (vs <40% if continuing current training)"
echo "  Pattern:  75-85% (vs 50-65% if continuing)"
echo "  Assessment: 45-50% (vs 35-42% if continuing)"
echo ""
read -p "Press Enter to start recovery training (or Ctrl+C to cancel)..."

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
    echo "Please verify the checkpoint directory exists."
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}/model.safetensors" ]; then
    echo "ERROR: model.safetensors not found in checkpoint"
    exit 1
fi

echo ""
echo "Checkpoint verified. Starting recovery training..."
echo ""

uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size micro \
  --model-type roberta \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --streaming \
  --tokenizer-path ./tokenizer.json \
  --resume-from-checkpoint "./${CHECKPOINT_PATH}" \
  --max-steps 400000 \
  --warmup-steps 0 \
  --scheduler-type cosine \
  --learning-rate 0.0003 \
  --weight-decay 0.00005 \
  --max-length 512 \
  --mixed-precision \
  --gradient-accumulation-steps 24 \
  --max-grad-norm 4.0 \
  --batch-size 32 \
  --run-assessment \
  --assessment-steps 1000 \
  --save-steps 10000 \
  --save-total-limit 5 \
  --use-wandb \
  --wandb-project "glaurung-micro-v2" \
  --wandb-run-name "micro_v2_recover_150k_${TIMESTAMP}" \
  --wandb-tags "micro" \
  --wandb-tags "v2-recovery" \
  --wandb-tags "cosine-schedule" \
  --wandb-tags "resume-150k" \
  --wandb-notes "RECOVERY: Resumed from checkpoint-150000 (before collapse) with cosine scheduler to avoid catastrophic acceleration seen @ 190k. Original v2 collapsed from L2 query 55.7%->39.9% between 150k-190k due to 60% plateau. Recovery uses cosine schedule + larger batch (393K tokens) for stability." \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Please check final metrics:"
echo "  - L2 query rank should be >55%"
echo "  - Pattern learning should be 75-85%"
echo "  - Assessment should be 45-50%"
