#!/usr/bin/bash

# MICRO ARCHITECTURE TRAINING - Fixed & Optimized
#
# This replaces the old NANO training script with:
# - New MICRO architecture: 192 hidden / 3 layers / 14M params (vs old 128/4/9.3M)
# - Fixed architecture: 64 dims/layer (prevents rank collapse)
# - Fixed warmup: 1000 steps (was 50k - 113× too long!)
# - Same proven hyperparameters from experiments
#
# Architecture rationale:
# - Old NANO (128/4=32 dims/layer) suffered rank collapse in Layer 3
# - MICRO (192/3=64 dims/layer) maintains healthy gradient flow
# - Proven by BERT architectures (all use 64 dims/layer)
#
# Hyperparameter rationale:
# - Warmup 1000 steps: matches data exposure of successful exp06 (7.5k × 32batch = 240K seqs)
#   vs old 50k × 544batch = 27.2M seqs (113× too much!)
# - LR 3e-4: Proven optimal in exp06 (35.85% peak)
# - Gradient accumulation 16: Proven stable, effective batch 544
# - All other params from successful experiments

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="micro_${TIMESTAMP}"

# Architecture: MICRO (192 hidden, 3 layers, 14M params)
MODEL_SIZE="micro"

# Training parameters - OPTIMIZED
MAX_STEPS=400000        # 2+ full epochs through 14GB data
WARMUP_STEPS=1000       # FIXED: was 50k (113× too long!)
LR=0.0003               # 3e-4 (proven optimal from exp06)
WD=0.00005              # 5e-5 (proven optimal)
BATCH_SIZE=34           # Stable batch size
GRAD_ACCUM=16           # Effective batch = 544 sequences
# Note: Trapezoidal scheduler uses default plateau_ratio=0.6 (60% of training at peak LR)

echo "==========================================="
echo "MICRO ARCHITECTURE TRAINING (v2)"
echo "==========================================="
echo "Architecture: MICRO (192 hidden / 3 layers / 14M params)"
echo "  vs old NANO: 128 hidden / 4 layers / 9.3M params"
echo "  Dims/layer:  64 (vs 32 - prevents rank collapse!)"
echo ""
echo "Training Configuration:"
echo "  Total steps:        ${MAX_STEPS}"
echo "  Warmup:             ${WARMUP_STEPS} steps (FIXED from 50k!)"
echo "  Learning rate:      ${LR}"
echo "  Effective batch:    $((BATCH_SIZE * GRAD_ACCUM)) sequences"
echo "  Gradient accum:     ${GRAD_ACCUM}"
echo "  Weight decay:       ${WD}"
echo "  Plateau:            60% of training"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "==========================================="
echo ""

# Calculate expected training time
# Assume similar speed to old NANO: ~0.4-0.5 min/step with gradient accumulation
HOURS_ESTIMATE=$(python3 -c "print(f'{(400000 * 0.5 / 60):.1f}')")
echo "Estimated duration: ${HOURS_ESTIMATE} hours"
echo ""
echo "Expected outcomes:"
echo "  - NO rank collapse (64 dims/layer prevents this)"
echo "  - Peak at ~10-15k steps (vs 29k in broken warmup)"
echo "  - Stable performance throughout training"
echo "  - Target peak: 38-42% (vs old NANO 32-35%)"
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
  --scheduler-type trapezoidal \
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
  --wandb-project "glaurung-micro-v2" \
  --wandb-run-name "micro_fixed_arch_${TIMESTAMP}" \
  --wandb-tags "micro" \
  --wandb-tags "v2-architecture" \
  --wandb-tags "fixed-warmup" \
  --wandb-tags "64-dims-per-layer" \
  --wandb-notes "MICRO v2: Fixed architecture (192/3 layers, 64 dims/layer prevents rank collapse). Fixed warmup (1000 steps vs old 50k). LR 3e-4, gradient accumulation 16, effective batch 544. Expected: stable training, no degradation, peak 38-42%." \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "==========================================="
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Check layer SVD analysis at checkpoints"
echo "  2. Verify no rank collapse in final layers"
echo "  3. Compare to old NANO baseline (32-35%)"
echo "==========================================="
