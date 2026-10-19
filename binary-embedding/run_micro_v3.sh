#!/usr/bin/bash

# MICRO ARCHITECTURE TRAINING v3 - Lessons Learned
#
# Changes from v2 (which had partial collapse @ 180k steps):
#
# 1. SCHEDULER: Trapezoidal 60% plateau → Cosine (continuous decay)
#    - v2 issue: Model peaked @ 80k but stayed at full LR until 240k
#    - Result: Query layers collapsed (-35% pattern learning)
#    - Fix: Cosine provides smooth continuous LR decay
#    - No sharp transitions, natural adaptation throughout training
#
# 2. BATCH SIZE: 544 seqs (278K tokens) → 768 seqs (393K tokens)
#    - v2: Below standard MLM recommendation (500K-2M tokens)
#    - Larger batch = more stable gradients at high LR
#    - Should help prevent query layer instability
#    - Tradeoff: 1.5x training time (6.3 days → 9.5 days)
#
# 3. KEPT SAME:
#    - Architecture: MICRO (192/3, 64 dims/layer) ✅ Prevented catastrophic collapse
#    - Warmup: 1000 steps ✅ Worked well
#    - LR: 3e-4 ✅ Good initial peak (49% @ 80k)
#    - Weight decay: 5e-5 ✅ No issues
#
# What we learned from v2:
# - Architecture fix (64 dims/layer) worked - prevented catastrophic collapse
# - Warmup fix (1000 steps) worked - fast convergence
# - BUT: 60% plateau was too long → selective collapse in query layers
# - Pattern learning: 92% → 60% (-35%)
# - L2 query rank: 61.8% → 50.4% (-18%)
# - L2 value rank: 80.4% → 81.0% (stable!)
#
# Expected outcome with v3:
# - Peak around 80-100k steps (same as v2)
# - LR begins decaying immediately after warmup
# - Query layers should stay stable (no extended high-LR stress)
# - Should maintain 45-50% throughout training (vs v2's 49% → 38% decline)
# - Final model target: 45-50% assessment (vs v2's expected 38-42%)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="micro_v3_${TIMESTAMP}"

# Architecture: MICRO (192 hidden, 3 layers, 14M params)
MODEL_SIZE="micro"

# Training parameters - v3 OPTIMIZED
MAX_STEPS=400000        # Same: 2+ full epochs through 14GB data
WARMUP_STEPS=1000       # Same: FIXED from old 50k
SCHEDULER="cosine"      # CHANGED: was trapezoidal (60% plateau) → continuous decay
LR=0.0003               # Same: 3e-4 (proven optimal, worked great @ 80k)
WD=0.00005              # Same: 5e-5 (proven optimal)
BATCH_SIZE=32           # Changed from 34 (memory optimization)
GRAD_ACCUM=24           # CHANGED: 16 → 24 (effective batch 768 seqs = 393K tokens)
                        # Was: 544 seqs = 278K tokens (below 500K-2M recommendation)
                        # Now: 768 seqs = 393K tokens (closer to recommendation)

echo "==========================================="
echo "MICRO ARCHITECTURE TRAINING (v3)"
echo "==========================================="
echo "Architecture: MICRO (192 hidden / 3 layers / 14M params)"
echo "  Dims/layer: 64 (prevents rank collapse)"
echo ""
echo "Training Configuration:"
echo "  Total steps:        ${MAX_STEPS}"
echo "  Warmup:             ${WARMUP_STEPS} steps"
echo "  Scheduler:          ${SCHEDULER} (continuous decay, no plateau)"
echo "  Learning rate:      ${LR} peak"
echo "  Batch size:         ${BATCH_SIZE}"
echo "  Gradient accum:     ${GRAD_ACCUM}"
echo "  Effective batch:    $((BATCH_SIZE * GRAD_ACCUM)) sequences ($((BATCH_SIZE * GRAD_ACCUM * 512)) tokens)"
echo "  Weight decay:       ${WD}"
echo ""
echo "Key Changes from v2:"
echo "  - Scheduler: Trapezoidal 60% plateau → Cosine continuous decay"
echo "  - Batch: 544 seqs (278K tok) → 768 seqs (393K tok)"
echo "  - Rationale: v2 had query layer collapse from extended high LR"
echo ""
echo "Output: ${OUTPUT_DIR}"
echo "==========================================="
echo ""

# Calculate expected training time (1.5x slower due to larger batch)
HOURS_ESTIMATE=$(python3 -c "print(f'{(400000 * 0.5 * 1.5 / 60):.1f}')")
echo "Estimated duration: ${HOURS_ESTIMATE} hours (~$(python3 -c "print(f'{${HOURS_ESTIMATE}/24:.1f}')") days)"
echo ""
echo "Expected outcomes (based on v2 lessons):"
echo "  - Peak: 45-50% @ 80-100k steps (similar to v2's 49% @ 80k)"
echo "  - Stability: Assessments should stay 45-50% throughout"
echo "  - No query layer collapse (continuous LR decay prevents stress)"
echo "  - Pattern learning should stay 70-90% (vs v2's 92% → 60% collapse)"
echo "  - Final model: 45-50% assessment (vs v2's 38-42%)"
echo ""
echo "v2 Comparison:"
echo "  v2 @ 80k:  49% assessment, 92% pattern, 61.8% L2 query rank"
echo "  v2 @ 180k: 44% assessment, 60% pattern, 50.4% L2 query rank (collapse!)"
echo "  v3 goal:   Maintain 45-50% throughout, 70-90% pattern, stable query"
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
  --wandb-project "glaurung-micro-v3" \
  --wandb-run-name "micro_v3_cosine_${TIMESTAMP}" \
  --wandb-tags "micro" \
  --wandb-tags "v3-architecture" \
  --wandb-tags "cosine-schedule" \
  --wandb-tags "fixed-plateau" \
  --wandb-tags "64-dims-per-layer" \
  --wandb-notes "MICRO v3: Cosine scheduler (fixes v2's 60% plateau issue). Larger batch (768 seqs, 393K tokens) for stability. Architecture: 192/3 layers, 64 dims/layer. Expected: stable training, no query layer collapse, 45-50% assessment throughout." \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "==========================================="
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Compare assessment trends vs v2 (should be stable vs declining)"
echo "  2. Check layer SVD - query layers should stay >55% (vs v2's 50.4%)"
echo "  3. Pattern learning should stay >70% (vs v2's 60%)"
echo "  4. If successful, this becomes the new baseline!"
echo "==========================================="
