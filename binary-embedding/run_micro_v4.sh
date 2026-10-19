#!/usr/bin/bash

# MICRO v4 - The Minimal Fix
#
# v2 was EXCELLENT until 80k (49% assessment, 92% pattern)
# v2's ONLY problem: 60% plateau = stayed at full LR for 160k steps AFTER peak
#
# v4 = v2 with ONE change: 60% plateau → 15% plateau
#
# This makes LR start decaying @ 60k, BEFORE the 80k peak
# By the time model peaks, LR is already naturally reducing
# Should prevent query layer overfit while maintaining v2's strong performance

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="micro_v4_${TIMESTAMP}"

MODEL_SIZE="micro"

# v2's EXACT config (proven to work):
MAX_STEPS=400000
WARMUP_STEPS=10000       # ✅ Worked great
LR=0.0002               # ✅ Achieved 49% peak
WD=0.00005              # ✅ No issues
BATCH_SIZE=34           # ✅ Stable
GRAD_ACCUM=8           # ✅ Effective batch 544 = 278K tokens

# ONLY CHANGE: Short plateau
PLATEAU_RATIO=0.15      # Was 0.6, now 0.15 → decay starts @ 60k not 240k

echo "==========================================="
echo "MICRO v4 - Minimal Fix from v2"
echo "==========================================="
echo "v2 results:"
echo "  @ 80k:  49% assessment ✅ 92% pattern ✅"
echo "  @ 180k: 44% assessment 💀 60% pattern 💀"
echo "  Problem: Stayed at 3e-4 LR until 240k"
echo ""
echo "v4 change:"
echo "  Plateau: 60% → 15% (240k steps → 60k steps)"
echo "  LR decay starts @ 60k (before 80k peak)"
echo "  By 80k: LR already reducing naturally"
echo ""
echo "Configuration:"
echo "  Batch:         ${BATCH_SIZE} × ${GRAD_ACCUM} = 544 seqs (278K tokens)"
echo "  Learning rate: ${LR} (peak)"
echo "  Scheduler:     Trapezoidal with ${PLATEAU_RATIO} plateau"
echo "  Schedule:"
echo "    0-1k:    Warmup to 3e-4"
echo "    1k-60k:  Plateau @ 3e-4  (15% of training)"
echo "    60k-400k: Decay to 0     (85% of training)"
echo ""
echo "Expected: Maintain 49% peak, NO collapse"
echo "==========================================="
echo ""
read -p "Press Enter to start v4 (or Ctrl+C to cancel)..."

uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size $MODEL_SIZE \
  --model-type roberta \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --streaming \
  --tokenizer-path mjbommar/glaurung-binary-tokenizer-001 \
  --max-steps $MAX_STEPS \
  --warmup-steps $WARMUP_STEPS \
  --scheduler-type trapezoidal \
  --learning-rate $LR \
  --plateau-ratio $PLATEAU_RATIO \
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
  --wandb-run-name "micro_v4_short_plateau_${TIMESTAMP}" \
  --wandb-tags "micro" \
  --wandb-tags "v4" \
  --wandb-tags "short-plateau" \
  --wandb-tags "15pct-plateau" \
  --wandb-notes "MICRO v4: Exact v2 config (544 seqs, 3e-4 LR, 278K tokens) but with 15% plateau instead of 60%. Decay starts @ 60k (before 80k peak) to prevent query layer overfit that killed v2 @ 180k. Expected: 49% peak maintained, no collapse." \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "==========================================="
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Key checkpoints to compare with v2:"
echo "  @ 80k:  Should match v2's 49% peak"
echo "  @ 180k: Should maintain ~45-48% (vs v2's 44% collapse)"
echo "  @ 400k: Should be 45-50% stable"
echo "==========================================="
