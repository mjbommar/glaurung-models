#!/usr/bin/bash

# MICRO v5 - "v2 Power with Controlled Decay"
#
# HYPOTHESIS: v2's collapse was due to plateau ratio, NOT LR/batch
#
# v2: 3e-4 LR, 544 batch, 60% plateau → 49% peak @ 80k, collapsed @ 180k 💀
# v4: 2e-4 LR, 272 batch, 15% plateau → 29% peak @ 120k, stable but WAY behind 📉
#
# v5 STRATEGY: Keep v2's proven power, fix ONLY the plateau
# - Use v2's EXACT LR (3e-4) and batch (544) that achieved 49%
# - Use 25% plateau (split between v4's 15% and v2's 60%)
# - Decay starts @ 107.5k (well before v2's 180k collapse)
#
# EXPECTED RESULTS:
# - Match v2's 49% peak @ 80k (same LR/batch)
# - Maintain through 180k (decay started @ 107.5k prevents overfit)
# - Final performance: 45-48% (vs v2's 44% collapse, v4's 29% stable)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="micro_v5_${TIMESTAMP}"

MODEL_SIZE="micro"

# v2's PROVEN hyperparameters (achieved 49% peak):
MAX_STEPS=400000
WARMUP_STEPS=10000       # ✅ Worked in v2
LR=0.0003                # ✅ v2's proven LR (3e-4)
WD=0.00005               # ✅ v2's proven weight decay
BATCH_SIZE=34            # ✅ v2's proven batch
GRAD_ACCUM=16            # ✅ v2's proven grad accum (effective batch 544)

# ONLY CHANGE: Reduce plateau from 60% to 25%
PLATEAU_RATIO=0.25       # 25% = middle ground (v4: 15%, v2: 60%)

echo "==========================================="
echo "MICRO v5 - v2 Power with Controlled Decay"
echo "==========================================="
echo "Strategy: Isolate the ONE variable that matters"
echo ""
echo "v2 results (baseline):"
echo "  @ 80k:  49% avg ✅ 92% pattern ✅ (PEAK)"
echo "  @ 180k: 44% avg 💀 60% pattern 💀 (COLLAPSE)"
echo "  Problem: 60% plateau = 240k steps at full LR"
echo ""
echo "v4 results (stable but slow):"
echo "  @ 80k:  27% avg 📉 52% pattern 📉"
echo "  @ 120k: 29% avg 📉 66% pattern 📉"
echo "  Problem: Too conservative (nerfed LR, batch, AND plateau)"
echo ""
echo "v5 changes:"
echo "  LR:     3e-4 (SAME as v2 ✅)"
echo "  Batch:  544  (SAME as v2 ✅)"
echo "  Plateau: 25% (FIXED from v2's 60%)"
echo ""
echo "Configuration:"
echo "  Batch:         ${BATCH_SIZE} × ${GRAD_ACCUM} = 544 seqs (v2's power ✅)"
echo "  Learning rate: ${LR} (v2's peak LR ✅)"
echo "  Scheduler:     Trapezoidal with ${PLATEAU_RATIO} plateau"
echo "  Schedule:"
echo "    0-10k:    Warmup to 3e-4"
echo "    10k-107.5k:  Plateau @ 3e-4 (25% = 97.5k steps)"
echo "    107.5k-400k: Decay to 0 (292.5k steps)"
echo ""
echo "Expected results:"
echo "  @ 80k:  Match v2's 49% peak (same LR/batch)"
echo "  @ 107.5k: Peak ends, decay starts (vs v2's 240k)"
echo "  @ 180k: Maintain 45-48% (vs v2's 44% collapse)"
echo "  @ 400k: Stable 45-50% (best of both worlds)"
echo ""
echo "Key insight:"
echo "  v4 proved early decay prevents collapse"
echo "  v5 tests if we can have v2's speed WITHOUT collapse"
echo "==========================================="
echo ""
read -p "Press Enter to start v5 (or Ctrl+C to cancel)..."

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
  --wandb-run-name "micro_v5_v2power_25plateau_${TIMESTAMP}" \
  --wandb-tags "micro" \
  --wandb-tags "v5" \
  --wandb-tags "v2-power" \
  --wandb-tags "25pct-plateau" \
  --wandb-tags "controlled-decay" \
  --wandb-notes "MICRO v5: v2's exact hyperparameters (3e-4 LR, 544 batch) but with 25% plateau instead of 60%. Tests if plateau ratio is THE critical variable. Decay starts @ 107.5k (vs v2's 240k) to prevent 180k collapse. Expected: 49% peak maintained, no collapse." \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "==========================================="
echo "Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Comparison points:"
echo "  @ 80k:  Compare to v2's 49% peak"
echo "  @ 107.5k: Decay starts (watch query layers)"
echo "  @ 180k: v2 collapsed here - will v5 hold?"
echo "  @ 400k: Final performance vs v2 (44%) and v4 (29%)"
echo "==========================================="
