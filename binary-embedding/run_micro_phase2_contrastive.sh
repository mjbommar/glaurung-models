#!/usr/bin/bash

# MICRO Phase 2 - Contrastive Fine-Tuning
#
# WHY CONTRASTIVE PHASE?
# v6 @ 150k showed classic overfitting:
#   - MLM loss improving (5.11 → 5.04)
#   - Assessment scores declining (41% → 36.6%)
#   - Pattern learning STRONG (63.3%) ✅
#   - Embedding quality WEAK (24.9%) 💀
#
# DIAGNOSIS: MLM optimizes token prediction, NOT embedding quality
# SOLUTION: Contrastive learning optimizes embedding space directly
#
# RESEARCH BASIS:
#   - NeoBERT (June 2025): Contrastive learning → 66.60 MTEB score
#   - VIKCSE (May 2025): +2.79% improvement on semantic similarity
#   - SimCSE: Fixes BERT's "anisotropy problem" in embeddings
#
# EXPECTED RESULTS:
#   Pattern Learning:  63% → 60%+ (maintained by MLM weight)
#   Embedding Quality: 25% → 35%+ (improved by contrastive)
#   File Header:       25% → 30%+ (better semantics)
#   Context:           40% → 45%+ (better representations)
#   Overall Average:   38% → 45%+

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="micro_phase2_contrastive_${TIMESTAMP}"

MODEL_SIZE="micro"

# Phase 2 Configuration
MAX_STEPS=75000           # Shorter than Phase 1 (150k)
WARMUP_STEPS=5000         # Short warmup (fine-tuning, not pretraining)
LR=0.00001                # 1e-5 (MUCH lower - this is fine-tuning!)
WD=0.00005                # Same as Phase 1
BATCH_SIZE=8             # HALVED for pairs (17 pairs × 2 views = 34 sequences)
GRAD_ACCUM=16             # Effective batch: 17×2×16 = 544 (same as Phase 1!)

# Loss weights (balanced multi-task approach)
MLM_WEIGHT=1.0            # Keep MLM to maintain pattern learning
VIEW_WEIGHT=0.5           # Duplicate-view contrastive (same chunk, different masks)
SAMEFILE_WEIGHT=0.5       # Same-file contrastive (different chunks, same binary)

# Contrastive settings
CONTRASTIVE_TEMP=0.07     # Temperature for InfoNCE loss
POOLING="mean"            # Mean pooling for embeddings
DUP_PROB=0.5              # 50% duplicate pairs, 50% same-file pairs

# Scheduler: Cosine with 30% plateau
SCHEDULER_TYPE="trapezoidal"
PLATEAU_RATIO=0.30        # Less aggressive than v6's 10%

# Data configuration
CHUNK_SIZE=4096
MIN_CHUNK_SEPARATION=4096 # Chunks must be 4KB apart for same-file pairs

echo "==========================================="
echo "MICRO Phase 2 - Contrastive Fine-Tuning"
echo "==========================================="
echo "Starting from: models/micro-v6-best/ (checkpoint-150000)"
echo ""
echo "WHY THIS PHASE?"
echo "  v6 Peak @ 150k: 38.3% avg, but:"
echo "    Pattern:  63.3% ✅ (MLM did its job)"
echo "    Embedding: 24.9% 💀 (CRITICAL - embeddings not semantic)"
echo "    Header:    25.0% ⚠️  (needs better semantics)"
echo "    Context:   40.0% ⚠️  (needs better representations)"
echo ""
echo "  v6 After 150k: Overfitting to MLM"
echo "    Loss kept improving → Scores kept declining"
echo "    Model memorizing patterns, not learning meaning"
echo ""
echo "SOLUTION: Contrastive Learning"
echo "  MLM = predict tokens (optimizes pattern matching)"
echo "  Contrastive = optimize embeddings (creates semantic space)"
echo "  Multi-task = MLM + Contrastive (best of both worlds)"
echo ""
echo "Configuration:"
echo "  Starting checkpoint: models/micro-v6-best/"
echo "  Phase 1 performance:  38.3% avg @ 150k steps"
echo ""
echo "Training Setup:"
echo "  Max steps:        ${MAX_STEPS} (shorter than Phase 1)"
echo "  Warmup:           ${WARMUP_STEPS} steps (fine-tuning warmup)"
echo "  Learning rate:    ${LR} (1e-5 - much lower than Phase 1's 3e-4)"
echo "  Batch:            ${BATCH_SIZE} pairs × 2 views × ${GRAD_ACCUM} = 544 seqs (same as Phase 1)"
echo "  Weight decay:     ${WD} (same as Phase 1)"
echo "  Scheduler:        ${SCHEDULER_TYPE} with ${PLATEAU_RATIO} plateau"
echo ""
echo "MEMORY NOTE: Contrastive uses pairs, so batch_size=17 (not 34)"
echo "  17 pairs × 2 views per pair = 34 sequences per micro-batch"
echo "  34 sequences × 16 grad_accum = 544 effective batch ✓"
echo ""
echo "Loss Configuration:"
echo "  MLM weight:       ${MLM_WEIGHT}  (maintain pattern learning)"
echo "  View weight:      ${VIEW_WEIGHT}  (learn: same chunk = similar)"
echo "  SameFile weight:  ${SAMEFILE_WEIGHT}  (learn: same file = related)"
echo "  Temperature:      ${CONTRASTIVE_TEMP} (InfoNCE loss)"
echo ""
echo "Data Configuration:"
echo "  Pooling method:   ${POOLING} (mean pooling for embeddings)"
echo "  Duplicate prob:   ${DUP_PROB} (50% dup pairs, 50% same-file pairs)"
echo "  Chunk size:       ${CHUNK_SIZE} bytes"
echo "  Min separation:   ${MIN_CHUNK_SEPARATION} bytes (same-file pairs)"
echo ""
echo "Schedule (trapezoidal ${PLATEAU_RATIO} plateau):"
echo "  0-5k:     Warmup to 1e-5"
echo "  5k-27k:   Plateau @ 1e-5 (30% = 22.5k steps at peak)"
echo "  27k-75k:  Decay to 0 (48k steps)"
echo ""
echo "Expected Results:"
echo "  Pattern Learning:  63% → 60%+ (MLM maintains this)"
echo "  Embedding Quality: 25% → 35%+ (contrastive TARGET)"
echo "  File Header:       25% → 30%+ (better semantics)"
echo "  Context:           40% → 45%+ (better representations)"
echo "  Overall Average:   38% → 45%+ (target)"
echo ""
echo "Key Monitoring Points:"
echo "  @ 10k:  Early check - embeddings should improve"
echo "  @ 25k:  Mid-training - pattern loss should stay stable"
echo "  @ 50k:  Late training - overall scores should peak"
echo "  @ 75k:  End - compare to Phase 1 (150k @ 38.3%)"
echo ""
echo "Research Backing:"
echo "  - NeoBERT (June 2025): Contrastive → 66.60 MTEB"
echo "  - VIKCSE (May 2025): +2.79% semantic similarity"
echo "  - SimCSE: Fixes BERT anisotropy problem"
echo "  - Multi-task approach prevents catastrophic forgetting"
echo "==========================================="
echo ""
read -p "Press Enter to start Phase 2 (or Ctrl+C to cancel)..."

uv run python -m binary_embedding.cli contrastive-train \
  --model-size $MODEL_SIZE \
  --model-type roberta \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --streaming \
  --tokenizer-path mjbommar/glaurung-binary-tokenizer-001 \
  --init-from-checkpoint models/micro-v6-best/ \
  --max-steps $MAX_STEPS \
  --warmup-steps $WARMUP_STEPS \
  --scheduler-type $SCHEDULER_TYPE \
  --plateau-ratio $PLATEAU_RATIO \
  --learning-rate $LR \
  --weight-decay $WD \
  --max-length 512 \
  --mixed-precision \
  --gradient-accumulation-steps $GRAD_ACCUM \
  --max-grad-norm 4.0 \
  --batch-size $BATCH_SIZE \
  --pooling $POOLING \
  --dup-prob $DUP_PROB \
  --chunk-size $CHUNK_SIZE \
  --min-chunk-separation $MIN_CHUNK_SEPARATION \
  --contrastive-temp $CONTRASTIVE_TEMP \
  --mlm-weight $MLM_WEIGHT \
  --view-weight $VIEW_WEIGHT \
  --samefile-weight $SAMEFILE_WEIGHT \
  --run-assessment \
  --assessment-steps 1000 \
  --save-steps 5000 \
  --save-total-limit 5 \
  --use-wandb \
  --wandb-project "glaurung-micro-phase2" \
  --wandb-run-name "micro_phase2_contrastive_${TIMESTAMP}" \
  --wandb-tags "micro" \
  --wandb-tags "phase2" \
  --wandb-tags "contrastive" \
  --wandb-tags "multi-task" \
  --wandb-tags "embedding-quality" \
  --wandb-notes "MICRO Phase 2: Contrastive fine-tuning from v6 best checkpoint (150k @ 38.3%). Addresses weak embedding quality (24.9%) using multi-task MLM+Contrastive. Research-backed approach (NeoBERT, VIKCSE, SimCSE). Expected: 38% → 45%+ by improving embeddings while maintaining pattern learning." \
  --output-dir "$OUTPUT_DIR"

echo ""
echo "==========================================="
echo "Phase 2 Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Comparison to Phase 1:"
echo "  Phase 1 (v6 @ 150k): 38.3% avg"
echo "    Pattern:  63.3% ✅"
echo "    Embedding: 24.9% 💀"
echo "    Header:    25.0%"
echo "    Context:   40.0%"
echo ""
echo "Check Phase 2 results:"
echo "  Look for embedding quality improvement (target: 35%+)"
echo "  Pattern learning should stay stable (target: 60%+)"
echo "  Overall average should improve (target: 45%+)"
echo ""
echo "Next Steps:"
echo "  1. Check assessment_history.json for progression"
echo "  2. Compare best Phase 2 checkpoint to Phase 1"
echo "  3. If embedding quality improved → SUCCESS"
echo "  4. If not → may need stronger contrastive weights"
echo "==========================================="
