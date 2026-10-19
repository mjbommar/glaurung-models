#!/usr/bin/bash

# Round 2: Learning Rate Exploration
# Round 1 showed clear trend: LR 1.2e-4 > 1e-4 > 8e-5
# Goal: Find optimal LR and test with both top schedulers (trap, two-phase)

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="experiments_round2_${TIMESTAMP}"
mkdir -p "$BASE_DIR"

# Shared parameters
DATA_DIR="/nas4/data/glaurung-data/binaries-small/"
TOKENIZER="./tokenizer.json"
MAX_STEPS=50000
BATCH_SIZE=32
WARMUP=7500  # 15% warmup proven optimal in round 1
WD_DEFAULT=0.00005  # 5e-5 proven optimal in round 1

# Function to run a single experiment
run_experiment() {
    local name=$1
    local scheduler=$2
    local lr=$3
    local wd=$4
    local extra_args=$5

    echo "========================================="
    echo "Starting: $name"
    echo "LR: $lr, WD: $wd, Scheduler: $scheduler"
    echo "========================================="

    OUTPUT_DIR="${BASE_DIR}/${name}"
    mkdir -p "$OUTPUT_DIR"

    uv run python -m binary_embedding.cli mlm-pretrain \
       --model-size nano \
       --model-type roberta \
       --data-dir "$DATA_DIR" \
       --streaming \
       --tokenizer-path "$TOKENIZER" \
       --max-steps $MAX_STEPS \
       --warmup-steps $WARMUP \
       --scheduler-type $scheduler \
       --learning-rate $lr \
       --weight-decay $wd \
       --max-length 512 \
       --mixed-precision \
       --gradient-accumulation-steps 1 \
       --max-grad-norm 2.0 \
       --batch-size $BATCH_SIZE \
       --run-assessment \
       --assessment-steps 1000 \
       --save-steps 5000 \
       --use-wandb \
       --wandb-project "glaurung-nano-lr-exploration" \
       --wandb-run-name "${name}_${TIMESTAMP}" \
       --wandb-tags "nano" \
       --wandb-tags "round2" \
       --wandb-tags "lr-exploration" \
       --wandb-tags "$name" \
       --wandb-notes "$extra_args" \
       --output-dir "$OUTPUT_DIR"

    echo "Completed: $name"
    echo ""
}

echo "========================================="
echo "ROUND 2: AGGRESSIVE LEARNING RATE EXPLORATION"
echo "Testing LRs: 1.5e-4 to 3.0e-4 (1.5x to 3x baseline)"
echo "NANO is tiny (9.3M params) - can handle high LRs"
echo "Schedulers: trapezoidal, two_phase"
echo "========================================="
echo ""

# ============================================================================
# PART 1: Trapezoidal LR Sweep (6 experiments)
# Wide sweep from 1.5e-4 to 3.0e-4 to find ceiling
# NANO (9.3M params) is tiny - small models tolerate high LRs like ViT-Tiny
# ============================================================================

# Experiment 1: LR 1.5e-4 (conservative start, +25% vs 1.2e-4)
run_experiment \
    "exp01_trap_lr_1.5e-4" \
    "trapezoidal" \
    "0.00015" \
    "$WD_DEFAULT" \
    "Trap LR sweep: 1.5e-4 (1.5x baseline)"

# Experiment 2: LR 1.8e-4 (mid-range)
run_experiment \
    "exp02_trap_lr_1.8e-4" \
    "trapezoidal" \
    "0.00018" \
    "$WD_DEFAULT" \
    "Trap LR sweep: 1.8e-4 (1.8x baseline)"

# Experiment 3: LR 2.0e-4 (2x baseline - psychological barrier)
run_experiment \
    "exp03_trap_lr_2.0e-4" \
    "trapezoidal" \
    "0.00020" \
    "$WD_DEFAULT" \
    "Trap LR sweep: 2.0e-4 (2x baseline 1e-4)"

# Experiment 4: LR 2.5e-4 (aggressive - 2.5x baseline)
run_experiment \
    "exp04_trap_lr_2.5e-4" \
    "trapezoidal" \
    "0.00025" \
    "$WD_DEFAULT" \
    "Trap LR sweep: 2.5e-4 (2.5x baseline, approaching ViT-Tiny range)"

# Experiment 5: LR 2.8e-4 (very aggressive)
run_experiment \
    "exp05_trap_lr_2.8e-4" \
    "trapezoidal" \
    "0.00028" \
    "$WD_DEFAULT" \
    "Trap LR sweep: 2.8e-4 (2.8x baseline, expect possible instability)"

# Experiment 6: LR 3.0e-4 (ceiling test - 3x baseline)
run_experiment \
    "exp06_trap_lr_3.0e-4" \
    "trapezoidal" \
    "0.00030" \
    "$WD_DEFAULT" \
    "Trap LR sweep: 3.0e-4 (3x baseline, finding ceiling)"

# ============================================================================
# PART 2: Two-Phase at Promising LRs (2 experiments)
# Test best scheduler at 2x and 2.5x baseline LR
# ============================================================================

# Experiment 7: Two-phase with LR 2.0e-4
run_experiment \
    "exp07_two_phase_lr_2.0e-4" \
    "two_phase" \
    "0.00020" \
    "$WD_DEFAULT" \
    "Two-phase (best scheduler) with LR 2.0e-4 (2x baseline)"

# Experiment 8: Two-phase with LR 2.5e-4
run_experiment \
    "exp08_two_phase_lr_2.5e-4" \
    "two_phase" \
    "0.00025" \
    "$WD_DEFAULT" \
    "Two-phase (best scheduler) with LR 2.5e-4 (2.5x baseline)"

echo "========================================="
echo "All Round 2 experiments complete!"
echo "Results saved in: $BASE_DIR"
echo "========================================="

# Generate comparison summary
echo "Generating comparison table..."
python3 << EOF
import json
import os
from pathlib import Path

base_dir = Path("${BASE_DIR}")
results = []

for exp_dir in sorted(base_dir.iterdir()):
    if not exp_dir.is_dir():
        continue

    history_file = exp_dir / "assessment_history.json"
    if history_file.exists():
        with open(history_file) as f:
            data = json.load(f)
            if data:
                scores = [a["assessment/average_score"] for a in data]
                results.append({
                    "name": exp_dir.name,
                    "peak": max(scores),
                    "final": scores[-1],
                    "peak_step": data[scores.index(max(scores))]["step"]
                })

if results:
    print("\n" + "="*80)
    print("ROUND 2: LEARNING RATE EXPLORATION RESULTS")
    print("="*80)
    print(f"{'Experiment':<35} {'Peak Score':<12} {'Peak Step':<12} {'Final Score':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<35} {r['peak']:>11.2%} {r['peak_step']:>11,} {r['final']:>11.2%}")
    print("="*80)

    # Best performers
    best_peak = max(results, key=lambda x: x["peak"])
    best_final = max(results, key=lambda x: x["final"])
    print(f"\nBest Peak:  {best_peak['name']} - {best_peak['peak']:.2%} @ step {best_peak['peak_step']:,}")
    print(f"Best Final: {best_final['name']} - {best_final['final']:.2%}")

    # Find best LR
    trap_results = [r for r in results if "trap" in r["name"] and "two_phase" not in r["name"]]
    two_phase_results = [r for r in results if "two_phase" in r["name"]]

    if trap_results:
        best_trap = max(trap_results, key=lambda x: x["final"])
        print(f"\nBest Trapezoidal: {best_trap['name']} - {best_trap['final']:.2%}")

    if two_phase_results:
        best_two_phase = max(two_phase_results, key=lambda x: x["final"])
        print(f"Best Two-Phase:   {best_two_phase['name']} - {best_two_phase['final']:.2%}")

    # Compare to round 1 best (29.23% final, 30.44% peak)
    round1_best_final = 0.2923
    round1_best_peak = 0.3044

    if best_final["final"] > round1_best_final:
        improvement = ((best_final["final"] - round1_best_final) / round1_best_final) * 100
        print(f"\n🎉 IMPROVEMENT over Round 1: {improvement:+.1f}%")
    else:
        regression = ((best_final["final"] - round1_best_final) / round1_best_final) * 100
        print(f"\n⚠️  Regression from Round 1: {regression:.1f}%")
EOF
