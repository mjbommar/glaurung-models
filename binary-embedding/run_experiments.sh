#!/usr/bin/bash

# Systematic hyperparameter exploration for NANO model
# All runs: 50k steps to avoid overfitting window seen in 80k run
# Target: beat 29.06% peak from Run 2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="experiments_${TIMESTAMP}"
mkdir -p "$BASE_DIR"

# Shared parameters
DATA_DIR="/nas4/data/glaurung-data/binaries-small/"
TOKENIZER="./tokenizer.json"
MAX_STEPS=50000
BATCH_SIZE=32

# Function to run a single experiment
run_experiment() {
    local name=$1
    local scheduler=$2
    local lr=$3
    local wd=$4
    local warmup=$5
    local extra_args=$6

    echo "========================================="
    echo "Starting: $name"
    echo "LR: $lr, WD: $wd, Warmup: $warmup"
    echo "Scheduler: $scheduler"
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
       --warmup-steps $warmup \
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
       --wandb-project "glaurung-nano-experiments" \
       --wandb-run-name "${name}_${TIMESTAMP}" \
       --wandb-tags "nano" \
       --wandb-tags "experiment" \
       --wandb-tags "$name" \
       --wandb-notes "$extra_args" \
       --output-dir "$OUTPUT_DIR"

    echo "Completed: $name"
    echo ""
}

# Experiment 1: Baseline trapezoid (from Run 2 config but 50k steps)
run_experiment \
    "exp01_baseline_trap" \
    "trapezoidal" \
    "0.0001" \
    "0.00005" \
    "7500" \
    "Baseline: trap, 15% warmup, 60% plateau"

# Experiment 2: Higher learning rate
run_experiment \
    "exp02_higher_lr" \
    "trapezoidal" \
    "0.00012" \
    "0.00005" \
    "7500" \
    "Higher LR: 1.2e-4 vs 1e-4 baseline"

# Experiment 3: Lower weight decay (less regularization)
run_experiment \
    "exp03_lower_wd" \
    "trapezoidal" \
    "0.0001" \
    "0.00002" \
    "7500" \
    "Lower WD: 2e-5 vs 5e-5, less regularization"

# Experiment 4: Longer plateau (70% vs 60%)
run_experiment \
    "exp04_longer_plateau" \
    "trapezoidal" \
    "0.0001" \
    "0.00005" \
    "7500" \
    "Longer plateau: maintain peak LR longer"

# Experiment 5: Shorter warmup (10% vs 15%)
run_experiment \
    "exp05_short_warmup" \
    "trapezoidal" \
    "0.0001" \
    "0.00005" \
    "5000" \
    "Shorter warmup: 10% vs 15%, faster to peak"

# Experiment 6: Two-phase scheduler
run_experiment \
    "exp06_two_phase" \
    "two_phase" \
    "0.0001" \
    "0.00005" \
    "7500" \
    "Two-phase: 30k high LR, 20k at 30% reduction"

# Experiment 7: Conservative (lower LR, higher WD)
run_experiment \
    "exp07_conservative" \
    "trapezoidal" \
    "0.00008" \
    "0.0001" \
    "10000" \
    "Conservative: lower LR, higher WD, longer warmup"

# Experiment 8: Aggressive (higher LR, lower WD, short warmup)
run_experiment \
    "exp08_aggressive" \
    "trapezoidal" \
    "0.00013" \
    "0.00003" \
    "5000" \
    "Aggressive: higher LR, lower WD, fast warmup"

echo "========================================="
echo "All experiments complete!"
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
                scores = [a["average_score"] for a in data]
                results.append({
                    "name": exp_dir.name,
                    "peak": max(scores),
                    "final": scores[-1],
                    "peak_step": data[scores.index(max(scores))]["step"]
                })

if results:
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    print(f"{'Experiment':<25} {'Peak Score':<12} {'Peak Step':<12} {'Final Score':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<25} {r['peak']:>11.2%} {r['peak_step']:>11,} {r['final']:>11.2%}")
    print("="*80)

    # Best performers
    best_peak = max(results, key=lambda x: x["peak"])
    best_final = max(results, key=lambda x: x["final"])
    print(f"\nBest Peak:  {best_peak['name']} - {best_peak['peak']:.2%} @ step {best_peak['peak_step']:,}")
    print(f"Best Final: {best_final['name']} - {best_final['final']:.2%}")
EOF
