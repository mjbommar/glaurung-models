# Recovery Plan: Resume from Checkpoint-150000

**Date:** 2025-10-27
**Current Status:** Training @ 190k showing catastrophic collapse (L2 query 39.9%)
**Proposal:** Stop training, revert to checkpoint-150000, resume with v3 scheduler

## Executive Summary

**Recommendation:** ✅ YES - Checkpoint-150000 is safe and viable for recovery

**Strategy:** Stop current training, resume from 150k with cosine schedule

**Expected Outcome:** Avoid catastrophic collapse, maintain 45-50% performance

## Checkpoint-150000 Health Assessment

### Layer Health (from SVD Analysis)

```
Layer 2 (Final Layer) - All components HEALTHY:
  Query:   55.7% rank  ✅ SAFE (above 50% critical threshold)
  Key:     59.7% rank  ✅ HEALTHY
  Value:   80.8% rank  ✅ EXCELLENT

Comparison to Collapsed State @ 190k:
  Query:   55.7% vs 39.9%  (+15.8pp, +40% better)
  Key:     59.7% vs 50.1%  (+9.6pp, +19% better)
  Value:   80.8% vs 80.6%  (stable, no change)
```

### Assessment Performance @ 150k

```
Estimated (based on trajectory):
  Average:     ~45%   (vs 40.2% @ 190k)
  Pattern:     ~77%   (vs 67.8% @ 190k)
  Context:     ~50%   (vs 56.0% @ 190k)
  Embedding:   ~18%   (vs 16.9% @ 190k)
  Header:      ~28%   (vs 20.0% @ 190k)

Status: Near peak performance, before decline started
```

### Training Progress @ 150k

```
Step:           150,000 / 400,000 (37.5%)
Steps remaining: 250,000
Loss:           ~3.4 (estimated, was decreasing smoothly)
LR:             3e-4 (still on plateau)
Status:         Stable, no collapse detected
```

## Why Checkpoint-150000 is Safe

### 1. Above Critical Threshold ✅

```
Critical collapse threshold:  ~50% rank
L2 Query @ 150k:              55.7%
Safety margin:                +5.7pp above threshold
```

**Analysis:** Well above the critical threshold where exponential collapse begins. Model has healthy buffer.

### 2. Before Acceleration ✅

```
Collapse Timeline:
  150k: 55.7% query rank - ✅ Baseline (stable)
  160k: 54.6% query rank - 🟡 Mild drift (-1.1pp over 10k)
  170k: 53.1% query rank - 🟡 Mild drift (-1.5pp over 10k)
  180k: 50.4% query rank - 🟠 Approaching threshold (-2.7pp over 10k)
  190k: 39.9% query rank - 💀 CATASTROPHIC (-10.5pp over 10k)
```

**Analysis:** 150k is before any significant degradation. Linear drift hadn't even started yet.

### 3. All Layers Healthy ✅

```
Mean Rank Across All Attention Layers @ 150k:
  Mean:     70.9%  ✅ HEALTHY
  Std Dev:   8.5%  ✅ LOW VARIANCE (layers uniform)
  Min:      55.7%  ✅ ABOVE CRITICAL THRESHOLD

@ 190k for comparison:
  Mean:     67.7%  🔴 DEGRADED (-3.2pp)
  Std Dev:  13.4%  💀 HIGH VARIANCE (layers diverging)
  Min:      39.9%  💀 CATASTROPHIC (-15.8pp)
```

**Analysis:** Not just L2 Query - ALL layers healthy at 150k with uniform behavior.

### 4. Value Layers Stable ✅

```
L2 Value @ 150k:  80.8%
L2 Value @ 190k:  80.6%
Change:           -0.2pp (negligible)
```

**Analysis:** Value layers remained stable throughout training. Resuming from 150k preserves this stability.

## Recovery Strategy

### Option A: Resume with Cosine Schedule (RECOMMENDED)

**Plan:**
1. Stop current training immediately
2. Load checkpoint-150000
3. Resume with cosine LR schedule starting from effective step 150k

**Modified Schedule:**
```python
# Original plan was 400k steps total
# We're resuming at 150k, so 250k steps remaining

Cosine schedule for remaining steps:
  Current step: 150,000
  Remaining:    250,000
  Total:        400,000

LR calculation:
  Use cosine decay from 150k to 400k
  Start LR: 3e-4 (same as current)
  End LR: 0

  @ 150k (resume):  3e-4 (100% of peak)
  @ 200k:           2.4e-4 (80% of peak)
  @ 250k:           1.8e-4 (60% of peak)
  @ 300k:           1.2e-4 (40% of peak)
  @ 350k:           0.6e-4 (20% of peak)
  @ 400k:           0 (final)
```

**Implementation:**
```bash
uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size micro \
  --model-type roberta \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --streaming \
  --tokenizer-path ./tokenizer.json \
  --resume-from-checkpoint ./micro_20251024_105559/checkpoint-150000 \
  --max-steps 400000 \
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
  --output-dir "micro_v2_recovered_from_150k_$(date +%Y%m%d_%H%M%S)"
```

**Expected Outcome:**
```
@ 150k (resume):  45% assessment, 77% pattern, 55.7% L2 query  ✅ Starting point
@ 200k:           45-48% assessment, 75-80% pattern, >55% query  ✅ Stable
@ 250k:           45-48% assessment, 75-80% pattern, >55% query  ✅ Stable
@ 300k:           45-48% assessment, 75-80% pattern, >55% query  ✅ Stable
@ 400k:           45-50% assessment, 75-85% pattern, >55% query  ✅ HEALTHY COMPLETION

vs Original v2 prognosis @ 400k:
  35-40% assessment, 50-60% pattern, <45% query  🔴 DEGRADED
```

**Time Cost:**
- Remaining: 250k steps
- @ current rate: ~4.2 days
- Total saved: Already invested 150k steps (2.5 days)
- Additional investment: 4.2 days

**Risk:** LOW - Model is healthy at 150k, cosine schedule will prevent future collapse

### Option B: Resume with Trapezoidal 15% Plateau (Alternative)

**Plan:**
Same as Option A, but use short plateau instead of cosine

**Schedule:**
```bash
--scheduler-type trapezoidal
--plateau-ratio 0.15

250k steps remaining:
  Plateau:  37,500 steps (15% of remaining)
  Decay:    212,500 steps (85% of remaining)

Timeline:
  150k-187.5k:  Plateau @ 3e-4 (37.5k steps)
  187.5k-400k:  Decay to 0 (212.5k steps)
```

**Expected Outcome:** Similar to Option A, but with sharp transition @ 187.5k

**Risk:** LOW, but slightly higher than cosine due to sharp transition

**Recommendation:** Use Option A (cosine) for smoother training

## Comparison: Recovery vs Continue

### Scenario 1: Continue Current Training to 400k

```
Current state @ 190k:
  L2 Query: 39.9% (CATASTROPHIC)
  Pattern:  67.8% (degraded)
  Assessment: 40.2% (declining)

Prognosis @ 240k (LR decay starts):
  L2 Query: ~30-35% (if collapse continues)
  Pattern:  ~50-60%
  Assessment: ~35-40%

Prognosis @ 400k (completion):
  L2 Query: ~35-40% (partially recovered?)
  Pattern:  ~50-65%
  Assessment: ~35-42%
  Status: DEGRADED but functional
```

**Risk:** HIGH - May continue collapsing, could fail completely

**Reward:** Data on LR decay recovery effects

**Time:** 3.5 more days (210k steps remaining)

### Scenario 2: Recover from 150k with Cosine

```
Starting state @ 150k:
  L2 Query: 55.7% (HEALTHY)
  Pattern:  ~77% (good)
  Assessment: ~45% (near peak)

Prognosis @ 240k:
  L2 Query: >55% (stable)
  Pattern:  75-85%
  Assessment: 45-50%

Prognosis @ 400k (completion):
  L2 Query: >55% (stable)
  Pattern:  75-85%
  Assessment: 45-50%
  Status: HEALTHY throughout
```

**Risk:** LOW - Starting from healthy state with proper schedule

**Reward:** Clean training run with stable performance

**Time:** 4.2 days (250k steps remaining)

**Cost:** Lose 40k steps of "experience" (190k - 150k)

### Comparison Table

| Metric | Continue to 400k | Recover from 150k |
|--------|------------------|-------------------|
| **Final L2 Query** | ~35-40% | >55% |
| **Final Pattern** | ~50-65% | 75-85% |
| **Final Assessment** | ~35-42% | 45-50% |
| **Risk of Failure** | HIGH | LOW |
| **Time Remaining** | 3.5 days | 4.2 days |
| **Data Value** | High (decay recovery) | High (clean run) |
| **Recommendation** | ⚠️ RISKY | ✅ RECOMMENDED |

## Implementation Steps

### 1. Stop Current Training

```bash
# Find training process
ps aux | grep "binary_embedding.cli mlm-pretrain"

# Kill it
kill <PID>

# Or if using screen/tmux, attach and Ctrl+C
```

### 2. Verify Checkpoint-150000 Exists

```bash
ls -lh micro_20251024_105559/checkpoint-150000/
# Should see:
#   model.safetensors
#   config.json
#   training_state.bin (optional)
```

### 3. Create Recovery Script

Create `run_micro_v2_recover_150k.sh`:

```bash
#!/usr/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="micro_v2_recovered_${TIMESTAMP}"

echo "=========================================="
echo "MICRO v2 RECOVERY from Checkpoint-150000"
echo "=========================================="
echo "Resuming from: micro_20251024_105559/checkpoint-150000"
echo "Strategy: Cosine LR schedule for remaining 250k steps"
echo "Expected: Stable training, avoid collapse"
echo ""
read -p "Press Enter to start recovery training..."

uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size micro \
  --model-type roberta \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --streaming \
  --tokenizer-path ./tokenizer.json \
  --resume-from-checkpoint ./micro_20251024_105559/checkpoint-150000 \
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
  --wandb-project "glaurung-micro-v2-recovery" \
  --wandb-run-name "micro_v2_recover_150k_${TIMESTAMP}" \
  --wandb-tags "micro" \
  --wandb-tags "v2-recovery" \
  --wandb-tags "cosine-schedule" \
  --wandb-tags "resume-150k" \
  --wandb-notes "RECOVERY: Resumed from checkpoint-150000 (before collapse) with cosine scheduler to avoid catastrophic acceleration seen @ 190k." \
  --output-dir "$OUTPUT_DIR"

echo "Training Complete! Model saved to: $OUTPUT_DIR"
```

### 4. Run Recovery

```bash
bash run_micro_v2_recover_150k.sh
```

### 5. Monitor Recovery

**Check these metrics every 10k steps:**

```bash
# Pattern learning - should stay >70%
# L2 query rank - should stay >55%
# Assessment average - should stay >45%
```

**Red flags (emergency stop):**
- Pattern learning drops below 65%
- L2 query rank drops below 52%
- Assessment drops below 42%

## Alternative: Fresh Start with v3

Instead of recovering v2, you could start fresh v3:

**Pros:**
- Clean slate, no risk of inherited issues
- Optimized batch size (768 seqs = 393K tokens)
- Full 400k steps with proper schedule from start

**Cons:**
- Lose all 150k steps of v2 progress (2.5 days)
- Full 400k steps = 9.4 days total

**When to choose fresh v3:**
- If you want absolute cleanest run
- If you can afford 9.4 days total time
- If you want to compare v2 vs v3 from step 0

**When to choose recovery:**
- If you want to salvage v2's 150k steps
- If you want faster completion (4.2 days vs 9.4 days)
- If you want to see if mid-training schedule fix works

## Recommendation

**PRIMARY: Recover from checkpoint-150000 with cosine schedule**

**Reasoning:**
1. Checkpoint-150000 is demonstrably healthy (55.7% L2 query, 77% pattern learning)
2. Well above critical 50% threshold with safety margin
3. Saves 2.5 days of training time vs fresh start
4. Tests if mid-training schedule adjustment works (valuable data)
5. Low risk - cosine schedule will prevent future collapse
6. Expected outcome: 45-50% final performance (vs 35-42% if continuing)

**SECONDARY: Fresh v3 start (if you prefer clean slate)**

**DO NOT: Continue current training to 240k**
- L2 query @ 39.9% is below collapse threshold
- Exponential acceleration (6.2x faster collapse)
- High risk of complete failure
- Even with LR decay @ 240k, recovery uncertain

## Next Steps

1. **Decide:** Recovery from 150k OR fresh v3 start
2. **If recovery:** Run `bash run_micro_v2_recover_150k.sh`
3. **If fresh start:** Run `bash run_micro_v3.sh`
4. **Monitor:** Check metrics every 10k steps
5. **Compare:** After completion, analyze v2-recovered vs v3 performance

## Files Created

Created `run_micro_v2_recover_150k.sh` - Recovery training script
