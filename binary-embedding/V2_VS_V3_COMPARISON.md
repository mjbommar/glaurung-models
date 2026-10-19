# MICRO v2 vs v3 - Quick Comparison

## Command Line Differences

### v2 (Current - Has Partial Collapse)

```bash
uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size micro \
  --model-type roberta \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --streaming \
  --tokenizer-path ./tokenizer.json \
  --max-steps 400000 \
  --warmup-steps 1000 \
  --scheduler-type trapezoidal \         # ← ISSUE: 60% plateau default
  --learning-rate 0.0003 \
  --weight-decay 0.00005 \
  --max-length 512 \
  --mixed-precision \
  --gradient-accumulation-steps 16 \    # ← 544 seqs = 278K tokens (low)
  --max-grad-norm 4.0 \
  --batch-size 34 \
  --run-assessment \
  --assessment-steps 1000 \
  --save-steps 10000 \
  --save-total-limit 5 \
  --use-wandb
```

### v3 Standard (RECOMMENDED)

```bash
uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size micro \
  --model-type roberta \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --streaming \
  --tokenizer-path ./tokenizer.json \
  --max-steps 400000 \
  --warmup-steps 1000 \
  --scheduler-type cosine \              # ← FIXED: Continuous decay
  --learning-rate 0.0003 \
  --weight-decay 0.00005 \
  --max-length 512 \
  --mixed-precision \
  --gradient-accumulation-steps 24 \    # ← INCREASED: 768 seqs = 393K tokens
  --max-grad-norm 4.0 \
  --batch-size 32 \                     # ← Adjusted for memory
  --run-assessment \
  --assessment-steps 1000 \
  --save-steps 10000 \
  --save-total-limit 5 \
  --use-wandb
```

### v3 Aggressive (If Standard Fails)

```bash
uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size micro \
  --model-type roberta \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --streaming \
  --tokenizer-path ./tokenizer.json \
  --max-steps 400000 \
  --warmup-steps 1000 \
  --scheduler-type cosine \              # ← Same as standard
  --learning-rate 0.00025 \              # ← LOWER: 2.5e-4 for extra stability
  --weight-decay 0.00005 \
  --max-length 512 \
  --mixed-precision \
  --gradient-accumulation-steps 64 \    # ← LARGE: 2048 seqs = 1M tokens
  --max-grad-norm 4.0 \
  --batch-size 32 \
  --run-assessment \
  --assessment-steps 1000 \
  --save-steps 10000 \
  --save-total-limit 5 \
  --use-wandb
```

## Parameter Comparison Table

| Parameter | v2 (Current) | v3 Standard | v3 Aggressive | Change Reason |
|-----------|--------------|-------------|---------------|---------------|
| **scheduler-type** | trapezoidal | **cosine** | **cosine** | Fix 60% plateau issue |
| **batch-size** | 34 | 32 | 32 | Memory optimization |
| **gradient-accum** | 16 | **24** | **64** | Increase effective batch |
| **Effective batch** | 544 seqs | **768 seqs** | **2048 seqs** | Better stability |
| **Effective tokens** | 278K | **393K** | **1M** | Meet 500K-2M standard |
| **learning-rate** | 3e-4 | 3e-4 | **2.5e-4** | Extra conservative |
| **Training time** | 6.3 days | 9.4 days | 25.1 days | Tradeoff for stability |

## Performance Comparison

### v2 Observed Performance

```
Step    Assessment  Pattern     L2 Query    Status
0       -           -           -           Start
80k     49%        92%         61.8%       🟢 PEAK
120k    43%        77%         58.4%       🟡 Declining
160k    45%        77%         54.6%       🟡 Stressed
180k    44%        60%         50.4%       🔴 COLLAPSE
187k    39%        62%         ?           🔴 Collapsing

Trend: Assessments declining, pattern learning collapsed 35%, query layers stressed
```

### v3 Expected Performance

```
Step    Assessment  Pattern     L2 Query    Status
0       -           -           -           Start
80k     45-50%     75-85%      >58%        🟢 PEAK (similar to v2)
120k    45-50%     75-85%      >58%        🟢 STABLE (vs v2 decline)
160k    45-50%     75-85%      >58%        🟢 STABLE (vs v2 stressed)
180k    45-50%     75-85%      >58%        🟢 STABLE (vs v2 collapse)
240k    45-50%     75-85%      >58%        🟢 STABLE (low LR by now)
400k    45-50%     75-85%      >58%        🟢 STABLE (fine-tuning)

Trend: Assessments stable throughout, no collapse, healthy query layers
```

## Learning Rate Schedules Visualized

### v2 Trapezoidal (60% plateau)

```
LR
3e-4 |----●-----------------------------------------------------------------●----
     |   /                                                                  \
     |  /                                                                    \
     | /                                                                      \
0    |/________________________________________________________________________\___
     0       1k                    240k                                   400k
     Warmup  |<------- Plateau @ 3e-4 (60%) ------->|<--- Decay (40%) --->|

Issue: Model peaked @ 80k but stayed at full LR until 240k (160k past peak!)
```

### v3 Cosine (continuous decay)

```
LR
3e-4 |----●
     |   /  ╲
     |  /    ╲___
     | /         ╲___
     |/              ╲___
0    |___________________╲_______________________________________________
     0       1k      80k      160k      240k      320k      400k
     Warmup  |<--------------- Smooth Cosine Decay ------------------>|

Fix: LR decays smoothly starting right after warmup
     @ 80k:  ~2.4e-4 (80% of peak) - natural reduction at peak
     @ 160k: ~1.5e-4 (50% of peak) - vs v2's 100%
     @ 240k: ~0.9e-4 (30% of peak) - vs v2's decay start
```

## Files Created

1. **`run_micro_v3.sh`** - Standard v3 training script (RECOMMENDED)
   - Cosine schedule
   - 768 seqs (393K tokens)
   - Same LR (3e-4)
   - 1.5x training time

2. **`run_micro_v3_aggressive.sh`** - Aggressive v3 (backup option)
   - Cosine schedule
   - 2048 seqs (1M tokens)
   - Lower LR (2.5e-4)
   - 4x training time

3. **`V3_TRAINING_RATIONALE.md`** - Detailed explanation
   - Why each change was made
   - Expected outcomes
   - Theoretical foundations

4. **`V2_VS_V3_COMPARISON.md`** - This file
   - Quick reference
   - Side-by-side comparison

5. **`PARTIAL_COLLAPSE_DIAGNOSIS.md`** - v2 problem analysis
   - What went wrong
   - Evidence and correlation
   - Prognosis

## Recommended Next Steps

### 1. Let v2 Complete

**Current:** Step 187k / 400k (46.9%)
**LR decay starts:** Step 240k (~20 hours away)
**Completion:** ~3.5 days

**Why:**
- Need to see if LR decay recovers performance
- Valuable data on long-plateau effects
- Only 20 hours until decay starts

### 2. Start v3 Standard

**After v2 completes:**
```bash
cd /home/mjbommar/projects/personal/glaurung-models/binary-embedding
bash run_micro_v3.sh
```

**Monitor for:**
- Assessments staying 45-50% (not declining like v2)
- Pattern learning >70% (not collapsing to 60%)
- Query layers >55% rank (not dropping to 50%)

### 3. If v3 Fails, Try Aggressive

**Only if v3 standard shows:**
- Pattern learning drops below 65%
- Query layers drop below 52%
- Assessments decline >5pp from peak

**Then run:**
```bash
bash run_micro_v3_aggressive.sh
```

## Key Takeaways

### v2 Lessons

✅ **Architecture fix (64 dims/layer) worked**
   - Prevented catastrophic uniform collapse
   - Only selective query layer stress

✅ **Warmup fix (1000 steps) worked**
   - Fast convergence to peak

❌ **Schedule was wrong (60% plateau)**
   - Model peaked @ 80k (20%)
   - Stayed at full LR for 160k more steps
   - Caused selective collapse

❌ **Batch slightly low (278K tokens)**
   - Below 500K-2M recommendation
   - Contributed to instability

### v3 Improvements

1. **Cosine schedule** - Fixes plateau issue
2. **Larger batch** - Improves stability
3. **Keep good parts** - LR, warmup, architecture

### Expected Results

**v2 final (estimated):**
- 38-42% assessment
- Pattern learning ~50-55%
- Query layers ~45-48%
- Functional but degraded

**v3 final (expected):**
- 45-50% assessment
- Pattern learning 75-85%
- Query layers >55%
- Stable and healthy

**Improvement:** +5-10pp assessment, +20-30pp pattern learning, healthier architecture
