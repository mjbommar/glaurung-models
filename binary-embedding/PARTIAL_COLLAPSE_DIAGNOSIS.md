# Partial Model Collapse Diagnosis - Step 187k

**Date:** 2025-10-27
**Run:** `micro_20251024_105559`
**Current Step:** 187,401 / 400,000 (46.9%)

## Executive Summary

The model is experiencing **partial selective collapse** - not the catastrophic uniform collapse seen in old NANO, but degradation in specific capabilities (pattern recognition) while improving in others (context understanding, embeddings). This is caused by an excessively long LR plateau (60% of training) keeping the model at full LR for 160k steps past its performance peak.

## Evidence

### Assessment Breakdown (Step 80k → 180k)

| Metric | @ 80k | @ 180k | Change | Status |
|--------|-------|--------|--------|--------|
| **Average** | 48.9% | 44.6% | -4.3pp | ⚠️ DOWN (-8.7%) |
| **Pattern Learning** | **92.2%** | **60.0%** | **-32.2pp** | 🔴 **COLLAPSE (-34.9%)** |
| **File Header** | 34.0% | 28.0% | -6.0pp | ⚠️ DOWN (-17.6%) |
| **Context Understanding** | 50.0% | 68.0% | +18.0pp | ✅ UP (+36.0%) |
| **Embedding Quality** | 19.2% | 22.3% | +3.1pp | ✅ UP (+16.3%) |

### Layer Health (SVD Analysis, Step 80k → 180k)

| Layer | @ 80k | @ 180k | Change | Status |
|-------|-------|--------|--------|--------|
| **L2 Query** | **61.8%** | **50.4%** | **-11.4pp** | 🔴 **COLLAPSING (-18.4%)** |
| **L2 Key** | 62.7% | 57.6% | -5.1pp | ⚠️ DRIFTING (-8.1%) |
| **L2 Value** | 80.4% | 81.0% | +0.6pp | ✅ STABLE (+0.7%) |
| **L1 Query** | 68.9% | 63.1% | -5.8pp | ⚠️ DRIFTING (-8.4%) |
| **Embeddings** | 95.4% | 94.3% | -1.1pp | ✅ STABLE (-1.2%) |

### Loss vs Assessment Divergence

```
Loss:       9.00 → 3.21 (↓64.3%) ✅ Still improving
Assessment: 48.9% → 44.6% (↓8.7%) ⚠️ Declining
```

**This divergence is the key symptom of overspecialization.**

## Root Cause Analysis

### 1. Plateau Too Long ⏱️

```
Current schedule:
  Warmup:  1,000 steps (0.25%) ✅ Fixed correctly
  Plateau: 239,000 steps (60%) ⚠️ TOO LONG
  Decay:   160,000 steps (40%)

Optimal schedule:
  Warmup:  1,000 steps (0.25%)
  Plateau: 40-60k steps (10-15%)
  Decay:   Remainder (85-90%)
```

**Problem:** Model peaked @ 80k but remained at full LR for 100k more steps, causing overspecialization.

### 2. Overspecialization 🔄

At high LR for extended periods, the model overoptimizes certain objectives at the expense of others:

- **Query layers** (pattern matching) → Degrading
- **Value layers** (contextual info) → Improving
- **Loss** → Improving (but on different objectives than we care about)

### 3. Differential Layer Stress 📉

Different layer types respond differently to extended high-LR training:

- **Query layers:** Sensitive to high LR, degrade when overtrained
- **Value layers:** Robust to high LR, can improve with more training
- **Result:** Model trades pattern capability for context capability

## Comparison: Old NANO vs New MICRO

### Old NANO (Catastrophic Collapse)
```
Issue:       32 dims/layer insufficient
Symptom:     Uniform collapse across all layers
Layer 3:     49.4% → 41.6% (↓7.8%)
Assessment:  Stuck ~32%, never improved
Result:      Training broken, irrecoverable
```

### New MICRO (Partial Selective Collapse)
```
Issue:       60% plateau too long
Symptom:     Selective collapse in query layers
L2 Query:    61.8% → 50.4% (↓11.4%)
L2 Value:    80.4% → 81.0% (↑0.6%) ✅ Healthy!
Result:      Functional but losing specific capabilities
```

**Key Difference:** Architecture fix (64 dims/layer) prevented catastrophic collapse, but schedule issue caused selective degradation.

## Correlation Analysis

### Pattern Learning Collapse ↔ Query Layer Collapse

```
Pattern learning: 92.2% → 60.0% (↓34.9%)
    ⬇
L2 Query rank:    61.8% → 50.4% (↓18.4%)
L1 Query rank:    68.9% → 63.1% (↓8.4%)
```

**Query layers in final layers are critical for pattern recognition. As they degrade, pattern learning collapses.**

### Context Improvement ↔ Value Layer Stability

```
Context:        50.0% → 68.0% (↑36.0%)
    ⬇
L2 Value rank:  80.4% → 81.0% (↑0.7%)
```

**Value layers remain healthy, maintaining/improving context understanding.**

## Current Status

### State: 🟡 STRESSED BUT FUNCTIONAL

- Not catastrophically broken like old NANO
- Still above critical thresholds (50.4% vs old NANO's 41.6%)
- Loss improving, embeddings healthy
- Value layers robust

### Risk Level: 🟠 MODERATE

- 53k steps @ full LR remain before decay starts
- Pattern learning may degrade further (60% → 50%?)
- Query layers may drift more (50.4% → 45%?)

### Recovery Potential: ✅ GOOD

- LR decay starts in ~20 hours (step 240,400)
- Value layers healthy to support recovery
- 160k steps of decay/fine-tuning remaining
- Model still functional, not catastrophically broken

## Prognosis

### Scenario A: Most Likely (60% probability)

```
Steps 180k-240k: Continued degradation
  - Pattern: 60% → 50%
  - Query: 50.4% → 45%

Steps 240k-300k: Stabilization during LR decay
  - Drift arrests
  - Metrics stabilize

Steps 300k-400k: Partial recovery
  - Pattern: 50% → 55%
  - Query: 45% → 48%

Final Model: Functional but weaker on pattern tasks than @ 80k
```

### Scenario B: Optimistic (25% probability)

```
LR decay immediately arrests degradation
Pattern/query metrics stabilize quickly
Minimal further degradation before decay starts
Final Model: Near-peak performance (pattern 55-60%)
```

### Scenario C: Pessimistic (15% probability)

```
53k more steps @ full LR cause severe further collapse
Query layers approach critical threshold (~45%)
LR decay stabilizes but can't recover much
Final Model: Significantly impaired (pattern 40-45%)
```

## Recommendations

### 1. Continue Current Training ✅

**Rationale:**
- Only 20 hours until LR decay
- Need to see if decay phase recovers performance
- Valuable data for understanding long-plateau effects
- Model still functional, worth completing

### 2. For Future Runs: Shorter Plateau 🔄

**Option A: Trapezoidal with 15% plateau**
```bash
uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size micro \
  --max-steps 400000 \
  --warmup-steps 1000 \
  --scheduler-type trapezoidal \
  --plateau-ratio 0.15 \  # Plateau ends @ 60k
  --learning-rate 3e-4 \
  ...
```

**Option B: Cosine schedule (no plateau)**
```bash
uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size micro \
  --max-steps 400000 \
  --warmup-steps 1000 \
  --scheduler-type cosine \  # Smooth decay, no plateau
  --learning-rate 3e-4 \
  ...
```

### 3. Monitor Closely 📊

- Check assessments every 10k steps
- Run SVD at 190k, 200k, 210k to track query layer health
- Watch for pattern learning dropping below 50%
- If query layers drop below 45%, consider early stopping

## Key Learnings

### 1. Architecture Fix Worked ✅

The 64 dims/layer ratio successfully prevented catastrophic uniform collapse:
- Old NANO: All layers collapsed uniformly (irrecoverable)
- New MICRO: Selective collapse in stressed layers, stable layers remain healthy

### 2. Schedule Issue Identified ⚠️

The 60% plateau was too aggressive for this model/task:
- Model peaked @ 80k steps (~20% of training)
- Continuing at full LR for 160k more steps caused overspecialization
- Future runs should use 10-15% plateau or cosine schedule

### 3. Query vs Value Layer Differential 📊

Different attention components have different robustness:
- **Query layers:** Sensitive, degrade under extended high-LR training
- **Value layers:** Robust, can improve with more training
- This differential explains selective collapse pattern

### 4. Loss-Assessment Divergence as Early Warning 🚨

Loss improving while assessments declining is a key indicator:
- Signals overspecialization to loss-optimized objectives
- May not align with downstream task performance
- Should be monitored as early warning signal

## Conclusion

We have identified and diagnosed **partial model collapse due to overspecialization**. This is distinct from the catastrophic collapse in old NANO:

- **Architecture (64 dims/layer):** ✅ Working correctly, prevented catastrophic failure
- **Warmup (1k steps):** ✅ Working correctly
- **Plateau (60%):** ⚠️ Too long, causing selective degradation

The model is still functional and has good recovery potential through the upcoming LR decay phase. The architecture fix successfully prevented the worst-case scenario, but the schedule needs adjustment for future runs.

**Bottom line:** Continue training to completion, use shorter plateau (10-15%) or cosine schedule for next run.
