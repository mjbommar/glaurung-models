# MICRO v3 Training Scripts - Rationale & Recommendations

## Executive Summary

Created two new training scripts based on lessons learned from v2's partial collapse:

1. **`run_micro_v3.sh`** (RECOMMENDED) - Balanced approach
2. **`run_micro_v3_aggressive.sh`** - Maximum stability (use if v3 fails)

## What Went Wrong in v2

### Primary Issue: 60% Plateau Too Long

```
v2 schedule:
  Warmup:  1,000 steps (0.25%) ✅
  Plateau: 239,000 steps (60%) ❌ TOO LONG
  Decay:   160,000 steps (40%)

Result:
  - Peaked @ 80k steps (20% of training)
  - Stayed at full LR (3e-4) for 160k MORE steps
  - Query layers collapsed: 61.8% → 50.4% rank (-18%)
  - Pattern learning collapsed: 92% → 60% (-35%)
  - Assessment declined: 49% → 38-44%
```

### Secondary Issue: Batch Size Slightly Low

```
v2 batch: 544 sequences = 278K tokens
Standard MLM recommendation: 500K - 2M tokens
```

Lower batch + extended high LR = unstable gradients → query layer collapse

## Key Insights from v2 Data

### What Worked ✅

1. **Architecture (64 dims/layer)** - Prevented catastrophic collapse
   - Old NANO: All layers collapsed uniformly (irrecoverable)
   - New MICRO: Only query layers stressed, value layers healthy

2. **Warmup (1000 steps)** - Fast convergence
   - Peaked @ 80k vs old NANO's 29k

3. **Learning Rate (3e-4)** - Good initial performance
   - 49% peak @ 80k
   - Problem was holding it too long, not the magnitude

4. **Weight Decay (5e-5)** - No issues

### What Failed ❌

1. **Plateau Duration** - 60% was too aggressive
   - Should be 10-20% or use cosine

2. **Batch Size** - 278K tokens slightly low
   - Below 500K recommendation
   - Contributed to instability at high LR

### Differential Layer Behavior 📊

```
ROBUST layers (improved or stable):
  - L2 Value:   80.4% → 81.0% (↑0.6%)
  - Embeddings: 95.4% → 94.3% (↓1.1%, minimal)
  - Context understanding: 50% → 68% (↑36%)

VULNERABLE layers (collapsed):
  - L2 Query:  61.8% → 50.4% (↓18%)
  - L1 Query:  68.9% → 63.1% (↓8%)
  - Pattern learning: 92% → 60% (↓35%)
```

**Insight:** Query layers handle pattern matching and are sensitive to extended high-LR training. Value layers handle context and are robust.

## v3 Changes Explained

### 1. Scheduler: Trapezoidal → Cosine (PRIMARY FIX)

#### Why Cosine?

**Trapezoidal issues:**
- Sharp transitions (warmup → plateau → decay)
- Requires guessing optimal plateau duration
- v2 showed plateau was too long

**Cosine benefits:**
- Smooth continuous decay (no sharp transitions)
- Natural adaptation throughout training
- Popular in modern LLM training (GPT, BERT, LLaMA)
- Conservative - less risky

**Schedule comparison:**

```
v2 Trapezoidal (60% plateau):
  0-1k:     Warmup 0 → 3e-4
  1k-240k:  Plateau at 3e-4 (239k steps) ← TOO LONG
  240k-400k: Decay 3e-4 → 0

v3 Cosine:
  0-1k:     Warmup 0 → 3e-4
  1k-400k:  Smooth cosine decay 3e-4 → 0

  @ 80k:    ~2.4e-4 (80% of peak, vs v2's 100%)
  @ 160k:   ~1.5e-4 (50% of peak, vs v2's 100%)
  @ 240k:   ~0.9e-4 (30% of peak, vs v2's decay start)
  @ 320k:   ~0.3e-4 (10% of peak)
```

**Expected impact:**
- LR naturally decays soon after peak (80k)
- Query layers experience less stress
- Should maintain stability throughout training

### 2. Batch Size: 544 seqs → 768 seqs (SECONDARY FIX)

#### Why Larger Batch?

**Problem with small batch + high LR:**
- Noisy gradients
- More variance in updates
- Harder for layers to converge smoothly

**Benefit of larger batch:**
- More stable gradients (averaged over more examples)
- Smoother convergence
- Better for high learning rates
- Aligns with standard MLM recommendations

**Size choice:**

```
Current (v2):   544 seqs × 512 tokens = 278K tokens  (too low)
Standard range:                         500K - 2M tokens

v3 Options:
  Conservative:   768 seqs = 393K tokens  (closer, 1.5x time) ← RECOMMENDED
  Aggressive:    2048 seqs = 1M tokens    (optimal, 4x time)
```

**Tradeoff analysis:**

```
Configuration    Batch    Tokens    Training Time    Stability
v2               544      278K      150 hrs (6.3d)   Low (collapsed)
v3 standard      768      393K      226 hrs (9.4d)   Good (expected)
v3 aggressive   2048      1.0M      603 hrs (25.1d)  Maximum
```

**Recommendation:** Start with v3 standard (768 seqs). Use aggressive only if standard still shows instability.

### 3. Keep Same: LR, Warmup, Architecture ✅

**Learning rate: 3e-4**
- Worked great initially (49% peak)
- Problem was duration, not magnitude
- With cosine, it'll decay naturally

**Warmup: 1000 steps**
- Fast convergence (peaked @ 80k)
- No issues observed

**Architecture: 64 dims/layer**
- Prevented catastrophic collapse
- Only stressed specific layers (query)
- Value layers remained healthy

## Recommended Training Strategy

### Phase 1: Let v2 Complete (Currently @ 187k)

**Why:**
- Only 20 hours until LR decay starts (240k)
- Need data on whether decay recovers performance
- Valuable learning about long-plateau effects

**Monitor:**
- Pattern learning (currently 60%, watch for drop to 50%)
- L2 query rank (currently 50.4%, critical threshold ~45%)

### Phase 2: Start v3 Standard

**Script:** `run_micro_v3.sh`

**Expected outcomes:**
```
@ 80k:   45-50% assessment (similar to v2's 49%)
@ 160k:  45-50% assessment (vs v2's 44% decline)
@ 240k:  45-50% assessment (vs v2's 38-42% expected)
@ 400k:  45-50% assessment (stable throughout)

Pattern learning: 70-90% throughout (vs v2's 92% → 60%)
L2 query rank: >55% throughout (vs v2's 61.8% → 50.4%)
```

**Success criteria:**
- Assessments stay 45-50% (no decline)
- Pattern learning >70%
- Query layers >55% rank
- No collapse in any metric

### Phase 3: If Needed, Try v3 Aggressive

**Script:** `run_micro_v3_aggressive.sh`

**When to use:**
- If v3 standard still shows instability
- If query layers still degrade
- If pattern learning still drops

**Changes:**
- Batch: 2048 seqs (1M tokens) - maximum stability
- LR: 2.5e-4 (lower than 3e-4) - extra conservative
- Still cosine schedule

**Tradeoff:**
- 4x training time (25 days vs 6 days)
- But maximum stability

## Comparison Table

| Metric | v2 (Current) | v3 Standard | v3 Aggressive |
|--------|--------------|-------------|---------------|
| **Schedule** | Trap 60% | Cosine | Cosine |
| **Batch** | 544 seq (278K) | 768 seq (393K) | 2048 seq (1M) |
| **LR** | 3e-4 | 3e-4 | 2.5e-4 |
| **Training Time** | 6.3 days | 9.4 days | 25.1 days |
| **Stability** | Low (collapsed) | Good (expected) | Maximum |
| **@ 80k Peak** | 49% | 45-50% | 45-50% |
| **@ 180k** | 38-44% (decline) | 45-50% (stable) | 45-50% (stable) |
| **Pattern** | 92% → 60% | 70-90% | 70-90% |
| **L2 Query** | 61.8% → 50.4% | >55% | >55% |

## Additional Considerations

### Alternative: Trapezoidal with 15% Plateau

If you prefer trapezoidal over cosine:

```bash
--scheduler-type trapezoidal
--plateau-ratio 0.15

Schedule:
  0-1k:    Warmup
  1k-60k:  Plateau (15% of training)
  60k-400k: Decay (85% of training)
```

**Pros:**
- Familiar (same as v2, just shorter)
- Explicit control over plateau duration
- Plateau ends right around peak (80k)

**Cons:**
- Sharp transition @ 60k
- Still requires guessing optimal plateau
- Cosine is more popular in modern training

**My recommendation:** Stick with cosine for v3. Save trapezoidal 15% as backup option.

### Monitoring During Training

**Critical metrics to watch:**

1. **Pattern learning** - Most sensitive indicator
   - v2: 92% → 60% (collapsed)
   - v3 target: >70% throughout

2. **L2 query rank** - Direct layer health
   - v2: 61.8% → 50.4% (stressed)
   - v3 target: >55% throughout

3. **Assessment average** - Overall performance
   - v2: 49% → 38-44% (declined)
   - v3 target: 45-50% stable

4. **Loss** - Should decrease smoothly regardless
   - v2: Decreased throughout (even during collapse)
   - Not sufficient alone - must check assessments!

**Red flags:**
- Pattern learning drops below 60%
- Query rank drops below 50%
- Assessments decline >5pp from peak

**If red flags appear:**
- Consider early stopping
- Switch to v3 aggressive for next run
- May need to revisit architecture (though 64 dims/layer worked)

## Theoretical Foundations

### Why Cosine Schedule Works

**Learning dynamics:**
- Early training: Large updates needed for coarse features
- Mid training: Medium updates for fine-tuning
- Late training: Small updates for refinement

**Cosine properties:**
- Smooth continuous decay matches learning dynamics
- No sharp transitions that could destabilize
- Popular in GPT-3, BERT-large, RoBERTa, LLaMA

**Mathematical form:**
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))

Where:
  t = current step
  T = total steps
  lr_max = peak LR (3e-4)
  lr_min = final LR (0)
```

### Why Larger Batch Helps

**Gradient variance:**
```
Var(gradient) ∝ 1 / batch_size

Small batch (544):  Higher variance → noisier updates
Large batch (768):  Lower variance → smoother updates
Large batch (2048): Lowest variance → smoothest updates
```

**Tradeoff:**
- Too small: Noisy, unstable, but explores more
- Too large: Stable, but may converge to suboptimal minima
- Sweet spot: 500K-2M tokens (standard for transformers)

### Why Query Layers Are Vulnerable

**Attention mechanism:**
```
Attention(Q, K, V) = softmax(QK^T / √d) * V

Query (Q): "What am I looking for?" - Pattern matching
Key (K):   "What features do I have?"
Value (V): "What content do I carry?" - Contextual info
```

**Under high LR stress:**
- Query matrices must precisely match patterns
- Small changes → large attention shifts
- More sensitive to noise and drift

- Value matrices carry semantic content
- Robust to moderate changes
- Less sensitive to optimization stress

**Result:** Query layers degrade first under extended high-LR training.

## Conclusion

**Primary fix:** Cosine schedule (replaces 60% plateau)
**Secondary fix:** Larger batch (768 seqs, 393K tokens)
**Keep:** LR 3e-4, warmup 1k, architecture 64 dims/layer

**Expected outcome:** Stable training with 45-50% performance throughout, no query layer collapse.

**Recommendation:** Start with `run_micro_v3.sh` after v2 completes. Monitor closely. Use aggressive version only if needed.
