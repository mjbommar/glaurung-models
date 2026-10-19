# Hyperparameter Analysis for New Architecture

## Critical Finding from Current Run

**Peak occurred DURING warmup, not after:**
- Step 29k: 32.34% (LR = 1.74e-4, still in warmup)
- Step 50k: Warmup completes, LR = 3e-4
- Steps 50k-239k: Never exceeded warmup peak, volatile 20-32%

**But loss was still decreasing:**
- Step 1k: 10.97
- Step 239k: 4.55
- Smooth monotonic decrease throughout
- No instability

**Interpretation:** Assessment volatility is measurement noise, NOT training instability.

## Warmup Problem

### Old Script (BROKEN)
```
warmup_steps = 50,000
batch_size = 34
gradient_accumulation = 16
effective_batch = 544 sequences

Data seen during warmup: 50k × 544 = 27.2M sequences
```

### Best Experiment (exp06)
```
warmup_steps = 7,500
batch_size = 32
effective_batch = 32 sequences

Data seen during warmup: 7.5k × 32 = 240K sequences
```

### Ratio: 27.2M / 240K = 113× MORE data in warmup!

This is why warmup took forever. Model doesn't reach full LR until seeing 237 epochs of data.

## Optimal Warmup Calculation

**Rule:** Scale warmup inversely with batch size to maintain same data exposure.

```
warmup_new = warmup_old × (batch_old / batch_new)
warmup_new = 7,500 × (32 / 544)
warmup_new = 441 steps
```

Round up for stability: **500 or 1000 steps**

At 1000 steps × 544 batch = 544K sequences (2.3× exp06, still reasonable)

## Learning Rate Analysis

### Old Architecture (128/4 layers, 9.3M params)
- Best LR: 3e-4 (exp06: 35.85% peak)
- This was with batch=32

### New Architectures

**NANO (128/2 layers, 8.8M params):**
- Same hidden size as old NANO
- Fewer layers = fewer parameters
- Effective batch now 544 (17× larger)

LR scaling rules:
1. **Larger batch → higher LR** (linear or sqrt scaling)
2. **Same hidden size → same LR** (hidden size affects gradient magnitudes)
3. **Fewer params → slightly lower LR** (less capacity to absorb high LR)

Effects cancel out. **Recommendation: Start with 3e-4**

**MICRO (192/3 layers, 14M params):**
- Larger hidden (192 vs 128) = 1.5× scale
- More params (14M vs 9.3M) = 1.5× scale
- Same effective batch (544)

Larger models can typically handle slightly higher LR.
**Recommendation: Try 3e-4 or 3.5e-4**

**SMALL (256/4 layers, 20M params):**
- Much larger (2× params vs old NANO)
- Can handle higher LR
**Recommendation: Try 4e-4 or 5e-4**

### Conservative Approach

For all new architectures, **start with 3e-4** (proven safe).
Once training stability is confirmed, can sweep [3e-4, 4e-4, 5e-4] for optimal.

## Other Hyperparameters

### Gradient Accumulation: 16 ✓
- Proven stable in current run
- Effective batch 544 is good size
- No need to change

### Batch Size: 34 ✓
- Works well with gradient accumulation
- Fits in memory
- No need to change

### Weight Decay: 5e-5 ✓
- Proven optimal in Round 2
- No need to change

### Scheduler: Trapezoidal ✓
- Beat two-phase consistently
- Plateau ratio 0.8 (60% plateau) proven optimal
- No need to change

### Max Grad Norm: 4.0 ✓
- Current run stable
- Higher than old (2.0) to accommodate larger updates
- Keep at 4.0

### Max Steps: 400k
**Question:** Do new architectures need more/fewer steps?

Old NANO (4 layers):
- Peaked at 23k steps (exp06)
- Trained to 50k steps

New architectures (2-3 layers):
- Fewer parameters to tune → might converge faster
- But less capacity → might need more iterations

**Recommendation:** Keep 400k steps (safe, can early stop if converged)

## Final Recommendations

### NANO (128/2 layers, 8.8M)
```bash
LR=3e-4                 # Same as old NANO
WARMUP=1000             # Fix warmup (was 50k)
BATCH=34                # Keep
GRAD_ACCUM=16           # Keep
MAX_STEPS=400000        # Keep
WD=5e-5                 # Keep
MAX_GRAD_NORM=4.0       # Keep
```

### MICRO (192/3 layers, 14M) - **RECOMMENDED**
```bash
LR=3e-4                 # Conservative start (can try 3.5e-4)
WARMUP=1000             # Fix warmup
BATCH=34                # Keep
GRAD_ACCUM=16           # Keep
MAX_STEPS=400000        # Keep
WD=5e-5                 # Keep
MAX_GRAD_NORM=4.0       # Keep
```

### SMALL (256/4 layers, 20M)
```bash
LR=4e-4                 # Slightly higher for larger model
WARMUP=1500             # Bit more for stability
BATCH=34                # Keep
GRAD_ACCUM=16           # Keep
MAX_STEPS=500000        # More steps for larger model
WD=5e-5                 # Keep
MAX_GRAD_NORM=4.0       # Keep
```

## Expected Performance

Based on:
- Fixed architecture (no rank collapse)
- Fixed warmup (proper schedule)
- Proven hyperparameters

**Predictions:**

| Model | Peak (Conservative) | Peak (Optimistic) | Notes |
|-------|-------------------|-------------------|-------|
| NANO | 32-35% | 36-38% | Only 2 layers, limited |
| MICRO | 36-40% | 40-44% | Good balance |
| SMALL | 42-48% | 48-52% | More capacity |

These should be **stable** throughout training (no degradation).

## Migration from Old Script

**Key changes:**
1. ✅ Model: `nano` → `micro` (or new `nano`)
2. ✅ Warmup: 50000 → 1000 steps (113× reduction!)
3. ✅ LR: 3e-4 (keep, proven)
4. ✅ All other params: Keep

**Expected improvement:**
- Reach peak performance much faster (~10k steps vs ~30k)
- Stable rank throughout training
- Higher final performance (40%+ vs 32%)
