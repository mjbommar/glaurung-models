# MICRO Phase 2: Contrastive Fine-Tuning Plan

## Executive Summary

Phase 1 (v6) achieved **38.3% average** @ 150k steps but showed **classic overfitting**:
- MLM loss kept improving (5.11 → 5.04)
- Assessment scores kept declining (41% → 36.6%)
- **Pattern learning STRONG** (63.3%) ✅
- **Embedding quality CRITICAL** (24.9%) 💀

**Solution:** Phase 2 contrastive fine-tuning to optimize embedding space while maintaining pattern learning.

**Expected Outcome:** 38% → 45%+ average by improving embeddings (25% → 35%+) while keeping patterns (60%+).

---

## Problem Analysis

### Phase 1 (v6) Performance Breakdown @ 150k

| Task | Score | Status | What It Measures |
|------|-------|--------|------------------|
| Pattern Learning | 63.3% | ✅ STRONG | Token-level predictions (MLM's strength) |
| Embedding Quality | 24.9% | 💀 CRITICAL | Semantic similarity of representations |
| File Header | 25.0% | ⚠️ WEAK | Understanding binary file types |
| Context | 40.0% | ⚠️ MODERATE | Long-range dependencies |
| **Average** | **38.3%** | **TARGET** | Overall performance |

### Root Cause: Training Objective Mismatch

**MLM (Masked Language Modeling):**
- Optimizes: "Predict the correct token at each position"
- Good at: Pattern matching, local context
- **Does NOT optimize:** Global embedding quality, semantic similarity
- Result: Model learned patterns (63.3%) but embeddings aren't semantically meaningful (24.9%)

**What We Need:**
- Maintain pattern learning (it's working!)
- Improve embedding quality (it's broken!)
- Better semantic understanding (headers, context)

---

## Solution: Contrastive Learning

### Why Contrastive?

**1. Different Training Signal**
- **MLM:** Token A should predict token B (local)
- **Contrastive:** Chunk A and Chunk B from same binary should have similar embeddings (global)

**2. Directly Optimizes Embeddings**
- Pulls together: Similar items (same file, duplicate views)
- Pushes apart: Dissimilar items (different files)
- Creates semantically meaningful embedding space

**3. Research-Validated (2024-2025)**

| Model | Approach | Result |
|-------|----------|--------|
| **NeoBERT** (June 2025) | Contrastive learning | 66.60 MTEB score (SOTA <250M params) |
| **VIKCSE** (May 2025) | Contrastive + prompts | +2.79% semantic similarity improvement |
| **SimCSE** | Contrastive fine-tuning | Fixes BERT's "anisotropy problem" |

**Key Finding:** "Contrastive learning can alleviate the anisotropy problem and significantly improve sentence representation performance." (2024 research)

**4. Multi-Task Prevents Catastrophic Forgetting**
- MLM weight = 1.0 → Maintains pattern learning
- Contrastive weights = 0.5 each → Adds embedding optimization
- Research shows MLM auxiliary loss helps avoid forgetting token-level knowledge

---

## Phase 2 Configuration

### Training Setup

```bash
# Starting Point
--init-from-checkpoint models/micro-v6-best/  # 150k steps, 38.3% avg

# Training Duration
--max-steps 75000           # Shorter than Phase 1 (fine-tuning, not pretraining)
--warmup-steps 5000         # Short warmup

# Learning Rate (CRITICAL - much lower!)
--learning-rate 0.00001     # 1e-5 (vs Phase 1's 3e-4)
                            # This is fine-tuning, not pretraining!

# Scheduler
--scheduler-type trapezoidal
--plateau-ratio 0.30        # 30% plateau (less aggressive than v6's 10%)
                            # Schedule: 0-5k warmup, 5k-27k plateau, 27k-75k decay

# Batch Configuration (same as Phase 1)
--batch-size 34
--gradient-accumulation-steps 16  # Effective batch: 544
--weight-decay 0.00005
--max-grad-norm 4.0
```

### Loss Configuration (Multi-Task)

```bash
# MLM Loss (maintain pattern learning)
--mlm-weight 1.0

# Contrastive Losses (improve embeddings)
--view-weight 0.5           # Duplicate-view pairs
--samefile-weight 0.5       # Same-file pairs
--contrastive-temp 0.07     # InfoNCE temperature
```

**What Each Loss Does:**

1. **MLM Loss (weight=1.0)**
   - Predicts masked tokens
   - Maintains pattern learning (63.3% score)
   - Prevents forgetting Phase 1 knowledge

2. **View Contrastive Loss (weight=0.5)**
   - Takes same chunk with different masks
   - Learns: "Different views of same data should be similar"
   - Improves: Robust representations, embedding quality

3. **Same-File Contrastive Loss (weight=0.5)**
   - Takes different chunks from same binary
   - Learns: "Different parts of same file should be related"
   - Improves: File-level understanding, semantic grouping

### Data Configuration

```bash
--pooling mean              # Mean pooling for embeddings
--dup-prob 0.5              # 50% duplicate pairs, 50% same-file pairs
--chunk-size 4096           # 4KB chunks
--min-chunk-separation 4096 # Same-file chunks must be 4KB apart
--streaming                 # Use streaming dataloader for pairs
```

---

## Expected Training Progression

### Assessment Milestones

| Step | Expected Avg | Embedding | Pattern | Notes |
|------|-------------|-----------|---------|-------|
| 0 (start) | 38.3% | 24.9% | 63.3% | Phase 1 checkpoint |
| 10k | 39-40% | 26-28% | 62-64% | Early improvement, embedding quality starting to rise |
| 25k | 41-42% | 29-32% | 60-62% | Mid-training, approaching plateau end |
| 50k | 43-45% | 33-35% | 59-61% | Late training, embeddings should be much better |
| 75k | 44-46% | 34-36% | 58-60% | Final, best checkpoint likely 50-75k |

### Key Indicators of Success

**✅ Success Signals:**
1. Embedding quality: 25% → 35%+ (primary goal)
2. Pattern learning: Stays 58-64% (no catastrophic forgetting)
3. Overall average: 38% → 44%+ (combined improvement)
4. File header: 25% → 30%+ (better semantics)
5. Loss curves: MLM stable, contrastive losses decreasing

**⚠️ Warning Signals:**
1. Pattern learning drops below 55% (too much forgetting)
2. Embedding quality doesn't improve by 20k (weights too low?)
3. All losses increasing (LR too high, overfitting)

**💀 Failure Signals:**
1. Pattern learning collapses (<50%)
2. Embedding quality gets worse (<25%)
3. Overall average below Phase 1 (<38%)

### What to Do at Each Milestone

**@ 10k steps:**
- Check: Embedding quality should be 26-28% (up from 24.9%)
- If not improving: Consider increasing contrastive weights (0.5 → 0.75)
- If pattern dropping fast: Decrease contrastive weights (0.5 → 0.3)

**@ 25k steps:**
- Check: Embedding quality should be 29-32%
- Check: Pattern learning should be 60-62%
- If both good: Continue to end
- If embedding stalled: May have peaked, check if best checkpoint

**@ 50k steps:**
- Check: Embedding quality should be 33-35%
- This is likely near peak performance
- Compare to Phase 1: Should be clearly better

**@ 75k steps:**
- Final checkpoint
- If peak was earlier (40-60k), use that checkpoint instead
- Document best checkpoint and performance

---

## Why NOT Another MLM Pass?

### Tempting but Wrong

"We already have MLM working, why not just do it again with better settings?"

**Problems:**

1. **Already overfit to MLM objective**
   - v6 saw 81.6M sequences doing MLM
   - Loss kept improving but scores declined
   - Model stuck in local optimum

2. **MLM doesn't optimize embeddings**
   - MLM objective = predict tokens (pattern matching)
   - Embedding quality = semantic similarity (different goal)
   - No amount of MLM will fix embeddings

3. **Diminishing returns**
   - Pattern learning already at 63.3%
   - More MLM won't improve it much (maybe 63% → 65%?)
   - But embedding quality (24.9%) is the bottleneck

4. **Waste of compute**
   - Another 150k steps of MLM = 67 more hours
   - Likely outcome: 38% → 40% at best
   - Phase 2 expected: 38% → 45%+ in 75k steps

### If You Still Want to Try MLM

If you absolutely must try another MLM pass (not recommended), you would need:

1. **Much stronger regularization**
   - Weight decay: 5e-5 → 1e-4 (2× stronger)
   - Add dropout: 0.1 to all layers
   - Lower batch size: 544 → 256

2. **Drastically lower LR**
   - Current: 2e-4 decaying
   - Need: 5e-6 constant (100× lower!)

3. **Different data distribution**
   - Filter out easy patterns (high entropy)
   - Focus on harder examples

But this is fighting the wrong battle. MLM did its job. Fix the embeddings.

---

## Implementation Details

### Script Usage

```bash
# Run the training
./run_micro_phase2_contrastive.sh

# Monitor with tail
tail -f micro_phase2_contrastive_*/mlm_training_log.jsonl

# Check assessments
grep '"event": "assessment"' micro_phase2_contrastive_*/mlm_training_log.jsonl | tail -20
```

### WandB Monitoring

Project: `glaurung-micro-phase2`
Run name: `micro_phase2_contrastive_{timestamp}`

**Key Metrics to Watch:**
1. `train/mlm_loss` - Should stay stable (not increase)
2. `train/view_contrastive_loss` - Should decrease steadily
3. `train/same_file_contrastive_loss` - Should decrease steadily
4. `assessment/embedding_quality_score` - Should increase (PRIMARY GOAL)
5. `assessment/pattern_learning_score` - Should stay 58-64% (maintenance)
6. `assessment/average_score` - Should increase to 44%+

### Checkpoints

Saved every 5k steps:
- `checkpoint-5000/`
- `checkpoint-10000/`
- `checkpoint-15000/`
- ...
- `checkpoint-75000/`

**Save total limit: 5** - Keeps only last 5 checkpoints

**Best checkpoint selection:**
- Compare embedding quality scores across all checkpoints
- Best checkpoint likely between 40k-60k steps
- Don't automatically use final checkpoint (may be overfit)

---

## Success Criteria

### Minimum Success (45th percentile)
- Embedding quality: 25% → 30% (+5 points)
- Overall average: 38% → 40% (+2 points)
- Pattern learning: Stays above 58%

### Target Success (75th percentile)
- Embedding quality: 25% → 35% (+10 points)
- Overall average: 38% → 44% (+6 points)
- Pattern learning: Stays above 60%

### Excellent Success (90th percentile)
- Embedding quality: 25% → 38%+ (+13 points)
- Overall average: 38% → 46%+ (+8 points)
- All tasks improve simultaneously
- Pattern learning: Maintains 60%+

---

## Troubleshooting

### Embedding Quality Not Improving

**Symptoms:** Still at 25-27% after 20k steps

**Causes:**
1. Contrastive weights too low (0.5)
2. LR too low (1e-5)
3. Temperature too low (0.07)

**Fixes:**
- Increase contrastive weights: 0.5 → 0.75 or 1.0
- Increase LR: 1e-5 → 2e-5 (careful!)
- Increase temperature: 0.07 → 0.1

### Pattern Learning Collapsing

**Symptoms:** Drops below 55%

**Causes:**
1. Contrastive weights too high
2. LR too high (1e-5)
3. Catastrophic forgetting

**Fixes:**
- Decrease contrastive weights: 0.5 → 0.3
- Decrease LR: 1e-5 → 5e-6
- Increase MLM weight: 1.0 → 1.5

### All Scores Getting Worse

**Symptoms:** Everything declining

**Causes:**
1. LR too high (overfitting)
2. Batch size wrong
3. Wrong checkpoint loaded

**Fixes:**
- Verify checkpoint loaded correctly
- Reduce LR: 1e-5 → 5e-6
- Check logs for errors

---

## References

### Research Papers

1. **NeoBERT: A Next-Generation BERT** (June 2025)
   - arXiv:2502.19587
   - Uses contrastive learning for embeddings
   - Achieves 66.60 MTEB score

2. **VIKCSE: Visual-knowledge enhanced contrastive learning** (May 2025)
   - +2.79% improvement on semantic textual similarity
   - Combines visual knowledge with contrastive learning

3. **SimCSE: Simple Contrastive Learning of Sentence Embeddings**
   - Fixes BERT's anisotropy problem
   - Standard approach for improving BERT embeddings

4. **Self-Supervised Contrastive Pre-Training For Time Series**
   - Multi-task learning prevents catastrophic forgetting
   - MLM auxiliary loss maintains token-level knowledge

### Key Insights from Research

> "The anisotropy of the learned embedding space prevents BERT sentence embeddings from achieving good results in semantic textual similarity tasks, and it has been shown that contrastive learning can alleviate the anisotropy problem and significantly improve sentence representation performance."

> "Multi-stage contrastive learning, task-specific LoRA adapters, and batch clustering with contextual corpus embeddings have all pushed the limits of the leaderboard."

> "Research shows an optional MLM auxiliary objective loss alongside contrastive learning to help avoid catastrophic forgetting of token-level knowledge."

---

## Conclusion

Phase 2 contrastive training is the right approach because:

1. ✅ **Addresses root cause:** MLM can't optimize embeddings, contrastive can
2. ✅ **Research-validated:** NeoBERT, VIKCSE, SimCSE all use this approach
3. ✅ **Infrastructure ready:** ContrastiveTrainer already implemented
4. ✅ **Balanced approach:** Multi-task learning maintains patterns while fixing embeddings
5. ✅ **Escape overfitting:** Different training signal gets model unstuck

Expected to improve overall performance from 38% to 45%+ by specifically targeting the weak embedding quality (25% → 35%+) while maintaining strong pattern learning (63% → 60%+).

**Next Steps:**
1. Run `./run_micro_phase2_contrastive.sh`
2. Monitor embedding quality metric closely
3. Compare to Phase 1 at equivalent compute
4. Document results for Phase 3 planning
