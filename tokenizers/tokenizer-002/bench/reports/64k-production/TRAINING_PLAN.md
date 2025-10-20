# Glaurung Tokenizer 002 - Final Production Training
## 65k Vocabulary on binaries-small Dataset

**Date**: October 19, 2025
**Purpose**: Production-ready binary tokenizer for Glaurung models
**Status**: Planning → Execution

---

## Training Configuration

### Dataset
- **Path**: `/nas4/data/glaurung-data/binaries-small/`
- **Size**: 13 GB
- **Files**: 30,738 binaries
- **Platforms**: Linux (Alpine, Debian, Ubuntu), Windows (8/10/11), ARM64, x86-64
- **Content**: Real-world compiled binaries across multiple architectures

### Parameters
```bash
--output bench/glaurung-tokenizer-002/glaurung-tokenizer-002.json
--vocab-size 65536
--min-frequency 4
--chunk-size 8192
# No --allowed-lengths flag → Uses DEFAULT (1-16, optimal mode)
```

### Rationale for 65k Vocabulary

**Why double from 32k to 65k?**

1. **Vocabulary size vs compression** (empirical):
   - 8k vocab: ~2.2 bytes/token
   - 16k vocab: ~2.5 bytes/token
   - 32k vocab: ~2.8 bytes/token
   - 65k vocab: ~3.1 bytes/token (estimated)
   - **Expected improvement**: +10-15% over 32k

2. **Rare pattern coverage**:
   - 32k captures common patterns (top 99%)
   - 65k captures rare but important patterns:
     - Less common instructions (SIMD, FPU)
     - Windows-specific patterns (PE headers)
     - ARM64 instructions (less frequent in corpus)
     - Debug info patterns
     - Exception handling tables

3. **Model size tradeoff**:
   - Embedding layer: 65k × hidden_dim
   - For 768-dim: 50M parameters (acceptable)
   - For 1024-dim: 67M parameters (still reasonable)
   - Modern GPUs can handle 65k vocab easily

4. **Industry standard**:
   - GPT-2: 50,257 tokens
   - GPT-3/4: ~100k tokens (includes byte fallback)
   - LLaMA: 32k tokens (text)
   - **65k is sweet spot for binary data**

5. **Diminishing returns analysis**:
   - 32k → 65k: Good ROI (expect +10-15% improvement)
   - 65k → 128k: Marginal (expect +3-5% improvement)
   - **65k is optimal price/performance**

---

## Expected Performance

### Based on 32k Training Results

**32k training metrics**:
- Iterations: 32,505
- Time: 4.66 hours (16,775 seconds)
- Rate: 1.94 tokens/second
- Peak RAM: 67.8 GB
- Distinct pairs: 83.98M at completion

**65k projections**:
- Iterations: ~65,000 (target vocab - 256 base - 7 special)
- Expected rate: 1.5-1.8 tokens/second (slower due to larger heap)
- **Estimated time**: 10-12 hours
- **Expected RAM**: 90-100 GB peak
- **Expected distinct pairs**: 150-200M

### Training Timeline

| Milestone | Expected Time | Vocab Progress |
|-----------|---------------|----------------|
| Start | T+0h | 256 (base) |
| 25% complete | T+2.5h | ~16,384 |
| 50% complete | T+5h | ~32,768 |
| 75% complete | T+8h | ~49,152 |
| **Completion** | **T+10-12h** | **65,536** |

**Expected completion**: ~10:00 PM EDT (if started at 10:00 AM)

---

## Monitoring Strategy

### Automated Monitoring
- Check progress every 30 minutes
- Alert if:
  - Training stalls (no progress for 1 hour)
  - Memory exceeds 110 GB (approaching system limits)
  - Process crashes

### Manual Checkpoints
- **T+1h**: Verify training progressing normally
- **T+3h**: Check if on pace (should be ~20k vocab)
- **T+6h**: Midpoint check (should be ~32k vocab)
- **T+9h**: Final stretch check (should be ~50k vocab)

### Success Criteria
- ✓ Reaches 65,536 vocab (±20 tokens)
- ✓ Training completes without errors
- ✓ Memory usage stays under 110 GB
- ✓ Iteration times remain <1 second on average
- ✓ Final tokenizer file is valid JSON

---

## Resource Requirements

### Minimum Requirements
- **RAM**: 100 GB available
- **CPU**: 16+ cores (will use ~20-24)
- **Disk**: 5 GB free (for output + logs)
- **Time**: 12 hours uninterrupted

### Expected Usage
- **RAM**: 90-100 GB peak (gradual increase)
- **CPU**: 2000-2500% (20-25 cores at 100%)
- **Disk I/O**: Moderate (reading corpus once)
- **Network**: None (local training)

---

## Risk Assessment

### Risk 1: Out of Memory (Moderate)
**Probability**: 20%
**Impact**: Training crash, restart needed
**Mitigation**:
- System has 128 GB RAM
- 65k vocab used 68 GB for 32k → estimate 90-100 GB
- **Mitigation**: Monitor memory, reduce chunk_size if needed

### Risk 2: Training Time Exceeds 15 Hours (Low)
**Probability**: 10%
**Impact**: Delays delivery
**Mitigation**:
- Conservative estimate is 10-12h
- 32k took 4.66h, scaling suggests ~10h
- **Mitigation**: None needed, just wait

### Risk 3: Disk Space Exhaustion (Very Low)
**Probability**: <5%
**Impact**: Training fails to save
**Mitigation**:
- Need ~2 MB for tokenizer file
- Need ~100 MB for training log
- Have 5+ GB available
- **Mitigation**: Pre-check disk space

### Risk 4: System Reboot/Power Loss (Low)
**Probability**: 5%
**Impact**: Complete restart needed
**Mitigation**:
- Training is single-pass, can't resume
- **Mitigation**: Use stable system, UPS recommended

---

## Post-Training Validation

### Immediate Checks (automated)
1. ✓ File exists and is valid JSON
2. ✓ Vocabulary size == 65,536 (±20)
3. ✓ Can load with tokenizers library
4. ✓ Token length distribution is reasonable

### Quality Checks (manual)
1. ✓ Compare compression with 32k tokenizer
2. ✓ Inspect top 100 tokens for sanity
3. ✓ Test on held-out binaries
4. ✓ Verify length-3 tokens present (breakthrough feature)

### Expected Results
- **Average token length**: 3.5-3.8 bytes
- **Length-3 tokens**: 15-20% of vocabulary
- **Compression improvement**: 10-15% better than 32k
- **File size**: ~2.0-2.5 MB

---

## Deliverables

### Primary Output
- **glaurung-tokenizer-002.json** (final production tokenizer)

### Supporting Files
- **TRAINING_LOG.txt** (complete training output)
- **VALIDATION_REPORT.md** (post-training analysis)
- **token_analysis.json** (token statistics)
- **compression_benchmarks.txt** (performance tests)

---

## Training Command

```bash
cd /home/mjbommar/src/glaurung-models/tokenizers/tokenizer-002

time cargo run --release --bin train -- \
  --output bench/glaurung-tokenizer-002/glaurung-tokenizer-002.json \
  /nas4/data/glaurung-data/binaries-small/ \
  --vocab-size 65536 \
  --min-frequency 4 \
  --chunk-size 8192 \
  2>&1 | tee bench/glaurung-tokenizer-002/TRAINING_LOG.txt
```

**Start time**: TBD
**Expected completion**: Start + 10-12 hours
**Priority**: High (blocking model training)

---

## Success Definition

Training is successful if:
1. ✅ Completes without errors
2. ✅ Produces valid 65k tokenizer
3. ✅ Achieves 10-15% better compression than 32k
4. ✅ Token distribution shows length-3 breakthrough
5. ✅ Ready for production model training

---

**Status**: READY TO EXECUTE
**Approval**: Pending user confirmation
