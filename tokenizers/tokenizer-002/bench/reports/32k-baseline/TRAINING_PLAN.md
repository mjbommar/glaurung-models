# Production Tokenizer Training Plan
## DEFAULT Mode (lengths 1-16) on binaries-small Dataset

### Dataset
- **Path**: `/nas4/data/glaurung-data/binaries-small/`
- **Size**: 13 GB
- **Files**: 30,738 binaries
- **Platforms**: Linux (Alpine, Debian, Ubuntu, Busybox), Windows (8/10/11), SOREL-20M samples
- **Architectures**: Primarily x86-64, some x86-32

### Old Tokenizer (Baseline)
- **Allowed lengths**: 1,2,4,8,16,32 (powers of 2 only)
- **Vocab size**: 32,761 tokens
- **Token distribution**:
  - 71.9% length-2 (highly compositional)
  - 23.8% length-4
  - 3.5% length-8
  - 0.6% length-16
  - 0.1% length-32
  - **0% length-3** ← Missing the sweet spot!
- **Average token length**: 2.81 bytes/token
- **File size**: 996 KB
- **Training time**: ~5 hours 9 minutes

### New Tokenizer (Target)
- **Allowed lengths**: 1-16 (ALL lengths, unrestricted)
- **Vocab size**: 32,768 tokens (target)
- **Expected distribution** (based on /usr/bin test):
  - ~28% length-2 (prefixes)
  - **~28% length-3** ← Capture x86-64 instruction sweet spot
  - ~16% length-4
  - ~28% length-5 to 16 (varied patterns)
- **Expected average**: ~4.2 bytes/token (+49% vs old)
- **Expected file size**: ~1.2-1.4 MB
- **Expected training time**: ~5.5-6 hours (+10% overhead)
- **Expected compression improvement**: +21% (fewer tokens for same data)

### Training Parameters
```bash
cargo run --release --bin train -- \
  --output bench/binaries-small-32k/tokenizer-default-1-16.json \
  /nas4/data/glaurung-data/binaries-small/ \
  --vocab-size 32768 \
  --min-frequency 4 \
  --chunk-size 8192
```

**Rationale for each parameter**:

1. **No `--allowed-lengths` flag** → Uses DEFAULT (1..=16)
   - Allows tokenizer to discover natural instruction boundaries
   - Captures 3-byte x86-64 instructions
   - More flexible than constrained modes

2. **`--vocab-size 32768`** → Production-size vocabulary
   - Same as old tokenizer for fair comparison
   - Good balance between coverage and model size
   - Standard size for binary tokenization

3. **`--min-frequency 4`** → Filter noise, keep signal
   - For 13GB corpus, patterns appearing <4 times are likely noise
   - Focuses vocabulary on truly common patterns
   - Standard default value

4. **`--chunk-size 8192`** → Memory/performance trade-off
   - Breaks files into 8KB chunks for processing
   - Prevents memory issues with large files
   - Maintains pattern locality

5. **No `--plateau-stop`** → Let it reach full vocab
   - Disabled by default
   - For production, we want the full 32k vocabulary
   - Ensures maximum compression capability

### Key Hypotheses to Validate

1. **Compression Hypothesis**:
   - New tokenizer will use ~21% fewer tokens than old tokenizer
   - Test on held-out binaries not in training set

2. **Length-3 Hypothesis**:
   - ~28% of learned tokens will be length-3
   - These will be complete x86-64 instructions (REX + opcode + ModR/M)

3. **Average Length Hypothesis**:
   - Average token length will increase from 2.81 → ~4.2 bytes
   - This indicates better pattern capture

4. **Vocabulary Efficiency Hypothesis**:
   - Tokens will be more "information dense"
   - Each token captures more semantic meaning
   - Better generalization to unseen binaries

### Monitoring Strategy

During training:
- Monitor memory usage (should stay under 32GB)
- Check iteration times (should be ~1-2ms per iteration for large corpus)
- Verify progress toward 32,768 vocab target

After training:
- Analyze token length distribution
- Compare compression ratios on test set
- Examine top tokens for sensibility
- Validate against specific test cases (bash, python3.12, gcc)

### Success Criteria

**Must achieve**:
1. ✓ Vocab size: 32,768 ± 20 tokens
2. ✓ Training completes without errors
3. ✓ Tokenizer file is valid JSON and loads correctly

**Should achieve**:
1. ✓ Average token length: 3.8-4.5 bytes (vs 2.81 old)
2. ✓ Length-3 tokens: 20-35% of vocabulary
3. ✓ Compression improvement: 15-25% vs old tokenizer

**Nice to have**:
1. ✓ Training time: <6 hours
2. ✓ Memory usage: <32GB peak
3. ✓ Length-5,6,7 tokens: 15-20% combined (captures complex instructions)

### Post-Training Validation Plan

1. **Distribution Analysis**:
   ```bash
   python3 analyze_tokenizer.py tokenizer-default-1-16.json
   ```

2. **Compression Testing**:
   ```bash
   python3 compare_compression.py \
     tokenizer.json \
     tokenizer-default-1-16.json \
     /nas4/data/glaurung-data/binaries-small/debian/usr/bin/bash
   ```

3. **Token Inspection**:
   - Examine top 100 tokens by ID
   - Look for sensible x86-64 instruction patterns
   - Verify length-3 tokens are complete instructions

4. **Edge Case Testing**:
   - Test on Windows PE binaries
   - Test on ARM binaries (if any in corpus)
   - Test on packed/obfuscated samples

### Risk Mitigation

**Risk**: Training takes too long (>8 hours)
- **Mitigation**: Can interrupt and use partial vocabulary
- **Fallback**: Reduce vocab size to 16k or 8k

**Risk**: Memory usage exceeds available RAM
- **Mitigation**: Reduce chunk size from 8192 to 4096
- **Fallback**: Train on smaller subset first

**Risk**: New tokenizer performs worse than old
- **Mitigation**: Comprehensive testing showed 21% improvement
- **Fallback**: Keep old tokenizer as default, make new one optional

**Risk**: Token distribution doesn't match expectations
- **Analysis**: May indicate corpus differences vs /usr/bin sample
- **Action**: Investigate why (e.g., Windows binaries dominate corpus)

### Timeline

- **T+0h**: Start training
- **T+1h**: Check first progress update, verify memory usage
- **T+3h**: Midpoint check, estimate completion time
- **T+5-6h**: Training completes
- **T+6h**: Run validation suite
- **T+6.5h**: Generate comparison report
- **T+7h**: Decision on promotion to default

### Deliverables

1. `tokenizer-default-1-16.json` - New production tokenizer
2. `TRAINING_LOG.txt` - Full training output with metrics
3. `VALIDATION_REPORT.md` - Comparison with old tokenizer
4. `token_distribution.png` - Visualization of length distribution
5. Updated `config.rs` with new default (pending validation)

---

**Ready to execute: YES**

Command to run:
```bash
cd /home/mjbommar/src/glaurung-models/tokenizers/tokenizer-002 && \
  time cargo run --release --bin train -- \
    --output bench/binaries-small-32k/tokenizer-default-1-16.json \
    /nas4/data/glaurung-data/binaries-small/ \
    --vocab-size 32768 \
    --min-frequency 4 \
    --chunk-size 8192 \
    2>&1 | tee bench/binaries-small-32k/TRAINING_LOG.txt
```
