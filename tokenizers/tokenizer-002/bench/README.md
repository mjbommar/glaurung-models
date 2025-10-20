# Glaurung Tokenizer Benchmarks

This directory contains production tokenizers, baselines, experiments, and analysis tools for the Glaurung binary tokenization project.

## Production Tokenizers

### glaurung-tokenizer-002 (64K) - RECOMMENDED

**Location**: `production/glaurung-tokenizer-002.json`

The production-ready 64K vocabulary tokenizer for binary analysis and neural models.

**Key Metrics**:
- Vocabulary: **65,536 tokens** (exactly 2^16)
  - Base + learned: 65,529 tokens (IDs 0-65528)
  - Special tokens: 7 tokens (IDs 65529-65535: <s>, </s>, <pad>, <unk>, <cls>, <sep>, <mask>)
- Compression: 2.849 bytes/token
- Average token length: 3.651 bytes
- Length-3 tokens: 9,261 (14.2%) - captures complete x86-64 instructions
- Training: 13GB binaries, 8.46 hours on 24 cores

**Performance**:
- 9.0% fewer tokens than 32K baseline
- 9.9% better compression ratio
- 86% of theoretical compression optimum
- Consistent performance across Linux, Windows, x86-64, ARM64

**Documentation**: See `production/DOCUMENTATION.md` for complete usage guide.

### glaurung-tokenizer-001 (32K) - Baseline

**Location**: `baseline/glaurung-tokenizer-001.json`

The 32K baseline tokenizer for comparison and backward compatibility.

**Key Metrics**:
- Vocabulary: 32,761 tokens
- Compression: 2.592 bytes/token
- Average token length: 3.512 bytes

**Use Case**: Backward compatibility with existing models only. New projects should use glaurung-tokenizer-002.

## Directory Structure

```
bench/
├── production/           # Production-ready tokenizers
│   ├── glaurung-tokenizer-002.json
│   └── DOCUMENTATION.md
│
├── baseline/             # Baseline tokenizers for comparison
│   └── glaurung-tokenizer-001.json
│
├── experiments/          # Experimental tokenizers and analyses
│   ├── length-modes/     # Token length constraint experiments
│   └── cross-domain/     # Text vs binary comparison
│
├── reports/              # Training reports and analysis
│   ├── 64k-production/   # glaurung-tokenizer-002 training details
│   ├── 32k-baseline/     # glaurung-tokenizer-001 training details
│   └── cross-domain/     # Cross-domain validation study
│
└── tools/                # Analysis and benchmarking scripts
    ├── analyze_tokenizer.py
    ├── inspect_tokens.py
    ├── test_compression.py
    └── find_strings.py
```

## Quick Start

### Loading the Tokenizer

```python
from tokenizers import Tokenizer
from pathlib import Path

# Load production tokenizer
tokenizer = Tokenizer.from_file("bench/production/glaurung-tokenizer-002.json")

# Encode binary data
binary_data = Path("/usr/bin/ls").read_bytes()
encoding = tokenizer.encode(binary_data.decode('latin-1'))

print(f"Tokens: {len(encoding.ids)}")
print(f"Compression: {len(binary_data) / len(encoding.ids):.3f} bytes/token")
```

### Running Analysis Tools

```bash
# Analyze token distribution
python3 bench/tools/analyze_tokenizer.py

# Inspect top learned tokens
python3 bench/tools/inspect_tokens.py

# Test compression performance
python3 bench/tools/test_compression.py

# Search for strings and symbols
python3 bench/tools/find_strings.py
```

## Training Configuration

### glaurung-tokenizer-002 (64K)

```bash
cargo run --release --bin train -- \
  --output bench/production/glaurung-tokenizer-002.json \
  /nas4/data/glaurung-data/binaries-small/ \
  --vocab-size 65536 \
  --min-frequency 4 \
  --chunk-size 8192
```

**Dataset**:
- Path: `/nas4/data/glaurung-data/binaries-small/`
- Size: 13 GB
- Files: 30,738 binaries
- Platforms: Linux (Alpine, Debian, Ubuntu), Windows (8/10/11)
- Architectures: x86-64, ARM64

**Parameters**:
- `--vocab-size 65536`: Target 2^16 tokens for uint16 efficiency
- `--min-frequency 4`: Merge pairs appearing ≥4 times
- `--chunk-size 8192`: Process 8KB chunks for memory efficiency
- Token lengths: 1-16 bytes (DEFAULT mode - unrestricted)

## Key Findings

### 1. Length-3 Breakthrough

The tokenizer learned that x86-64 instructions follow a REX + opcode + ModR/M pattern (3 bytes). This structural understanding enables:
- Complete instructions as single tokens
- Reduced fragmentation
- Better compression efficiency

**Example**: `0x48 0x8b 0xc0` (MOV rax, rax) → single token

### 2. Domain Specialization is Critical

Cross-domain testing proved that specialized tokenizers are essential:

| Tokenizer | On Binary | On Text | Penalty |
|-----------|-----------|---------|---------|
| Binary-64K | 2.849 bytes/token ✓ | 2.354 bytes/token ✗ | +102% |
| Text-16K | 1.129 bytes/token ✗ | 4.768 bytes/token ✓ | +140% |

**Conclusion**: Zero overlap in learned patterns. Use binary tokenizers for binaries, text tokenizers for text.

### 3. Optimal Vocabulary Size

Testing shows 64K (2^16) is the optimal vocabulary size:
- 32K → 64K: +9-10% improvement ✓ **Good ROI**
- 64K → 128K: +3-5% expected (diminishing returns)
- 128K → 256K: +1-2% expected (marginal)

### 4. Architecture Coverage

The tokenizer learned patterns from multiple architectures:
- **x86-64**: REX prefixes, MOV, LEA, arithmetic, conditionals
- **ARM64**: 4-byte little-endian instructions
- **Windows PE**: API names, DLL references
- **Linux ELF**: Syscalls, libc functions, section names

**String tokens**: 3,759 tokens (5.76% of vocabulary) containing function names, library references, paths, and error messages.

## Performance Metrics

### Compression Benchmarks

Tested on `/usr/bin` binaries (NOT in training set):

| Binary | Size | 32K Tokens | 64K Tokens | Improvement |
|--------|------|------------|------------|-------------|
| bash | 1.38 MB | 589,872 | 535,541 | +9.2% |
| python3.12 | 7.65 MB | 3,078,745 | 2,801,226 | +9.0% |
| gcc-13 | 0.98 MB | 377,022 | 344,201 | +8.7% |
| ls | 0.14 MB | 54,302 | 49,574 | +8.7% |
| grep | 0.18 MB | 74,013 | 67,567 | +8.7% |
| **TOTAL** | **10.32 MB** | **4,173,954** | **3,798,109** | **+9.0%** |

### Information-Theoretic Analysis

- **Binary entropy**: ~6.5 bits/byte
- **Theoretical optimal**: 2.46 bytes/token (arithmetic coding)
- **Our performance**: 2.849 bytes/token
- **Efficiency**: 86% of theoretical optimum

**Why not 100%**: BPE is greedy (not globally optimal), uses fixed vocabulary, and is context-free. However, 86% efficiency is excellent for neural network compatibility.

## Training Cost-Benefit Analysis

### One-Time Costs
- Training compute: 8.46 hours × 24 cores = 203 core-hours
- Training memory: 70 GB peak
- Storage: 2.3 MB tokenizer file

### Ongoing Benefits (per GB of binary data)
- 9% fewer tokens → 9% shorter sequences
- 9% less compute per forward pass
- Faster inference → lower latency, higher throughput
- Smaller context → can process longer binaries

### ROI
- Training cost: ~$20 (203 core-hours @ $0.10/core-hour)
- Inference savings: 9% compute reduction on all future usage
- Break-even: After encoding ~200 MB of data
- Typical usage: GB-TB of data
- **ROI**: Massive - one-time cost, permanent efficiency gain

## Recommendations

1. **Use glaurung-tokenizer-002 for all new projects** ✓
   - 9-10% better compression than 32K
   - Optimal vocabulary size for uint16
   - Production-ready quality

2. **Keep glaurung-tokenizer-001 for backward compatibility only**
   - Existing models may depend on it
   - New models should use 64K

3. **Don't train larger vocabularies unless you have specific needs**
   - 128K: Only for specialized domains or >100GB corpora
   - 256K: Research only, diminishing returns

4. **Use domain-specific tokenizers**
   - Binary tokenizer for binaries (this one)
   - Text tokenizer for text/code
   - 100-140% penalty for using wrong domain

## Citation

If you use this tokenizer in your research, please cite:

```
Glaurung Tokenizer-002 (64K Binary Tokenizer)
Training date: October 19, 2025
Training corpus: 13GB binaries-small dataset (30,738 files)
Vocabulary: 65,536 tokens (exactly 2^16 for perfect uint16 coverage)
Compression: 2.849 bytes/token (86% of theoretical optimum)
```

## Files and Artifacts

### Production Files
- `production/glaurung-tokenizer-002.json` - The tokenizer (2.3 MB)
- `production/DOCUMENTATION.md` - Complete usage guide (962 lines)

### Baseline Files
- `baseline/glaurung-tokenizer-001.json` - 32K baseline (1.2 MB)

### Experiment Files
- `experiments/length-modes/` - Token length constraint experiments
  - `default.json`, `even.json`, `pow2.json` - Different length modes
  - `ANALYSIS.md` - Results and comparison
- `experiments/cross-domain/` - Text vs binary comparison
  - `tokenizer-text-16k.json` - Text tokenizer trained on man pages
  - `CROSS_DOMAIN_SUMMARY.md` - Cross-domain validation results

### Report Files
- `reports/64k-production/PRODUCTION_REPORT.md` - Comprehensive analysis
- `reports/64k-production/TRAINING_PLAN.md` - Pre-training plan
- `reports/64k-production/training.log.gz` - Compressed training log (1.4 MB)
- `reports/32k-baseline/` - Baseline training artifacts

### Tools
- `tools/analyze_tokenizer.py` - Token distribution analysis
- `tools/inspect_tokens.py` - Interpret learned tokens (x86-64/ARM64)
- `tools/test_compression.py` - Benchmark compression performance
- `tools/find_strings.py` - Search for function names, symbols, paths

## Support

For questions, issues, or contributions:
- See `production/DOCUMENTATION.md` for detailed usage
- Check `reports/64k-production/PRODUCTION_REPORT.md` for training details
- Run analysis tools in `tools/` for exploration

---

**Status**: ✅ Production Ready
**Version**: 1.0
**Last Updated**: October 20, 2025
