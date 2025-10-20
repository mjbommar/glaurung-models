# Glaurung Tokenizer-002 (Production)

**Status**: ✅ Production Ready
**Version**: 1.0
**Vocabulary**: 65,536 tokens (exactly 2^16 = 64K)
**Compression**: 2.849 bytes/token
**Training Date**: October 19, 2025

## Quick Start

### Python

```python
from tokenizers import Tokenizer
from pathlib import Path

# Load tokenizer
tokenizer = Tokenizer.from_file("glaurung-tokenizer-002.json")

# Encode binary
binary = Path("/usr/bin/ls").read_bytes()
encoding = tokenizer.encode(binary.decode('latin-1'))

print(f"File size: {len(binary):,} bytes")
print(f"Tokens: {len(encoding.ids):,}")
print(f"Compression: {len(binary) / len(encoding.ids):.3f} bytes/token")
```

### Expected Output
```
File size: 142,144 bytes
Tokens: 49,574
Compression: 2.866 bytes/token
```

## Key Features

- **64K vocabulary**: Optimal size for uint16 token IDs
- **9% better compression** than 32K baseline
- **Multi-architecture**: x86-64, ARM64, Windows PE, Linux ELF
- **Instruction-aware**: Captures complete x86-64 instructions (REX + opcode + ModR/M)
- **String-rich**: 5.76% of vocabulary contains function names, paths, library references

## Files

- `glaurung-tokenizer-002.json` - The tokenizer (2.3 MB)
- `DOCUMENTATION.md` - Complete usage guide (962 lines)
- `README.md` - This file

## Performance

### Compression Benchmarks
Tested on `/usr/bin` binaries (not in training set):

| Binary | Size | Tokens | bytes/token |
|--------|------|--------|-------------|
| bash | 1.38 MB | 535,541 | 2.698 |
| python3.12 | 7.65 MB | 2,801,226 | 2.863 |
| gcc-13 | 0.98 MB | 344,201 | 2.986 |
| ls | 0.14 MB | 49,574 | 2.866 |
| grep | 0.18 MB | 67,567 | 2.667 |

**Average**: 2.849 bytes/token

### Information-Theoretic Efficiency
- Binary entropy: ~6.5 bits/byte
- Theoretical optimal: 2.46 bytes/token
- Our performance: 2.849 bytes/token
- **Efficiency: 86%** of theoretical optimum

## Training

**Dataset**: 13GB, 30,738 binaries (Linux, Windows, x86-64, ARM64)
**Duration**: 8.46 hours on 24 cores
**Memory**: 70 GB peak

**Command**:
```bash
cargo run --release --bin train -- \
  --output glaurung-tokenizer-002.json \
  /nas4/data/glaurung-data/binaries-small/ \
  --vocab-size 65536 \
  --min-frequency 4 \
  --chunk-size 8192
```

## Token Distribution

| Length | Count | Percentage | Examples |
|--------|-------|------------|----------|
| 2 bytes | 31,528 | 48.3% | `0x48 0x8b`, `0xcc 0xcc` |
| 3 bytes | 9,261 | 14.2% | `0x48 0x8b 0xc0` (MOV rax, rax) |
| 4 bytes | 11,520 | 17.6% | `0x48 0x89 0x45 0xf8` |
| 5+ bytes | 13,164 | 20.2% | Multi-instruction sequences |

**Average**: 3.651 bytes/token

## Use Cases

✅ **Recommended for**:
- Binary neural language models
- Malware analysis
- Reverse engineering tools
- Binary similarity detection
- Code pattern recognition

❌ **Not recommended for**:
- Text/source code (use text tokenizer, 100%+ penalty)
- Sub-1MB binaries (overhead too high)
- Real-time streaming (load time ~100ms)

## Documentation

For complete documentation including:
- Installation instructions
- Advanced usage examples
- API reference
- Token interpretation
- Troubleshooting

See: **DOCUMENTATION.md** in this directory.

## Comparison with Baseline

| Metric | 32K Baseline | 64K Production | Improvement |
|--------|--------------|----------------|-------------|
| Vocabulary | 32,761 | 65,536 | 2.00x |
| bytes/token | 2.592 | 2.849 | +9.9% |
| Avg length | 3.512 | 3.651 | +4.0% |
| Length-3 | 5,405 (16.6%) | 9,261 (14.2%) | +71% more |
| Training | 4.66 hours | 8.46 hours | 1.81x |

**Result**: 9% fewer tokens needed to encode the same data.

## Citation

```
Glaurung Tokenizer-002
64K Binary Tokenizer for Neural Language Models
Vocabulary: 65,536 tokens (exactly 2^16)
Training: October 19, 2025
Dataset: 13GB binaries-small (30,738 files)
Performance: 2.849 bytes/token (86% of theoretical optimum)
```

## Support

- Main README: `../README.md`
- Training report: `../reports/64k-production/PRODUCTION_REPORT.md`
- Analysis tools: `../tools/`

---

**Production Ready**: ✅
**Approved for Deployment**: October 20, 2025
