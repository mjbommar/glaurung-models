---
language:
- code
license: apache-2.0
tags:
- binary-analysis
- tokenizer
- bpe
- malware-analysis
- reverse-engineering
- security
- x86-64
- arm64
- elf
- pe
library_name: tokenizers
pipeline_tag: feature-extraction
---

# Glaurung Binary Tokenizer 001

**Production-ready 64K vocabulary BPE tokenizer for binary executables**

🔗 **GitHub**: [mjbommar/glaurung](https://github.com/mjbommar/glaurung)

---

## Overview

**Glaurung Binary Tokenizer 001** is a specialized Byte Pair Encoding (BPE) tokenizer optimized for compiled binary data across multiple architectures (x86-64, ARM64, Windows PE, Linux ELF). This is the production successor to [binary-tokenizer-005](https://huggingface.co/mjbommar/binary-tokenizer-005).

### Key Specifications

- **Vocabulary Size**: 65,536 tokens (exactly 2^16 = 64K)
- **Compression**: 2.849 bytes/token average
- **Training Data**: 13GB corpus, 30,738 binaries
- **Architectures**: x86-64, x86-32, ARM64
- **Platforms**: Linux (Alpine, Debian, Ubuntu), Windows (8, 10, 11)
- **Encoding**: Latin-1 (each byte 0-255 maps to a single character)

### Performance Highlights

- **9-10% better compression** than 32K baseline
- **86% of theoretical maximum** compression efficiency
- **Instruction-aware**: Captures complete x86-64 instructions (REX + opcode + ModR/M)
- **String-rich**: 5.76% of vocabulary contains function names, paths, library references

---

## Installation

```bash
pip install tokenizers transformers
```

---

## Quick Start

### Method 1: Using the tokenizers library (Recommended)

```python
from tokenizers import Tokenizer
from pathlib import Path

# Load tokenizer directly from Hugging Face Hub
tokenizer = Tokenizer.from_pretrained("mjbommar/glaurung-binary-tokenizer-001")

# Process binary data - MUST use latin-1 encoding
binary_path = Path("/usr/bin/ls")
raw_bytes = binary_path.read_bytes()
text = raw_bytes.decode('latin-1')  # Convert bytes to latin-1 string

# Tokenize
encoded = tokenizer.encode(text)
tokens = encoded.ids

print(f"File size: {len(raw_bytes):,} bytes")
print(f"Tokens: {len(tokens):,}")
print(f"Compression: {len(raw_bytes) / len(tokens):.3f} bytes/token")

# Decode back to text (note: adds spaces between tokens due to BPE behavior)
decoded = tokenizer.decode(tokens)
```

**Expected Output** (for `/usr/bin/ls`):
```
File size: 142,144 bytes
Tokens: 49,574
Compression: 2.866 bytes/token
```

### Method 2: Using transformers library

```python
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

# Load the base tokenizer
base_tokenizer = Tokenizer.from_pretrained("mjbommar/glaurung-binary-tokenizer-001")

# Wrap with PreTrainedTokenizerFast for transformers compatibility
tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)

# Process binary data
with open("/usr/bin/ls", "rb") as f:
    raw_bytes = f.read()
    text = raw_bytes.decode('latin-1')

# Tokenize (returns dict with input_ids, attention_mask, etc.)
result = tokenizer(text)
tokens = result["input_ids"]
```

---

## Important: Data Format

The tokenizer expects binary data encoded as **latin-1 strings**, NOT hex strings:

```python
# ✅ CORRECT - Use latin-1 encoded bytes
raw_bytes = b'\x7fELF\x01\x01'
text = raw_bytes.decode('latin-1')  # → '\x7fELF\x01\x01'
encoded = tokenizer.encode(text)

# ❌ WRONG - Do not use hex strings
hex_str = "7f 45 4c 46 01 01"  # Will not work correctly
```

**Why latin-1?** Every byte value (0-255) maps to exactly one latin-1 character, ensuring lossless round-trip conversion between bytes and text.

---

## Performance Benchmarks

### Compression on Real-World Binaries

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

---

## Example: Tokenizing an ELF Header

```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_pretrained("mjbommar/glaurung-binary-tokenizer-001")

# ELF header bytes
elf_header = b'\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00'
text = elf_header.decode('latin-1')

# Tokenize
encoded = tokenizer.encode(text)
print(f"Original bytes: {elf_header.hex()}")
print(f"Tokens: {encoded.ids}")
print(f"Token count: {len(encoded.ids)}")
print(f"Compression: {len(elf_header) / len(encoded.ids):.2f} bytes/token")

# Examine individual tokens
for token_id, token_str in zip(encoded.ids, encoded.tokens):
    token_bytes = token_str.encode('latin-1')
    print(f"  Token {token_id:5d}: {token_bytes.hex():16s} ({len(token_bytes)} bytes)")
```

---

## Token Distribution

| Length | Count | Percentage | Examples |
|--------|-------|------------|----------|
| 2 bytes | 31,528 | 48.3% | `0x48 0x8b` (REX.W prefix), `0xcc 0xcc` (int3 padding) |
| 3 bytes | 9,261 | 14.2% | `0x48 0x8b 0xc0` (MOV rax, rax) |
| 4 bytes | 11,520 | 17.6% | `0x48 0x89 0x45 0xf8` (MOV [rbp-8], rax) |
| 5+ bytes | 13,164 | 20.2% | Multi-instruction sequences, string literals |

**Average token length**: 3.651 bytes

---

## Training Details

### Dataset

**Source**: `/nas4/data/glaurung-data/binaries-small/`

- **Size**: 13 GB
- **Files**: 30,738 binaries
- **Content**: Real-world compiled binaries including system utilities, libraries, and applications

**Platform Distribution**:
- Linux: Alpine, Debian, Ubuntu (ELF format)
- Windows: 8, 10, 11 (PE format)

**Architecture Distribution**:
- x86-64 (primary)
- x86-32
- ARM64

### Training Parameters

```bash
cargo run --release --bin train -- \
  --output glaurung-tokenizer-002.json \
  /nas4/data/glaurung-data/binaries-small/ \
  --vocab-size 65536 \
  --min-frequency 4 \
  --chunk-size 8192
```

**Training Duration**: 8.46 hours on 24 cores
**Peak Memory**: 70 GB

---

## Use Cases

### ✅ Recommended For

- Binary neural language models
- Malware analysis and classification
- Reverse engineering tools
- Binary similarity detection
- Code pattern recognition
- Vulnerability research
- Firmware analysis

### ❌ Not Recommended For

- Text/source code (use text tokenizer like GPT-2, 100%+ penalty)
- Very small binaries <1KB (overhead too high)
- Real-time streaming (load time ~100ms)

---

## Comparison with Predecessor

| Metric | binary-tokenizer-005 | glaurung-binary-tokenizer-001 | Improvement |
|--------|---------------------|------------------------------|-------------|
| Vocabulary | 65,536 | 65,536 | Same |
| Training data | ~5GB mixed | 13GB binaries-small | 2.6x larger |
| bytes/token | ~2.6 | 2.849 | +9.6% |
| Platforms | Mixed | Multi-OS (Linux, Windows) | More diverse |
| Architecture awareness | Basic | Advanced (instruction-aware) | Significant |
| Documentation | Basic | Comprehensive | Extensive |

**Key Improvements**:
- Larger, more diverse training corpus
- Better cross-platform coverage
- Instruction-boundary awareness
- Production-ready quality

---

## Advanced Usage

### Batch Processing Multiple Files

```python
from tokenizers import Tokenizer
from pathlib import Path
import numpy as np

tokenizer = Tokenizer.from_pretrained("mjbommar/glaurung-binary-tokenizer-001")

def tokenize_binary_file(file_path):
    """Tokenize a single binary file."""
    raw_bytes = Path(file_path).read_bytes()
    text = raw_bytes.decode('latin-1')
    encoded = tokenizer.encode(text)
    return {
        'file': file_path,
        'size_bytes': len(raw_bytes),
        'token_count': len(encoded.ids),
        'compression_ratio': len(raw_bytes) / len(encoded.ids),
        'token_ids': encoded.ids
    }

# Process directory
binary_dir = Path("/usr/bin")
results = []
for binary_path in binary_dir.glob("*"):
    if binary_path.is_file():
        try:
            result = tokenize_binary_file(binary_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {binary_path}: {e}")

# Analyze compression statistics
compression_ratios = [r['compression_ratio'] for r in results]
print(f"Mean compression: {np.mean(compression_ratios):.3f} bytes/token")
print(f"Std deviation: {np.std(compression_ratios):.3f}")
```

### Using with PyTorch/TensorFlow Models

```python
from tokenizers import Tokenizer
import torch
from pathlib import Path

tokenizer = Tokenizer.from_pretrained("mjbommar/glaurung-binary-tokenizer-001")

def prepare_binary_for_model(file_path, max_length=512):
    """Prepare binary data for neural network input."""
    raw_bytes = Path(file_path).read_bytes()
    text = raw_bytes.decode('latin-1')

    # Tokenize
    encoded = tokenizer.encode(text)
    token_ids = encoded.ids

    # Truncate or pad to max_length
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        # Pad with token ID 0 (or use a dedicated padding token)
        token_ids = token_ids + [0] * (max_length - len(token_ids))

    # Convert to tensor
    return torch.tensor(token_ids, dtype=torch.long)

# Use in model
binary_tensor = prepare_binary_for_model("/usr/bin/ls", max_length=1024)
print(f"Tensor shape: {binary_tensor.shape}")  # torch.Size([1024])
```

---

## Troubleshooting

### Issue: UnicodeDecodeError when processing binary

**Solution**: Always use `latin-1` encoding, never `utf-8`:
```python
# ✅ Correct
text = raw_bytes.decode('latin-1')

# ❌ Wrong
text = raw_bytes.decode('utf-8')  # Will fail on non-UTF-8 bytes
```

### Issue: Decoded output doesn't match original

**Cause**: BPE tokenizers add spaces between tokens during decoding.

**Solution**: Use the raw token IDs and decode manually if exact byte recovery is needed:
```python
# Get tokens without spaces
tokens_no_spaces = ''.join(encoded.tokens)
original_bytes = tokens_no_spaces.encode('latin-1')
```

### Issue: Poor compression on specific binary types

**Cause**: The tokenizer may not be optimized for highly specialized formats (e.g., bytecode for Python .pyc, Java .class).

**Solution**: Consider domain-specific tokenizers for specialized formats, or use this as a general-purpose baseline.

---

## Related Projects

- **Predecessor**: [mjbommar/binary-tokenizer-005](https://huggingface.co/mjbommar/binary-tokenizer-005) - Earlier 64K binary tokenizer
- **Framework**: [mjbommar/glaurung](https://github.com/mjbommar/glaurung) - Binary analysis framework
- **Training Code**: [mjbommar/glaurung-models](https://github.com/mjbommar/glaurung-models) - Binary embedding models and tokenizers

---

## Technical Architecture

### Vocabulary Structure

- **Base tokens**: 256 single-byte tokens (0x00 to 0xFF)
- **Merged tokens**: 65,280 learned byte-pair combinations
- **Total**: 65,536 tokens (exactly 2^16)

### Special Tokens

The tokenizer includes boundary markers for file-level segmentation:
- `<|start|>` (ID: 0)
- `<|end|>` (ID: 1)

These help models distinguish between concatenated files and identify file headers.

### Token Properties

**Instruction-aware patterns** (x86-64 examples):
- REX prefixes: `0x48`, `0x4c`, `0x4d`
- Common opcodes: `0x8b` (MOV), `0x89` (MOV), `0xe8` (CALL)
- ModR/M patterns: `0xc0`, `0x45`, `0x5d`

**Common patterns**:
- Padding: `0xcc 0xcc` (int3), `0x90 0x90` (nop)
- Alignment: `0x00 0x00 0x00 0x00`
- String terminators: `0x00` at word boundaries

---

## Performance Characteristics

### Load Time

- **Tokenizer size**: 2.3 MB on disk
- **Load time**: ~100ms (cold), ~20ms (cached)
- **Memory footprint**: ~15 MB in RAM

### Encoding Speed

On a modern CPU (tested on Intel i9-12900K):

| Operation | Speed |
|-----------|-------|
| Encode 1 MB binary | ~50 ms |
| Encode 10 MB binary | ~450 ms |
| Encode 100 MB binary | ~4.2 s |

**Throughput**: ~20-25 MB/second

---

## Limitations

1. **Cross-domain penalty**: Using on text data causes 100-140% efficiency loss
2. **Small file overhead**: Files <1KB have proportionally higher tokenization overhead
3. **Deterministic decoding**: Spaces inserted between tokens during decode (BPE behavior)
4. **Architecture bias**: Trained primarily on x86-64; may be less optimal for RISC-V, MIPS, etc.

---

## Citation

If you use this tokenizer in research, please cite:

```
Glaurung Binary Tokenizer 001
64K Binary Tokenizer for Neural Language Models
Vocabulary: 65,536 tokens (exactly 2^16)
Training: October 2025
Dataset: 13GB binaries-small (30,738 files)
Performance: 2.849 bytes/token (86% of theoretical optimum)
HuggingFace: mjbommar/glaurung-binary-tokenizer-001
```

---

## License

Apache License 2.0

This tokenizer is part of the [Glaurung](https://github.com/mjbommar/glaurung) project. See the [glaurung-models repository](https://github.com/mjbommar/glaurung-models) for full license details.

---

## Support & Issues

- **GitHub Issues**: [mjbommar/glaurung-models/issues](https://github.com/mjbommar/glaurung-models/issues)
- **Documentation**: Full training report available in the [glaurung-models repository](https://github.com/mjbommar/glaurung-models/tree/master/tokenizers/tokenizer-002)
- **Email**: Contact maintainer via GitHub

---

**Production Status**: ✅ Ready for deployment
**Version**: 1.0.0
**Release Date**: October 2025
