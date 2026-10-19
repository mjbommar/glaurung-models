# Tokenizer Comparison: binary-tokenizer-005 vs glaurung-binary-tokenizer-001

**Date**: October 2025
**Comparison**: Top 100 learned multi-byte tokens (excluding special tokens)

---

## Executive Summary

**Key Finding**: 64% overlap between the top 100 learned tokens demonstrates both tokenizers successfully learned fundamental binary patterns, while the 36% unique tokens in each reflect their different training data characteristics.

- **binary-tokenizer-005**: ~5GB mixed binary data
- **glaurung-binary-tokenizer-001** (tokenizer-002): 13GB binaries-small (30,738 files, multi-platform)

---

## Overlap Analysis

### Statistics

| Metric | Value |
|--------|-------|
| **Shared tokens** | 64 / 100 (64.0%) |
| **Unique to binary-tokenizer-005** | 36 tokens |
| **Unique to glaurung-tokenizer-001** | 36 tokens |
| **Token ID range (005)** | 391-490 |
| **Token ID range (001)** | 256-355 |

Note: binary-tokenizer-005 has 135 special tokens (e.g., `<|start|>`, `<|end|>`, `<|reserved:0|>`) at IDs 0-134, which is why learned tokens start at ID 391 instead of 256.

---

## Shared Patterns (64 tokens)

These fundamental binary patterns appear in both tokenizers' top 100:

### 1. Null Byte Sequences (Padding/Alignment)
```
0x0000                  (2 bytes)
0x00000000              (4 bytes)
0x0000000000000000      (8 bytes)
```

### 2. x86-64 REX Prefixes
```
0x488b                  REX.W + MOV
0x4889                  REX.W + MOV
0x488d                  REX.W + LEA
0x4883                  REX.W + OP
0x4c8b                  REX.WR + MOV
```

### 3. Common Small Integers (Little Endian)
```
0x0100, 0x0200, 0x0300, 0x0400, 0x0600, 0x0800
0x01000000              (1 as 32-bit LE)
```

### 4. Padding Patterns
```
0xcccc                  int3 int3 (debugger breakpoint)
0xcccccccc              int3 padding (4 bytes)
0xcccccccccccccccc      int3 padding (8 bytes)
```

### 5. All-Ones Patterns
```
0xffff                  (2 bytes)
0xffffffff              (4 bytes)
```

### 6. ARM64 Patterns
```
0x40f9                  LDR (load register, ARM64)
0xe003                  MOV (ARM64)
```

### 7. Stack Operations
```
0x4424                  [rsp+0x24]
```

---

## Unique to binary-tokenizer-005 (36 tokens)

### Characteristics
- More Windows/MSVC debug patterns
- More SSE/AVX instruction prefixes
- Repeated printable ASCII patterns

### Notable Patterns

**Debug Fill Patterns (MSVC)**:
```
0xcdcd                  Clean memory fill (debug heap)
0xcdcdcdcd              (4 bytes)
```

**Tilde Patterns** (possibly text/debug data):
```
0x7e7e                  ~~
0x7e7e7e7e              ~~~~
0x7e7e7e7e7e7e7e7e      ~~~~~~~~
```

**SSE/AVX Instruction Prefixes**:
```
0x0f10                  MOVUPS
0x0f11                  MOVUPS (store)
0x0f28                  MOVAPS
0x660f                  SSE prefix
0xf30f                  SSE3 prefix
0x440f, 0x410f, 0x450f  REX + SSE combinations
```

**Other x86 Patterns**:
```
0x448b                  REX.R + MOV
0x8000, 0x0080          High bit patterns
```

---

## Unique to glaurung-tokenizer-001 (36 tokens)

### Characteristics
- More ARM64 instruction patterns
- More complete x86 instruction sequences
- String fragments from binaries
- Linux ELF patterns

### Notable Patterns

**ARM64 Instructions**:
```
0x0091                  ADD (immediate)
0x0071                  SUBS
0xf353                  ARM64 pattern
0x13aa, 0x14aa          ARM64 patterns
0x5fd6                  RET (ARM64)
0xc0035fd6              Instruction + RET
0x2303d5                MSR/MRS instruction
0x1f2003d5              NOP (ARM64)
0xfe97                  BL (branch with link)
```

**x86-64 Complete Instructions**:
```
0x0f1f                  Multi-byte NOP prefix
0x0f1f440000            5-byte NOP (alignment)
0x4883ec                SUB rsp, imm8 (stack allocation)
0x48ff15                CALL [rip+disp32] (position-independent)
0xff54                  CALL [reg+disp8]
```

**String Fragments** (from binary data):
```
0x696e                  "in"
0x6572                  "er"
0x7265                  "re"
0x7400                  "t\0"
0x006500                "\0e\0" (possibly UTF-16)
0x000054                "\0\0T"
```

**NOP Padding**:
```
0x9090                  NOP NOP
```

**Long Null Sequence**:
```
0x00000000000000000000000000000000    (16 bytes)
```

---

## Length Distribution

### binary-tokenizer-005
| Length | Count | Percentage |
|--------|-------|------------|
| 2 bytes | 76 | 76.0% |
| 3 bytes | 11 | 11.0% |
| 4 bytes | 6 | 6.0% |
| 5+ bytes | 7 | 7.0% |

### glaurung-tokenizer-001
| Length | Count | Percentage |
|--------|-------|------------|
| 2 bytes | 70 | 70.0% |
| 3 bytes | 15 | 15.0% |
| 4 bytes | 6 | 6.0% |
| 5+ bytes | 9 | 9.0% |

**Observation**: glaurung-tokenizer-001 has slightly more 3+ byte tokens, suggesting it learned more complete instruction sequences.

---

## Pattern Analysis Summary

| Pattern Category | binary-tokenizer-005 | glaurung-tokenizer-001 |
|------------------|---------------------|----------------------|
| **x86 REX prefixes** | 11 (11.0%) | 13 (13.0%) |
| **Padding patterns** | 16 (16.0%) | 20 (20.0%) |
| **Printable strings** | 11 (11.0%) | 10 (10.0%) |
| **High entropy (>0x7f)** | 13 (13.0%) | 13 (13.0%) |
| **x86 opcodes** | 7 (7.0%) | 7 (7.0%) |
| **Null-terminated** | 36 (36.0%) | 32 (32.0%) |

---

## Key Insights

### 1. Platform Differences

**binary-tokenizer-005**:
- More Windows/MSVC patterns (0xcdcd debug fill)
- More SSE/AVX prefixes (suggests Windows binaries with advanced instructions)
- Tilde patterns (possibly text files or debug output)

**glaurung-tokenizer-001**:
- Significantly more ARM64 patterns
- More Linux ELF patterns (position-independent code: `CALL [rip+disp32]`)
- String fragments from actual binaries ("in", "er", "re")

### 2. Training Data Quality

The 64% overlap confirms both tokenizers learned core binary fundamentals:
- Memory alignment (null bytes)
- x86-64 instruction structure (REX prefixes + opcodes)
- Compiler padding (int3, nop)

The 36% unique tokens demonstrate:
- **005**: Trained on more Windows/x86 data with debug symbols
- **001**: Trained on more diverse multi-architecture data (x86-64 + ARM64) with production binaries

### 3. Instruction Awareness

**glaurung-tokenizer-001 shows better instruction awareness**:
- Complete instructions: `0x4883ec` (SUB rsp), `0x0f1f440000` (5-byte NOP)
- Position-independent code patterns: `0x48ff15` (CALL [rip+disp32])
- ARM64 multi-instruction patterns

**binary-tokenizer-005 shows more prefix fragmentation**:
- Separate tokens for SSE prefixes (0x0f10, 0xf30f, 0x660f)
- Suggests tokenizer learned prefixes separately from full instructions

### 4. Vocabulary Efficiency

Both tokenizers use ~70-76% of top 100 tokens for 2-byte patterns, which is optimal for BPE on binary data given:
- x86-64 instructions are typically 2-4 bytes
- Most common patterns are 2-byte combinations

---

## Recommendations

### Use binary-tokenizer-005 when:
- Working primarily with Windows binaries
- Analyzing x86/x86-64 only
- Need compatibility with existing models
- Working with debug builds (MSVC patterns)

### Use glaurung-binary-tokenizer-001 when:
- Working with multi-architecture binaries (x86-64 + ARM64)
- Analyzing Linux ELF binaries
- Need better compression (2.849 bytes/token vs ~2.6)
- Working with production (stripped) binaries
- Need position-independent code awareness

---

## Conclusion

The 64% overlap demonstrates both tokenizers successfully learned fundamental binary patterns. The differences reflect their training data:

- **005**: Windows/MSVC-heavy, x86-only, debug patterns
- **001**: Multi-platform, multi-architecture, production-focused

For new projects requiring broad binary analysis coverage, **glaurung-binary-tokenizer-001** is recommended due to:
1. Larger, more diverse training corpus (13GB vs ~5GB)
2. Multi-architecture support (ARM64 + x86-64)
3. Better instruction-level patterns
4. Superior compression (9-10% improvement)
5. Production-ready status with comprehensive documentation

---

## Appendix: Top 20 Shared Tokens

| Token ID (005) | Token ID (001) | Hex | Description |
|---------------|---------------|-----|-------------|
| 391 | 256 | 0x0000 | Null bytes (alignment) |
| 392 | 257 | 0x00000000 | 4-byte null alignment |
| 393 | 260 | 0xffff | All ones (2 bytes) |
| 394 | 258 | 0x0000000000000000 | 8-byte null alignment |
| 395 | 259 | 0x000000 | 3-byte null |
| 396 | 292 | 0xffffffff | All ones (4 bytes) |
| 397 | 261 | 0xcccc | int3 int3 padding |
| 398 | 262 | 0x488b | REX.W + MOV |
| 401 | 269 | 0xcccccccc | int3 padding (4 bytes) |
| 402 | 265 | 0x0100 | Integer 1 (little-endian) |
| 403 | 268 | 0x4889 | REX.W + MOV |
| 404 | 270 | 0x0200 | Integer 2 (little-endian) |
| 405 | 267 | 0x488d | REX.W + LEA |
| 406 | 279 | 0x010000 | Integer 1 (3 bytes) |
| 407 | 264 | 0x0300 | Integer 3 (little-endian) |
| 408 | 276 | 0x01000000 | Integer 1 (4 bytes) |
| 410 | 273 | 0x0000000000 | 5-byte null |
| 411 | 271 | 0x0400 | Integer 4 (little-endian) |
| 412 | 275 | 0x4883 | REX.W + OP (various) |
| 413 | 266 | 0x40f9 | LDR (ARM64) |

---

**Generated**: October 2025
**Tools**: `compare_tokenizers_learned.py`
**Tokenizers Compared**:
- binary-tokenizer-005: `binary-embedding/binary-tokenizer-005/tokenizer.json`
- glaurung-binary-tokenizer-001: `tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json`
