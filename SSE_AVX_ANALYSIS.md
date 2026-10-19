# SSE/AVX Instruction Analysis: binary-tokenizer-005 vs glaurung-binary-tokenizer-001

**Question**: Did glaurung-tokenizer-001 fail to learn SSE/AVX patterns, or did it learn different specialized instruction sets?

**Answer**: **Both tokenizers learned SSE/AVX patterns**, but glaurung-tokenizer-001 prioritized them differently due to its multi-architecture training data.

---

## Executive Summary

| Metric | binary-tokenizer-005 | glaurung-tokenizer-001 |
|--------|---------------------|----------------------|
| **Total SSE/AVX tokens** | 2,570 | 1,509 |
| **Complete instructions** | 805 | 471 |
| **Strategy** | Complete-instruction | Complete-instruction |
| **Ratio** | 268:1 | 157:1 |

**Both tokenizers use the same strategy** (complete instructions vs prefix-based), but binary-tokenizer-005 has **70% more SSE/AVX tokens** (2,570 vs 1,509).

---

## Key Findings

### 1. Both Learned SSE/AVX Instructions

**binary-tokenizer-005**:
- 154 two-byte SSE instructions (e.g., `0x0f10` = MOVUPS, `0x0f28` = MOVAPS)
- 250 three-byte SSE instructions (with prefixes `0x660f`, `0xf30f`, `0xf20f`)
- 401 four-byte SSE instructions (with ModR/M)
- 264 REX + SSE combinations (e.g., `0x440f`, `0x410f`)
- 1,173 AVX prefix tokens (VEX/EVEX)
- 325 complete AVX instructions

**glaurung-tokenizer-001**:
- 157 two-byte SSE instructions
- 171 three-byte SSE instructions
- 143 four-byte SSE instructions
- 109 REX + SSE combinations
- 728 AVX prefix tokens
- 198 complete AVX instructions

### 2. Why SSE/AVX in Top 100 of binary-tokenizer-005 but NOT glaurung-tokenizer-001?

**Frequency, not strategy difference:**

**binary-tokenizer-005** top 100 includes:
- `0x0f10` (ID 419) - MOVUPS - extremely common in Windows binaries
- `0x660f` (ID 446) - SSE2 prefix
- `0xf30f` (ID 439) - SSE3 prefix
- `0x440f`, `0x410f`, `0x450f` - REX + SSE combinations

These appear in top 100 because **Windows binaries heavily use SSE/SSE2 for memcpy/memset** and general computation.

**glaurung-tokenizer-001** top 100 includes:
- `0x0f1f` (ID 315) - NOP variant (alignment/padding)
- ARM64 patterns (13 tokens)
- More diverse x86 patterns

SSE/AVX starts appearing around **ID 315-450**, but not as densely because:
1. Training corpus includes ARM64 binaries (no SSE/AVX)
2. Linux binaries may use different optimization profiles
3. Multi-platform data dilutes x86-specific patterns

### 3. Training Data Characteristics

**binary-tokenizer-005** (~5GB):
- Likely Windows-heavy
- MSVC compiler (aggressive SSE/AVX optimization)
- Debug builds with specific patterns

**glaurung-tokenizer-001** (13GB):
- Multi-platform (Linux, Windows)
- Multi-architecture (x86-64, ARM64)
- Production binaries (stripped)
- More diverse compiler toolchains (GCC, Clang, MSVC)

---

## Detailed Breakdown

### SSE Instruction Types Found

#### Two-Byte SSE (0x0f + opcode)

Common in both:
- `0x0f10` - MOVUPS (move unaligned packed single)
- `0x0f11` - MOVUPS (store)
- `0x0f28` - MOVAPS (move aligned packed single)
- `0x0f84` - JE (conditional jump - technically not SSE but uses 0x0f escape)
- `0x0f85` - JNE (conditional jump)

#### Three-Byte SSE (prefix + 0x0f + opcode)

Prefixes:
- `0x660f` - SSE2 packed double-precision
- `0xf30f` - SSE scalar single-precision
- `0xf20f` - SSE2 scalar double-precision

**binary-tokenizer-005** has these as **separate tokens in top 500**, suggesting they're used as building blocks.

**glaurung-tokenizer-001** has these **only as part of complete instructions**, suggesting less frequent usage.

#### Four-Byte SSE (prefix + 0x0f + opcode + ModR/M)

Complete instructions with addressing modes, e.g.:
- `0x660f6fc0` - MOVDQA xmm0, xmm0
- `0xf30f10??` - MOVSS xmm?, [mem]

Both tokenizers have hundreds of these, but binary-tokenizer-005 has **2.8x more** (401 vs 143).

### AVX Instructions (VEX/EVEX)

AVX uses different encoding:
- `0xc4` - 3-byte VEX prefix
- `0xc5` - 2-byte VEX prefix
- `0x62` - EVEX prefix (AVX-512)

**binary-tokenizer-005**: 1,173 AVX prefixes, 325 complete AVX instructions
**glaurung-tokenizer-001**: 728 AVX prefixes, 198 complete AVX instructions

This **1.6x difference** suggests binary-tokenizer-005 trained on more AVX-optimized code (possibly newer Windows builds or scientific computing binaries).

---

## What glaurung-tokenizer-001 Prioritized Instead

### ARM64 Instructions (Not in binary-tokenizer-005)

In top 100 of glaurung-tokenizer-001:
- `0x5fd6` - RET (ARM64)
- `0x2303d5` - MSR/MRS instruction
- `0x1f2003d5` - NOP (ARM64)
- `0xfe97` - BL (branch with link)
- `0x0091`, `0x0071`, `0xf353`, etc.

Total: **~13 ARM64 tokens in top 100**, representing significant training corpus diversity.

### Position-Independent Code (Linux/ELF)

- `0x48ff15` - CALL [rip+disp32] (ID 340)
- `0x4883ec` - SUB rsp, imm8 (stack allocation, ID 335)

These are characteristic of modern Linux ELF binaries with PIE (Position Independent Executable).

### Multi-byte NOP Sequences

- `0x0f1f` (ID 315) - 2-byte NOP prefix
- `0x0f1f440000` (ID 333) - 5-byte NOP (alignment)
- `0x9090` (ID 346) - NOP NOP

Compiler alignment varies by platform; glaurung-tokenizer-001 captured more diversity.

---

## Tokenization Strategy: Complete-Instruction vs Prefix-Based

### What We Initially Thought

Seeing `0x660f`, `0xf30f` as separate tokens in binary-tokenizer-005's top 100 suggested a **prefix-based strategy** where:
- Prefixes are learned separately
- Combined with various opcodes during tokenization

### What's Actually Happening

**Both tokenizers use complete-instruction strategy:**
- **binary-tokenizer-005**: 268:1 ratio (complete:prefix)
- **glaurung-tokenizer-001**: 157:1 ratio (complete:prefix)

The presence of `0x660f` and `0xf30f` in top 100 of binary-tokenizer-005 is NOT because it's prefix-based, but because:
1. These 2-byte sequences are **so frequent** they become tokens themselves
2. BPE will merge any frequently occurring byte pair
3. In Windows binaries with heavy SSE usage, even partial prefixes appear often enough

Think of it like natural language: if you train on English, you might learn "ing" as a token (suffix), but you'll also learn complete words like "running", "jumping", etc. The presence of "ing" doesn't mean you're using a suffix-based strategy.

---

## Conclusion

### Direct Answer to Your Question

> "So do you think that the new tokenizer didn't learn specialized instruction sets like SSE/AVX? Or just different specialized instruction sets?"

**Answer: Different specialized instruction sets** ✅

glaurung-tokenizer-001 learned:
- ✅ SSE/AVX instructions (1,509 tokens)
- ✅ ARM64 instructions (~13 in top 100, hundreds total)
- ✅ Position-independent x86-64 patterns
- ✅ More diverse NOP/padding patterns
- ✅ Multi-platform compiler patterns

binary-tokenizer-005 learned:
- ✅ SSE/AVX instructions (2,570 tokens - **70% more**)
- ✅ Windows/MSVC-specific patterns
- ✅ Debug heap patterns (0xcdcd)
- ✅ Aggressive SSE optimization patterns
- ❌ No ARM64 patterns

### Why the Difference?

**Vocabulary allocation:**
- Binary tokenizers have 65,536 tokens total
- glaurung-tokenizer-001 allocated vocabulary across:
  - x86-64 instructions
  - ARM64 instructions
  - Multiple compiler toolchains
  - Multiple operating systems
- binary-tokenizer-005 allocated vocabulary mainly to:
  - x86/x86-64 instructions (single architecture)
  - Specific compiler patterns (likely MSVC-heavy)

**Training data composition:**
- glaurung-tokenizer-001: 13GB multi-platform → more breadth
- binary-tokenizer-005: ~5GB likely x86-focused → more depth on x86 patterns

### Recommendation by Use Case

**Use binary-tokenizer-005 if:**
- Analyzing Windows binaries exclusively
- Working with SSE/AVX-heavy code (multimedia, scientific computing)
- Need deep x86-64 instruction coverage
- MSVC/Visual Studio toolchain

**Use glaurung-tokenizer-001 if:**
- Analyzing ARM64 binaries (mobile, Apple Silicon, AWS Graviton)
- Working with Linux ELF binaries
- Need multi-architecture support
- Cross-platform malware analysis
- Modern compiler toolchains (GCC, Clang, MSVC)

---

## Appendix: Example SSE Instructions by Token ID

### binary-tokenizer-005 (Top SSE/AVX tokens)

| ID | Hex | Instruction | Description |
|----|-----|-------------|-------------|
| 419 | 0x0f10 | MOVUPS | Move unaligned packed single |
| 425 | 0x440f | REX.R + 0x0f | REX prefix + SSE escape |
| 432 | 0x410f | REX.B + 0x0f | REX prefix + SSE escape |
| 439 | 0xf30f | Prefix | SSE3 scalar single prefix |
| 446 | 0x660f | Prefix | SSE2 packed double prefix |
| 448 | 0x450f | REX.RB + 0x0f | REX prefix + SSE escape |
| 468 | 0x0f28 | MOVAPS | Move aligned packed single |
| 469 | 0x0f11 | MOVUPS | Store unaligned packed single |

### glaurung-tokenizer-001 (Top SSE/AVX tokens)

| ID | Hex | Instruction | Description |
|----|-----|-------------|-------------|
| 315 | 0x0f1f | NOP variant | Multi-byte NOP (not SSE, but uses 0x0f) |
| 323 | 0x0f84 | JE | Conditional jump (0x0f escape) |
| 333 | 0x0f1f440000 | 5-byte NOP | Complete NOP instruction |
| 382 | 0x0f85 | JNE | Conditional jump |
| 418 | 0x410f | REX.B + 0x0f | REX prefix + SSE/jump escape |
| 439 | 0x0f00 | SLDT/STR | System instructions (rare) |
| 452 | 0x0f10 | MOVUPS | Move unaligned packed single |

Notice: glaurung-tokenizer-001 prioritizes control flow (0x0f84, 0x0f85) and alignment (0x0f1f) over data movement SSE instructions.

---

**Generated**: October 2025
**Analysis Tool**: `analyze_sse_avx_patterns.py`
