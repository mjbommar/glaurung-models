#!/usr/bin/env python3
"""Inspect and interpret the top learned tokens from 64K tokenizer."""

import json
from pathlib import Path
from collections import Counter

def load_tokenizer(path):
    """Load tokenizer vocabulary."""
    with open(path) as f:
        data = json.load(f)
    return data['model']['vocab']

def interpret_x86_64(token_bytes):
    """Interpret x86-64 instruction patterns."""
    if len(token_bytes) == 0:
        return None

    b0 = token_bytes[0]

    # REX prefixes (0x40-0x4f)
    if b0 >= 0x40 and b0 <= 0x4f:
        rex_parts = []
        if b0 & 0x08: rex_parts.append("W")  # 64-bit operand
        if b0 & 0x04: rex_parts.append("R")  # Extension of ModR/M reg
        if b0 & 0x02: rex_parts.append("X")  # Extension of SIB index
        if b0 & 0x01: rex_parts.append("B")  # Extension of ModR/M r/m or SIB base
        rex_str = "REX." + "".join(rex_parts) if rex_parts else "REX"

        if len(token_bytes) == 1:
            return f"x86-64: {rex_str}"
        elif len(token_bytes) >= 2:
            b1 = token_bytes[1]
            # Common opcodes after REX.W (0x48)
            if b0 == 0x48:
                if b1 == 0x8b: return "x86-64: REX.W MOV r64, r/m64"
                if b1 == 0x89: return "x86-64: REX.W MOV r/m64, r64"
                if b1 == 0x85: return "x86-64: REX.W TEST r/m64, r64"
                if b1 == 0x83: return "x86-64: REX.W arith r/m64, imm8"
                if b1 == 0x8d: return "x86-64: REX.W LEA r64, m"
                if b1 == 0x01: return "x86-64: REX.W ADD r/m64, r64"
                if b1 == 0x29: return "x86-64: REX.W SUB r/m64, r64"
                if b1 == 0x31: return "x86-64: REX.W XOR r/m64, r64"
                if b1 == 0x39: return "x86-64: REX.W CMP r/m64, r64"

            if len(token_bytes) >= 3:
                b2 = token_bytes[2]
                # Complete 3-byte instruction
                if b0 == 0x48 and b1 == 0x8b:
                    if b2 == 0x00: return "x86-64: MOV rax, [rax]"
                    if b2 == 0x40: return "x86-64: MOV rax, [rax+offset]"
                    if b2 == 0xc0: return "x86-64: MOV rax, rax"
                    return f"x86-64: REX.W MOV r64, r/m64 (ModR/M={b2:02x})"
                if b0 == 0x48 and b1 == 0x89:
                    return f"x86-64: REX.W MOV r/m64, r64 (ModR/M={b2:02x})"
                if b0 == 0x48 and b1 == 0x85:
                    return f"x86-64: REX.W TEST (ModR/M={b2:02x})"

            return f"x86-64: {rex_str} + opcode {b1:02x}"

    # Multi-byte NOP (0x0f 0x1f)
    if len(token_bytes) >= 2 and b0 == 0x0f and token_bytes[1] == 0x1f:
        return "x86-64: Multi-byte NOP (alignment)"

    # Two-byte opcodes (0x0f prefix)
    if b0 == 0x0f and len(token_bytes) >= 2:
        b1 = token_bytes[1]
        if b1 == 0x84: return "x86-64: JE (near conditional)"
        if b1 == 0x85: return "x86-64: JNE (near conditional)"
        if b1 == 0xb6: return "x86-64: MOVZX r32, r/m8"
        if b1 == 0xb7: return "x86-64: MOVZX r32, r/m16"

    # Single-byte opcodes
    if b0 == 0xe8: return "x86-64: CALL rel32"
    if b0 == 0xe9: return "x86-64: JMP rel32"
    if b0 == 0xc3: return "x86-64: RET"
    if b0 == 0x55: return "x86-64: PUSH rbp"
    if b0 == 0x5d: return "x86-64: POP rbp"
    if b0 == 0x90: return "x86-64: NOP"
    if b0 == 0xcc: return "x86-64: INT3 (breakpoint)"

    # Common patterns
    if all(b == 0x00 for b in token_bytes):
        return f"Padding: NULL × {len(token_bytes)}"
    if all(b == 0xcc for b in token_bytes):
        return f"Padding: INT3 × {len(token_bytes)}"
    if all(b == 0xff for b in token_bytes):
        return f"Padding: 0xFF × {len(token_bytes)}"

    return None

def interpret_arm64(token_bytes):
    """Interpret ARM64 instruction patterns."""
    if len(token_bytes) != 4:
        return None

    instr = int.from_bytes(token_bytes, 'little')

    if instr == 0xd65f03c0:
        return "ARM64: RET"
    if (instr & 0xffe0ffff) == 0xaa0003e0:
        return "ARM64: MOV (register)"
    if (instr & 0xff000000) == 0x94000000:
        return "ARM64: BL (branch with link)"
    if (instr & 0xff000000) == 0x14000000:
        return "ARM64: B (unconditional branch)"

    return None

def interpret_token(token_bytes):
    """Interpret a token and return description."""
    # Try x86-64
    x86_interp = interpret_x86_64(token_bytes)
    if x86_interp:
        return x86_interp

    # Try ARM64
    arm_interp = interpret_arm64(token_bytes)
    if arm_interp:
        return arm_interp

    # Check for little-endian integers
    if len(token_bytes) == 4:
        val = int.from_bytes(token_bytes, 'little')
        if val < 0x10000:
            return f"LE int32: {val}"

    if len(token_bytes) == 8:
        val = int.from_bytes(token_bytes, 'little')
        if val < 0x100000000:
            return f"LE int64: {val}"

    # Check for patterns
    if len(set(token_bytes)) == 1:
        return f"Repeated byte: 0x{token_bytes[0]:02x} × {len(token_bytes)}"

    return "Binary pattern"

def main():
    base_dir = Path(__file__).parent

    print("="*80)
    print("  64K TOKENIZER - TOP 100 LEARNED TOKENS")
    print("="*80)

    vocab = load_tokenizer(base_dir / "glaurung-tokenizer-002.json")
    id_to_token = {v: k for k, v in vocab.items()}

    print("\nTokens 256-355 (first 100 learned tokens):\n")

    length_counts = Counter()
    category_counts = Counter()

    for token_id in range(256, min(356, len(id_to_token) + 256)):
        if token_id not in id_to_token:
            continue

        token_str = id_to_token[token_id]
        token_bytes = token_str.encode('latin-1')
        length = len(token_bytes)

        length_counts[length] += 1

        # Format bytes
        hex_str = ' '.join(f'{b:02x}' for b in token_bytes)
        ascii_repr = ''.join(chr(b) if 32 <= b < 127 else '.' for b in token_bytes)

        # Interpret
        interpretation = interpret_token(token_bytes)

        # Categorize
        if 'x86-64' in interpretation:
            category_counts['x86-64 instructions'] += 1
        elif 'ARM64' in interpretation:
            category_counts['ARM64 instructions'] += 1
        elif 'Padding' in interpretation:
            category_counts['Padding patterns'] += 1
        elif 'LE int' in interpretation:
            category_counts['Little-endian integers'] += 1
        else:
            category_counts['Other patterns'] += 1

        # Show first 50
        if token_id < 306:
            length_marker = {
                1: " ", 2: "²", 3: "³", 4: "⁴", 5: "⁵",
                6: "⁶", 7: "⁷", 8: "⁸",
            }.get(length, f"^{length}")

            print(f"  {length_marker} ID {token_id:3d} (len={length:2d}): {hex_str:48s}  [{ascii_repr:16s}]")
            print(f"         {interpretation}")

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY STATISTICS (First 100 Tokens)")
    print("="*80)

    print(f"\nLength distribution:")
    for length in sorted(length_counts.keys()):
        count = length_counts[length]
        pct = 100 * count / 100
        bar = '█' * (count // 2)
        print(f"  Length {length:2d}: {count:3d} tokens ({pct:5.1f}%) {bar}")

    avg_length = sum(l * c for l, c in length_counts.items()) / sum(length_counts.values())
    print(f"\nAverage token length: {avg_length:.2f} bytes")

    print(f"\nPattern categories:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / 100
        print(f"  {category}: {count} tokens ({pct:.0f}%)")

    # Key insights
    print(f"\n{'='*80}")
    print("  KEY INSIGHTS")
    print("="*80)

    len3_count = length_counts.get(3, 0)
    print(f"\n1. LENGTH-3 TOKENS (THE BREAKTHROUGH):")
    print(f"   Found {len3_count} length-3 tokens in top 100 ({100*len3_count/100:.0f}%)")
    print(f"   These capture complete x86-64 instructions (REX + opcode + ModR/M)")

    x86_count = category_counts.get('x86-64 instructions', 0)
    print(f"\n2. X86-64 INSTRUCTION DOMINANCE:")
    print(f"   {x86_count} x86-64 instructions in top 100 ({100*x86_count/100:.0f}%)")
    print(f"   Tokenizer learned the ISA, not just byte patterns!")

    pad_count = category_counts.get('Padding patterns', 0)
    print(f"\n3. PADDING PATTERNS:")
    print(f"   {pad_count} padding patterns (NULL, INT3, 0xFF)")
    print(f"   Critical for section alignment in binaries")

    print(f"\n4. MULTI-ARCHITECTURE SUPPORT:")
    arm_count = category_counts.get('ARM64 instructions', 0)
    print(f"   x86-64: {x86_count} tokens")
    print(f"   ARM64: {arm_count} tokens")
    print(f"   Tokenizer learned patterns from multiple ISAs")

    print()

if __name__ == '__main__':
    main()
