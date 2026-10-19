#!/usr/bin/env python3
"""Analyze SSE/AVX instruction patterns in both tokenizers."""

from tokenizers import Tokenizer
from pathlib import Path
from collections import defaultdict

def load_tokenizer_vocab(tokenizer_path):
    """Load tokenizer and extract vocabulary."""
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    return tokenizer, vocab, id_to_token

def is_special_token(token_str):
    """Check if token is a special token."""
    return token_str.startswith('<|') and token_str.endswith('|>')

def analyze_sse_avx_patterns(id_to_token, name):
    """Find and categorize SSE/AVX related tokens."""

    # SSE/AVX instruction prefixes and opcodes
    sse_patterns = {
        'sse_prefixes': [],           # 0x0f, 0x660f, 0xf30f, 0xf20f
        'sse_complete_2byte': [],     # 0x0f?? (2-byte SSE)
        'sse_complete_3byte': [],     # 0x660f??, 0xf30f??, etc (3-byte SSE)
        'sse_complete_4byte': [],     # Complete 4-byte SSE instructions
        'sse_with_rex': [],           # REX + SSE (0x??0f)
        'avx_prefixes': [],           # VEX/EVEX prefixes (0xc4, 0xc5, 0x62)
        'avx_complete': [],           # Complete AVX instructions
    }

    for token_id, token_str in id_to_token.items():
        if is_special_token(token_str):
            continue

        try:
            token_bytes = token_str.encode('latin-1')
            hex_str = token_bytes.hex()

            if len(token_bytes) < 2:
                continue

            # Check for SSE prefix patterns
            if hex_str == '0f':
                sse_patterns['sse_prefixes'].append((token_id, hex_str, len(token_bytes), 'plain 0x0f'))
            elif hex_str in ['660f', 'f30f', 'f20f']:
                sse_patterns['sse_prefixes'].append((token_id, hex_str, len(token_bytes), f'prefix {hex_str}'))

            # Check for 2-byte SSE instructions (0x0f + opcode)
            elif hex_str.startswith('0f') and len(token_bytes) == 2:
                opcode = hex_str[2:4]
                sse_patterns['sse_complete_2byte'].append((token_id, hex_str, len(token_bytes), f'0f+{opcode}'))

            # Check for 3-byte SSE instructions (prefix + 0x0f + opcode)
            elif len(token_bytes) == 3:
                if hex_str.startswith('660f'):
                    sse_patterns['sse_complete_3byte'].append((token_id, hex_str, len(token_bytes), 'SSE2 packed double'))
                elif hex_str.startswith('f30f'):
                    sse_patterns['sse_complete_3byte'].append((token_id, hex_str, len(token_bytes), 'SSE scalar single'))
                elif hex_str.startswith('f20f'):
                    sse_patterns['sse_complete_3byte'].append((token_id, hex_str, len(token_bytes), 'SSE2 scalar double'))
                elif hex_str[2:4] == '0f':  # Middle byte is 0f
                    sse_patterns['sse_complete_3byte'].append((token_id, hex_str, len(token_bytes), f'?+0f+?'))

            # Check for 4+ byte SSE instructions
            elif len(token_bytes) >= 4:
                if '0f' in [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]:
                    if hex_str.startswith(('660f', 'f30f', 'f20f')):
                        sse_patterns['sse_complete_4byte'].append((token_id, hex_str, len(token_bytes), 'SSE with ModR/M'))
                    elif hex_str[2:4] == '0f':
                        sse_patterns['sse_complete_4byte'].append((token_id, hex_str, len(token_bytes), 'REX+SSE with ModR/M'))

            # Check for REX + SSE patterns (REX byte + 0x0f)
            if len(token_bytes) >= 2 and token_bytes[0] in [0x40, 0x41, 0x44, 0x45, 0x48, 0x49, 0x4c, 0x4d]:
                if len(token_bytes) >= 2 and token_bytes[1] == 0x0f:
                    sse_patterns['sse_with_rex'].append((token_id, hex_str, len(token_bytes), f'REX+0f'))
                elif len(token_bytes) >= 3 and token_bytes[2] == 0x0f:
                    sse_patterns['sse_with_rex'].append((token_id, hex_str, len(token_bytes), f'REX+?+0f'))

            # Check for AVX prefixes (VEX/EVEX)
            if token_bytes[0] in [0xc4, 0xc5, 0x62]:
                if token_bytes[0] == 0xc4:
                    sse_patterns['avx_prefixes'].append((token_id, hex_str, len(token_bytes), 'VEX 3-byte'))
                elif token_bytes[0] == 0xc5:
                    sse_patterns['avx_prefixes'].append((token_id, hex_str, len(token_bytes), 'VEX 2-byte'))
                elif token_bytes[0] == 0x62:
                    sse_patterns['avx_prefixes'].append((token_id, hex_str, len(token_bytes), 'EVEX'))

                if len(token_bytes) >= 4:
                    sse_patterns['avx_complete'].append((token_id, hex_str, len(token_bytes), 'Complete AVX'))

        except:
            pass

    return sse_patterns

def print_pattern_summary(patterns, name, top_n=500):
    """Print summary of SSE/AVX patterns found."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    total = sum(len(v) for v in patterns.values())
    print(f"\nTotal SSE/AVX-related tokens found: {total}")

    for category, tokens_list in patterns.items():
        if not tokens_list:
            continue

        print(f"\n{category.upper().replace('_', ' ')} ({len(tokens_list)} tokens)")
        print(f"  {'ID':<8} {'Bytes':<4} {'Hex':<20} {'Description'}")
        print(f"  {'-'*8} {'-'*4} {'-'*20} {'-'*30}")

        # Sort by token ID (frequency proxy)
        tokens_list.sort(key=lambda x: x[0])

        # Show top N in this category
        for token_id, hex_str, byte_len, desc in tokens_list[:20]:
            if token_id < top_n:  # Only show tokens in top N
                print(f"  {token_id:<8} {byte_len:<4} {hex_str:<20} {desc}")

        if len(tokens_list) > 20:
            print(f"  ... and {len(tokens_list) - 20} more")
            # Show ID range
            min_id = min(t[0] for t in tokens_list)
            max_id = max(t[0] for t in tokens_list)
            print(f"  ID range: {min_id} to {max_id}")

def compare_strategies(patterns1, patterns2, name1, name2):
    """Compare tokenization strategies between the two tokenizers."""
    print(f"\n{'='*80}")
    print(f"STRATEGY COMPARISON")
    print(f"{'='*80}")

    # Count prefix-style vs complete-instruction style
    prefixes1 = len(patterns1['sse_prefixes'])
    complete1 = sum(len(patterns1[k]) for k in ['sse_complete_2byte', 'sse_complete_3byte', 'sse_complete_4byte'])

    prefixes2 = len(patterns2['sse_prefixes'])
    complete2 = sum(len(patterns2[k]) for k in ['sse_complete_2byte', 'sse_complete_3byte', 'sse_complete_4byte'])

    print(f"\n{name1}:")
    print(f"  Prefix tokens (reusable):        {prefixes1:>4}")
    print(f"  Complete instruction tokens:     {complete1:>4}")
    print(f"  Ratio (complete/prefix):         {complete1/prefixes1 if prefixes1 > 0 else 'N/A'}")

    print(f"\n{name2}:")
    print(f"  Prefix tokens (reusable):        {prefixes2:>4}")
    print(f"  Complete instruction tokens:     {complete2:>4}")
    print(f"  Ratio (complete/prefix):         {complete2/prefixes2 if prefixes2 > 0 else 'N/A'}")

    print(f"\n📊 INTERPRETATION:")
    if prefixes1 > complete1 and complete2 > prefixes2:
        print(f"  • {name1} uses PREFIX-BASED strategy (more flexible)")
        print(f"  • {name2} uses COMPLETE-INSTRUCTION strategy (more efficient for common patterns)")
    elif complete1 > prefixes1 and prefixes2 > complete2:
        print(f"  • {name1} uses COMPLETE-INSTRUCTION strategy")
        print(f"  • {name2} uses PREFIX-BASED strategy")
    elif complete1 > prefixes1 and complete2 > prefixes2:
        print(f"  • Both use COMPLETE-INSTRUCTION strategy")
        print(f"  • {name1} ratio: {complete1/prefixes1:.1f}x, {name2} ratio: {complete2/prefixes2:.1f}x")
    else:
        print(f"  • Both use PREFIX-BASED strategy")
        print(f"  • Similar approaches to SSE/AVX encoding")

def main():
    # Paths
    tok005_path = Path("binary-embedding/binary-tokenizer-005/tokenizer.json")
    tok002_path = Path("tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json")

    print("="*80)
    print("SSE/AVX INSTRUCTION PATTERN ANALYSIS")
    print("="*80)
    print("\nAnalyzing top 500 tokens for SSE/AVX patterns...")

    # Load tokenizers
    tok005, vocab005, id_to_token005 = load_tokenizer_vocab(tok005_path)
    tok002, vocab002, id_to_token002 = load_tokenizer_vocab(tok002_path)

    # Analyze patterns
    patterns005 = analyze_sse_avx_patterns(id_to_token005, "binary-tokenizer-005")
    patterns002 = analyze_sse_avx_patterns(id_to_token002, "glaurung-tokenizer-001")

    # Print summaries
    print_pattern_summary(patterns005, "binary-tokenizer-005", top_n=500)
    print_pattern_summary(patterns002, "glaurung-tokenizer-001", top_n=500)

    # Compare strategies
    compare_strategies(patterns005, patterns002, "binary-tokenizer-005", "glaurung-tokenizer-001")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
