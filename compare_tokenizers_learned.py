#!/usr/bin/env python3
"""Compare top 100 LEARNED (non-special) tokens between binary-tokenizer-005 and glaurung-tokenizer-002."""

from tokenizers import Tokenizer
from pathlib import Path
import json
from collections import Counter

def load_tokenizer_vocab(tokenizer_path):
    """Load tokenizer and extract vocabulary."""
    print(f"Loading {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()

    # Invert to get id -> token mapping
    id_to_token = {v: k for k, v in vocab.items()}

    return tokenizer, vocab, id_to_token

def is_special_token(token_str):
    """Check if token is a special token (like <|start|>)."""
    return token_str.startswith('<|') and token_str.endswith('|>')

def filter_learned_multibyte_tokens(id_to_token):
    """Filter to only learned multi-byte tokens (no special tokens, no single bytes)."""
    learned_tokens = {}

    for token_id, token_str in id_to_token.items():
        # Skip special tokens
        if is_special_token(token_str):
            continue

        # Try to get byte length
        try:
            token_bytes = token_str.encode('latin-1')
            if len(token_bytes) > 1:
                learned_tokens[token_id] = {
                    'str': token_str,
                    'bytes': token_bytes,
                    'hex': token_bytes.hex(),
                    'length': len(token_bytes)
                }
        except:
            # Skip tokens that can't be encoded
            pass

    return learned_tokens

def get_top_n_tokens(multibyte_tokens, n=100):
    """Get top N tokens by ID (lower IDs = higher frequency in BPE)."""
    # Sort by token ID (lower = learned earlier = more frequent)
    sorted_ids = sorted(multibyte_tokens.keys())
    top_ids = sorted_ids[:n]

    return {tid: multibyte_tokens[tid] for tid in top_ids}

def compare_token_sets(tokens1, tokens2, name1, name2):
    """Compare two sets of tokens."""
    print(f"\n{'='*80}")
    print(f"COMPARISON: {name1} vs {name2}")
    print(f"{'='*80}\n")

    # Get token byte representations for comparison
    bytes1 = {info['hex']: (tid, info) for tid, info in tokens1.items()}
    bytes2 = {info['hex']: (tid, info) for tid, info in tokens2.items()}

    hex_set1 = set(bytes1.keys())
    hex_set2 = set(bytes2.keys())

    # Calculate overlap
    overlap = hex_set1 & hex_set2
    only_in_1 = hex_set1 - hex_set2
    only_in_2 = hex_set2 - hex_set1

    print(f"📊 STATISTICS")
    print(f"  {name1}: {len(tokens1)} tokens")
    print(f"  {name2}: {len(tokens2)} tokens")
    print(f"  Overlap: {len(overlap)} tokens ({len(overlap)/len(tokens1)*100:.1f}%)")
    print(f"  Only in {name1}: {len(only_in_1)} tokens")
    print(f"  Only in {name2}: {len(only_in_2)} tokens")

    # Analyze length distribution
    print(f"\n📏 LENGTH DISTRIBUTION")

    len_dist1 = Counter(info['length'] for info in tokens1.values())
    len_dist2 = Counter(info['length'] for info in tokens2.values())

    print(f"\n  {name1}:")
    for length in sorted(len_dist1.keys()):
        print(f"    {length} bytes: {len_dist1[length]} tokens ({len_dist1[length]/len(tokens1)*100:.1f}%)")

    print(f"\n  {name2}:")
    for length in sorted(len_dist2.keys()):
        print(f"    {length} bytes: {len_dist2[length]} tokens ({len_dist2[length]/len(tokens2)*100:.1f}%)")

    # Show overlap examples
    if overlap:
        print(f"\n🔄 OVERLAP EXAMPLES (first 30 shared tokens by ID in {name1})")
        print(f"  {'ID1':<8} {'ID2':<8} {'Len':<5} {'Hex':<24} {'Printable'}")
        print(f"  {'-'*8} {'-'*8} {'-'*5} {'-'*24} {'-'*30}")

        overlap_list = []
        for hex_bytes in overlap:
            tid1, info1 = bytes1[hex_bytes]
            tid2, info2 = bytes2[hex_bytes]
            overlap_list.append((tid1, tid2, info1))

        # Sort by ID in first tokenizer
        overlap_list.sort(key=lambda x: x[0])

        for tid1, tid2, info in overlap_list[:30]:
            printable = ''.join(c if 32 <= ord(c) < 127 else '.' for c in info['str'])
            print(f"  {tid1:<8} {tid2:<8} {info['length']:<5} {info['hex']:<24} {printable[:30]}")

    # Show unique examples
    print(f"\n🔹 UNIQUE TO {name1} (first 30 by ID)")
    print(f"  {'ID':<8} {'Len':<5} {'Hex':<24} {'Printable'}")
    print(f"  {'-'*8} {'-'*5} {'-'*24} {'-'*30}")

    unique1_list = []
    for hex_bytes in only_in_1:
        tid1, info1 = bytes1[hex_bytes]
        unique1_list.append((tid1, info1))

    unique1_list.sort(key=lambda x: x[0])

    for tid, info in unique1_list[:30]:
        printable = ''.join(c if 32 <= ord(c) < 127 else '.' for c in info['str'])
        print(f"  {tid:<8} {info['length']:<5} {info['hex']:<24} {printable[:30]}")

    print(f"\n🔹 UNIQUE TO {name2} (first 30 by ID)")
    print(f"  {'ID':<8} {'Len':<5} {'Hex':<24} {'Printable'}")
    print(f"  {'-'*8} {'-'*5} {'-'*24} {'-'*30}")

    unique2_list = []
    for hex_bytes in only_in_2:
        tid2, info2 = bytes2[hex_bytes]
        unique2_list.append((tid2, info2))

    unique2_list.sort(key=lambda x: x[0])

    for tid, info in unique2_list[:30]:
        printable = ''.join(c if 32 <= ord(c) < 127 else '.' for c in info['str'])
        print(f"  {tid:<8} {info['length']:<5} {info['hex']:<24} {printable[:30]}")

    # Pattern analysis
    print(f"\n🔍 PATTERN ANALYSIS")

    def analyze_patterns(tokens, name):
        patterns = {
            'x86_rex': [],  # REX prefixes (0x48, 0x4c, 0x4d, etc.)
            'padding': [],  # Common padding (0xcc, 0x90, 0x00)
            'elf_magic': [],  # ELF related
            'pe_magic': [],  # PE related
            'printable_strings': [],  # Contains mostly printable ASCII
            'high_entropy': [],  # All bytes > 0x7f
            'opcodes': [],  # Common x86 opcodes
            'null_terminated': [],  # Ends with null byte
        }

        for tid, info in tokens.items():
            hex_str = info['hex']
            token_str = info['str']
            token_bytes = info['bytes']

            # Check for x86 REX prefixes
            if hex_str.startswith(('48', '4c', '4d', '49')):
                patterns['x86_rex'].append((tid, info))

            # Check for padding
            if any(pad in hex_str for pad in ['cccc', '9090', '0000']):
                patterns['padding'].append((tid, info))

            # Check for ELF magic
            if '7f454c46' in hex_str:  # 0x7f ELF
                patterns['elf_magic'].append((tid, info))

            # Check for PE magic
            if '4d5a' in hex_str:  # MZ
                patterns['pe_magic'].append((tid, info))

            # Check if mostly printable ASCII (>50% printable)
            printable_count = sum(1 for c in token_str if 32 <= ord(c) < 127)
            if printable_count > len(token_str) * 0.5:
                patterns['printable_strings'].append((tid, info))

            # Check for high entropy (all bytes > 0x7f)
            if all(b > 0x7f for b in token_bytes):
                patterns['high_entropy'].append((tid, info))

            # Check for common x86 opcodes
            if hex_str[:2] in ['8b', '89', 'e8', 'c3', 'ff', 'eb']:
                patterns['opcodes'].append((tid, info))

            # Null terminated
            if hex_str.endswith('00'):
                patterns['null_terminated'].append((tid, info))

        return patterns

    patterns1 = analyze_patterns(tokens1, name1)
    patterns2 = analyze_patterns(tokens2, name2)

    print(f"\n  {name1}:")
    for pattern, tokens_list in patterns1.items():
        print(f"    {pattern:<25} {len(tokens_list):>4} ({len(tokens_list)/len(tokens1)*100:>5.1f}%)")

    print(f"\n  {name2}:")
    for pattern, tokens_list in patterns2.items():
        print(f"    {pattern:<25} {len(tokens_list):>4} ({len(tokens_list)/len(tokens2)*100:>5.1f}%)")

    # Show example patterns
    print(f"\n📝 EXAMPLE PATTERNS")

    for pattern_name in ['x86_rex', 'padding', 'printable_strings']:
        if patterns1[pattern_name] or patterns2[pattern_name]:
            print(f"\n  {pattern_name.upper()}:")

            if patterns1[pattern_name]:
                print(f"    {name1} (first 3):")
                for tid, info in patterns1[pattern_name][:3]:
                    printable = ''.join(c if 32 <= ord(c) < 127 else '.' for c in info['str'])
                    print(f"      ID {tid:<6} {info['hex']:<20} {printable[:20]}")

            if patterns2[pattern_name]:
                print(f"    {name2} (first 3):")
                for tid, info in patterns2[pattern_name][:3]:
                    printable = ''.join(c if 32 <= ord(c) < 127 else '.' for c in info['str'])
                    print(f"      ID {tid:<6} {info['hex']:<20} {printable[:20]}")

    return {
        'overlap_count': len(overlap),
        'overlap_pct': len(overlap) / len(tokens1) * 100 if tokens1 else 0,
        'unique1_count': len(only_in_1),
        'unique2_count': len(only_in_2),
        'patterns1': {k: len(v) for k, v in patterns1.items()},
        'patterns2': {k: len(v) for k, v in patterns2.items()},
    }

def main():
    # Paths
    tok005_path = Path("binary-embedding/binary-tokenizer-005/tokenizer.json")
    tok002_path = Path("tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json")

    # Load tokenizers
    print("="*80)
    print("LOADING TOKENIZERS")
    print("="*80)

    tok005, vocab005, id_to_token005 = load_tokenizer_vocab(tok005_path)
    print(f"  binary-tokenizer-005: {tok005.get_vocab_size()} tokens")

    tok002, vocab002, id_to_token002 = load_tokenizer_vocab(tok002_path)
    print(f"  glaurung-tokenizer-002: {tok002.get_vocab_size()} tokens")

    # Count special tokens in 005
    special_count = sum(1 for t in id_to_token005.values() if is_special_token(t))
    print(f"  binary-tokenizer-005 special tokens: {special_count}")

    # Filter to learned multibyte tokens only
    print(f"\n{'='*80}")
    print("FILTERING LEARNED MULTI-BYTE TOKENS (excluding special tokens)")
    print(f"{'='*80}")

    learned005 = filter_learned_multibyte_tokens(id_to_token005)
    print(f"  binary-tokenizer-005: {len(learned005)} learned multi-byte tokens")

    learned002 = filter_learned_multibyte_tokens(id_to_token002)
    print(f"  glaurung-tokenizer-002: {len(learned002)} learned multi-byte tokens")

    # Get top 100
    print(f"\n{'='*80}")
    print("EXTRACTING TOP 100 LEARNED TOKENS")
    print(f"{'='*80}")

    top100_005 = get_top_n_tokens(learned005, 100)
    print(f"  binary-tokenizer-005: {len(top100_005)} tokens")
    print(f"    ID range: {min(top100_005.keys())} to {max(top100_005.keys())}")

    top100_002 = get_top_n_tokens(learned002, 100)
    print(f"  glaurung-tokenizer-002: {len(top100_002)} tokens")
    print(f"    ID range: {min(top100_002.keys())} to {max(top100_002.keys())}")

    # Compare
    results = compare_token_sets(
        top100_005, top100_002,
        "binary-tokenizer-005",
        "glaurung-tokenizer-002"
    )

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n✅ Overlap: {results['overlap_pct']:.1f}% of top 100 learned tokens are shared")

    if results['overlap_pct'] < 50:
        print(f"⚠️  Low overlap suggests different training data characteristics")
    else:
        print(f"✨ Good overlap indicates similar binary patterns learned")

if __name__ == "__main__":
    main()
