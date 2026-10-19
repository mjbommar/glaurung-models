#!/usr/bin/env python3
"""Compare top 100 multi-byte tokens between binary-tokenizer-005 and glaurung-tokenizer-002."""

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

def filter_multibyte_tokens(id_to_token):
    """Filter out single-byte tokens, return sorted by ID (proxy for frequency)."""
    multibyte_tokens = {}

    for token_id, token_str in id_to_token.items():
        # Try to get byte length
        try:
            token_bytes = token_str.encode('latin-1')
            if len(token_bytes) > 1:
                multibyte_tokens[token_id] = {
                    'str': token_str,
                    'bytes': token_bytes,
                    'hex': token_bytes.hex(),
                    'length': len(token_bytes)
                }
        except:
            # Skip tokens that can't be encoded
            pass

    return multibyte_tokens

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
    only_in_2 = hex_set2 - hex_set2

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
    print(f"\n🔄 OVERLAP EXAMPLES (first 20 shared tokens by ID in {name1})")
    print(f"  {'ID1':<8} {'ID2':<8} {'Len':<5} {'Hex':<20} {'Printable'}")
    print(f"  {'-'*8} {'-'*8} {'-'*5} {'-'*20} {'-'*20}")

    overlap_list = []
    for hex_bytes in overlap:
        tid1, info1 = bytes1[hex_bytes]
        tid2, info2 = bytes2[hex_bytes]
        overlap_list.append((tid1, tid2, info1))

    # Sort by ID in first tokenizer
    overlap_list.sort(key=lambda x: x[0])

    for tid1, tid2, info in overlap_list[:20]:
        printable = ''.join(c if 32 <= ord(c) < 127 else '.' for c in info['str'])
        print(f"  {tid1:<8} {tid2:<8} {info['length']:<5} {info['hex']:<20} {printable[:20]}")

    # Show unique examples
    print(f"\n🔹 UNIQUE TO {name1} (first 20 by ID)")
    print(f"  {'ID':<8} {'Len':<5} {'Hex':<20} {'Printable'}")
    print(f"  {'-'*8} {'-'*5} {'-'*20} {'-'*20}")

    unique1_list = []
    for hex_bytes in only_in_1:
        tid1, info1 = bytes1[hex_bytes]
        unique1_list.append((tid1, info1))

    unique1_list.sort(key=lambda x: x[0])

    for tid, info in unique1_list[:20]:
        printable = ''.join(c if 32 <= ord(c) < 127 else '.' for c in info['str'])
        print(f"  {tid:<8} {info['length']:<5} {info['hex']:<20} {printable[:20]}")

    print(f"\n🔹 UNIQUE TO {name2} (first 20 by ID)")
    print(f"  {'ID':<8} {'Len':<5} {'Hex':<20} {'Printable'}")
    print(f"  {'-'*8} {'-'*5} {'-'*20} {'-'*20}")

    unique2_list = []
    for hex_bytes in only_in_2:
        tid2, info2 = bytes2[hex_bytes]
        unique2_list.append((tid2, info2))

    unique2_list.sort(key=lambda x: x[0])

    for tid, info in unique2_list[:20]:
        printable = ''.join(c if 32 <= ord(c) < 127 else '.' for c in info['str'])
        print(f"  {tid:<8} {info['length']:<5} {info['hex']:<20} {printable[:20]}")

    # Pattern analysis
    print(f"\n🔍 PATTERN ANALYSIS")

    def analyze_patterns(tokens, name):
        patterns = {
            'x86_rex': 0,  # REX prefixes (0x48, 0x4c, 0x4d, etc.)
            'padding': 0,  # Common padding (0xcc, 0x90, 0x00)
            'elf_magic': 0,  # ELF related
            'pe_magic': 0,  # PE related
            'printable': 0,  # Contains printable ASCII
            'high_entropy': 0,  # All bytes > 0x7f
        }

        for info in tokens.values():
            hex_str = info['hex']
            token_str = info['str']

            # Check for x86 REX prefixes
            if hex_str.startswith(('48', '4c', '4d', '49')):
                patterns['x86_rex'] += 1

            # Check for padding
            if any(pad in hex_str for pad in ['cccc', '9090', '0000']):
                patterns['padding'] += 1

            # Check for ELF magic
            if '7f454c46' in hex_str:  # 0x7f ELF
                patterns['elf_magic'] += 1

            # Check for PE magic
            if '4d5a' in hex_str:  # MZ
                patterns['pe_magic'] += 1

            # Check if contains printable ASCII
            if any(32 <= ord(c) < 127 for c in token_str):
                patterns['printable'] += 1

            # Check for high entropy (all bytes > 0x7f)
            if all(b > 0x7f for b in info['bytes']):
                patterns['high_entropy'] += 1

        return patterns

    patterns1 = analyze_patterns(tokens1, name1)
    patterns2 = analyze_patterns(tokens2, name2)

    print(f"\n  {name1}:")
    for pattern, count in patterns1.items():
        print(f"    {pattern:<20} {count:>4} ({count/len(tokens1)*100:>5.1f}%)")

    print(f"\n  {name2}:")
    for pattern, count in patterns2.items():
        print(f"    {pattern:<20} {count:>4} ({count/len(tokens2)*100:>5.1f}%)")

    return {
        'overlap_count': len(overlap),
        'overlap_pct': len(overlap) / len(tokens1) * 100,
        'unique1_count': len(only_in_1),
        'unique2_count': len(only_in_2),
        'patterns1': patterns1,
        'patterns2': patterns2,
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

    # Filter multibyte tokens
    print(f"\n{'='*80}")
    print("FILTERING MULTI-BYTE TOKENS")
    print(f"{'='*80}")

    multi005 = filter_multibyte_tokens(id_to_token005)
    print(f"  binary-tokenizer-005: {len(multi005)} multi-byte tokens")

    multi002 = filter_multibyte_tokens(id_to_token002)
    print(f"  glaurung-tokenizer-002: {len(multi002)} multi-byte tokens")

    # Get top 100
    print(f"\n{'='*80}")
    print("EXTRACTING TOP 100 TOKENS")
    print(f"{'='*80}")

    top100_005 = get_top_n_tokens(multi005, 100)
    print(f"  binary-tokenizer-005: {len(top100_005)} tokens")

    top100_002 = get_top_n_tokens(multi002, 100)
    print(f"  glaurung-tokenizer-002: {len(top100_002)} tokens")

    # Compare
    results = compare_token_sets(
        top100_005, top100_002,
        "binary-tokenizer-005",
        "glaurung-tokenizer-002"
    )

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n✅ Overlap: {results['overlap_pct']:.1f}% of tokens are shared")
    print(f"📊 Each tokenizer has unique patterns suited to their training data")

if __name__ == "__main__":
    main()
