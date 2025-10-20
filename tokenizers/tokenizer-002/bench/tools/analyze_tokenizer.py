#!/usr/bin/env python3
"""Analyze the 64K tokenizer and compare with 32K."""

import json
from pathlib import Path
from collections import Counter

def load_tokenizer(path):
    """Load tokenizer vocabulary."""
    with open(path) as f:
        data = json.load(f)
    return data['model']['vocab']

def analyze_length_distribution(vocab, name):
    """Analyze token length distribution."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")

    lengths = Counter()
    id_to_token = {v: k for k, v in vocab.items()}

    # Analyze all learned tokens (256+)
    for token_id in range(256, len(id_to_token) + 256):
        if token_id not in id_to_token:
            continue
        token_str = id_to_token[token_id]
        token_bytes = token_str.encode('latin-1')
        lengths[len(token_bytes)] += 1

    total_learned = sum(lengths.values())
    avg_length = sum(l * c for l, c in lengths.items()) / total_learned

    print(f"\nVocabulary: {len(vocab)} tokens ({total_learned} learned)")
    print(f"Average token length: {avg_length:.3f} bytes\n")

    print("Length distribution:")
    for length in sorted(lengths.keys()):
        count = lengths[length]
        pct = 100 * count / total_learned
        bar = '█' * int(pct / 2)
        print(f"  Length {length:2d}: {count:6d} tokens ({pct:5.1f}%) {bar}")

    # Find length-3 count
    len3_count = lengths.get(3, 0)
    len3_pct = 100 * len3_count / total_learned

    return {
        'total': len(vocab),
        'learned': total_learned,
        'avg_length': avg_length,
        'lengths': dict(lengths),
        'length_3_count': len3_count,
        'length_3_pct': len3_pct
    }

def main():
    base_dir = Path(__file__).parent

    print("="*80)
    print("  64K vs 32K TOKENIZER COMPARISON")
    print("="*80)

    # Load tokenizers
    tok_64k = load_tokenizer(base_dir / "glaurung-tokenizer-002.json")
    tok_32k = load_tokenizer(base_dir.parent / "binaries-small-32k" / "tokenizer-default-1-16.json")

    # Analyze both
    stats_64k = analyze_length_distribution(tok_64k, "64K TOKENIZER (glaurung-tokenizer-002)")
    stats_32k = analyze_length_distribution(tok_32k, "32K TOKENIZER (baseline)")

    # Comparison
    print(f"\n{'='*80}")
    print("  KEY COMPARISONS")
    print(f"{'='*80}")

    print("\n1. VOCABULARY SIZE:")
    print(f"   32K: {stats_32k['total']:,} tokens")
    print(f"   64K: {stats_64k['total']:,} tokens")
    print(f"   Increase: {stats_64k['total'] / stats_32k['total']:.2f}x")

    print("\n2. AVERAGE TOKEN LENGTH:")
    print(f"   32K: {stats_32k['avg_length']:.3f} bytes/token")
    print(f"   64K: {stats_64k['avg_length']:.3f} bytes/token")
    improvement = (stats_64k['avg_length'] / stats_32k['avg_length'] - 1) * 100
    print(f"   Change: {improvement:+.1f}%")

    print("\n3. LENGTH-3 TOKENS (x86-64 instruction breakthrough):")
    print(f"   32K: {stats_32k['length_3_count']:,} tokens ({stats_32k['length_3_pct']:.1f}%)")
    print(f"   64K: {stats_64k['length_3_count']:,} tokens ({stats_64k['length_3_pct']:.1f}%)")
    print(f"   Increase: {stats_64k['length_3_count'] / max(stats_32k['length_3_count'], 1):.1f}x more")

    print("\n4. EXPECTED COMPRESSION IMPROVEMENT:")
    expected_improvement = (stats_64k['avg_length'] / stats_32k['avg_length'] - 1) * 100
    print(f"   Theoretical: ~{expected_improvement:.1f}% better compression")
    print(f"   (Longer tokens = fewer tokens needed for same data)")

    print("\n5. TRAINING TIME:")
    print(f"   32K: 4.66 hours (16,775s)")
    print(f"   64K: 8.46 hours (30,443s)")
    print(f"   Ratio: {30443 / 16775:.2f}x longer (expected for 2x vocab)")

    print()

if __name__ == '__main__':
    main()
