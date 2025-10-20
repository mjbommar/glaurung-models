#!/usr/bin/env python3
"""Test compression: 64K vs 32K tokenizer on real binaries."""

import json
from pathlib import Path
from tokenizers import Tokenizer

def test_compression(tokenizer_path, binary_path, name):
    """Test tokenizer compression on binary."""
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    with open(binary_path, 'rb') as f:
        data = f.read()

    # Encode
    encoding = tokenizer.encode(data.decode('latin-1'))
    tokens = encoding.ids

    bytes_per_token = len(data) / len(tokens)

    return {
        'name': name,
        'size': len(data),
        'tokens': len(tokens),
        'bytes_per_token': bytes_per_token
    }

def main():
    base_dir = Path(__file__).parent

    # Test binaries
    test_files = [
        ('/usr/bin/bash', 'bash'),
        ('/usr/bin/python3.12', 'python3.12'),
        ('/usr/bin/gcc-13', 'gcc-13'),
        ('/usr/bin/ls', 'ls'),
        ('/usr/bin/grep', 'grep'),
    ]

    print("="*80)
    print("  64K vs 32K COMPRESSION BENCHMARK")
    print("="*80)
    print("\nTesting on /usr/bin binaries (not in training set)\n")

    # Load tokenizers
    tok_64k_path = base_dir / "glaurung-tokenizer-002.json"
    tok_32k_path = base_dir.parent / "binaries-small-32k" / "tokenizer-default-1-16.json"

    results = []
    total_bytes = 0
    total_tokens_32k = 0
    total_tokens_64k = 0

    for binary_path, name in test_files:
        if not Path(binary_path).exists():
            print(f"⚠️  Skipping {name} (not found)")
            continue

        result_32k = test_compression(tok_32k_path, binary_path, name)
        result_64k = test_compression(tok_64k_path, binary_path, name)

        improvement = (result_32k['tokens'] - result_64k['tokens']) / result_32k['tokens'] * 100
        compression_improvement = (result_64k['bytes_per_token'] / result_32k['bytes_per_token'] - 1) * 100

        print(f"{name:15s} ({result_32k['size']/1024/1024:6.2f} MB)")
        print(f"  32K: {result_32k['tokens']:8,} tokens ({result_32k['bytes_per_token']:.3f} bytes/token)")
        print(f"  64K: {result_64k['tokens']:8,} tokens ({result_64k['bytes_per_token']:.3f} bytes/token)")
        print(f"  Improvement: {improvement:+.1f}% fewer tokens, {compression_improvement:+.1f}% better compression")
        print()

        total_bytes += result_32k['size']
        total_tokens_32k += result_32k['tokens']
        total_tokens_64k += result_64k['tokens']

        results.append({
            'name': name,
            'size': result_32k['size'],
            'tokens_32k': result_32k['tokens'],
            'tokens_64k': result_64k['tokens'],
            'improvement': improvement,
            'compression_improvement': compression_improvement
        })

    # Aggregate results
    print("="*80)
    print("  AGGREGATE RESULTS")
    print("="*80)

    total_mb = total_bytes / 1024 / 1024
    avg_improvement = (total_tokens_32k - total_tokens_64k) / total_tokens_32k * 100

    bytes_per_token_32k = total_bytes / total_tokens_32k
    bytes_per_token_64k = total_bytes / total_tokens_64k
    compression_improvement = (bytes_per_token_64k / bytes_per_token_32k - 1) * 100

    print(f"\nTotal tested: {total_mb:.2f} MB across {len(results)} binaries")
    print(f"\n32K tokenizer: {total_tokens_32k:,} tokens ({bytes_per_token_32k:.3f} bytes/token)")
    print(f"64K tokenizer: {total_tokens_64k:,} tokens ({bytes_per_token_64k:.3f} bytes/token)")
    print(f"\n✓ Improvement: {avg_improvement:.1f}% fewer tokens")
    print(f"✓ Compression: {compression_improvement:+.1f}% better bytes/token")

    # Show improvement per binary
    print(f"\nPer-binary improvements:")
    for r in sorted(results, key=lambda x: -x['improvement']):
        print(f"  {r['name']:15s}: {r['improvement']:+.1f}% fewer tokens")

    print(f"\n{'='*80}")
    print("  CONCLUSION")
    print("="*80)
    print(f"\nDoubling vocabulary from 32K → 64K provides:")
    print(f"  • {avg_improvement:.1f}% reduction in token count")
    print(f"  • {compression_improvement:.1f}% improvement in compression ratio")
    print(f"  • {bytes_per_token_64k:.3f} bytes/token (vs {bytes_per_token_32k:.3f})")
    print(f"\nThis translates to:")
    print(f"  • Shorter sequence lengths for transformers")
    print(f"  • Less compute per byte of binary")
    print(f"  • Better model efficiency")

    print()

if __name__ == '__main__':
    main()
