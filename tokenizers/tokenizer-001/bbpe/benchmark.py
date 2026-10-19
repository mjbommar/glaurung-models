#!/usr/bin/env python3
"""Benchmark different tokenizer configurations."""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple
import gzip
import statistics

def train_tokenizer(config: Dict, output_path: str) -> float:
    """Train a tokenizer with given configuration and return training time."""
    cmd = [
        "./target/release/bbpe", "train",
        "/usr/bin", "/usr/local/bin",  # Training directories
        "--mode", "fixed",
        "--fixed-bytes", str(config.get("chunk_size", 4096)),
        "--vocab-size", str(config.get("vocab_size", 8192)),
        "--min-frequency", str(config.get("min_freq", 2)),
        "--output", output_path,
        "--progress", "false",
        "--entropy-filter", "false",
        "--reserved", "20"
    ]
    
    # Add pattern configuration
    if config.get("patterns_enabled", False):
        cmd.append("--enable-patterns")
        cmd.extend(["--min-pattern-pow2", str(config.get("min_pow", 3))])
        cmd.extend(["--max-pattern-pow2", str(config.get("max_pow", 11))])
        
        # Add individual pattern flags
        for pattern in ["null", "nop_x86", "int3", "ff", "space"]:
            if config.get(f"pattern_{pattern}", True):
                cmd.append(f"--pattern-{pattern.replace('_', '-')}")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    training_time = time.time() - start
    
    if result.returncode != 0:
        print(f"Training failed: {result.stderr}")
        return -1
    
    return training_time

def test_tokenizer(tokenizer_path: str, test_files: List[str]) -> Dict:
    """Test tokenizer on files and return metrics."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    
    from tokenizers import Tokenizer
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    metrics = {
        'total_bytes': 0,
        'total_tokens': 0,
        'pattern_tokens': 0,
        'compression_ratios': [],
        'files_tested': 0
    }
    
    # Load tokenizer data to identify pattern tokens
    with open(tokenizer_path) as f:
        tok_data = json.load(f)
    
    pattern_ids = set()
    for token in tok_data.get('added_tokens', []):
        content = token['content']
        if len(content) >= 8 and all(c == content[0] for c in content):
            pattern_ids.add(token['id'])
    
    for filepath in test_files[:10]:  # Test on first 10 files
        try:
            with open(filepath, 'rb') as f:
                data = f.read(100_000)  # Read first 100KB
                text = data.decode('latin-1', errors='replace')
            
            encoded = tokenizer.encode(text)
            tokens = encoded.ids
            
            metrics['total_bytes'] += len(data)
            metrics['total_tokens'] += len(tokens)
            metrics['pattern_tokens'] += sum(1 for t in tokens if t in pattern_ids)
            metrics['compression_ratios'].append(len(tokens) / len(data))
            metrics['files_tested'] += 1
            
        except Exception as e:
            pass
    
    if metrics['compression_ratios']:
        metrics['avg_ratio'] = statistics.mean(metrics['compression_ratios'])
        metrics['median_ratio'] = statistics.median(metrics['compression_ratios'])
    
    return metrics

def compare_with_gzip(test_files: List[str]) -> Dict:
    """Compare with gzip compression."""
    total_original = 0
    total_compressed = 0
    
    for filepath in test_files[:10]:
        try:
            with open(filepath, 'rb') as f:
                data = f.read(100_000)
            compressed = gzip.compress(data, compresslevel=6)
            total_original += len(data)
            total_compressed += len(compressed)
        except:
            pass
    
    return {
        'original_bytes': total_original,
        'compressed_bytes': total_compressed,
        'ratio': total_compressed / total_original if total_original > 0 else 1
    }

def main():
    # Test configurations
    configs = [
        {
            'name': 'Baseline (no patterns)',
            'patterns_enabled': False,
            'vocab_size': 8192,
            'chunk_size': 4096
        },
        {
            'name': 'Basic patterns (null, nop, ff)',
            'patterns_enabled': True,
            'pattern_null': True,
            'pattern_nop_x86': True,
            'pattern_int3': True,
            'pattern_ff': True,
            'pattern_space': False,
            'min_pow': 3,
            'max_pow': 10,
            'vocab_size': 8192,
            'chunk_size': 4096
        },
        {
            'name': 'All patterns',
            'patterns_enabled': True,
            'pattern_null': True,
            'pattern_nop_x86': True,
            'pattern_int3': True,
            'pattern_ff': True,
            'pattern_space': True,
            'min_pow': 3,
            'max_pow': 11,
            'vocab_size': 8192,
            'chunk_size': 4096
        },
        {
            'name': 'Optimized (larger chunks, all patterns)',
            'patterns_enabled': True,
            'pattern_null': True,
            'pattern_nop_x86': True,
            'pattern_int3': True,
            'pattern_ff': True,
            'pattern_space': True,
            'min_pow': 2,
            'max_pow': 12,
            'vocab_size': 16384,
            'chunk_size': 8192
        }
    ]
    
    # Get test files
    test_files = list(Path('/usr/bin').glob('*'))[:50]
    
    print("=" * 80)
    print("TOKENIZER BENCHMARK")
    print("=" * 80)
    
    # Compare with gzip first
    print("\nBaseline comparison:")
    gzip_stats = compare_with_gzip(test_files)
    print(f"gzip compression ratio: {gzip_stats['ratio']:.3f}")
    
    results = []
    
    for config in configs:
        print(f"\n{'-' * 60}")
        print(f"Testing: {config['name']}")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tokenizer_path = f.name
        
        # Train
        train_time = train_tokenizer(config, tokenizer_path)
        if train_time < 0:
            print("  Training failed!")
            continue
        
        print(f"  Training time: {train_time:.2f}s")
        
        # Test
        metrics = test_tokenizer(tokenizer_path, test_files)
        
        print(f"  Files tested: {metrics['files_tested']}")
        print(f"  Avg compression: {metrics.get('avg_ratio', 0):.3f} tokens/byte")
        print(f"  Pattern tokens used: {metrics['pattern_tokens']:,}")
        
        if metrics['pattern_tokens'] > 0 and metrics['total_tokens'] > 0:
            pattern_pct = metrics['pattern_tokens'] / metrics['total_tokens'] * 100
            print(f"  Pattern token usage: {pattern_pct:.1f}%")
        
        # Calculate effective compression vs gzip
        token_bytes = metrics['total_tokens'] * 2  # 16-bit token IDs
        compression_vs_gzip = token_bytes / gzip_stats['compressed_bytes'] if gzip_stats['compressed_bytes'] > 0 else 1
        print(f"  Size vs gzip: {compression_vs_gzip:.2f}x")
        
        results.append({
            'config': config['name'],
            'train_time': train_time,
            'metrics': metrics,
            'vs_gzip': compression_vs_gzip
        })
        
        # Clean up
        Path(tokenizer_path).unlink(missing_ok=True)
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    
    print(f"\n{'Config':<40} {'Ratio':<12} {'vs gzip':<10} {'Train(s)':<10}")
    print("-" * 72)
    for r in results:
        ratio = r['metrics'].get('avg_ratio', 0)
        print(f"{r['config']:<40} {ratio:<12.3f} {r['vs_gzip']:<10.2f} {r['train_time']:<10.1f}")
    
    # Find best configuration
    if results:
        best = min(results, key=lambda x: x['metrics'].get('avg_ratio', 999))
        print(f"\nBest configuration: {best['config']}")
        print(f"Compression ratio: {best['metrics'].get('avg_ratio', 0):.3f} tokens/byte")

if __name__ == '__main__':
    main()