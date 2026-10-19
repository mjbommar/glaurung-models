#!/usr/bin/env python3
"""Analyze common byte patterns in binary files."""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import struct

def analyze_runs(data, min_run=4):
    """Find runs of repeated bytes."""
    runs = []
    if not data:
        return runs
    
    current_byte = data[0]
    run_start = 0
    
    for i in range(1, len(data)):
        if data[i] != current_byte:
            run_length = i - run_start
            if run_length >= min_run:
                runs.append((current_byte, run_start, run_length))
            current_byte = data[i]
            run_start = i
    
    # Check final run
    run_length = len(data) - run_start
    if run_length >= min_run:
        runs.append((current_byte, run_start, run_length))
    
    return runs

def analyze_common_sequences(data, seq_len=4, top_n=20):
    """Find common byte sequences of a given length."""
    sequences = Counter()
    for i in range(len(data) - seq_len + 1):
        seq = data[i:i+seq_len]
        sequences[seq] += 1
    return sequences.most_common(top_n)

def analyze_binary(filepath, max_size=10*1024*1024):  # 10MB limit for analysis
    """Analyze patterns in a single binary file."""
    try:
        size = os.path.getsize(filepath)
        if size > max_size:
            # Sample the file
            with open(filepath, 'rb') as f:
                # Read beginning, middle, and end
                data = f.read(max_size // 3)
                f.seek(size // 2)
                data += f.read(max_size // 3)
                f.seek(-max_size // 3, 2)
                data += f.read(max_size // 3)
        else:
            with open(filepath, 'rb') as f:
                data = f.read()
        
        # Analyze runs
        runs = analyze_runs(data)
        run_stats = defaultdict(lambda: {'count': 0, 'total_bytes': 0})
        for byte_val, _, length in runs:
            run_stats[byte_val]['count'] += 1
            run_stats[byte_val]['total_bytes'] += length
        
        # Analyze common 4-byte sequences
        seq4 = analyze_common_sequences(data, 4, 10)
        
        # Analyze common 8-byte sequences
        seq8 = analyze_common_sequences(data, 8, 10)
        
        return {
            'size': size,
            'run_stats': dict(run_stats),
            'seq4': seq4,
            'seq8': seq8,
            'total_runs': len(runs),
            'largest_run': max(runs, key=lambda x: x[2]) if runs else None
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_directory(directory, extensions=None):
    """Analyze all binary files in a directory."""
    path = Path(directory)
    results = {}
    
    # Get all files
    if extensions:
        files = []
        for ext in extensions:
            files.extend(path.glob(f'*{ext}'))
    else:
        files = [f for f in path.iterdir() if f.is_file()]
    
    for filepath in files:
        if filepath.is_file():
            result = analyze_binary(filepath)
            if 'error' not in result:
                results[filepath.name] = result
    
    return results

def summarize_results(results):
    """Summarize analysis results across multiple files."""
    total_files = len(results)
    total_size = sum(r['size'] for r in results.values())
    
    # Aggregate run statistics
    byte_runs = defaultdict(lambda: {'files': 0, 'total_bytes': 0})
    all_seq4 = Counter()
    all_seq8 = Counter()
    
    for filename, result in results.items():
        for byte_val, stats in result.get('run_stats', {}).items():
            byte_runs[byte_val]['files'] += 1
            byte_runs[byte_val]['total_bytes'] += stats['total_bytes']
        
        for seq, count in result.get('seq4', []):
            all_seq4[seq] += count
        
        for seq, count in result.get('seq8', []):
            all_seq8[seq] += count
    
    print(f"\nAnalyzed {total_files} files, total size: {total_size:,} bytes")
    print(f"\n{'='*60}")
    print("Most common run patterns (bytes that repeat):")
    print(f"{'Byte':<10} {'Files':<10} {'Total Bytes':<15} {'Description'}")
    print(f"{'-'*50}")
    
    sorted_runs = sorted(byte_runs.items(), key=lambda x: x[1]['total_bytes'], reverse=True)[:10]
    for byte_val, stats in sorted_runs:
        desc = {
            0x00: 'NULL (padding/uninitialized)',
            0x90: 'x86 NOP',
            0xFF: 'FF (flash/erase)',
            0xCC: 'INT3 (breakpoint)',
            0x20: 'SPACE',
            0x0A: 'NEWLINE',
            0x41: 'A (padding)',
        }.get(byte_val, f'0x{byte_val:02X}')
        print(f"0x{byte_val:02X}       {stats['files']:<10} {stats['total_bytes']:<15,} {desc}")
    
    print(f"\n{'='*60}")
    print("Most common 4-byte sequences:")
    print(f"{'Sequence':<20} {'Count':<10} {'Hex'}")
    print(f"{'-'*50}")
    for seq, count in all_seq4.most_common(10):
        hex_str = ' '.join(f'{b:02X}' for b in seq)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in seq)
        print(f"{ascii_str:<20} {count:<10} {hex_str}")
    
    print(f"\n{'='*60}")
    print("Most common 8-byte sequences:")
    print(f"{'Sequence':<20} {'Count':<10} {'Hex'}")
    print(f"{'-'*50}")
    for seq, count in all_seq8.most_common(10):
        hex_str = ' '.join(f'{b:02X}' for b in seq)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in seq)[:20]
        print(f"{ascii_str:<20} {count:<10} {hex_str}")

if __name__ == '__main__':
    directories = [
        '/usr/bin',
        '/usr/lib',
        '/usr/local/bin',
        '/bin',
        '/sbin'
    ]
    
    all_results = {}
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nAnalyzing {directory}...")
            results = analyze_directory(directory)
            all_results.update(results)
    
    summarize_results(all_results)