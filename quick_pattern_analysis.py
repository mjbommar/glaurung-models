#!/usr/bin/env python3
"""Quick pattern analysis on a sample of binaries."""

import os
import random
from collections import Counter, defaultdict
from pathlib import Path

def find_repeated_patterns(data, min_length=8, max_length=64):
    """Find sections of repeated single bytes."""
    patterns = defaultdict(int)
    
    i = 0
    while i < len(data):
        # Check if we have a run
        start_byte = data[i]
        j = i + 1
        while j < len(data) and j < i + max_length and data[j] == start_byte:
            j += 1
        
        run_length = j - i
        if run_length >= min_length:
            # Round to power of 2 for categorization
            for power in range(3, 15):  # 8 to 16384
                size = 2 ** power
                if run_length >= size:
                    patterns[(start_byte, size)] += 1
        
        i = j if run_length >= min_length else i + 1
    
    return patterns

def quick_sample_analysis():
    """Do a quick analysis on a sample of files."""
    # Sample some binaries
    dirs = ['/usr/bin', '/bin', '/usr/local/bin']
    sample_files = []
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            files = [f for f in Path(dir_path).iterdir() if f.is_file()]
            # Take up to 10 random files from each directory
            sample_files.extend(random.sample(files, min(10, len(files))))
    
    print(f"Analyzing {len(sample_files)} sampled files...")
    
    all_patterns = defaultdict(int)
    file_sizes = []
    
    for filepath in sample_files:
        try:
            size = os.path.getsize(filepath)
            file_sizes.append(size)
            
            # Read first 100KB
            with open(filepath, 'rb') as f:
                data = f.read(100_000)
            
            patterns = find_repeated_patterns(data)
            for key, count in patterns.items():
                all_patterns[key] += count
                
        except:
            pass
    
    # Report findings
    print(f"\nAnalyzed {len(file_sizes)} files")
    print(f"Average size: {sum(file_sizes)/len(file_sizes)/1024:.1f} KB")
    
    print("\nMost common repeated byte patterns:")
    print(f"{'Byte':<10} {'Size':<10} {'Occurrences':<15} {'Description'}")
    print("-" * 50)
    
    sorted_patterns = sorted(all_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
    for (byte_val, size), count in sorted_patterns:
        desc = {
            0x00: 'NULL',
            0x90: 'NOP(x86)',
            0xFF: 'FF',
            0xCC: 'INT3',
            0x20: 'SPACE',
            0x0A: 'LF',
            0x2F: 'SLASH',
        }.get(byte_val, f'0x{byte_val:02X}')
        print(f"0x{byte_val:02X}       {size:<10} {count:<15} {desc}")
    
    # Suggest additional patterns
    print("\nSuggested additional patterns to add:")
    unique_bytes = set(byte_val for (byte_val, _), _ in sorted_patterns[:10])
    for byte_val in unique_bytes:
        if byte_val not in [0x00, 0x90, 0xFF, 0xCC]:  # Already supported
            print(f"  - 0x{byte_val:02X} patterns")

if __name__ == '__main__':
    quick_sample_analysis()