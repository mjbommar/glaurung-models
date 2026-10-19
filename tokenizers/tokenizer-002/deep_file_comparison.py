#!/usr/bin/env python3
"""
Deep comparison of tokenizer files - beyond just vocabulary
Check if files are byte-for-byte identical or just token sets match
"""

import json
import urllib.request
import hashlib

def download_hf_tokenizer():
    """Download the HuggingFace tokenizer JSON"""
    print("Downloading HuggingFace tokenizer JSON...")
    url = "https://huggingface.co/mjbommar/glaurung-binary-tokenizer-001/resolve/main/tokenizer.json"

    with urllib.request.urlopen(url) as response:
        content = response.read()

    print(f"  Downloaded {len(content):,} bytes")
    return content

def load_local_file(path):
    """Load local file as bytes"""
    print(f"Loading local file: {path}...")
    with open(path, 'rb') as f:
        content = f.read()
    print(f"  Loaded {len(content):,} bytes")
    return content

def compute_hash(data):
    """Compute SHA256 hash"""
    return hashlib.sha256(data).hexdigest()

def compare_json_structure(json1, json2):
    """Compare JSON structure deeply"""
    print("\n" + "="*80)
    print("DEEP JSON STRUCTURE COMPARISON")
    print("="*80)

    # Parse both
    data1 = json.loads(json1)
    data2 = json.loads(json2)

    # Compare top-level keys
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    print(f"\nTop-level keys:")
    print(f"  HF: {sorted(keys1)}")
    print(f"  Local: {sorted(keys2)}")
    print(f"  Common: {sorted(keys1 & keys2)}")
    print(f"  Only in HF: {sorted(keys1 - keys2)}")
    print(f"  Only in Local: {sorted(keys2 - keys1)}")

    # Deep diff
    print("\n" + "-"*80)
    print("Deep Diff Analysis:")
    print("-"*80)

    diff = DeepDiff(data1, data2, ignore_order=True, verbose_level=2)

    if not diff:
        print("  ✓ NO DIFFERENCES FOUND - Structures are identical!")
        return True
    else:
        print("  ✗ DIFFERENCES DETECTED:")
        for key, value in diff.items():
            print(f"\n  {key}:")
            print(f"    {value}")
        return False

def compare_sections(json1, json2):
    """Compare specific sections of the tokenizer"""
    data1 = json.loads(json1)
    data2 = json.loads(json2)

    sections = [
        'version',
        'truncation',
        'padding',
        'normalizer',
        'pre_tokenizer',
        'post_processor',
        'decoder',
        'model'
    ]

    print("\n" + "="*80)
    print("SECTION-BY-SECTION COMPARISON")
    print("="*80)

    for section in sections:
        print(f"\n[{section}]")
        val1 = data1.get(section)
        val2 = data2.get(section)

        if val1 == val2:
            print(f"  ✓ IDENTICAL")
        else:
            print(f"  ✗ DIFFERENT")
            diff = DeepDiff(val1, val2, ignore_order=True)
            if diff:
                print(f"    Diff: {diff}")

def main():
    # Download and load files
    hf_bytes = download_hf_tokenizer()
    local_bytes = load_local_file('tokenizer-003.json')

    # Byte-level comparison
    print("\n" + "="*80)
    print("BYTE-LEVEL COMPARISON")
    print("="*80)

    hf_hash = compute_hash(hf_bytes)
    local_hash = compute_hash(local_bytes)

    print(f"\nFile sizes:")
    print(f"  HF:    {len(hf_bytes):,} bytes")
    print(f"  Local: {len(local_bytes):,} bytes")
    print(f"  Match: {len(hf_bytes) == len(local_bytes)}")

    print(f"\nSHA256 hashes:")
    print(f"  HF:    {hf_hash}")
    print(f"  Local: {local_hash}")
    print(f"  Match: {hf_hash == local_hash}")

    if hf_bytes == local_bytes:
        print("\n" + "="*80)
        print("RESULT: FILES ARE BYTE-FOR-BYTE IDENTICAL!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("RESULT: FILES DIFFER AT BYTE LEVEL")
        print("="*80)

        # Continue with structural comparison
        compare_json_structure(hf_bytes, local_bytes)
        compare_sections(hf_bytes, local_bytes)

if __name__ == '__main__':
    main()
