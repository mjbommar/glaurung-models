#!/usr/bin/env python3
"""Find the exact differences between tokenizers"""

import json

with open('/tmp/hf_tokenizer.json', 'r') as f:
    hf_data = json.load(f)

with open('tokenizer-003.json', 'r') as f:
    local_data = json.load(f)

def deep_compare(obj1, obj2, path=""):
    """Recursively compare two objects and report differences"""
    if type(obj1) != type(obj2):
        print(f"Type mismatch at {path}: {type(obj1).__name__} vs {type(obj2).__name__}")
        return False

    if isinstance(obj1, dict):
        keys1 = set(obj1.keys())
        keys2 = set(obj2.keys())

        if keys1 != keys2:
            print(f"Key mismatch at {path}:")
            print(f"  Only in HF: {keys1 - keys2}")
            print(f"  Only in Local: {keys2 - keys1}")
            return False

        all_match = True
        for key in keys1:
            if not deep_compare(obj1[key], obj2[key], f"{path}.{key}" if path else key):
                all_match = False
        return all_match

    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            print(f"Length mismatch at {path}: {len(obj1)} vs {len(obj2)}")
            return False

        all_match = True
        for i, (item1, item2) in enumerate(zip(obj1, obj2)):
            if not deep_compare(item1, item2, f"{path}[{i}]"):
                all_match = False
        return all_match

    else:
        if obj1 != obj2:
            # Only print if values are small enough
            if isinstance(obj1, str) and len(str(obj1)) < 100:
                print(f"Value mismatch at {path}:")
                print(f"  HF: {repr(obj1)}")
                print(f"  Local: {repr(obj2)}")
            elif not isinstance(obj1, str):
                print(f"Value mismatch at {path}:")
                print(f"  HF: {repr(obj1)}")
                print(f"  Local: {repr(obj2)}")
            else:
                print(f"Value mismatch at {path} (values too long to display)")
            return False
        return True

print("="*80)
print("DEEP COMPARISON - Finding all differences")
print("="*80)

result = deep_compare(hf_data, local_data)

if result:
    print("\n✓ Files are structurally identical!")
else:
    print("\n✗ Differences found above")
