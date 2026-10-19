#!/usr/bin/env python3
"""Simple diff between the two tokenizer files"""

import json

# Load both files
with open('/tmp/hf_tokenizer.json', 'r') as f:
    hf_data = json.load(f)

with open('tokenizer-003.json', 'r') as f:
    local_data = json.load(f)

# Compare top-level keys
hf_keys = set(hf_data.keys())
local_keys = set(local_data.keys())

print("Top-level keys:")
print(f"  HF: {sorted(hf_keys)}")
print(f"  Local: {sorted(local_keys)}")
print(f"  Only in HF: {sorted(hf_keys - local_keys)}")
print(f"  Only in Local: {sorted(local_keys - hf_keys)}")

# Compare vocab sizes
hf_vocab = hf_data['model']['vocab']
local_vocab = local_data['model']['vocab']
print(f"\nVocab sizes:")
print(f"  HF: {len(hf_vocab)}")
print(f"  Local: {len(local_vocab)}")
print(f"  Match: {len(hf_vocab) == len(local_vocab)}")

# Check if vocabs are identical
print(f"\nVocab token sets identical: {set(hf_vocab.keys()) == set(local_vocab.keys())}")

# Compare added_tokens
hf_added = hf_data.get('added_tokens', [])
local_added = local_data.get('added_tokens', [])
print(f"\nAdded tokens:")
print(f"  HF: {len(hf_added)}")
print(f"  Local: {len(local_added)}")

# Check specific sections
sections = ['version', 'truncation', 'padding', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder']
print(f"\nSection comparison:")
for section in sections:
    hf_val = hf_data.get(section)
    local_val = local_data.get(section)
    match = hf_val == local_val
    print(f"  {section}: {'✓' if match else '✗'}")

# Check model merges
hf_merges = hf_data['model'].get('merges', [])
local_merges = local_data['model'].get('merges', [])
print(f"\nModel merges:")
print(f"  HF: {len(hf_merges)}")
print(f"  Local: {len(local_merges)}")
print(f"  Match: {hf_merges == local_merges}")

# File size difference explanation
print(f"\n" + "="*80)
print("FILE SIZE ANALYSIS")
print("="*80)

# Estimate sizes of different sections
import sys
hf_vocab_str = json.dumps(hf_vocab)
local_vocab_str = json.dumps(local_vocab)
hf_merges_str = json.dumps(hf_merges)
local_merges_str = json.dumps(local_merges)

print(f"\nVocab JSON size:")
print(f"  HF: {len(hf_vocab_str):,} bytes")
print(f"  Local: {len(local_vocab_str):,} bytes")
print(f"  Difference: {len(hf_vocab_str) - len(local_vocab_str):,} bytes")

print(f"\nMerges JSON size:")
print(f"  HF: {len(hf_merges_str):,} bytes")
print(f"  Local: {len(local_merges_str):,} bytes")
print(f"  Difference: {len(hf_merges_str) - len(local_merges_str):,} bytes")
