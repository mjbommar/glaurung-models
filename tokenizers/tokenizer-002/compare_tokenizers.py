#!/usr/bin/env python3
"""
Compare glaurung-binary-tokenizer-001 from HuggingFace with tokenizer-003.json
Analyze vocabulary overlap and differences
"""

import json
import urllib.request
from collections import defaultdict

def load_hf_tokenizer():
    """Download and load the HuggingFace tokenizer JSON"""
    print("Downloading HuggingFace tokenizer: mjbommar/glaurung-binary-tokenizer-001...")
    url = "https://huggingface.co/mjbommar/glaurung-binary-tokenizer-001/resolve/main/tokenizer.json"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode('utf-8'))
        print("  ✓ Download complete")
        return data
    except Exception as e:
        print(f"  ✗ Error downloading: {e}")
        return None

def load_local_tokenizer(path):
    """Load the local tokenizer-003.json"""
    print(f"Loading local tokenizer: {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    print("  ✓ Load complete")
    return data

def extract_vocab(tokenizer_data):
    """Extract vocabulary from tokenizer JSON"""
    # Both should be raw JSON format
    model = tokenizer_data.get('model', {})
    vocab = model.get('vocab', {})
    return vocab

def extract_added_tokens(tokenizer_data):
    """Extract added_tokens from tokenizer JSON"""
    added = tokenizer_data.get('added_tokens', [])
    return added

def analyze_tokens(vocab):
    """Analyze token characteristics"""
    analysis = {
        'total_tokens': len(vocab),
        'single_byte': 0,
        'multi_byte': 0,
        'special_tokens': 0,
        'max_token_length': 0,
        'token_length_dist': defaultdict(int)
    }

    for token, idx in vocab.items():
        token_len = len(token.encode('latin-1', errors='ignore'))
        analysis['token_length_dist'][token_len] += 1
        analysis['max_token_length'] = max(analysis['max_token_length'], token_len)

        if token.startswith('<|') and token.endswith('|>'):
            analysis['special_tokens'] += 1
        elif token_len == 1:
            analysis['single_byte'] += 1
        else:
            analysis['multi_byte'] += 1

    return analysis

def compare_vocabularies(vocab1, vocab2, name1="HF", name2="Local"):
    """Create Venn diagram analysis of vocabularies"""
    tokens1 = set(vocab1.keys())
    tokens2 = set(vocab2.keys())

    only_in_1 = tokens1 - tokens2
    only_in_2 = tokens2 - tokens1
    common = tokens1 & tokens2

    print("\n" + "="*80)
    print("VENN DIAGRAM ANALYSIS")
    print("="*80)
    print(f"\n{name1} total tokens: {len(tokens1):,}")
    print(f"{name2} total tokens: {len(tokens2):,}")
    print(f"\nCommon tokens: {len(common):,} ({100*len(common)/max(len(tokens1), len(tokens2)):.2f}% of larger set)")
    print(f"Only in {name1}: {len(only_in_1):,}")
    print(f"Only in {name2}: {len(only_in_2):,}")

    return {
        'common': common,
        'only_in_1': only_in_1,
        'only_in_2': only_in_2
    }

def analyze_token_samples(tokens, vocab, name, limit=20):
    """Analyze sample tokens"""
    print(f"\n{name} Sample Tokens (up to {limit}):")
    print("-" * 80)

    sample = list(tokens)[:limit]
    for token in sample:
        token_id = vocab.get(token, '?')
        # Try to display token safely
        try:
            display = repr(token)
            if len(display) > 60:
                display = display[:60] + "..."
        except:
            display = f"<binary: {len(token)} bytes>"
        print(f"  ID {token_id:6}: {display}")

def deep_analysis(vocab1, vocab2, venn_data):
    """Perform deep analysis and interpretation"""
    print("\n" + "="*80)
    print("DEEP ANALYSIS & INTERPRETATION")
    print("="*80)

    # Analyze common tokens
    common_tokens = venn_data['common']

    # Check if all single bytes are common
    single_bytes_1 = {k for k in vocab1.keys() if len(k.encode('latin-1', errors='ignore')) == 1}
    single_bytes_2 = {k for k in vocab2.keys() if len(k.encode('latin-1', errors='ignore')) == 1}
    single_bytes_common = single_bytes_1 & single_bytes_2

    print(f"\nSingle-byte tokens:")
    print(f"  HF: {len(single_bytes_1)}")
    print(f"  Local: {len(single_bytes_2)}")
    print(f"  Common: {len(single_bytes_common)}")

    # Analyze multi-byte tokens
    multi_bytes_common = len([t for t in common_tokens if len(t.encode('latin-1', errors='ignore')) > 1])
    print(f"\nMulti-byte tokens in common: {multi_bytes_common:,}")

    # Check special tokens
    special_1 = {k for k in vocab1.keys() if k.startswith('<|') and k.endswith('|>')}
    special_2 = {k for k in vocab2.keys() if k.startswith('<|') and k.endswith('|>')}

    print(f"\nSpecial tokens:")
    print(f"  HF: {special_1}")
    print(f"  Local: {special_2}")
    print(f"  Common: {special_1 & special_2}")
    print(f"  Only in HF: {special_1 - special_2}")
    print(f"  Only in Local: {special_2 - special_1}")

    # Analyze vocabulary size relationship
    print(f"\nVocabulary size relationship:")
    vocab1_size = len(vocab1)
    vocab2_size = len(vocab2)
    size_diff = abs(vocab1_size - vocab2_size)
    print(f"  HF size: {vocab1_size:,}")
    print(f"  Local size: {vocab2_size:,}")
    print(f"  Difference: {size_diff:,}")

    if vocab1_size == 65536:
        print(f"  HF = 2^16 (exact power of 2)")
    if vocab2_size == 65536:
        print(f"  Local = 2^16 (exact power of 2)")

    # Calculate Jaccard similarity
    jaccard = len(common_tokens) / len(vocab1.keys() | vocab2.keys())
    print(f"\nJaccard similarity: {jaccard:.4f} ({jaccard*100:.2f}%)")

    # Analyze token length distribution for unique tokens
    print(f"\nUnique token characteristics:")

    only_hf = venn_data['only_in_1']
    only_local = venn_data['only_in_2']

    if only_hf:
        hf_lengths = [len(t.encode('latin-1', errors='ignore')) for t in list(only_hf)[:1000]]
        print(f"  HF-only tokens avg length: {sum(hf_lengths)/len(hf_lengths):.2f} bytes")

    if only_local:
        local_lengths = [len(t.encode('latin-1', errors='ignore')) for t in list(only_local)[:1000]]
        print(f"  Local-only tokens avg length: {sum(local_lengths)/len(local_lengths):.2f} bytes")

def main():
    # Load tokenizers
    hf_tokenizer = load_hf_tokenizer()
    if hf_tokenizer is None:
        print("Failed to load HuggingFace tokenizer. Exiting.")
        return

    local_tokenizer = load_local_tokenizer('tokenizer-003.json')

    # Extract vocabularies
    print("\nExtracting vocabularies...")
    hf_vocab = extract_vocab(hf_tokenizer)
    local_vocab = extract_vocab(local_tokenizer)

    # Extract added tokens
    print("\nExtracting added tokens...")
    hf_added = extract_added_tokens(hf_tokenizer)
    local_added = extract_added_tokens(local_tokenizer)
    print(f"  HF added tokens: {len(hf_added)}")
    print(f"  Local added tokens: {len(local_added)}")
    if hf_added:
        print(f"  HF added tokens list: {hf_added}")
    if local_added:
        print(f"  Local added tokens list: {local_added}")

    # Analyze individual tokenizers
    print("\n" + "="*80)
    print("HF TOKENIZER ANALYSIS")
    print("="*80)
    hf_analysis = analyze_tokens(hf_vocab)
    for key, value in hf_analysis.items():
        if key != 'token_length_dist':
            print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("LOCAL TOKENIZER ANALYSIS")
    print("="*80)
    local_analysis = analyze_tokens(local_vocab)
    for key, value in local_analysis.items():
        if key != 'token_length_dist':
            print(f"  {key}: {value}")

    # Compare vocabularies
    venn_data = compare_vocabularies(hf_vocab, local_vocab, "HuggingFace", "Local-003")

    # Sample tokens
    if venn_data['only_in_1']:
        analyze_token_samples(venn_data['only_in_1'], hf_vocab, "Only in HuggingFace", 15)

    if venn_data['only_in_2']:
        analyze_token_samples(venn_data['only_in_2'], local_vocab, "Only in Local-003", 15)

    if venn_data['common']:
        analyze_token_samples(venn_data['common'], hf_vocab, "Common Tokens", 15)

    # Deep analysis
    deep_analysis(hf_vocab, local_vocab, venn_data)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
