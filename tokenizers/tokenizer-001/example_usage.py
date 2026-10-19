#!/usr/bin/env python3
"""
Example usage of the binary BPE tokenizer for arbitrary binary data.

This script demonstrates how to:
1. Load a pre-trained binary tokenizer
2. Encode arbitrary binary data (including machine code, images, etc.)
3. Decode token IDs back to binary data
4. Handle special tokens properly
"""

import json
from tokenizers import Tokenizer
from pathlib import Path
import sys

def bytes_to_latin1_string(data: bytes) -> str:
    """Convert raw bytes to Latin-1 string for tokenizer input."""
    return ''.join(chr(b) for b in data)

def latin1_string_to_bytes(s: str) -> bytes:
    """Convert Latin-1 string back to raw bytes."""
    return bytes(ord(c) for c in s)

def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Load the binary tokenizer from a tokenizer.json file."""
    return Tokenizer.from_file(tokenizer_path)

def encode_binary_data(tokenizer: Tokenizer, data: bytes, add_special_tokens: bool = True) -> dict:
    """
    Encode binary data using the tokenizer.
    
    Returns a dict with:
    - 'ids': list of token IDs
    - 'tokens': list of token strings
    - 'offsets': list of (start, end) byte offsets
    """
    # Convert bytes to Latin-1 string
    text = bytes_to_latin1_string(data)
    
    # Encode with the tokenizer
    encoding = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    return {
        'ids': encoding.ids,
        'tokens': encoding.tokens,
        'offsets': encoding.offsets,
        'attention_mask': encoding.attention_mask,
    }

def decode_to_binary(tokenizer: Tokenizer, token_ids: list) -> bytes:
    """Decode token IDs back to binary data."""
    # Decode to string
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    # Convert Latin-1 string back to bytes
    return latin1_string_to_bytes(text)

def analyze_tokenizer(tokenizer: Tokenizer) -> dict:
    """Analyze tokenizer properties."""
    vocab = tokenizer.get_vocab()
    
    # Find special tokens
    special_tokens = {}
    for token, id_ in vocab.items():
        if token.startswith('<|') and token.endswith('|>'):
            special_tokens[token] = id_
    
    return {
        'vocab_size': len(vocab),
        'special_tokens': special_tokens,
        'has_post_processor': tokenizer.post_processor is not None,
    }

def main():
    # Path to the tokenizer - adjust as needed
    if len(sys.argv) > 1:
        tokenizer_path = sys.argv[1]
    else:
        # Default to the sampled tokenizer in the repo
        tokenizer_path = "bbpe/sampled-tokenizer.json"
    
    if not Path(tokenizer_path).exists():
        print(f"Tokenizer file not found: {tokenizer_path}")
        print("Usage: python example_usage.py [path/to/tokenizer.json]")
        sys.exit(1)
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Analyze the tokenizer
    info = analyze_tokenizer(tokenizer)
    print(f"\nTokenizer Info:")
    print(f"  Vocab size: {info['vocab_size']}")
    print(f"  Special tokens: {len(info['special_tokens'])}")
    print(f"  Has post-processor: {info['has_post_processor']}")
    
    print(f"\nSpecial tokens:")
    for token, id_ in sorted(info['special_tokens'].items(), key=lambda x: x[1]):
        if not token.startswith('<|reserved:'):  # Don't print all reserved tokens
            print(f"  {token}: {id_}")
    
    # Example 1: Encode some text as binary
    print(f"\n--- Example 1: Text as Binary ---")
    text_data = b"Hello, world! This is binary data."
    result = encode_binary_data(tokenizer, text_data, add_special_tokens=True)
    
    print(f"Original data: {text_data}")
    print(f"Token IDs: {result['ids']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Number of tokens: {len(result['ids'])}")
    
    # Decode back
    decoded = decode_to_binary(tokenizer, result['ids'])
    print(f"Decoded: {decoded}")
    print(f"Round-trip successful: {decoded == text_data}")
    
    # Example 2: Encode arbitrary binary data (simulated machine code)
    print(f"\n--- Example 2: Simulated Machine Code ---")
    # Some x86-64 machine code bytes (nop, mov, ret instructions)
    machine_code = bytes([0x90, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x10, 0xC3])
    result2 = encode_binary_data(tokenizer, machine_code, add_special_tokens=True)
    
    print(f"Machine code bytes: {list(machine_code)}")
    print(f"Token IDs: {result2['ids']}")
    print(f"Tokens: {result2['tokens']}")
    print(f"Number of tokens: {len(result2['ids'])}")
    
    # Decode back
    decoded2 = decode_to_binary(tokenizer, result2['ids'])
    print(f"Decoded bytes: {list(decoded2)}")
    print(f"Round-trip successful: {decoded2 == machine_code}")
    
    # Example 3: Test compression efficiency
    print(f"\n--- Example 3: Compression Analysis ---")
    # Test with repetitive data that should compress well
    repetitive_data = b"AAAA" * 100  # 400 bytes of repeated pattern
    result3 = encode_binary_data(tokenizer, repetitive_data, add_special_tokens=False)
    
    compression_ratio = len(repetitive_data) / len(result3['ids'])
    print(f"Original size: {len(repetitive_data)} bytes")
    print(f"Tokenized: {len(result3['ids'])} tokens")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    
    # Example 4: Demonstrate special token handling
    print(f"\n--- Example 4: Special Token Usage ---")
    sample_data = b"Sample binary data for demonstration"
    
    # Encode with special tokens
    with_special = encode_binary_data(tokenizer, sample_data, add_special_tokens=True)
    without_special = encode_binary_data(tokenizer, sample_data, add_special_tokens=False)
    
    print(f"With special tokens: {with_special['tokens'][:5]}...{with_special['tokens'][-5:]}")
    print(f"Without special tokens: {without_special['tokens'][:5]}...{without_special['tokens'][-5:]}")
    
    print(f"\nTokenizer usage examples completed successfully!")

if __name__ == "__main__":
    main()