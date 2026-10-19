#!/usr/bin/env python3
"""Update glaurung-tokenizer-002 special tokens to match binary-tokenizer-005 format.

Changes:
  <s>     -> <|start|>
  </s>    -> <|end|>
  <pad>   -> <|pad|>
  <unk>   -> <|unk|>
  <cls>   -> <|cls|>
  <sep>   -> <|sep|>
  <mask>  -> <|mask|>
"""

import json
from pathlib import Path

def update_special_tokens(input_path: str, output_path: str):
    """Update special token names in tokenizer JSON."""

    # Token mapping: old format -> new format
    token_mapping = {
        '<s>': '<|start|>',
        '</s>': '<|end|>',
        '<pad>': '<|pad|>',
        '<unk>': '<|unk|>',
        '<cls>': '<|cls|>',
        '<sep>': '<|sep|>',
        '<mask>': '<|mask|>',
    }

    print(f"Loading tokenizer from: {input_path}")
    with open(input_path, 'r') as f:
        tokenizer_json = json.load(f)

    # Track changes
    changes_made = []

    # 1. Update added_tokens section
    if 'added_tokens' in tokenizer_json:
        print(f"\nUpdating {len(tokenizer_json['added_tokens'])} added_tokens...")
        for token_info in tokenizer_json['added_tokens']:
            old_content = token_info['content']
            if old_content in token_mapping:
                new_content = token_mapping[old_content]
                token_info['content'] = new_content
                changes_made.append(f"  ID {token_info['id']}: {old_content:10} -> {new_content}")
                print(f"  ✓ Updated ID {token_info['id']}: {old_content} -> {new_content}")

    # 2. Update model vocabulary
    if 'model' in tokenizer_json and 'vocab' in tokenizer_json['model']:
        print(f"\nUpdating model vocabulary...")
        vocab = tokenizer_json['model']['vocab']

        # Create new vocab with updated token names
        updated_vocab = {}
        for token_str, token_id in vocab.items():
            if token_str in token_mapping:
                new_token = token_mapping[token_str]
                updated_vocab[new_token] = token_id
                print(f"  ✓ Vocab: {token_str} -> {new_token} (ID {token_id})")
            else:
                updated_vocab[token_str] = token_id

        tokenizer_json['model']['vocab'] = updated_vocab

    # 3. Update decoder (if present)
    if 'decoder' in tokenizer_json and tokenizer_json['decoder'] is not None:
        if 'vocab' in tokenizer_json['decoder']:
            print(f"\nUpdating decoder vocabulary...")
            decoder_vocab = tokenizer_json['decoder']['vocab']

            updated_decoder = {}
            for token_str, token_id in decoder_vocab.items():
                if token_str in token_mapping:
                    new_token = token_mapping[token_str]
                    updated_decoder[new_token] = token_id
                    print(f"  ✓ Decoder: {token_str} -> {new_token}")
                else:
                    updated_decoder[token_str] = token_id

            tokenizer_json['decoder']['vocab'] = updated_decoder

    # Save updated tokenizer
    print(f"\nSaving updated tokenizer to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(tokenizer_json, f, indent=2)

    print(f"\n✅ Successfully updated {len(changes_made)} special tokens!")
    print(f"\nChanges summary:")
    for change in changes_made:
        print(change)

    return tokenizer_json

def verify_special_tokens(tokenizer_path: str):
    """Verify special tokens in tokenizer."""
    from tokenizers import Tokenizer

    print(f"\n{'='*70}")
    print(f"Verifying special tokens in: {Path(tokenizer_path).name}")
    print(f"{'='*70}")

    tok = Tokenizer.from_file(tokenizer_path)

    expected_tokens = [
        '<|start|>', '<|end|>', '<|pad|>', '<|unk|>',
        '<|cls|>', '<|sep|>', '<|mask|>'
    ]

    all_found = True
    for token_str in expected_tokens:
        token_id = tok.token_to_id(token_str)
        if token_id is not None:
            print(f"  ✓ {token_str:15} -> ID {token_id}")
        else:
            print(f"  ✗ {token_str:15} -> NOT FOUND")
            all_found = False

    if all_found:
        print(f"\n✅ All special tokens found!")
    else:
        print(f"\n⚠️  Some special tokens missing!")

    return all_found

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update glaurung-tokenizer-002 special tokens")
    parser.add_argument(
        "--input",
        type=str,
        default="tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json",
        help="Input tokenizer JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="binary-embedding/glaurung-tokenizer-002-updated.json",
        help="Output tokenizer JSON file"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the updated tokenizer after creation"
    )

    args = parser.parse_args()

    # Update tokens
    update_special_tokens(args.input, args.output)

    # Verify if requested
    if args.verify:
        verify_special_tokens(args.output)

    print(f"\n{'='*70}")
    print("Next steps:")
    print(f"{'='*70}")
    print(f"1. Test the updated tokenizer:")
    print(f"   cd binary-embedding")
    print(f"   PYTHONPATH=src uv run python -c 'from binary_embedding.tokenizer import load_tokenizer; tok = load_tokenizer(\"{args.output}\"); print(f\"Loaded! start_token_id={{tok.start_token_id}}\")'")
    print(f"\n2. Update run.sh to use new tokenizer:")
    print(f"   --tokenizer-path {args.output} \\")
    print(f"\n3. Run quick training test (100 steps)")
