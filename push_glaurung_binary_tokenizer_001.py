#!/usr/bin/env python3
"""Push glaurung-binary-tokenizer-001 to Hugging Face Hub."""

from huggingface_hub import HfApi
from tokenizers import Tokenizer
import json
import os
from pathlib import Path

def push_tokenizer_to_hub(
    tokenizer_path: str,
    repo_id: str,
    readme_path: str,
    private: bool = False,
    commit_message: str = "Upload glaurung-binary-tokenizer-001"
):
    """Push glaurung-binary-tokenizer-001 to Hugging Face Hub."""

    # Load and verify the tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Get vocab size and basic info
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer loaded successfully. Vocab size: {vocab_size}")

    # Initialize HF API
    api = HfApi()

    # Create repo if it doesn't exist
    try:
        repo_url = api.create_repo(repo_id=repo_id, private=private, repo_type="model")
        print(f"Created new repository: {repo_id}")
    except Exception as e:
        print(f"Repository might already exist or error creating: {e}")

    # Upload the tokenizer file
    print(f"Uploading tokenizer to {repo_id}")
    api.upload_file(
        path_or_fileobj=tokenizer_path,
        path_in_repo="tokenizer.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message
    )

    # Create tokenizer_config.json with comprehensive metadata
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": vocab_size,
        "model_type": "glaurung-binary-tokenizer",
        "version": "001",
        "encoding": "latin-1",
        "description": "BPE tokenizer for binary executables and malware (x86-64, ARM64, Windows PE, Linux ELF)",
        "compression_ratio": 2.849,
        "training_data": {
            "size_gb": 13,
            "files": 30738,
            "platforms": ["Linux (Alpine, Debian, Ubuntu)", "Windows (8, 10, 11)"],
            "architectures": ["x86-64", "x86-32", "ARM64"]
        },
        "performance": {
            "bytes_per_token": 2.849,
            "theoretical_efficiency": 0.86,
            "improvement_over_32k": 0.099
        },
        "predecessor": "mjbommar/binary-tokenizer-005"
    }

    config_path = "/tmp/tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Upload config
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo="tokenizer_config.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add tokenizer config"
    )

    # Upload README if provided
    if readme_path and os.path.exists(readme_path):
        print(f"Uploading README from {readme_path}")
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Add README"
        )

    print(f"✅ Successfully pushed tokenizer to https://huggingface.co/{repo_id}")
    print(f"\nTo use this tokenizer:")
    print(f'  from tokenizers import Tokenizer')
    print(f'  tokenizer = Tokenizer.from_pretrained("{repo_id}")')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Push glaurung-binary-tokenizer-001 to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Upload to HuggingFace (public)
  python push_glaurung_binary_tokenizer_001.py

  # Upload as private repository
  python push_glaurung_binary_tokenizer_001.py --private

  # Specify custom paths
  python push_glaurung_binary_tokenizer_001.py \\
    --tokenizer-path ./custom/path/tokenizer.json \\
    --readme-path ./custom/README.md
        """
    )
    parser.add_argument("--repo-id", type=str, default="mjbommar/glaurung-binary-tokenizer-001",
                       help="Repository ID (e.g., username/model-name)")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private (default: public)")
    parser.add_argument("--tokenizer-path", type=str,
                       default="./tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json",
                       help="Path to tokenizer.json file")
    parser.add_argument("--readme-path", type=str,
                       default="./glaurung_binary_tokenizer_001_README.md",
                       help="Path to README.md file")

    args = parser.parse_args()

    # Verify tokenizer exists
    if not os.path.exists(args.tokenizer_path):
        print(f"❌ Error: Tokenizer not found at {args.tokenizer_path}")
        exit(1)

    # Verify README exists
    if not os.path.exists(args.readme_path):
        print(f"⚠️  Warning: README not found at {args.readme_path}")
        response = input("Continue without README? (y/N): ")
        if response.lower() != 'y':
            exit(1)
        args.readme_path = None

    push_tokenizer_to_hub(
        tokenizer_path=args.tokenizer_path,
        repo_id=args.repo_id,
        readme_path=args.readme_path,
        private=args.private
    )
