#!/usr/bin/env python3
"""Push binary-tokenizer-005 to Hugging Face Hub."""

from huggingface_hub import HfApi
from tokenizers import Tokenizer
import json
import os
from pathlib import Path

def push_tokenizer_to_hub(
    tokenizer_path: str,
    repo_id: str,
    readme_path: str,
    private: bool = True,
    commit_message: str = "Upload binary-tokenizer-005"
):
    """Push a tokenizer to Hugging Face Hub."""
    
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
    
    # Create a basic tokenizer_config.json
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": vocab_size,
        "model_type": "binary-tokenizer",
        "version": "005",
        "encoding": "latin-1",
        "description": "BPE tokenizer trained on binary executables"
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
    
    print(f"Successfully pushed tokenizer to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Push binary-tokenizer-005 to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="mjbommar/binary-tokenizer-005",
                       help="Repository ID (e.g., username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--tokenizer-path", type=str, 
                       default="./binary-embedding/binary-tokenizer-005/tokenizer.json",
                       help="Path to tokenizer.json file")
    parser.add_argument("--readme-path", type=str,
                       default="./tokenizer_readme.md",
                       help="Path to README.md file")
    
    args = parser.parse_args()
    
    push_tokenizer_to_hub(
        tokenizer_path=args.tokenizer_path,
        repo_id=args.repo_id,
        readme_path=args.readme_path,
        private=args.private
    )