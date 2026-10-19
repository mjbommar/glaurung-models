#!/usr/bin/env python3
"""Upload binary-tokenizer-005 to Hugging Face Hub."""

from pathlib import Path
from huggingface_hub import HfApi, login

if __name__ == "__main__":
    # Tokenizer directory path
    tokenizer_path = Path("./binary-embedding/binary-tokenizer-005")
    
    # Repository ID
    repo_id = "mjbommar/binary-tokenizer-005"
    
    # Create HF API instance with token
    api = HfApi()
    
    # Create repository first
    print(f"Creating repository: {repo_id}")
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,  # Make it public
            exist_ok=True,  # Don't fail if it exists
        )
        print(f"Repository ready: {repo_url}")
    except Exception as e:
        print(f"Error with repository: {e}")
        # Try to continue anyway
    
    # Upload the entire folder
    print(f"Uploading {tokenizer_path} to {repo_id}")
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=tokenizer_path,
            repo_type="model",
            commit_message="Upload binary-tokenizer-005"
        )
        print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Upload error: {e}")
        print("\nTrying alternative approach...")
        # Try uploading files individually
        for file_path in tokenizer_path.glob("*"):
            if file_path.is_file():
                print(f"Uploading {file_path.name}...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_path.name,
                        repo_id=repo_id,
                        repo_type="model",
                    )
                except Exception as file_error:
                    print(f"  Error uploading {file_path.name}: {file_error}")