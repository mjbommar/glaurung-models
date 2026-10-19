#!/usr/bin/env python3
"""
Upload Glaurung Binary Embedding MICRO model to HuggingFace Hub.

This script uploads the trained model, tokenizer, and documentation to HuggingFace,
making it available for public use.

Usage:
    python upload_to_huggingface.py [--repo-name REPO] [--private] [--dry-run]

Requirements:
    pip install huggingface_hub transformers
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, login, whoami
from huggingface_hub.utils import RepositoryNotFoundError


def check_login():
    """Check if user is logged in to HuggingFace."""
    try:
        user_info = whoami()
        username = user_info["name"]
        print(f"✓ Logged in as: {username}")
        return username
    except Exception:
        print("❌ Not logged in to HuggingFace")
        print("\nPlease run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        sys.exit(1)


def verify_model_files(model_dir: Path):
    """Verify all required model files exist."""
    required_files = {
        "model.safetensors": "Model weights",
        "config.json": "Model configuration",
        "tokenizer.json": "Tokenizer configuration",
        "README.md": "Model documentation",
    }

    print("\nVerifying model files:")
    print("=" * 60)

    missing_files = []
    for filename, description in required_files.items():
        filepath = model_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename:25s} ({size_mb:6.2f} MB) - {description}")
        else:
            print(f"❌ {filename:25s} MISSING - {description}")
            missing_files.append(filename)

    # Check for training_state.pt (should NOT be uploaded)
    training_state = model_dir / "training_state.pt"
    if training_state.exists():
        size_mb = training_state.stat().st_size / (1024 * 1024)
        print(
            f"⚠️  training_state.pt ({size_mb:.2f} MB) - Will NOT be uploaded (training only)"
        )

    print("=" * 60)

    if missing_files:
        print(f"\n❌ Missing required files: {', '.join(missing_files)}")
        sys.exit(1)

    print("✓ All required files present\n")


def calculate_upload_size(model_dir: Path):
    """Calculate total size of files to upload."""
    total_bytes = 0
    files_to_upload = ["model.safetensors", "config.json", "tokenizer.json", "README.md"]

    for filename in files_to_upload:
        filepath = model_dir / filename
        if filepath.exists():
            total_bytes += filepath.stat().st_size

    total_mb = total_bytes / (1024 * 1024)
    return total_mb


def create_repo_if_needed(api: HfApi, repo_id: str, private: bool):
    """Create repository if it doesn't exist."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✓ Repository exists: https://huggingface.co/{repo_id}")
        return False  # Already exists
    except RepositoryNotFoundError:
        print(f"Creating new repository: {repo_id}")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        print(f"✓ Created repository: https://huggingface.co/{repo_id}")
        return True  # Newly created


def upload_files(api: HfApi, model_dir: Path, repo_id: str, dry_run: bool):
    """Upload model files to HuggingFace Hub."""
    files_to_upload = [
        ("model.safetensors", "Model weights"),
        ("config.json", "Model configuration"),
        ("tokenizer.json", "Tokenizer"),
        ("README.md", "Documentation"),
    ]

    print("\nUploading files:")
    print("=" * 60)

    for filename, description in files_to_upload:
        filepath = model_dir / filename

        if dry_run:
            print(f"[DRY RUN] Would upload: {filename} - {description}")
            continue

        print(f"Uploading {filename}... ", end="", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
            )
            print("✓")
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    print("=" * 60)
    print("✓ All files uploaded successfully\n")
    return True


def create_gitattributes(model_dir: Path):
    """Create .gitattributes for Git LFS."""
    gitattributes_content = """*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.json filter=lfs diff=lfs merge=lfs -text
"""

    gitattributes_path = model_dir / ".gitattributes"
    if not gitattributes_path.exists():
        gitattributes_path.write_text(gitattributes_content)
        print("✓ Created .gitattributes for Git LFS")
    else:
        print("✓ .gitattributes already exists")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Glaurung Binary Embedding MICRO to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload to default repo (mjbommar/glaurung-binary-embedding-micro)
  python upload_to_huggingface.py

  # Upload to custom repo
  python upload_to_huggingface.py --repo-name username/custom-repo-name

  # Create private repo
  python upload_to_huggingface.py --private

  # Dry run (check files without uploading)
  python upload_to_huggingface.py --dry-run
        """,
    )

    parser.add_argument(
        "--repo-name",
        type=str,
        default="mjbommar/glaurung-binary-embedding-micro",
        help="HuggingFace repository name (default: mjbommar/glaurung-binary-embedding-micro)",
    )

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/micro-v6-best"),
        help="Path to model directory (default: models/micro-v6-best)",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository (default: public)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify files without uploading",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Glaurung Binary Embedding MICRO - HuggingFace Upload")
    print("=" * 70)
    print()

    # Verify model directory exists
    if not args.model_dir.exists():
        print(f"❌ Model directory not found: {args.model_dir}")
        sys.exit(1)

    print(f"Model directory: {args.model_dir}")
    print(f"Repository: {args.repo_name}")
    print(f"Private: {args.private}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Check HuggingFace login
    username = check_login()

    # Verify all required files exist
    verify_model_files(args.model_dir)

    # Calculate upload size
    upload_size_mb = calculate_upload_size(args.model_dir)
    print(f"Total upload size: {upload_size_mb:.2f} MB\n")

    # Create .gitattributes
    create_gitattributes(args.model_dir)

    if args.dry_run:
        print("\n[DRY RUN] Skipping upload")
        print(f"✓ All checks passed - ready to upload to {args.repo_name}")
        return

    # Confirm upload
    print("\n" + "=" * 70)
    print("Ready to upload!")
    print("=" * 70)
    print(f"Repository: https://huggingface.co/{args.repo_name}")
    print(f"Size: {upload_size_mb:.2f} MB")
    print(f"Visibility: {'Private' if args.private else 'Public'}")
    print()

    response = input("Continue with upload? [y/N]: ")
    if response.lower() != "y":
        print("Upload cancelled")
        sys.exit(0)

    # Initialize API
    api = HfApi()

    # Create repo if needed
    create_repo_if_needed(api, args.repo_name, args.private)

    # Upload files
    success = upload_files(api, args.model_dir, args.repo_name, args.dry_run)

    if success:
        print("\n" + "=" * 70)
        print("✅ Upload Complete!")
        print("=" * 70)
        print(f"\nYour model is now available at:")
        print(f"🤗 https://huggingface.co/{args.repo_name}")
        print()
        print("To use the model:")
        print(f'  from transformers import AutoModelForMaskedLM, AutoTokenizer')
        print(f'  model = AutoModelForMaskedLM.from_pretrained("{args.repo_name}")')
        print(f'  tokenizer = AutoTokenizer.from_pretrained("{args.repo_name}")')
        print()

        if args.private:
            print("⚠️  Repository is PRIVATE")
            print("To make it public, go to:")
            print(f"  https://huggingface.co/{args.repo_name}/settings")
    else:
        print("\n❌ Upload failed - please check errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
