#!/usr/bin/env python3
"""
End-to-end regression for tokenizer-002 training and Hugging Face interoperability.

Steps
-----
1. Generate a tiny binary corpus with repeating 2-byte patterns.
2. Invoke the tokenizer-002 training CLI to build a tokenizer.json.
3. Load the tokenizer via `tokenizers.Tokenizer`.
4. Verify:
   * the expected merged token is present in the vocabulary
   * encoding the sample payload emits the merged token.

Run with (from project root):
    uv run --with tokenizers python3 tokenizer-002/tests/e2e_python.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_BIN = REPO_ROOT / "src" / "bin" / "train.rs"


def latin1(bytes_like: bytes) -> str:
    return bytes_like.decode("latin-1")


def main() -> None:
    if not TRAIN_BIN.exists():
        raise SystemExit("train binary source missing; run from tokenizer-002 repository")

    sample_bytes = bytes([0xAA, 0xBB]) * 128 + bytes([0xCC, 0xDD]) * 16
    merged_token = latin1(bytes([0xAA, 0xBB]))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        corpus_path = tmpdir_path / "sample.bin"
        model_path = tmpdir_path / "tokenizer.json"

        corpus_path.write_bytes(sample_bytes)

        cargo_home = REPO_ROOT / ".cargo"
        cargo_home.mkdir(parents=True, exist_ok=True)

        cmd = [
            "cargo",
            "run",
            "--quiet",
            "--bin",
            "train",
            "--",
            "--output",
            str(model_path),
            str(corpus_path),
            "--vocab-size",
            "272",
            "--min-frequency",
            "2",
            "--chunk-size",
            "0",
            "--allowed-lengths",
            "1,2",
        ]
        print("Running:", " ".join(cmd))
        env = {
            **os.environ,
            "CARGO_TERM_COLOR": "never",
            "CARGO_HOME": str(cargo_home),
        }
        subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            env=env,
        )

        tokenizer = Tokenizer.from_file(str(model_path))

        vocab = tokenizer.get_vocab(with_added_tokens=False)
        if merged_token not in vocab:
            raise AssertionError("expected merged token missing from vocab")

        encoded = tokenizer.encode(latin1(sample_bytes), add_special_tokens=False)
        if merged_token not in encoded.tokens:
            raise AssertionError("merged token not produced during encoding")

        print("ok: merged token present with id", vocab[merged_token])
        print("ok: encode() emitted merged token", merged_token.encode("latin-1").hex())


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as err:
        raise SystemExit(f"training command failed: {err}") from err
