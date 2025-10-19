# tokenizer-002

High-performance byte-level BPE trainer tuned for binary corpora (DLL, ELF, Mach-O, etc.). The trainer produces Hugging Face–compatible JSON models without depending on the official `tokenizers` training code.

## Prerequisites

- Rust 1.75 or newer (`cargo` must be on `PATH`).
- Optional: [`uv`](https://github.com/astral-sh/uv) for quick Python spot-checks with the `tokenizers` crate.

Workspace directories you may want to create ahead of time:

- `bench/` – holds training corpora (sample `/usr/bin` subsets are already referenced in `log.md`).
- `.cargo/` – use `CARGO_HOME=$PWD/.cargo` to keep cargo caches local.
- `.uv-cache/` – set `UV_CACHE_DIR=$PWD/.uv-cache` when invoking `uv` to avoid permission issues.

## Building

```bash
cd tokenizer-002
cargo build --release
```

All binaries are emitted under `target/release/`.

## Training

The CLI is exposed as `train`:

```bash
cd tokenizer-002
env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
    --output bench/usr-bin-sample/tokenizer.json \
    bench/usr-bin-sample \
    --vocab-size 8192 \
    --min-frequency 8 \
    --chunk-size 32768 \
    --allowed-lengths 1,2,4,8
```

Key flags:

- `--output`: destination path for the Hugging Face tokenizer JSON.
- Positional arguments: files or directories containing binary data. Use `--chunk-size` to split large files. A value of `0` reads each file as a single sequence.
- `--vocab-size`: total vocabulary (base 256 bytes + special tokens + merges).
- `--min-frequency`: minimum pair count required for merging.
- `--allowed-lengths`: comma-separated even token lengths (default `1,2,4,8`). Extending to `16,32` is supported.
- Plateau controls (`--plateau-frequency`, `--plateau-divisor`, `--plateau-patience`, `--plateau-stop`) can cap merges early; leave defaults for full vocab training.

All timing and memory snapshots from our experiments are recorded in `log.md`.

## Testing & Validation

Unit tests:

```bash
cd tokenizer-002
env CARGO_HOME=$PWD/.cargo cargo test
```

Python compatibility check (requires `uv`):

```bash
cd tokenizer-002
UV_CACHE_DIR=$PWD/.uv-cache uv run --with tokenizers python -c "
from tokenizers import Tokenizer
tok = Tokenizer.from_file('bench/usr-bin-100mb/tokenizer-v32k.json')
print('vocab_size', tok.get_vocab_size(False))
print(tok.encode('Hello, world!').tokens[:8])
"
```

Token sanity inspection on binaries:

```bash
UV_CACHE_DIR=$PWD/.uv-cache uv run --with tokenizers python -c "
from tokenizers import Tokenizer
from pathlib import Path
tok = Tokenizer.from_file('bench/usr-bin-100mb/tokenizer-v32k.json')
data = Path('/usr/lib/cargo/bin/coreutils/ls').read_bytes()[:512]
enc = tok.encode(data.decode('latin-1'))
print('tokens', enc.tokens[:16])
"
```

## Metrics & Benchmarking

Training emits per-iteration progress when `--show-progress` is enabled (default). Additional metrics (RSS, iteration timing, plateau status) are collected in-memory and summarized at the end of each run. Use `/usr/bin/time -v` to capture wall/CPU/RSS more precisely; copy the output into `log.md` to track regressions.

Reference runs (8 k, 16 k, 32 k vocab across `/usr/bin` subsets and the full tree) are already logged for comparison.
