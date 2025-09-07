**Binary BPE Tokenizer — Architecture & Implementation Plan**

**Overview**
- Goal: Train and use a byte-native BPE tokenizer for arbitrary binary data (machine code, images, compressed data, etc.) using Hugging Face tokenizers (Rust core) for maximum compatibility with existing Python/Rust tooling.
- Approach: Use the built-in BPE model/trainer from `tokenizers` with all 256 byte values in the initial alphabet, disable text-focused normalizers/pre-tokenizers, stream binary corpora to avoid OOM, and serialize to standard `tokenizer.json`.
- Compatibility: Ensure round-trip with `tokenizers` Rust/Python by saving the canonical JSON format and avoiding custom model types. Special tokens use `<|name|>` style.

**Key Design Decisions**
- Byte mapping: Latin-1 (ISO-8859-1) reversible mapping (byte 0x00–0xFF ↔ Unicode code points U+0000–U+00FF). We map bytes→String for training/encoding and String→bytes for decoding. No GPT‑2 byte-level mapping to keep behavior transparent for binary.
- Model: `tokenizers::models::bpe::BPE` with `tokenizers::models::bpe::BpeTrainer`.
- Pipeline: No normalizer, no pre-tokenizer, no decoder. Post-processor is optional; a `TemplateProcessing` processor can be enabled to add special tokens around sequences when desired.
- Alphabet: Initialize trainer with the full 256-char alphabet to guarantee no OOV for binary inputs; optional `unk` special token for framework compatibility, but not required for bytes.
- Streaming: Walk corpora and feed data in configurable windows to avoid building huge in-memory strings. Favor sampling and windowing for memory control over raw throughput.

**Rust API Surface (verified in reference/tokenizers)**
- BPE model and trainer
  - `tokenizers::models::bpe::{BPE, BpeBuilder, BpeTrainer, BpeTrainerBuilder}`
  - `BpeTrainerBuilder` supports: `min_frequency`, `vocab_size`, `special_tokens(Vec<AddedToken>)`, `limit_alphabet`, `initial_alphabet(HashSet<char>)`, `continuing_subword_prefix`, `end_of_word_suffix`, `max_token_length`, `show_progress`.
  - `BpeTrainer::feed<I, S, F>(&mut self, iterator, process)` accumulates word counts; `BpeTrainer::train(&self, &mut BPE)` runs merges. Also available: `BpeTrainer::do_train(&self, &AHashMap<CompactString, u64>, &mut BPE)` for direct word-count input.
  - `BPE::builder()` to construct model; supports `unk_token`, `dropout`, `byte_fallback`, `ignore_merges`, etc. We won’t use byte_fallback because we include all 256 chars.
- Tokenizer assembly and JSON I/O
  - `tokenizers::TokenizerBuilder` or `Tokenizer::new(BPE)` to assemble model + optional components.
  - No normalizer/pre-tokenizer/decoder for binary; optional `processors::template::TemplateProcessing` for special-token templates.
  - `Tokenizer` is `Serialize`/`Deserialize` via `tokenizer::serialization`; save by `serde_json::to_writer` to produce `tokenizer.json` (current Rust API does not expose `Tokenizer::save`; the docs example uses a now-removed helper).
- Processors
  - `tokenizers::processors::template::TemplateProcessing` lets us define templates like `<|start|> $A <|end|>` for single and `<|start|> $A <|sep|> $B <|end|>` for pairs. Must provide `(token, id)` pairs via `special_tokens(...)` that match the Tokenizer’s added vocabulary.

**Special Tokens**
- Required set: `<|start|>`, `<|end|>`, `<|sep|>`, `<|cls|>`, `<|pad|>`.
- Recommended extras: `<|mask|>` (MLM), `<|unk|>` (compat), and reserved placeholders `<|reserved:0|>` … `<|reserved:31|>` for future tasks.
- Storage: Add via `Tokenizer::add_special_tokens(&[AddedToken])` so special tokens live in `added_vocabulary`, separate from the BPE model vocab.
- Post-processing: Optional `TemplateProcessing` with templates:
  - Single: `<|start|> $A <|end|>`
  - Pair: `<|start|> $A <|sep|> $B <|end|>`
  - We’ll wire template ids to those assigned by `added_vocabulary`.
- Training data should not embed these literal strings; they would be treated as plain characters and distort the alphabet. Instead, rely on the post-processor to add them at encode time.

**Byte Mapping Semantics**
- Encoding: input bytes → Latin‑1 `String` (each byte becomes a single char). BPE acts on chars; merges represent byte sequences.
- Decoding: `Tokenizer::decode` yields a `String`; convert back to bytes by mapping each char’s code point to `u8` (`as u32 <= 0xFF`). Provide helper APIs for `encode_bytes(&[u8]) -> Encoding` and `decode_to_bytes(&[u32]) -> Vec<u8>`.
- Offsets: `Tokenizer::encode` uses byte offsets; they map 1:1 to our Latin‑1 bytes-per-char assumption.

**Data Ingestion & Memory Strategy**
- Walking: Use `walkdir` to discover files; options to follow symlinks (off by default), ignore `.git`, `.DS_Store`, zero-length files, and optionally limit by extension or size.
- Chunking modes (configurable):
  - `complete`: treat each file as one “document” (emit `<|start|>`, file content, `<|end|>` as separate items).
  - `fixed`: non-overlapping windows of N bytes (default N=4096); emit `<|start|>` before first window and `<|end|>` after last.
  - `random`: random window sizes between 2^min and 2^max bytes (defaults: 2^3=8 to 2^14=16384); same boundary behavior as `fixed`.
- Sampling & caps (to avoid OOM):
  - `max_documents`: global cap on the number of windows/documents considered.
  - `max_documents_per_file` and `sampling_stride`/`sampling_rate`.
  - `seed` for deterministic sampling order.
  - Optional “space-saving” heavy-hitter sketch (phase 2) to retain top‑K frequent windows without storing all unique windows.
- Pre-aggregation: Aggregate identical windows across corpus into a `AHashMap<CompactString, u64>` of counts to reduce duplicate storage before handing to `BpeTrainer::do_train`.
- Parallelism: Use rayon where it reduces wall time without increasing peak memory (e.g., hashing/dedup); guard with `RAYON_RS_NUM_THREADS` and a CLI flag. Trainer internally uses parallel fragments as well.

**Training Flow (Rust)**
- Build trainer with `BpeTrainerBuilder`:
  - `vocab_size`, `min_frequency`, `show_progress`.
  - `initial_alphabet`: Set to all 256 chars: `HashSet::from_iter((0u8..=255).map(|b| char::from(b)))`.
  - Leave `limit_alphabet` unset; no `continuing_subword_prefix` / `end_of_word_suffix`.
  - Add special tokens with `AddedToken::from("<|...|>".into(), true)`.
- Collect word_counts:
  - Iterate according to chosen chunking mode, convert bytes→Latin‑1 string, aggregate into `AHashMap<CompactString, u64>` with sampling/caps applied.
- Train model:
  - Construct `let mut bpe = BPE::builder().build()?;`
  - Call `trainer.do_train(&word_counts, &mut bpe)?;`
- Assemble tokenizer:
  - `let mut tokenizer = Tokenizer::new(bpe);`
  - No normalizer/pre-tokenizer/decoder. Optionally configure `TemplateProcessing` post-processor.
  - Add special tokens via `tokenizer.add_special_tokens(&tokens)` and wire IDs into the template.
- Save: `serde_json::to_writer(File::create("tokenizer.json")?, &tokenizer)`.

**CLI & Library Design**
- Crate layout
  - `binary-bpe-tokenizer/`
    - `src/lib.rs`: public API for training and encode/decode helpers.
    - `src/cli.rs`: CLI command wiring (train/encode/decode/inspect).
    - `src/ingest.rs`: walkdir, chunking, sampling, pre-aggregation.
    - `src/mapping.rs`: Latin‑1 mapping helpers and validations.
    - `src/train.rs`: build `BpeTrainer`, run `do_train`, assemble `Tokenizer`.
    - `src/post.rs`: optional `TemplateProcessing` construction from special tokens.
    - `src/io.rs`: save/load `tokenizer.json`, read corpus manifests.
    - `src/bin/bbpe.rs`: `main()` (CLI entry) using `clap`.
- Dependencies
  - `tokenizers` (path dep to `reference/tokenizers/tokenizers` during dev or crates.io version), `walkdir`, `rayon` (optional), `clap`, `anyhow`, `serde`, `serde_json`, `ahash`, `compact_str`.
- CLI commands
  - `train`: Train a tokenizer.
    - Inputs: `--input PATH...`, `--vocab-size`, `--min-frequency`, `--mode [file|fixed|delim]`, `--window-bytes`, `--delimiter-hex`, `--max-docs`, `--max-docs-per-file`, `--sample-rate`, `--seed`, `--progress/--no-progress`, `--threads`, `--output tokenizer.json`, `--pretty`.
    - Special tokens: `--with-standard-specials`, `--with-mask`, `--with-unk`, `--reserve N`.
    - Post-processor: `--with-template` (enables `<|start|> $A <|end|>` & pair form), `--no-template`.
  - `encode`: Encode bytes from a file/stdin; outputs token IDs, tokens, and byte offsets; supports adding special tokens.
  - `decode`: Decode IDs from stdin/file to bytes (latin‑1 back to bytes) and optionally write to a file.
  - `inspect`: Print vocab size, top merges, special tokens, etc.
- Config file
  - Support `--config config.toml` for reproducibility; CLI flags override.

**Training Defaults**
- `vocab_size`: 32768 (2^15).
- `min_frequency`: 1024.
- Chunking: `fixed` mode with `--window-bytes=4096` default.
- Special tokens: include `<|start|>`, `<|end|>`, `<|sep|>`, `<|cls|>`, `<|pad|>`, `<|mask|>`, `<|unk|>` plus 128 reserved placeholders `<|reserved:0|>`…`<|reserved:127|>`.
- Post-train padding: pad total vocabulary (model + added) up to the next power-of-two by appending additional reserved tokens if needed.
- Progress bar enabled unless `--no-progress`. Determinism via `--seed` and fixed walk order.

**Interoperability**
- Rust↔Python: `tokenizer.json` loads in both. In Python, callers must latin‑1 encode/decode around `Tokenizer.encode/Tokenizer.decode`:
  - `text = bytes_obj.decode('latin-1')`
  - `ids = tok.encode(text).ids`
  - `bytes_out = tok.decode(ids).encode('latin-1')`
- Transformers: Works like any BPE tokenizer. If needed, supply `tokenizer_config.json` with special token names matching Transformers conventions. Not required for the base deliverable.

**Memory & Performance Considerations**
- Peak memory drivers: number of unique “words” (documents/windows) in `word_counts`, and internal `trainer` `words` representation. Strategies:
  - Keep windows moderate (2–8 KiB) to increase duplication likelihood.
  - Raise `min_frequency` for large corpora.
  - Apply sampling caps; provide reproducible seeds.
  - Optionally prefilter extremely rare windows (phase 2: approximate counting or bloom filters) before calling the trainer.
- Parallelism: Trainer uses rayon under the hood; constrain threads via `RAYON_RS_NUM_THREADS` and a CLI option to reduce memory spikes.

**Validation Plan**
- Unit tests (Rust):
  - Latin‑1 map round trip for all 256 byte values.
  - Encode/decode round trip on random binary buffers (1 KiB–1 MiB) with various special-token settings (skip included specials on decode when asked).
  - Serialization/deserialization invariant for `Tokenizer` JSON.
- Integration tests:
  - Train on `old/v1` small sample; compare basic stats (vocab size, non-zero merges) and ensure round-trip on sample binaries.
  - Load `tokenizer.json` from Python and assert byte round-trip as above.
  - Stress test with caps to validate OOM avoidance.

**Risks & Mitigations**
- Too many unique documents/windows → high memory: default to fixed windows, provide sampling and frequency thresholds.
- Special tokens leaking into training: never inject literal `<|...|>` into training streams; add via `added_vocabulary` only.
- Decoding to bytes in Rust/Python: clarify Latin‑1 bridging in docs and provide helper utilities in the Rust crate and examples in README.
- Trainer API changes upstream: we embed against the local `reference/tokenizers` API (uses `do_train` and JSON serialization); pin the git SHA or crates.io version when we vendor.

**Implementation Steps**
1) Scaffold crate: library + CLI (clap), add deps (`tokenizers`, `walkdir`, `serde`, `serde_json`, `anyhow`, `ahash`, `compact_str`, optional `rayon`).
2) Implement Latin‑1 helpers (`mapping.rs`): `bytes_to_latin1_string`, `latin1_string_to_bytes`, plus validation tests.
3) Ingestion (`ingest.rs`): walkdir + chunking modes + sampling; produce `AHashMap<CompactString, u64>`.
4) Trainer setup (`train.rs`): build `BpeTrainer` with initial 256-char alphabet and specials; call `do_train` to obtain merges/vocab in a `BPE`.
5) Tokenizer assembly: create `Tokenizer`, add special tokens, optionally attach `TemplateProcessing` using generated ids.
6) Save/load (`io.rs`): JSON serialization/deserialization of `Tokenizer`.
7) CLI (`cli.rs`, `bin/bbpe.rs`): `train`, `encode`, `decode`, `inspect`.
8) Tests: unit + integration; include a smoke test using files under `old/v1`.
9) Docs: README with usage examples (Rust/Python), caveats, and performance tips.

**Sensible Defaults and Examples (Rust snippets)**
- Initial alphabet:
  - `let alphabet: std::collections::HashSet<char> = (0u16..=255).map(|b| char::from_u32(b as u32).unwrap()).collect();`
- Trainer:
  - `let mut trainer = BpeTrainer::builder().vocab_size(vs).min_frequency(minf).show_progress(pb).initial_alphabet(alphabet.into_iter().collect()).special_tokens(specials).build();`
- Train:
  - `trainer.do_train(&word_counts, &mut bpe)?;`
- Tokenizer assembly and save:
  - `let mut tok = Tokenizer::new(bpe); tok.add_special_tokens(&specials); /* attach template if requested */ serde_json::to_writer(std::fs::File::create(out)?, &tok)?;`

**Future Extensions**
- Heavy-hitter sketching (space-saving algorithm) to cap memory by tracking only top‑K windows.
- Boundary-aware training: treat each (BOS + file + EOS) as one document while still sampling windows for large files.
- Optional decompression for archives and common containers.
- Optional byte-fallback decoder to emit valid UTF‑8 on decode for mixed text/binary corpora (not required for strict byte round-trip).
- Streaming encode/decode for large inputs (`decode_stream`) wrapped with Latin‑1 translation helpers.

**Milestones**
- M1: CLI scaffolding + Latin‑1 helpers + ingestion + trainer + save JSON (end-to-end on small sample under `old/`).
- M2: Sampling/caps + template processor + encode/decode CLI.
- M3: Docs + Python interop examples + integration tests.
- M4: Optional memory-optimized counting.

**Notes on `old/` reference**
- The `old/v1` Python experiments used `ByteLevel` pre-tokenizer/decoder and Latin‑1 file decoding. For this Rust-first implementation we avoid byte-level pre-tokenizer and instead use direct Latin‑1 mapping, ensuring the BPE model sees genuine byte symbols and works uniformly in Rust and Python via `tokenizer.json`.
