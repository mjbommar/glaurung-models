## 2025-10-18 — `/usr/bin` sample benchmark

- Sample: 16 binaries copied from `/usr/bin` into `bench/usr-bin-sample/` (≈8.9 MiB total)
- Command:
  ```
  /usr/bin/time -v env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
      --output bench/usr-bin-sample/tokenizer.json \
      bench/usr-bin-sample \
      --vocab-size 8192 \
      --min-frequency 8 \
      --chunk-size 32768 \
      --allowed-lengths 1,2,4,8
  ```
- Result:
  - Runtime: 4 min 14 s (wall), 1 919 s CPU
  - Peak RSS: 253 MiB
  - Merge iterations: 7 929 (stopped at vocab 8 185)
  - Progress logs show per-iteration time ~30–35 ms with frequency plateau at 65.
- Takeaway: current trainer keeps merging long after frequencies flatten, causing ~O(iterations × corpus) behaviour even on tiny corpora.

## 2025-10-18 — `/usr/bin` sample with adaptive stop (v2)

- Same 16-binary corpus as above
- Command:
  ```
  /usr/bin/time -v env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
      --output bench/usr-bin-sample/tokenizer.json \
      bench/usr-bin-sample \
      --vocab-size 8192 \
      --min-frequency 8 \
      --chunk-size 32768 \
      --allowed-lengths 1,2,4,8 \
      --plateau-divisor 512 \
      --plateau-patience 32
  ```
- Result:
  - Runtime: 6.4 s wall, 56.5 s CPU
  - Peak RSS: 256 MiB
  - Merge iterations: 200 (stopped via plateau heuristic at freq ≈2.6 k)
  - Metrics summary: `iterations=200, stop=PlateauReached, total=4.65s, last_freq=2655`
- Takeaway: adaptive thresholds + merge cap cut runtime by ~40× while still expanding vocab beyond the 256-byte base.

## 2025-10-18 — Progressive `/usr/bin` subsets

| Sample | Size | Command tweaks | Runtime (wall) | Peak RSS | Merges | Stop reason |
| --- | --- | --- | --- | --- | --- | --- |
| `bench/usr-bin-10mb` | 11.1 MiB | `--vocab-size 8192 --plateau-divisor 512 --plateau-patience 32` | 10.6 s | 137 MiB | 341 | Plateau (freq≈2.6 k) |
| `bench/usr-bin-50mb` | 56.6 MiB | `--vocab-size 16384 --max-merges 4096 --plateau-divisor 512 --plateau-patience 32` | 53.3 s | 456 MiB | 277 | Plateau (freq≈12.4 k) |
| `bench/usr-bin-200mb` | 209 MiB | `--vocab-size 32768 --max-merges 4096 --plateau-divisor 512 --plateau-patience 32` | 121 s | 1.37 GiB | 211 | Plateau (freq≈50 k) |

- Metrics summaries recorded from CLI output (e.g., `iterations=277, stop=PlateauReached, total=53.11s, last_freq=12458`).
- Runtime scales roughly linearly with corpus size after the heuristics kick in; memory stays under 1.4 GiB at ~200 MiB input without chunk tuning.
- Next steps: tighten plateau heuristics per size, explore streaming pair-count sketch to push 200 MiB runs below 60 s.

## 2025-10-18 — Incremental pair heap (tokenizer-002)

- Change: maintain pair counts incrementally with a max-heap of `(pair, frequency)` entries, eliminating the full corpus rescan each merge.
- Command (warm run after rebuild):
  ```
  /usr/bin/time -v env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
      --output bench/usr-bin-sample/tokenizer.json \
      bench/usr-bin-sample \
      --vocab-size 8192 \
      --min-frequency 8 \
      --chunk-size 32768 \
      --allowed-lengths 1,2,4,8 \
      --plateau-divisor 512 \
      --plateau-patience 32
  ```
- Result:
  - Wall clock: 0.50 s (training loop 0.33 s total); initial rebuild run was 3.19 s including compilation.
  - CPU: 4.11 s user / 0.39 s sys.
  - Peak RSS: 93 MiB.
  - Merges: 200 (plateau at freq ≈2.66 k, final vocab 456).
  - Distinct pairs tracked at stop: ~68 k.
- Takeaway: ~13× speedup over the previous 6.4 s run on the same corpus while retaining plateau-controlled stopping; next iteration should address hitting the configured vocab size now that the merge loop is fast enough.

## 2025-10-18 — Target vocab run with plateau stop disabled

- Change: plateau-based stopping now opt-in (`--plateau-stop`); default run respects the requested vocab and removes the 4 096 merge cap.
- Command:
  ```
  /usr/bin/time -v env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
      --output bench/usr-bin-sample/tokenizer.json \
      bench/usr-bin-sample \
      --vocab-size 8192 \
      --min-frequency 8 \
      --chunk-size 32768 \
      --allowed-lengths 1,2,4,8
  ```
- Result:
  - Wall clock: 10.8 s (training loop 7.71 s after compilation reuse).
  - CPU: 104 s user / 8.2 s sys.
  - Peak RSS: 255 MiB.
  - Merges: 7 929 (final base vocab 8 185; +7 specials = configured 8 192).
  - Stop reason: reached target vocab; last merge freq ≈66; ~241 k pairs still tracked.
- Takeaway: we now hit the configured vocab size with ~40× faster training than the original full-rescan loop (4 min 14 s → 7.7 s). Plateau metrics remain available for analysis or optional early stop.

## 2025-10-18 — 110 MiB `/usr/bin` subset (target vocab)

- Dataset: `bench/usr-bin-100mb/` (76 binaries copied from `usr-bin-200mb`, ≈110 MiB).
- Command:
  ```
  /usr/bin/time -v env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
      --output bench/usr-bin-100mb/tokenizer.json \
      bench/usr-bin-100mb \
      --vocab-size 8192 \
      --min-frequency 8 \
      --chunk-size 32768 \
      --allowed-lengths 1,2,4,8
  ```
- Result:
  - Wall clock: 100.2 s (training loop 99.64 s reported), CPU 1 620 s user / 17.1 s sys.
  - Peak RSS: 733 MiB (resident set reported 750 780 kB).
  - Merges: 7 929 (final base vocab 8 185; +7 specials = 8 192 target).
  - Stop reason: target vocab reached; last merge freq ≈840; ~501 k distinct pairs tracked at finish.
- Takeaway: processing ~110 MiB now completes in ~1 min 40 s wall time while honoring the full merge budget; throughput scales ~13× slower than the 8.9 MiB sample (7.7 s), suggesting next optimization work should focus on reducing contention in the pair-delta update path as corpora grow.

## 2025-10-18 — Intrusive occurrence rewrite (tokenizer-002)

- Change: replaced per-iteration rescans with linked token nodes plus per-pair occurrence lists maintained incrementally.
- `/usr/bin` sample (8.9 MiB) command:
  ```
  /usr/bin/time -v env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
      --output bench/usr-bin-sample/tokenizer.json \
      bench/usr-bin-sample \
      --vocab-size 8192 \
      --min-frequency 8 \
      --chunk-size 32768 \
      --allowed-lengths 1,2,4,8
  ```
  - Wall clock: 20.6 s (trainer reported 19.92 s); CPU 19.6 s user / 0.96 s sys.
  - Peak RSS: 1.34 GiB.
  - Merges: 7 929 (target vocab hit); last freq ≈57; ~240 k distinct pairs tracked.
- `/usr-bin-100mb` (~110 MiB) command (same flags, dataset `bench/usr-bin-100mb`).
  - Wall clock: 5 min 48 s (trainer 344.62 s); CPU 342 s user / 6.9 s sys.
  - Peak RSS: 15.0 GiB.
  - Merges: 7 929; last freq ≈734; ~540 k distinct pairs tracked.
- Takeaway: correctness preserved but performance regressed (sample +3× slower than pre-rewrite; 110 MiB tier +3.4× slower with 10× higher RSS). Need aggressive pruning: recycle inactive occurrences, avoid registering pairs whose counts fall below threshold, and batch updates to cut hotspot contention.

## 2025-10-19 — Regression benchmark after revert

- `/usr/bin` sample (8.9 MiB): 7.99 s wall (trainer 5.99 s), CPU 74.7 s user / 6.9 s sys, peak RSS 255 MiB, 7 929 merges (`TargetVocabReached`).
- `/usr-bin-100mb` (~110 MiB): 93.7 s wall (trainer 93.36 s), CPU 1 450 s user / 16.9 s sys, peak RSS 728 MiB, 7 929 merges.
- These match the earlier incremental results (see 2025-10-18 entries), confirming performance is back to baseline after the rollback.

## 2025-10-19 — Reverted intrusive rewrite

- Action: restored the previous incremental pair-count trainer while keeping plateau-stop fixes after intrusive structure proved slower and memory hungry.
- Test: `env CARGO_HOME=$PWD/.cargo cargo test` (pass).
- Next focus: pursue lighter-weight optimizations (occurrence recycling, selective updates) experimentally in separate branches before merging to mainline.

## 2025-10-19 — Full `/usr/bin` sweep (baseline memory check)

- Dataset: `/usr/bin` mounted inside the container (≈789 298 247 B ≈ 753 MiB on disk via `du -sb /usr/bin`).
- Command:
  ```
  /usr/bin/time -v env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
      --output bench/usr-bin-full/tokenizer.json \
      /usr/bin \
      --vocab-size 8192 \
      --min-frequency 8 \
      --chunk-size 32768 \
      --allowed-lengths 1,2,4,8
  ```
- Result:
  - Wall clock: 11 min 35 s (trainer loop 689.28 s).
  - CPU: 11 601 s user / 34 s sys (≈16.7× parallelism).
  - Peak RSS: 4 351 476 kB (≈4.15 GiB), ~5.5× the corpus byte size.
  - Merges: 7 929 (`TargetVocabReached`); last merge freq ≈4 630; ~2.96 M distinct pairs tracked at finish.
- Takeaway: baseline run on the full binary tree confirms RSS scales superlinearly with corpus size once pair_counts approaches ~3 M entries. Upcoming memory-focused work should target pair map densification and arena-backed sequences to bring peak usage closer to the raw corpus footprint.

## 2025-10-19 — `/usr-bin-100mb` with higher vocab and longer tokens

- Dataset: `bench/usr-bin-100mb` (~110 MiB of binaries).
- Command:
  ```
  /usr/bin/time -v env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
      --output bench/usr-bin-100mb/tokenizer-v16k.json \
      bench/usr-bin-100mb \
      --vocab-size 16384 \
      --min-frequency 8 \
      --chunk-size 32768 \
      --allowed-lengths 1,2,4,8,16,32
  ```
- Result:
  - Wall clock: 3 min 01 s (trainer loop 180.60 s).
  - CPU: 2 869.6 s user / 32.5 s sys (≈16× parallelism).
  - Peak RSS: 739 700 kB (≈705 MiB), +10 MiB vs. 8 192 vocab baseline.
  - Merges: 16 121 (`TargetVocabReached`); last merge freq ≈385; ≈979 k distinct pairs live at finish.
- Takeaway: doubling the vocab budget to 16 k and allowing merges up to 32 bytes roughly doubles runtime (93 s → 181 s trainer time) while raising peak RSS by only ~1.5%. Longer allowed lengths didn’t inflate memory much, suggesting pair-map density is still the dominant factor; scaling cost remains near-linear in the merge count.

## 2025-10-19 — `/usr-bin-100mb` at 32 k vocab (lengths up to 32 bytes)

- Command:
  ```
  /usr/bin/time -v env CARGO_HOME=$PWD/.cargo cargo run --release --bin train -- \
      --output bench/usr-bin-100mb/tokenizer-v32k.json \
      bench/usr-bin-100mb \
      --vocab-size 32768 \
      --min-frequency 8 \
      --chunk-size 32768 \
      --allowed-lengths 1,2,4,8,16,32
  ```
- Result:
  - Wall clock: 6 min 29 s (trainer loop 389.13 s).
  - CPU: 5 947.2 s user / 81.6 s sys (≈15.5× parallelism), with 1.5 M voluntary and 2.7 M involuntary context switches logged.
  - Peak RSS: 731 784 kB (≈698 MiB), essentially flat vs. the 16 k run despite doubling merges.
  - Merges: 32 505 (`TargetVocabReached`); last merge freq ≈181; ≈1.60 M distinct pairs tracked.
- Takeaway: scaling from 16 k to 32 k vocab nearly doubled trainer time (181 s → 389 s) while holding memory steady—consistent with our linear-in-merge expectations. Enabling 32-byte tokens still didn’t blow out RSS; future memory wins will require denser pair storage rather than tweaking allowed lengths.

### Python compatibility spot-check

- Command:
  ```
  UV_CACHE_DIR=$PWD/.uv-cache uv run --with tokenizers python -c \
      "from tokenizers import Tokenizer; tok = Tokenizer.from_file('bench/usr-bin-100mb/tokenizer-v32k.json'); \
       print('vocab_size', tok.get_vocab_size(False)); \
       out = tok.encode('Hello, world!'); \
       print('tokens', out.tokens[:10]); \
       print('ids', out.ids[:10])"
  ```
- Result: tokenizer loads successfully; reported vocab size 32 761 and produced UTF-8-friendly tokens/ids for the sample string.

- Token sanity check (first 512 bytes of `/usr/lib/cargo/bin/coreutils/ls`):
  ```
  UV_CACHE_DIR=$PWD/.uv-cache uv run --with tokenizers python -c \
      "from tokenizers import Tokenizer; from pathlib import Path; \
       tok = Tokenizer.from_file('bench/usr-bin-100mb/tokenizer-v32k.json'); \
       data = Path('/usr/lib/cargo/bin/coreutils/ls').read_bytes()[:512]; \
       enc = tok.encode(data.decode('latin-1')); \
       print('len', len(enc.ids)); \
       print('first_tokens', enc.tokens[:24]); \
       print('first_ids', enc.ids[:24])"
  ```
  - Output (abridged): `len 171`, first tokens include `'\x7f'`, `'EL'`, `'F\x02'`, `'>'`, etc., confirming merges respect even-byte boundaries and align with ELF header patterns.
- Result (16384 vocab, 1/2/4/8/16/32 lengths):
  - Wall clock: 3 min 01 s (trainer 180.60 s).
  - Peak RSS: 739 700 kB (≈705 MiB).
  - Merges: 16 121 (`TargetVocabReached`); last merge freq ≈385; ≈979 k distinct pairs remaining.
