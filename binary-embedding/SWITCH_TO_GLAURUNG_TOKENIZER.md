# Quick Start: Switch to Glaurung Binary Tokenizer 001

**Date**: October 20, 2025
**Time to complete**: ~5 minutes

---

## What This Does

Switches training from `binary-tokenizer-005` to the new `glaurung-binary-tokenizer-001` (tokenizer-002) for:
- Better compression (2.849 vs 2.6 bytes/token)
- Multi-architecture support (x86-64 + ARM64)
- Larger training corpus (13GB vs 5GB)
- Instruction-aware patterns

---

## Prerequisites

✅ **Already complete**:
- Updated tokenizer created: `glaurung-tokenizer-002-updated.json`
- Training code supports both tokenizers (special token detection)
- Tokenizer verified and tested

---

## Step 1: Update Tokenizer Path (Choose One)

### Option A: Symlink (Easiest)

```bash
cd binary-embedding

# Remove old symlink
rm tokenizer.json

# Link to new tokenizer
ln -s glaurung-tokenizer-002-updated.json tokenizer.json
```

### Option B: Edit run.sh

Edit `run.sh` line 17:

```bash
# Change from:
--tokenizer-path ./tokenizer.json \

# To:
--tokenizer-path ./glaurung-tokenizer-002-updated.json \
```

---

## Step 2: Quick Test (Optional but Recommended)

Test tokenizer loading:

```bash
cd binary-embedding

PYTHONPATH=src uv run python -c '
from binary_embedding.tokenizer import load_tokenizer
tok = load_tokenizer("./glaurung-tokenizer-002-updated.json")
print(f"✓ Vocab: {tok.vocab_size:,}")
print(f"✓ start_token_id: {tok.start_token_id}")
print(f"✓ Ready to train!")
'
```

**Expected output**:
```
✓ Vocab: 65,536
✓ start_token_id: 65529
✓ Ready to train!
```

---

## Step 3: Run Training

### Quick 100-step Test

```bash
uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size nano \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --tokenizer-path ./glaurung-tokenizer-002-updated.json \
  --max-steps 100 \
  --batch-size 8 \
  --output-dir ./test-glaurung-100 \
  --no-wandb
```

### Full Training Run

Once quick test passes, run the full training:

```bash
./run.sh
```

The script will automatically use the new tokenizer if you updated the symlink or `run.sh`.

---

## What to Expect

### Training Differences

**Tokenization**:
- Slightly fewer tokens per binary (better compression)
- Different token IDs (ARM64 patterns present)
- Same special token behavior

**Model Performance**:
- May need different hyperparameters (test first)
- Potentially better generalization across architectures
- Should converge to similar or better scores

---

## Important Notes

### ⚠️ Do NOT Resume from Old Checkpoints

**CRITICAL**: Models trained with `binary-tokenizer-005` cannot use the new tokenizer.

```bash
# ❌ WRONG - This will fail or produce garbage
uv run python -m binary_embedding.cli mlm-pretrain \
  --init-from-checkpoint ./old-model-checkpoint-1000 \
  --tokenizer-path ./glaurung-tokenizer-002-updated.json \
  ...

# ✅ CORRECT - Train from scratch
uv run python -m binary_embedding.cli mlm-pretrain \
  --tokenizer-path ./glaurung-tokenizer-002-updated.json \
  ...
```

### ✅ Safe Operations

- Starting new training: ✅ Safe
- Assessment of new models: ✅ Safe
- Using existing binary-tokenizer-005 models: ✅ Keep original tokenizer

---

## Rollback (If Needed)

If you need to switch back:

```bash
# If using symlink:
cd binary-embedding
rm tokenizer.json
ln -s binary-tokenizer-005/tokenizer.json tokenizer.json

# If using run.sh:
# Change back to:
--tokenizer-path ./tokenizer.json \
```

---

## Verification Checklist

After switching, verify:

- [ ] Tokenizer loads without errors
- [ ] Special tokens found (start_token_id should be 65529)
- [ ] Quick test (100 steps) completes successfully
- [ ] Loss decreases during training
- [ ] No unusual errors in logs

---

## Expected Benefits

**Immediate**:
- ✅ Better compression (fewer tokens per binary)
- ✅ Multi-architecture token coverage

**Long-term** (after full training):
- ✅ Better embeddings for ARM64 binaries
- ✅ Better cross-platform malware detection
- ✅ Improved instruction-level understanding

---

## Troubleshooting

### Error: "special token not found"

**Solution**: Make sure you're using `glaurung-tokenizer-002-updated.json` (with updated special tokens), not the original `glaurung-tokenizer-002.json`.

### Error: "vocab size mismatch"

**Solution**: This happens if using old checkpoint with new tokenizer. Train from scratch (no `--init-from-checkpoint`).

### Warning: "Token IDs different than expected"

**Expected behavior**: Token IDs will differ between tokenizers. This is normal.

---

## Summary

1. ✅ Update tokenizer path (symlink or run.sh)
2. ✅ Test tokenizer loading
3. ✅ Run 100-step test
4. ✅ Launch full training

**Total time**: 5 minutes setup + training time

---

**Status**: Ready to use! 🚀
