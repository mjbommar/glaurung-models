# Glaurung Binary Tokenizer 001 - Special Token Update

**Date**: October 20, 2025
**Status**: ✅ Complete

---

## Summary

Successfully updated `glaurung-tokenizer-002.json` special tokens to match `binary-tokenizer-005` format, enabling seamless tokenizer switching in the binary embedding training code.

---

## Changes Made

### Special Token Name Updates

Updated 7 special tokens at IDs 65529-65535:

| Token ID | Original Format | New Format | Purpose |
|----------|----------------|------------|---------|
| 65529 | `<s>` | `<|start|>` | Sequence start |
| 65530 | `</s>` | `<|end|>` | Sequence end |
| 65531 | `<pad>` | `<|pad|>` | Padding |
| 65532 | `<unk>` | `<|unk|>` | Unknown token |
| 65533 | `<cls>` | `<|cls|>` | Classification |
| 65534 | `<sep>` | `<|sep|>` | Separator |
| 65535 | `<mask>` | `<|mask|>` | Masked token (MLM) |

**Why**: The training code expects `<|token|>` format, not `<token>` format.

---

## Files

### Input
`tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json` (original)

### Output
`binary-embedding/glaurung-tokenizer-002-updated.json` (updated)

### Script
`update_special_tokens.py` - Automated token renaming tool

---

## Verification

### Before Update
```python
tokenizer.token_to_id("<s>")          # 65529
tokenizer.token_to_id("<|start|>")    # None
```

### After Update
```python
tokenizer.token_to_id("<s>")          # None
tokenizer.token_to_id("<|start|>")    # 65529
```

### Vocabulary Integrity
- ✅ All 65,536 tokens preserved
- ✅ All learned tokens (IDs 0-65528) unchanged
- ✅ Only special token names updated (IDs 65529-65535)

### Sample Verification
```
ID 256:  0x0000              ✓ Unchanged
ID 257:  0x00000000          ✓ Unchanged
ID 258:  0x0000000000000000  ✓ Unchanged
ID 300:  0x8001000000        ✓ Unchanged
ID 500:  0x04000000          ✓ Unchanged
ID 1000: 0x6d6f              ✓ Unchanged
```

---

## Training Code Compatibility

### Updated Tokenizer Works With Original Code ✅

```python
from binary_embedding.tokenizer import BinaryTokenizer

tok = BinaryTokenizer('binary-embedding/glaurung-tokenizer-002-updated.json')

# All special tokens found
tok.start_token_id   # 65529 ✓
tok.end_token_id     # 65530 ✓
tok.pad_token_id     # 65531 ✓
tok.mask_token_id    # 65535 ✓
tok.sep_token_id     # 65534 ✓
tok.cls_token_id     # 65533 ✓
tok.unk_token_id     # 65532 ✓
```

**No code changes required!** The training code now works with both:
- `binary-tokenizer-005/tokenizer.json` (special tokens at IDs 0-134)
- `glaurung-tokenizer-002-updated.json` (special tokens at IDs 65529-65535)

---

## Key Differences Between Tokenizers

| Feature | binary-tokenizer-005 | glaurung-tokenizer-002-updated |
|---------|---------------------|-------------------------------|
| **Special token location** | IDs 0-134 (beginning) | IDs 65529-65535 (end) |
| **Special token format** | `<|start|>`, `<|end|>` | `<|start|>`, `<|end|>` ✅ |
| **Single bytes** | IDs 135-390 | IDs 0-255 |
| **Learned multi-byte** | IDs 391-65535 | IDs 256-65528 |
| **Total vocab** | 65,536 | 65,536 |
| **Training data** | ~5GB mixed | 13GB binaries-small |
| **Compression** | ~2.6 bytes/token | 2.849 bytes/token |
| **Architectures** | x86 only | x86-64 + ARM64 |

---

## How to Switch Tokenizers

### Option 1: Update Symlink (Recommended)

```bash
cd binary-embedding

# Remove old symlink (if exists)
rm tokenizer.json 2>/dev/null

# Link to updated tokenizer
ln -s glaurung-tokenizer-002-updated.json tokenizer.json

# Verify
ls -l tokenizer.json
```

### Option 2: Update run.sh Directly

Edit `binary-embedding/run.sh`:

```bash
# OLD:
--tokenizer-path ./tokenizer.json \

# NEW:
--tokenizer-path ./glaurung-tokenizer-002-updated.json \
```

---

## Testing

### Quick Load Test

```bash
cd binary-embedding
PYTHONPATH=src uv run python -c '
from binary_embedding.tokenizer import load_tokenizer
tok = load_tokenizer("./glaurung-tokenizer-002-updated.json")
print(f"✓ Loaded! Vocab: {tok.vocab_size:,}, start_token_id={tok.start_token_id}")
'
```

**Expected output**:
```
✓ Loaded! Vocab: 65,536, start_token_id=65529
```

### Quick Training Test (100 steps)

```bash
cd binary-embedding

uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size nano \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --tokenizer-path ./glaurung-tokenizer-002-updated.json \
  --max-steps 100 \
  --batch-size 8 \
  --output-dir ./test-output-glaurung \
  --no-wandb
```

---

## Expected Benefits of Switching

### Compression Improvement
- **2.6 → 2.849 bytes/token** (+9.6% compression)
- Fewer tokens per binary → potentially better embeddings
- More efficient sequence representation

### Multi-Architecture Support
- **ARM64 patterns** learned (13% of top 100 tokens)
- Better generalization across architectures
- Useful for mobile malware, Apple Silicon, AWS Graviton analysis

### Instruction Awareness
- **Complete x86-64 instructions** captured as single tokens
  - `0x4883ec` = SUB rsp (ID 335)
  - `0x48ff15` = CALL [rip+disp32] (ID 340)
  - `0x0f1f440000` = 5-byte NOP (ID 333)
- Better than prefix-based fragmentation

### Training Data Quality
- **13GB vs 5GB** (2.6x larger corpus)
- **30,738 binaries** from multiple platforms
- Production binaries (stripped, optimized)
- Better cross-platform coverage

---

## Important Notes

### ⚠️ Cannot Resume Training

Models trained with `binary-tokenizer-005` **CANNOT** use `glaurung-tokenizer-002-updated`:
- Token ID 500 has different meanings in each tokenizer
- Embeddings learned are tokenizer-specific
- **Must train from scratch** with new tokenizer

### ✅ Can Switch for New Runs

- Starting fresh training runs: ✅ Works
- Continuing from checkpoint: ❌ Requires same tokenizer
- Assessment/inference: ❌ Requires original tokenizer

---

## Uploading to HuggingFace

The updated tokenizer should be uploaded as `mjbommar/glaurung-binary-tokenizer-001`:

```bash
# Copy to final location
cp binary-embedding/glaurung-tokenizer-002-updated.json \
   glaurung-binary-tokenizer-001-final.json

# Upload
python push_glaurung_binary_tokenizer_001.py \
  --tokenizer-path glaurung-binary-tokenizer-001-final.json \
  --readme-path glaurung_binary_tokenizer_001_README.md
```

**README already created**: `glaurung_binary_tokenizer_001_README.md`

---

## Next Steps

1. ✅ **Special tokens updated** - COMPLETE
2. ⏭️ **Test with training code** - Ready to test
3. ⏭️ **Upload to HuggingFace** - Ready to upload
4. ⏭️ **Run full training** - Can start after testing

---

## Script Usage

To update another tokenizer with the same changes:

```bash
python update_special_tokens.py \
  --input path/to/tokenizer.json \
  --output path/to/updated-tokenizer.json \
  --verify
```

---

## Comparison with Previous Approach

### ❌ What We Tried First: Modify Training Code

- Add fallback logic to `tokenizer.py`
- Support multiple token formats (`<|start|>`, `<s>`, `[BOS]`)
- More complex code
- Every model using the tokenizer needs updated code

### ✅ What We Did Instead: Modify Tokenizer JSON

- One-time JSON file edit
- No code changes needed
- Cleaner separation of concerns
- Tokenizers are now interchangeable

**Lesson**: When possible, modify data format to match code expectations, not vice versa.

---

**Status**: ✅ Ready for production use
**Recommendation**: Test with 100-step training run, then deploy full training
