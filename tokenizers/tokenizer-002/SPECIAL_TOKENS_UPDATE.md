# Special Token Format Update

**Date**: October 20, 2025
**File**: `src/config.rs`
**Status**: ✅ Fixed

---

## Summary

Updated special token format in the Rust tokenizer trainer to match binary-tokenizer-005 format, ensuring future tokenizers are compatible with training code out-of-the-box.

---

## Changes Made

### Before (LLaMA-style format)

```rust
special_tokens: vec![
    "<s>".into(),
    "</s>".into(),
    "<pad>".into(),
    "<unk>".into(),
    "<cls>".into(),
    "<sep>".into(),
    "<mask>".into(),
],
```

### After (Binary-tokenizer format)

```rust
special_tokens: vec![
    "<|start|>".into(),
    "<|end|>".into(),
    "<|pad|>".into(),
    "<|unk|>".into(),
    "<|cls|>".into(),
    "<|sep|>".into(),
    "<|mask|>".into(),
],
```

---

## Location

File: `tokenizers/tokenizer-002/src/config.rs`
Lines: 35-43

This is in the `Default` implementation for `TrainerConfig`.

---

## Impact

### Future Tokenizers

All tokenizers trained with this code will now:
- ✅ Use `<|start|>` instead of `<s>` (ID 65529)
- ✅ Use `<|end|>` instead of `</s>` (ID 65530)
- ✅ Work directly with binary embedding training code
- ✅ No post-processing needed

### Existing Tokenizers

- `glaurung-tokenizer-002.json` (already trained): Uses old format, requires `update_special_tokens.py` to fix
- `glaurung-tokenizer-002-updated.json`: Already fixed, ready to use

---

## Why This Matters

### The Problem

The binary embedding training code (`binary-embedding/src/binary_embedding/tokenizer.py`) expects special tokens in this format:

```python
self.start_token_id = self.tokenizer.token_to_id("<|start|>")
self.end_token_id = self.tokenizer.token_to_id("<|end|>")
# ... etc
```

Tokenizers trained with the old Rust code used `<s>`, `</s>`, etc., causing:
- `token_to_id("<|start|>")` returned `None`
- Required post-processing script
- Extra manual step for every new tokenizer

### The Solution

By fixing the Rust training code:
- ✅ Future tokenizers work immediately
- ✅ No post-processing needed
- ✅ Consistent format across all tokenizers
- ✅ Less manual work

---

## Testing

### Compilation

```bash
cd tokenizers/tokenizer-002
cargo check
```

**Result**: ✅ Compiles successfully

### Future Test (When Training New Tokenizer)

```bash
# Train a new tokenizer
cargo run --release --bin train -- \
  --output test-tokenizer.json \
  /path/to/binaries \
  --vocab-size 65536

# Verify special tokens
python -c "
from tokenizers import Tokenizer
tok = Tokenizer.from_file('test-tokenizer.json')
print('start:', tok.token_to_id('<|start|>'))
print('end:', tok.token_to_id('<|end|>'))
"

# Expected output:
# start: 65529
# end: 65530
```

---

## Backwards Compatibility

### Training Code

The training code (`binary_embedding/tokenizer.py`) now supports **both** formats:

```python
def get_special_token_id(alternatives: list[str], default: int | None = None) -> int:
    """Try multiple token formats, return first match or default."""
    for token_str in alternatives:
        token_id = self.tokenizer.token_to_id(token_str)
        if token_id is not None:
            return token_id
    # ...

self.start_token_id = get_special_token_id(["<|start|>", "<s>", "[BOS]"])
```

This means:
- ✅ Works with old tokenizers (`<s>` format)
- ✅ Works with new tokenizers (`<|start|>` format)
- ✅ Works with BERT-style (`[BOS]` format)

### Existing Workflows

No impact on:
- Existing trained models
- Existing tokenizers in production
- Current training runs

---

## Related Files

### Updated in This Fix
- `tokenizers/tokenizer-002/src/config.rs` ✅

### Support Both Formats
- `binary-embedding/src/binary_embedding/tokenizer.py` ✅

### One-Time Fix Scripts
- `update_special_tokens.py` - For patching existing tokenizers
- `glaurung-tokenizer-002-updated.json` - Already patched

---

## Verification Checklist

When training a new tokenizer:

- [ ] Special tokens use `<|token|>` format
- [ ] Start token is `<|start|>` (not `<s>`)
- [ ] End token is `<|end|>` (not `</s>`)
- [ ] Training code loads without errors
- [ ] No post-processing needed

---

## Git Diff

```diff
--- a/tokenizers/tokenizer-002/src/config.rs
+++ b/tokenizers/tokenizer-002/src/config.rs
@@ -33,13 +33,13 @@ impl Default for TrainerConfig {
             allowed_token_lengths: vec![1, 2, 4, 8],
             show_progress: true,
             special_tokens: vec![
-                "<s>".into(),
-                "</s>".into(),
-                "<pad>".into(),
-                "<unk>".into(),
-                "<cls>".into(),
-                "<sep>".into(),
-                "<mask>".into(),
+                "<|start|>".into(),
+                "<|end|>".into(),
+                "<|pad|>".into(),
+                "<|unk|>".into(),
+                "<|cls|>".into(),
+                "<|sep|>".into(),
+                "<|mask|>".into(),
             ],
             plateau_frequency_floor: 128,
             plateau_patience: 32,
```

---

## Next Steps

### For Next Tokenizer Training

When training glaurung-binary-tokenizer-002 or future tokenizers:

1. ✅ Use updated Rust code (already done)
2. ✅ Train as normal:
   ```bash
   cargo run --release --bin train -- \
     --output tokenizer.json \
     /path/to/data \
     --vocab-size 65536
   ```
3. ✅ Verify special tokens:
   ```bash
   python -c "
   from tokenizers import Tokenizer
   tok = Tokenizer.from_file('tokenizer.json')
   assert tok.token_to_id('<|start|>') is not None
   print('✓ Special tokens correct!')
   "
   ```
4. ✅ Use directly in training (no post-processing)

---

## Root Cause Analysis

**Why did we use `<s>` format originally?**

The HuggingFace `tokenizers` library examples often use LLaMA-style tokens (`<s>`, `</s>`). The original code followed this convention without considering compatibility with the existing binary-tokenizer-005 format.

**Why does this matter?**

Consistency across the project:
- binary-tokenizer-005: Uses `<|token|>` format
- Binary embedding training code: Expects `<|token|>` format
- New tokenizers: Should match existing convention

**Lesson learned**: Always check existing conventions before choosing token formats.

---

**Status**: ✅ Fixed in source code
**Retroactive fix**: Use `update_special_tokens.py` for existing tokenizers
**Future tokenizers**: Will have correct format automatically
