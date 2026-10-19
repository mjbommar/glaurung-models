# Tokenizer Switching Guide

**Date**: October 2025
**Goal**: Switch binary embedding training from binary-tokenizer-005 to glaurung-binary-tokenizer-001

---

## Summary

✅ **YES**, the training code supports switching tokenizers easily via the `--tokenizer-path` CLI argument.

The tokenizer is completely decoupled from the model training code, making it straightforward to swap tokenizers without code changes.

---

## How Tokenizer Loading Works

### 1. Tokenizer Loading Function

Located in: `binary-embedding/src/binary_embedding/tokenizer.py:162-176`

```python
def load_tokenizer(
    tokenizer_path: str | Path | None = None,
) -> BinaryTokenizer:
    """Load the binary tokenizer.

    Args:
        tokenizer_path: Path to tokenizer.json file. If None, uses default path.

    Returns:
        BinaryTokenizer instance.
    """
    if tokenizer_path is None:
        tokenizer_path = Path("binary-tokenizer-005/tokenizer.json")

    return BinaryTokenizer(tokenizer_path)
```

**Default**: If no path is provided, it defaults to `binary-tokenizer-005/tokenizer.json`

### 2. CLI Integration

Located in: `binary-embedding/src/binary_embedding/cli.py:58-62`

```python
click.option(
    "--tokenizer-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to tokenizer.json file (uses default if not provided)",
),
```

The `--tokenizer-path` argument is available in both:
- `mlm-pretrain` command (line 583)
- `contrastive-train` command (line 871)
- `test` command (line 990)
- `assess` command (line 1041)

### 3. Current Usage

Current run script (`binary-embedding/run.sh:17`):

```bash
--tokenizer-path ./tokenizer.json \
```

This currently points to `binary-tokenizer-005/tokenizer.json` (presumably via symlink).

---

## How to Switch Tokenizers

### Option 1: Update Symlink (Recommended)

```bash
cd binary-embedding

# Remove old symlink
rm tokenizer.json

# Create new symlink to glaurung-tokenizer-001
ln -s ../tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json tokenizer.json

# Verify
ls -l tokenizer.json
```

**Pros**:
- No code changes needed
- Existing scripts work unchanged
- Easy to switch back

### Option 2: Update run.sh Directly

Edit `binary-embedding/run.sh:17`:

```bash
# OLD:
--tokenizer-path ./tokenizer.json \

# NEW:
--tokenizer-path ../tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json \
```

**Pros**:
- Explicit path makes it clear which tokenizer is used
- No symlink indirection

### Option 3: Set as Default in Code

Edit `binary-embedding/src/binary_embedding/tokenizer.py:174`:

```python
# OLD:
if tokenizer_path is None:
    tokenizer_path = Path("binary-tokenizer-005/tokenizer.json")

# NEW:
if tokenizer_path is None:
    tokenizer_path = Path("../tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json")
```

**Pros**:
- No CLI args needed
- All commands use new tokenizer by default

**Cons**:
- Code change required
- Less flexible

---

## Important Considerations

### 1. Special Tokens Compatibility

**binary-tokenizer-005** has special tokens at IDs 0-134:
- `<|start|>` (ID 0)
- `<|end|>` (ID 1)
- `<|sep|>`, `<|cls|>`, `<|pad|>`, `<|mask|>`, `<|unk|>`
- `<|reserved:0|>` through `<|reserved:126|>`

**glaurung-tokenizer-001** has NO special tokens defined:
- Learned tokens start immediately at ID 256
- No `<|start|>`, `<|end|>`, etc.

**Impact**: The tokenizer loading code expects these special tokens:

```python
# From tokenizer.py:26-32
self.start_token_id: int = self.tokenizer.token_to_id("<|start|>")
self.end_token_id: int = self.tokenizer.token_to_id("<|end|>")
self.sep_token_id: int = self.tokenizer.token_to_id("<|sep|>")
self.cls_token_id: int = self.tokenizer.token_to_id("<|cls|>")
self.pad_token_id: int = self.tokenizer.token_to_id("<|pad|>")
self.mask_token_id: int = self.tokenizer.token_to_id("<|mask|>")
self.unk_token_id: int = self.tokenizer.token_to_id("<|unk|>")
```

**Solution Required**: Add special tokens to glaurung-tokenizer-001 OR modify the tokenizer.py code to handle missing special tokens gracefully.

### 2. Vocabulary Size

Both tokenizers: 65,536 tokens (same)

This is critical because the model architecture is defined with `vocab_size` at initialization. Same vocab size means:
- ✅ Model embeddings layer stays the same size
- ✅ Output projection layer stays the same size
- ✅ No architecture changes needed

### 3. Model Checkpoints

**IMPORTANT**: Models trained with binary-tokenizer-005 **CANNOT** be used with glaurung-tokenizer-001 without retraining or special handling:

- Token ID 500 in binary-tokenizer-005 != Token ID 500 in glaurung-tokenizer-001
- The embeddings learned for each token are specific to that tokenizer's vocabulary
- Switching tokenizers requires **training from scratch**

**Checkpoint compatibility**:
- ✅ Can switch tokenizers for **new training runs**
- ❌ Cannot use old checkpoints with new tokenizer
- ❌ Cannot resume training with different tokenizer

---

## Step-by-Step Migration Plan

### Phase 1: Add Special Tokens to glaurung-tokenizer-001

**Option A: Modify Rust trainer to include special tokens**

Edit `tokenizers/tokenizer-002/src/main.rs` to add special tokens before training.

**Option B: Post-process existing tokenizer** (Easier)

Create a script to add special tokens:

```python
from tokenizers import Tokenizer

# Load original tokenizer
tokenizer = Tokenizer.from_file("tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json")

# Add special tokens
special_tokens = [
    "<|start|>",
    "<|end|>",
    "<|sep|>",
    "<|cls|>",
    "<|pad|>",
    "<|mask|>",
    "<|unk|>",
]

tokenizer.add_special_tokens(special_tokens)

# Save modified tokenizer
tokenizer.save("tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002-with-special.json")

print(f"New vocab size: {tokenizer.get_vocab_size()}")
```

**Issue**: This increases vocab size from 65,536 to 65,543 (adds 7 tokens), which breaks the architecture.

**Better Solution**: Modify tokenizer.py to use existing tokens as special tokens:

```python
# Use first few token IDs as special tokens if not found
self.pad_token_id: int = self.tokenizer.token_to_id("<|pad|>") or 0
self.unk_token_id: int = self.tokenizer.token_to_id("<|unk|>") or 1
# ... etc
```

### Phase 2: Update Tokenizer Path

Choose one of the options above (symlink, run.sh, or default code change).

### Phase 3: Start New Training Run

```bash
cd binary-embedding

# Use new tokenizer
./run.sh
```

**DO NOT** use `--init-from-checkpoint` or `--resume-from-checkpoint` with old binary-tokenizer-005 checkpoints.

### Phase 4: Validate

Check that:
1. Tokenizer loads without errors
2. Special tokens are assigned correctly
3. Model trains normally
4. Assessment tests pass

---

## Recommended Approach

### Immediate Solution (Low Risk)

1. **Create modified tokenizer with special tokens mapped to existing IDs**:

```python
#!/usr/bin/env python3
"""Create glaurung-tokenizer-001 compatible with training code."""

from tokenizers import Tokenizer

# Load original
tokenizer = Tokenizer.from_file(
    "tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json"
)

# Map existing tokens to special token names
# Use rarely-used byte combinations as special tokens
# IDs 0-255 are single bytes, so we use the first multi-byte tokens

# Option: Rename first few tokens
vocab = tokenizer.get_vocab()

# Create a new tokenizer with special tokens at specific IDs
# This requires manual manipulation of the JSON file

# Save
tokenizer.save("binary-embedding/glaurung-tokenizer-001-training.json")
```

**Issue**: The HuggingFace `tokenizers` library doesn't easily support renaming tokens. Special tokens must be added during training.

### Better Solution: Modify tokenizer.py

Update `binary-embedding/src/binary_embedding/tokenizer.py` to handle tokenizers without special tokens:

```python
def __init__(self, tokenizer_path: str | Path) -> None:
    """Initialize the binary tokenizer.

    Args:
        tokenizer_path: Path to the tokenizer.json file.
    """
    self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
    self.vocab_size: int = self.tokenizer.get_vocab_size()

    # Get special token IDs (use defaults if not found)
    self.start_token_id: int = self.tokenizer.token_to_id("<|start|>")
    if self.start_token_id is None:
        # Use first available ID
        self.start_token_id = 0
        print(f"Warning: <|start|> not found, using ID {self.start_token_id}")

    self.end_token_id: int = self.tokenizer.token_to_id("<|end|>")
    if self.end_token_id is None:
        self.end_token_id = 1
        print(f"Warning: <|end|> not found, using ID {self.end_token_id}")

    # ... similar for other special tokens
```

This allows using tokenizers with or without predefined special tokens.

---

## Testing Strategy

### 1. Small-Scale Test

Before full training, test tokenizer loading:

```bash
cd binary-embedding

uv run python -m binary_embedding.cli test \
  --checkpoint-path ./models/test \
  --tokenizer-path ../tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json \
  --text "48 65 6c 6c 6f"
```

### 2. Quick Training Run

Run 100 steps to verify everything works:

```bash
uv run python -m binary_embedding.cli mlm-pretrain \
  --model-size nano \
  --data-dir /nas4/data/glaurung-data/binaries-small/ \
  --tokenizer-path ../tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json \
  --max-steps 100 \
  --batch-size 8 \
  --output-dir ./test-output \
  --no-wandb
```

### 3. Assessment Test

After quick training, run assessment to verify model quality:

```bash
uv run python -m binary_embedding.cli assess \
  --checkpoint-path ./test-output/checkpoint-100 \
  --tokenizer-path ../tokenizers/tokenizer-002/bench/production/glaurung-tokenizer-002.json
```

---

## Expected Results

### Positive Outcomes

- **Better compression**: 2.849 bytes/token vs ~2.6 → fewer tokens per binary → potentially better embeddings
- **Multi-architecture awareness**: Model may generalize better across x86-64 and ARM64
- **Instruction-aware patterns**: Complete instructions captured as single tokens
- **Production-ready**: Trained on larger, more diverse corpus

### Potential Challenges

- **ARM64 tokens may confuse x86-only models**: If training data is x86-only, ARM64 tokens are wasted vocabulary
- **Different pattern distribution**: Model may need more training steps to converge
- **Special token handling**: Requires code modification or tokenizer modification

---

## Conclusion

### Answer: Yes, Easy to Switch ✅

The training code is **well-designed for tokenizer switching**:
- ✅ Single `--tokenizer-path` argument
- ✅ No hardcoded tokenizer dependencies in training logic
- ✅ Clean separation of concerns

### One Issue: Special Tokens ⚠️

The **only blocker** is special token handling. Two solutions:
1. **Modify tokenizer.py** to use default IDs when special tokens not found (Recommended)
2. **Add special tokens** to glaurung-tokenizer-001 (requires vocab size handling)

### Recommendation

**Modify tokenizer.py** to be more flexible, then switch to glaurung-tokenizer-001 for new training runs.

Expected timeline:
- 15 minutes: Modify tokenizer.py
- 15 minutes: Test loading
- 30 minutes: Quick training test (100 steps)
- **Total: ~1 hour** to verify everything works

Then launch full training run with confidence.

---

## Files to Modify

1. **`binary-embedding/src/binary_embedding/tokenizer.py`**
   - Add None checks for special token IDs
   - Use defaults when tokens not found

2. **`binary-embedding/run.sh`** (or use symlink)
   - Update `--tokenizer-path` to point to glaurung-tokenizer-002.json

3. **Optional**: Create wrapper script for testing
   - `test_new_tokenizer.sh` with reduced parameters

---

**Ready to proceed with migration? The infrastructure is already in place!**
