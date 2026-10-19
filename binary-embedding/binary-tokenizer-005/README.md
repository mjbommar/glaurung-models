# Binary Tokenizer 005

A BPE tokenizer trained on binary executable files for security research and binary analysis tasks.

## Overview

This tokenizer uses Byte Pair Encoding (BPE) trained on latin-1 encoded binary data from system executables across multiple operating systems and architectures (Alpine, Ubuntu, Debian, Rocky Linux, ARM64, x86-64).

- **Vocabulary Size**: 65,536 tokens
- **Training Data**: System binaries from various OS distributions
- **Encoding**: Latin-1 (each byte 0-255 maps to a single character)

## Installation

```bash
pip install tokenizers transformers
```

## Usage

### Method 1: Using the tokenizers library (Recommended)

```python
from tokenizers import Tokenizer

# Load tokenizer directly from Hugging Face Hub
tokenizer = Tokenizer.from_pretrained("mjbommar/binary-tokenizer-005")

# Process binary data - MUST use latin-1 encoding
with open("binary_file", "rb") as f:
    raw_bytes = f.read()
    text = raw_bytes.decode('latin-1')  # Convert bytes to latin-1 string
    encoded = tokenizer.encode(text)
    tokens = encoded.ids
    
# Decode back to text
decoded = tokenizer.decode(tokens)
```

### Method 2: Using transformers library

```python
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer

# Load the base tokenizer
base_tokenizer = Tokenizer.from_pretrained("mjbommar/binary-tokenizer-005")

# Wrap with PreTrainedTokenizerFast for transformers compatibility
tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)

# Process binary data
with open("binary_file", "rb") as f:
    raw_bytes = f.read()
    text = raw_bytes.decode('latin-1')
    
# Tokenize (returns dict with input_ids, attention_mask, etc.)
result = tokenizer(text)
tokens = result["input_ids"]
```

## Important: Data Format

The tokenizer expects binary data encoded as latin-1 strings, NOT hex strings:

```python
# CORRECT - Use latin-1 encoded bytes
raw_bytes = b'\x7fELF\x01\x01'
text = raw_bytes.decode('latin-1')  # → '\x7fELF\x01\x01'
encoded = tokenizer.encode(text)

# WRONG - Do not use hex strings
hex_str = "7f 45 4c 46 01 01"  # ❌ Will not work correctly
```

## Example: Tokenizing an ELF Header

```python
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_pretrained("mjbommar/binary-tokenizer-005")

# ELF header bytes
elf_header = b'\x7fELF\x01\x01\x01\x00'
text = elf_header.decode('latin-1')

# Tokenize
encoded = tokenizer.encode(text)
print(f"Tokens: {encoded.ids}")  
# Output: [0, 45689, 205, 22648, 1]
# Where: 0='<|start|>', 45689='\x7fEL', 205='F', 22648='\x01\x01\x01\x00', 1='<|end|>'

# Special tokens: <|start|> (id=0) and <|end|> (id=1) mark file boundaries
# This helps models identify file headers and distinguish between files
# Content tokens are: [45689, 205, 22648]

# Note: Decoding adds spaces between tokens (BPE tokenizer behavior)
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: {repr(decoded)}")  # '\x7fEL F \x01\x01\x01\x00'
```

## Related Projects

- [mjbommar/glaurung](https://github.com/mjbommar/glaurung) - Binary analysis framework
- [mjbommar/glaurung-models](https://github.com/mjbommar/glaurung-models) - Binary embedding models and tokenizers

## License

For research and educational purposes.