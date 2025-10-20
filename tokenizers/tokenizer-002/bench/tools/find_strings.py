#!/usr/bin/env python3
"""Search for strings, symbols, and names in the 64K tokenizer vocabulary."""

import json
from pathlib import Path
import re

def load_tokenizer(path):
    """Load tokenizer vocabulary."""
    with open(path) as f:
        data = json.load(f)
    return data['model']['vocab']

def is_printable_string(token_bytes, min_length=4):
    """Check if token is a printable ASCII string of sufficient length."""
    if len(token_bytes) < min_length:
        return False

    # Count printable characters
    printable_count = sum(1 for b in token_bytes if 32 <= b < 127)

    # Must be at least 80% printable
    if printable_count / len(token_bytes) < 0.8:
        return False

    # Decode and check
    try:
        text = token_bytes.decode('ascii', errors='ignore')
        # Filter out tokens that are just punctuation
        if not any(c.isalnum() for c in text):
            return False
        return True
    except:
        return False

def categorize_string(text):
    """Categorize what kind of string this is."""
    text_lower = text.lower()

    # Linux paths
    if text.startswith('/lib') or text.startswith('/usr') or text.startswith('/etc'):
        return 'Linux path'
    if '/' in text and any(c.isalpha() for c in text):
        return 'Unix path fragment'

    # Windows paths
    if ':\\' in text or text.startswith('C:'):
        return 'Windows path'
    if '\\' in text and any(c.isupper() for c in text):
        return 'Windows path fragment'

    # Function name patterns
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', text):
        # Common Linux syscalls
        if text in ['open', 'read', 'write', 'close', 'fork', 'exec', 'mmap', 'malloc', 'free', 'printf', 'scanf', 'exit', 'wait', 'pipe']:
            return 'Linux syscall/libc'
        # Common Windows APIs
        if text.startswith('Create') or text.startswith('Get') or text.startswith('Set'):
            return 'Windows API'
        if text.startswith('__') or text.startswith('_'):
            return 'Mangled/internal symbol'
        if text[0].isupper():
            return 'Function name (CamelCase)'
        return 'Function name (lowercase)'

    # Library names
    if 'lib' in text_lower and '.so' in text_lower:
        return 'Linux library (.so)'
    if '.dll' in text_lower:
        return 'Windows library (.dll)'

    # Section names
    if text in ['.text', '.data', '.bss', '.rodata', '.init', '.fini']:
        return 'ELF section name'

    # Compiler/linker markers
    if text.startswith('GCC:') or text.startswith('GNU'):
        return 'Compiler marker'

    # Error messages
    if any(word in text_lower for word in ['error', 'fail', 'invalid', 'null']):
        return 'Error message fragment'

    return 'String fragment'

def main():
    base_dir = Path(__file__).parent

    print("="*80)
    print("  STRING/SYMBOL ANALYSIS - 64K TOKENIZER")
    print("="*80)

    vocab = load_tokenizer(base_dir / "glaurung-tokenizer-002.json")
    id_to_token = {v: k for k, v in vocab.items()}

    # Find all string tokens
    string_tokens = []

    for token_id, token_str in id_to_token.items():
        if token_id < 256:  # Skip single bytes
            continue

        token_bytes = token_str.encode('latin-1')

        if is_printable_string(token_bytes, min_length=4):
            try:
                text = token_bytes.decode('ascii', errors='ignore')
                category = categorize_string(text)
                string_tokens.append({
                    'id': token_id,
                    'text': text,
                    'length': len(token_bytes),
                    'category': category
                })
            except:
                pass

    print(f"\nFound {len(string_tokens)} string/symbol tokens (out of {len(vocab)-256} learned tokens)")
    print(f"That's {100*len(string_tokens)/(len(vocab)-256):.2f}% of the vocabulary\n")

    # Group by category
    from collections import Counter
    categories = Counter(t['category'] for t in string_tokens)

    print("String categories:")
    for category, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {category:30s}: {count:4d} tokens")

    # Show examples from each category
    print(f"\n{'='*80}")
    print("  EXAMPLES FROM EACH CATEGORY")
    print("="*80)

    for category in sorted(categories.keys()):
        examples = [t for t in string_tokens if t['category'] == category][:10]
        if not examples:
            continue

        print(f"\n{category}:")
        for token in examples[:5]:  # Show first 5
            print(f"  ID {token['id']:5d}: {repr(token['text'])}")

    # Search for specific interesting patterns
    print(f"\n{'='*80}")
    print("  SPECIFIC PATTERN SEARCH")
    print("="*80)

    # Linux syscalls
    linux_syscalls = ['open', 'read', 'write', 'close', 'mmap', 'fork', 'exec']
    found_syscalls = []
    for token in string_tokens:
        for syscall in linux_syscalls:
            if syscall in token['text'].lower():
                found_syscalls.append(token)
                break

    print(f"\nLinux syscall-related tokens: {len(found_syscalls)}")
    for token in found_syscalls[:10]:
        print(f"  ID {token['id']:5d}: {repr(token['text'])}")

    # Windows APIs
    windows_apis = ['Create', 'Get', 'Set', 'Load', 'Find']
    found_winapi = []
    for token in string_tokens:
        for api in windows_apis:
            if api in token['text']:
                found_winapi.append(token)
                break

    print(f"\nWindows API-related tokens: {len(found_winapi)}")
    for token in found_winapi[:10]:
        print(f"  ID {token['id']:5d}: {repr(token['text'])}")

    # File extensions
    extensions = ['.so', '.dll', '.exe', '.a', '.o']
    found_extensions = []
    for token in string_tokens:
        for ext in extensions:
            if ext in token['text']:
                found_extensions.append(token)
                break

    print(f"\nFile extension tokens: {len(found_extensions)}")
    for token in found_extensions[:10]:
        print(f"  ID {token['id']:5d}: {repr(token['text'])}")

    # Look for common library names
    libraries = ['libc', 'libm', 'libpthread', 'kernel32', 'user32', 'ntdll']
    found_libs = []
    for token in string_tokens:
        for lib in libraries:
            if lib in token['text'].lower():
                found_libs.append(token)
                break

    print(f"\nLibrary name tokens: {len(found_libs)}")
    for token in found_libs[:10]:
        print(f"  ID {token['id']:5d}: {repr(token['text'])}")

    # Show longest strings
    print(f"\n{'='*80}")
    print("  LONGEST STRING TOKENS")
    print("="*80)

    longest = sorted(string_tokens, key=lambda x: -x['length'])[:20]
    for token in longest:
        print(f"  ID {token['id']:5d} (len={token['length']:2d}): {repr(token['text'][:60])}...")

    print()

if __name__ == '__main__':
    main()
