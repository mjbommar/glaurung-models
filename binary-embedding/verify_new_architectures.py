#!/usr/bin/env python3
"""Verify new architecture configurations."""

import sys
sys.path.insert(0, 'src')

from binary_embedding.models import ModelSize, get_model_config, create_model

print("="*80)
print("ARCHITECTURE VERIFICATION")
print("="*80)

for size in ModelSize:
    print(f"\n{size.value.upper()}:")
    print("-" * 40)

    config = get_model_config(size)

    # Calculate key metrics
    dims_per_layer = config.hidden_size / config.num_hidden_layers
    dims_per_head = config.hidden_size / config.num_attention_heads
    ffn_ratio = config.intermediate_size / config.hidden_size

    # Estimate parameters
    embedding_params = config.vocab_size * config.hidden_size + 520 * config.hidden_size
    # Per layer: 4 attention matrices (QKV + output) + 2 FFN + layer norms
    attention_params = 4 * config.hidden_size * config.hidden_size
    ffn_params = 2 * config.hidden_size * config.intermediate_size
    layer_params = attention_params + ffn_params
    encoder_params = config.num_hidden_layers * layer_params
    total_params = embedding_params + encoder_params

    print(f"  Hidden size:        {config.hidden_size}")
    print(f"  Num layers:         {config.num_hidden_layers}")
    print(f"  Num heads:          {config.num_attention_heads}")
    print(f"  FFN size:           {config.intermediate_size}")
    print()
    print(f"  Dims/layer:         {dims_per_layer:.1f}")
    print(f"  Dims/head:          {dims_per_head:.1f}")
    print(f"  FFN ratio:          {ffn_ratio:.1f}×")
    print()
    print(f"  Embedding params:   {embedding_params/1e6:.2f}M ({embedding_params/total_params*100:.1f}%)")
    print(f"  Encoder params:     {encoder_params/1e6:.2f}M ({encoder_params/total_params*100:.1f}%)")
    print(f"  TOTAL params:       {total_params/1e6:.2f}M")
    print()

    # Validation checks
    checks = []
    if dims_per_layer >= 48:
        checks.append("✅ Dims/layer ≥ 48 (safe)")
    else:
        checks.append(f"❌ Dims/layer = {dims_per_layer:.1f} < 48 (UNSAFE)")

    if dims_per_head >= 32:
        checks.append(f"✅ Dims/head = {dims_per_head:.0f} ≥ 32")
    else:
        checks.append(f"⚠️ Dims/head = {dims_per_head:.0f} < 32")

    if ffn_ratio == 4.0:
        checks.append("✅ FFN ratio = 4× (optimal)")
    else:
        checks.append(f"⚠️ FFN ratio = {ffn_ratio:.1f}× (non-standard)")

    for check in checks:
        print(f"  {check}")

print("\n" + "="*80)
print("COMPARISON TABLE")
print("="*80)
print(f"{'Size':<8} {'Hidden':>6} {'Layers':>6} {'Heads':>5} {'FFN':>6} {'D/L':>5} {'D/H':>5} {'Params':>8}")
print("-" * 80)

for size in ModelSize:
    config = get_model_config(size)
    dims_per_layer = config.hidden_size / config.num_hidden_layers
    dims_per_head = config.hidden_size / config.num_attention_heads

    embedding_params = config.vocab_size * config.hidden_size + 520 * config.hidden_size
    attention_params = 4 * config.hidden_size * config.hidden_size
    ffn_params = 2 * config.hidden_size * config.intermediate_size
    layer_params = attention_params + ffn_params
    encoder_params = config.num_hidden_layers * layer_params
    total_params = embedding_params + encoder_params

    print(f"{size.value:<8} {config.hidden_size:6d} {config.num_hidden_layers:6d} "
          f"{config.num_attention_heads:5d} {config.intermediate_size:6d} "
          f"{dims_per_layer:5.0f} {dims_per_head:5.0f} {total_params/1e6:7.1f}M")

print("\n" + "="*80)
print("CREATE MODEL TEST")
print("="*80)

try:
    model, config = create_model(size=ModelSize.NANO)
    print(f"✅ Successfully created NANO model")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
except Exception as e:
    print(f"❌ Failed to create NANO model: {e}")

print("\nAll architecture validations complete!")
