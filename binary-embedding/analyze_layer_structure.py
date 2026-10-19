#!/usr/bin/env python3
"""Analyze the structure of model layers using SVD/eigenvalue decomposition."""

import numpy as np
from safetensors import safe_open
import json
from pathlib import Path

checkpoint_path = Path("micro_20251024_105559/checkpoint-50000")

# Load model weights
print("Loading model weights from checkpoint...")
tensors = {}
with safe_open(checkpoint_path / "model.safetensors", framework="numpy") as f:
    for key in f.keys():
        tensors[key] = f.get_tensor(key)

print(f"\nLoaded {len(tensors)} tensors")
print("\nTensor shapes:")
for name, tensor in sorted(tensors.items()):
    print(f"  {name}: {tensor.shape}")

# Analyze weight matrices
print("\n" + "="*80)
print("SINGULAR VALUE ANALYSIS")
print("="*80)

def analyze_matrix(name, matrix):
    """Compute SVD statistics for a weight matrix."""
    if len(matrix.shape) != 2:
        print(f"\n{name}: Skipping (not a 2D matrix, shape={matrix.shape})")
        return None

    print(f"\n{name}:")
    print(f"  Shape: {matrix.shape}")
    print(f"  Norm: {np.linalg.norm(matrix):.4f}")

    # Compute SVD
    try:
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

        # Singular value statistics
        print(f"  Singular values:")
        print(f"    Max: {s[0]:.4f}")
        print(f"    Min: {s[-1]:.6f}")
        print(f"    Condition number: {s[0]/s[-1]:.2f}")

        # Effective rank (normalized entropy)
        s_norm = s / s.sum()
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
        max_entropy = np.log(len(s))
        effective_rank = np.exp(entropy)
        print(f"    Effective rank: {effective_rank:.2f} / {len(s)} ({effective_rank/len(s)*100:.1f}%)")

        # Energy concentration
        cumsum = np.cumsum(s) / s.sum()
        for pct in [0.5, 0.9, 0.95, 0.99]:
            n_components = np.searchsorted(cumsum, pct) + 1
            print(f"    {pct*100:.0f}% energy in {n_components}/{len(s)} components ({n_components/len(s)*100:.1f}%)")

        # Top singular values
        print(f"    Top 10 singular values: {s[:10]}")

        return {
            'name': name,
            'shape': matrix.shape,
            'singular_values': s.tolist(),
            'condition_number': float(s[0]/s[-1]),
            'effective_rank': float(effective_rank),
            'rank_ratio': float(effective_rank/len(s)),
        }
    except Exception as e:
        print(f"  Error computing SVD: {e}")
        return None

# Analyze all 2D weight matrices
results = {}
for name, tensor in sorted(tensors.items()):
    if 'weight' in name and len(tensor.shape) == 2:
        result = analyze_matrix(name, tensor)
        if result:
            results[name] = result

# Compare layers
print("\n" + "="*80)
print("LAYER COMPARISON")
print("="*80)

# Group by layer type
layer_types = {
    'attention': [k for k in results.keys() if 'attn' in k or 'attention' in k],
    'mlp': [k for k in results.keys() if 'mlp' in k or 'fc' in k or 'ffn' in k],
    'embedding': [k for k in results.keys() if 'embed' in k or 'wte' in k or 'wpe' in k],
    'other': [k for k in results.keys() if not any(x in k for x in ['attn', 'attention', 'mlp', 'fc', 'ffn', 'embed', 'wte', 'wpe'])]
}

for layer_type, names in layer_types.items():
    if not names:
        continue
    print(f"\n{layer_type.upper()} layers:")
    for name in names:
        r = results[name]
        print(f"  {name}:")
        print(f"    Effective rank: {r['effective_rank']:.2f} ({r['rank_ratio']*100:.1f}%)")
        print(f"    Condition: {r['condition_number']:.2f}")

# Save results
output_path = Path("layer_structure_analysis.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nDetailed results saved to {output_path}")
