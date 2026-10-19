#!/usr/bin/env python3
"""Compare layer structure across multiple checkpoints to track training dynamics."""

import numpy as np
from safetensors import safe_open
import json
from pathlib import Path
from collections import defaultdict

run_dir = Path("micro_20251024_105559")
checkpoints = [
    "checkpoint-150000",
    "checkpoint-160000",
    "checkpoint-170000",
    "checkpoint-180000",
    "checkpoint-190000",
]

def analyze_matrix_svd(matrix):
    """Compute SVD statistics for a weight matrix."""
    if len(matrix.shape) != 2:
        return None

    try:
        U, s, Vt = np.linalg.svd(matrix, full_matrices=False)

        # Effective rank (normalized entropy)
        s_norm = s / s.sum()
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
        effective_rank = np.exp(entropy)

        return {
            'max_sv': float(s[0]),
            'min_sv': float(s[-1]),
            'condition': float(s[0] / (s[-1] + 1e-12)),
            'effective_rank': float(effective_rank),
            'rank_ratio': float(effective_rank / len(s)),
            'singular_values': s.tolist(),
        }
    except Exception as e:
        return None

# Collect data across checkpoints
results_by_checkpoint = {}

for ckpt_name in checkpoints:
    print(f"\nProcessing {ckpt_name}...")
    ckpt_path = run_dir / ckpt_name

    tensors = {}
    with safe_open(ckpt_path / "model.safetensors", framework="numpy") as f:
        for key in f.keys():
            if 'weight' in key and len(f.get_tensor(key).shape) == 2:
                tensors[key] = f.get_tensor(key)

    results = {}
    for name, tensor in tensors.items():
        result = analyze_matrix_svd(tensor)
        if result:
            results[name] = result

    step = int(ckpt_name.split("-")[1])
    results_by_checkpoint[step] = results

# Track trends for key matrices
print("\n" + "="*80)
print("TRAINING DYNAMICS: Layer 2 Attention (Final Layer)")
print("="*80)

layer3_matrices = [
    'roberta.encoder.layer.2.attention.self.key.weight',
    'roberta.encoder.layer.2.attention.self.query.weight',
    'roberta.encoder.layer.2.attention.self.value.weight',
]

for matrix_name in layer3_matrices:
    print(f"\n{matrix_name}:")
    print(f"{'Step':>8} {'EffRank':>8} {'Rank%':>7} {'Condition':>12} {'MaxSV':>8} {'MinSV':>10}")
    print("-" * 65)

    for step in sorted(results_by_checkpoint.keys()):
        data = results_by_checkpoint[step][matrix_name]
        print(f"{step:8d} {data['effective_rank']:8.2f} {data['rank_ratio']*100:6.1f}% "
              f"{data['condition']:12.1f} {data['max_sv']:8.3f} {data['min_sv']:10.6f}")

# Compare all layers
print("\n" + "="*80)
print("EFFECTIVE RANK TRENDS (all attention layers)")
print("="*80)

attention_matrices = sorted([k for k in results_by_checkpoint[150000].keys()
                            if 'attention' in k and 'self' in k])

for matrix_name in attention_matrices:
    layer_num = matrix_name.split('.')[3]
    matrix_type = matrix_name.split('.')[-2]

    ranks = [results_by_checkpoint[step][matrix_name]['rank_ratio'] * 100
             for step in sorted(results_by_checkpoint.keys())]

    trend = "↓" if ranks[-1] < ranks[0] - 2 else "↑" if ranks[-1] > ranks[0] + 2 else "→"

    print(f"L{layer_num} {matrix_type:5s}: ", end="")
    for r in ranks:
        print(f"{r:5.1f}%", end=" ")
    print(f" {trend}")

# Compute rank collapse metric
print("\n" + "="*80)
print("RANK COLLAPSE ANALYSIS")
print("="*80)

for step in sorted(results_by_checkpoint.keys()):
    attention_ranks = [data['rank_ratio']
                      for name, data in results_by_checkpoint[step].items()
                      if 'attention.self' in name and 'weight' in name]

    mean_rank = np.mean(attention_ranks) * 100
    std_rank = np.std(attention_ranks) * 100
    min_rank = np.min(attention_ranks) * 100

    print(f"Step {step:6d}: Mean={mean_rank:5.1f}% Std={std_rank:4.1f}% Min={min_rank:5.1f}%")

# Check if any layers are getting worse
print("\n" + "="*80)
print("DEGRADATION DETECTION")
print("="*80)

for matrix_name in attention_matrices:
    ranks = [results_by_checkpoint[step][matrix_name]['rank_ratio'] * 100
             for step in sorted(results_by_checkpoint.keys())]
    conditions = [results_by_checkpoint[step][matrix_name]['condition']
                 for step in sorted(results_by_checkpoint.keys())]

    rank_change = ranks[-1] - ranks[0]
    condition_change = conditions[-1] / conditions[0]

    if rank_change < -5 or condition_change > 2.0:
        print(f"\n⚠️  {matrix_name}")
        print(f"   Rank: {ranks[0]:.1f}% → {ranks[-1]:.1f}% (Δ={rank_change:+.1f}%)")
        print(f"   Condition: {conditions[0]:.0f} → {conditions[-1]:.0f} ({condition_change:.2f}x)")

# Embedding stability
print("\n" + "="*80)
print("EMBEDDING LAYER STABILITY")
print("="*80)

embed_name = 'roberta.embeddings.word_embeddings.weight'
print(f"{'Step':>8} {'EffRank':>8} {'Rank%':>7} {'Condition':>12}")
print("-" * 37)
for step in sorted(results_by_checkpoint.keys()):
    data = results_by_checkpoint[step][embed_name]
    print(f"{step:8d} {data['effective_rank']:8.2f} {data['rank_ratio']*100:6.1f}% {data['condition']:12.1f}")

# Save full comparison
output = {
    'checkpoints': list(sorted(results_by_checkpoint.keys())),
    'results': results_by_checkpoint
}
output_path = Path("checkpoint_comparison.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nFull results saved to {output_path}")
