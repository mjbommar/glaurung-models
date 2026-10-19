# Binary Embedding Architecture Redesign

## Problem Analysis

From training experiments at 230k steps, we discovered:

### Layer 3 Rank Collapse
- **Effective rank dropped from 49.4% → 41.6%** over 40k steps (steps 190k-230k)
- Condition numbers exploded: 4.9k → 24.9k (query), 1.6k → 8.2k (value)
- Pattern: Degradation scales with depth (Layer 0 stable, Layer 3 collapsing)

### Root Cause: Hidden Size Too Small for Depth
- Current NANO: 128 hidden / 4 layers = **32 dims per layer**
- Gradients shrink exponentially through backprop
- Layer 3 gets insufficient gradient signal → rank collapse

### Key Insight: Width vs Depth Tradeoff
- **Loss still decreasing smoothly** → model compensates via Layers 0-2
- **Embeddings healthy (96.4% rank)** → foundation solid
- **MLP layers healthy (89-93% rank)** → FFN working well
- Problem is ATTENTION depth, not overall architecture

## Design Principles

Based on empirical findings + BERT architecture patterns:

### 1. Minimum Hidden Dims Per Layer
**Rule:** `hidden_size / num_layers ≥ 48`

Evidence:
- BERT-tiny: 128/2 = 64 ✓
- BERT-mini: 256/4 = 64 ✓
- Our NANO: 128/4 = 32 ✗ (FAILED)

### 2. Attention Head Dimensions
- **32 dims/head**: Minimum viable for local patterns (small models)
- **64 dims/head**: Better for complex patterns (medium+ models)
- Never go below 32 (loses too much info per head)

### 3. FFN Intermediate Size
- **4× hidden_size** (standard transformer ratio)
- Our healthy MLP layers (89-93% rank) confirm this works

### 4. Embeddings Dominate Small Models
- With vocab=65,536, embeddings are ~60-90% of params for models <50M
- Accept this, focus on encoder efficiency

### 5. Binary-Specific Considerations
- **Local patterns dominate**: File headers, opcodes (≤64 bytes)
- **Don't need deep semantic reasoning**: Unlike NLP, binary is more structural
- **Wider > Deeper** for binary data
- Suggests: Prefer 2-6 layers with good width over 12+ layers

## New Architecture Sizes

### MICRO (~6.6M params)
```python
hidden_size=96
num_hidden_layers=2
num_attention_heads=3       # 32 dims/head
intermediate_size=384       # 4× hidden
```
**Ratio:** 96/2 = 48 dims/layer ✓

**Params:**
- Embeddings: 6.3M (96%)
- Encoder: 280K (4%)
- **Total: 6.6M**

**Use case:** Ultra-fast training, embedded systems, baseline

---

### NANO (~11.5M params) ← **FIXED VERSION**
```python
hidden_size=160
num_hidden_layers=3         # REDUCED from 4
num_attention_heads=5       # 32 dims/head
intermediate_size=640       # 4× hidden
```
**Ratio:** 160/3 = 53 dims/layer ✓ (was 32, now fixed)

**Params:**
- Embeddings: 10.5M (91%)
- Encoder: 1.0M (9%)
- **Total: 11.5M**

**Changes from current:**
- Layers: 4 → 3 (fix collapse)
- Hidden: 128 → 160 (more capacity per layer)
- Heads: 4 → 5 (maintain 32 dims/head)

---

### SMALL (~20M params)
```python
hidden_size=256
num_hidden_layers=4
num_attention_heads=8       # 32 dims/head
intermediate_size=1024      # 4× hidden
```
**Ratio:** 256/4 = 64 dims/layer ✓

**Params:**
- Embeddings: 16.8M (84%)
- Encoder: 3.2M (16%)
- **Total: 20M**

**Use case:** Good balance of speed and performance

---

### MEDIUM (~40M params)
```python
hidden_size=384
num_hidden_layers=6
num_attention_heads=12      # 32 dims/head
intermediate_size=1536      # 4× hidden
```
**Ratio:** 384/6 = 64 dims/layer ✓

**Params:**
- Embeddings: 25.2M (63%)
- Encoder: 15M (37%)
- **Total: 40M**

**Use case:** Production models, good accuracy

---

### BASE (~78M params)
```python
hidden_size=512
num_hidden_layers=8
num_attention_heads=16      # 32 dims/head
intermediate_size=2048      # 4× hidden
```
**Ratio:** 512/8 = 64 dims/layer ✓

**Params:**
- Embeddings: 33.5M (43%)
- Encoder: 45M (57%)
- **Total: 78M**

**Use case:** High-quality embeddings, slower training

---

### LARGE (~160M params)
```python
hidden_size=768
num_hidden_layers=12
num_attention_heads=12      # 64 dims/head
intermediate_size=3072      # 4× hidden
```
**Ratio:** 768/12 = 64 dims/layer ✓

**Params:**
- Embeddings: 50M (31%)
- Encoder: 110M (69%)
- **Total: 160M**

**Note:** This is BERT-base architecture, proven for transformers

---

## Size Progression

| Model | Params | Layers | Hidden | Ratio | Heads | Head Dim |
|-------|--------|--------|--------|-------|-------|----------|
| MICRO | 6.6M   | 2      | 96     | 48    | 3     | 32       |
| NANO  | 11.5M  | 3      | 160    | 53    | 5     | 32       |
| SMALL | 20M    | 4      | 256    | 64    | 8     | 32       |
| MEDIUM| 40M    | 6      | 384    | 64    | 12    | 32       |
| BASE  | 78M    | 8      | 512    | 64    | 16    | 32       |
| LARGE | 160M   | 12     | 768    | 64    | 12    | 64       |

**Scaling pattern:** Each size ~2× larger in params

## Expected Improvements

### NANO (3 layers vs current 4 layers)
- **No rank collapse** in final layer
- More stable condition numbers
- Better gradient flow
- Slightly fewer params (9.3M → 11.5M, but more stable)

### All Sizes
- **Healthy rank utilization** across all layers (>70% effective rank)
- **Stable condition numbers** (< 1000 for attention matrices)
- **Better training dynamics** (no degradation over time)
- **Optimal width/depth tradeoff** for binary data

## Migration Strategy

1. **Immediate:** Update NANO to 3L/160H for ongoing experiments
2. **Short-term:** Train MICRO and SMALL to establish new baselines
3. **Long-term:** Deprecated old MICRO config, consolidate sizes

## Performance Predictions

Based on stable architecture + proven hyperparameters:

- **NANO (new):** Should exceed 35.85% (exp06 peak) and maintain stability
- **SMALL:** Should reach 40-45% with longer training
- **MEDIUM+:** Likely 45-50%+ given sufficient data and compute

Key: No more rank collapse → sustained improvement throughout training
