#!/usr/bin/env python3
"""Test torch.compile optimization."""

import torch
import time
from src.model.architecture import LightweightIOCModel, EfficientIOCTokenizer

print("=" * 60)
print("🚀 TORCH.COMPILE OPTIMIZATION TEST")
print("=" * 60)

# Check PyTorch version
print(f"\n📊 PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   Compile available: {hasattr(torch, 'compile')}")

if not hasattr(torch, 'compile'):
    print("\n⚠️  torch.compile not available (requires PyTorch 2.0+)")
    print("   Install with: pip install torch>=2.0")
    exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize model
model = LightweightIOCModel(
    vocab_size=258,
    hidden_dim=512,
    num_layers=4,
    nhead=8
).to(device)

tokenizer = EfficientIOCTokenizer(max_length=256)

# Prepare test batch
batch_size = 32
test_texts = [f"Test IOC detection sample {i} with IP 192.168.1.{i}" for i in range(batch_size)]
batch = tokenizer.batch_encode(test_texts)
input_ids = batch['input_ids'].to(device)
attention_mask = batch['attention_mask'].to(device)

# Function to benchmark
def benchmark_model(model, name, num_runs=100):
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids, attention_mask)
    
    # Benchmark
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_ids, attention_mask)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    avg_time = ((end - start) / num_runs) * 1000  # ms
    throughput = (batch_size * num_runs) / (end - start)
    
    print(f"\n{name}:")
    print(f"  Average latency: {avg_time:.2f} ms")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    
    return avg_time

print("\n" + "=" * 60)
print("BENCHMARKING")
print("=" * 60)

# Test 1: Regular model
print("\n1. REGULAR MODEL (No Compilation)")
regular_time = benchmark_model(model, "Regular Model")

# Test 2: Compiled model - default mode
print("\n2. COMPILED MODEL (default mode)")
compiled_model_default = torch.compile(model, mode='default')
compiled_default_time = benchmark_model(compiled_model_default, "Compiled (default)")

# Test 3: Compiled model - reduce-overhead mode
print("\n3. COMPILED MODEL (reduce-overhead mode)")
compiled_model_overhead = torch.compile(model, mode='reduce-overhead')
compiled_overhead_time = benchmark_model(compiled_model_overhead, "Compiled (reduce-overhead)")

# Test 4: Compiled model - max-autotune mode (best performance, slower compilation)
if device == 'cuda':
    print("\n4. COMPILED MODEL (max-autotune mode)")
    print("   ⏳ This may take a minute to compile...")
    compiled_model_autotune = torch.compile(model, mode='max-autotune')
    compiled_autotune_time = benchmark_model(compiled_model_autotune, "Compiled (max-autotune)")
else:
    compiled_autotune_time = float('inf')

# Summary
print("\n" + "=" * 60)
print("📊 RESULTS SUMMARY")
print("=" * 60)

print(f"\nSpeedup vs Regular Model:")
print(f"  Default mode: {regular_time/compiled_default_time:.2f}x")
print(f"  Reduce-overhead: {regular_time/compiled_overhead_time:.2f}x")
if device == 'cuda':
    print(f"  Max-autotune: {regular_time/compiled_autotune_time:.2f}x")

best_mode = 'default'
best_time = compiled_default_time

if compiled_overhead_time < best_time:
    best_mode = 'reduce-overhead'
    best_time = compiled_overhead_time

if device == 'cuda' and compiled_autotune_time < best_time:
    best_mode = 'max-autotune'
    best_time = compiled_autotune_time

print(f"\n✨ Best mode: {best_mode}")
print(f"   Speedup: {regular_time/best_time:.2f}x")
print(f"   Latency: {best_time:.2f}ms")

print("\n📝 Recommendation:")
print(f"   Use --compile --compile_mode={best_mode} for training")
print("\n   Example:")
print(f"   python scripts/train.py --generate_synthetic --compile --compile_mode={best_mode}")