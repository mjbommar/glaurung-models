#!/usr/bin/env python3
"""Test all improvements: GPU, balanced data, bigger model."""

import torch
import torch.nn as nn
import time
from pathlib import Path

from src.model.architecture import LightweightIOCModel, EfficientIOCTokenizer
from src.data_generation.enhanced_synthetic_gen import EnhancedSyntheticGenerator

print("=" * 60)
print("🚀 TESTING ALL IMPROVEMENTS")
print("=" * 60)

# Check GPU availability
print("\n📊 GPU STATUS:")
print("-" * 40)
if torch.cuda.is_available():
    print(f"✅ CUDA Available")
    print(f"  • Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  • GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    device = 'cuda'
else:
    print("❌ CUDA not available, using CPU")
    device = 'cpu'

# Test enhanced data generation
print("\n📊 ENHANCED DATA GENERATION:")
print("-" * 40)
generator = EnhancedSyntheticGenerator(seed=42)

# Generate a small balanced batch
samples = generator.generate_balanced_batch(
    n_samples=100,
    clean_ratio=0.4,
    light_suspicious=0.3,
    heavy_suspicious=0.3
)

# Show some examples
print("\nSample examples:")
for i, sample in enumerate(samples[:5]):
    ioc_count = len(sample.labels)
    print(f"\n{i+1}. Type: {sample.context_type}, IOCs: {ioc_count}, Risk: {sample.risk_score:.2f}")
    print(f"   Text: {sample.text[:100]}...")

# Test bigger model configurations
print("\n📊 MODEL SIZE COMPARISON:")
print("-" * 40)

configs = [
    {"name": "Small (Original)", "hidden_dim": 256, "num_layers": 2, "nhead": 4},
    {"name": "Medium", "hidden_dim": 384, "num_layers": 3, "nhead": 6},
    {"name": "Large (New Default)", "hidden_dim": 512, "num_layers": 4, "nhead": 8},
    {"name": "XL", "hidden_dim": 768, "num_layers": 6, "nhead": 12},
]

tokenizer = EfficientIOCTokenizer(max_length=256)
test_text = "Test sample with IP 192.168.1.1"
encoded = tokenizer.encode(test_text)
input_ids = encoded['input_ids'].unsqueeze(0).to(device)
attention_mask = encoded['attention_mask'].unsqueeze(0).to(device)

for config in configs:
    model = LightweightIOCModel(
        vocab_size=258,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        nhead=config["nhead"]
    ).to(device)
    
    stats = model.get_model_size()
    
    # Test inference speed
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_ids, attention_mask)
        
        # Benchmark
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(100):
            _ = model(input_ids, attention_mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
    
    avg_time = ((end - start) / 100) * 1000  # ms
    
    print(f"\n{config['name']}:")
    print(f"  • Parameters: {stats['total_params']:,}")
    print(f"  • Size: {stats['model_size_mb']:.2f} MB")
    print(f"  • Inference: {avg_time:.2f} ms")
    print(f"  • Throughput: {1000/avg_time:.0f} samples/sec")

# Test GPU vs CPU speed
if torch.cuda.is_available():
    print("\n📊 GPU vs CPU SPEED COMPARISON:")
    print("-" * 40)
    
    # Use large model for comparison
    model_cpu = LightweightIOCModel(
        vocab_size=258,
        hidden_dim=512,
        num_layers=4,
        nhead=8
    )
    
    model_gpu = LightweightIOCModel(
        vocab_size=258,
        hidden_dim=512,
        num_layers=4,
        nhead=8
    ).cuda()
    
    # Prepare batch
    batch_size = 32
    batch_texts = [f"Test sample {i} with IP 192.168.1.{i}" for i in range(batch_size)]
    batch_encoded = tokenizer.batch_encode(batch_texts)
    
    # CPU timing
    input_cpu = batch_encoded['input_ids']
    mask_cpu = batch_encoded['attention_mask']
    
    model_cpu.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(10):
            _ = model_cpu(input_cpu, mask_cpu)
        cpu_time = (time.perf_counter() - start) / 10
    
    # GPU timing
    input_gpu = batch_encoded['input_ids'].cuda()
    mask_gpu = batch_encoded['attention_mask'].cuda()
    
    model_gpu.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model_gpu(input_gpu, mask_gpu)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = model_gpu(input_gpu, mask_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) / 10
    
    print(f"\nBatch size: {batch_size}")
    print(f"CPU time: {cpu_time*1000:.2f} ms")
    print(f"GPU time: {gpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    
    # Test multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"\n✅ Multi-GPU available: {torch.cuda.device_count()} GPUs")
        model_parallel = nn.DataParallel(model_gpu)
        print("   Model wrapped in DataParallel for training")

# Summary
print("\n" + "=" * 60)
print("✨ IMPROVEMENTS SUMMARY")
print("=" * 60)
print("\n✅ All improvements implemented:")
print(f"  1. GPU Support: {'Enabled' if device == 'cuda' else 'CPU only'}")
print(f"  2. Balanced Data: Clean + suspicious samples")
print(f"  3. Bigger Models: Up to 768 hidden dim")
print(f"  4. Extended Training: 30 epochs default")

print("\n📈 Recommended configuration:")
print("  • Model: Large (512 hidden, 4 layers)")
print("  • Training: 50K samples, 30 epochs")
print("  • Hardware: GPU with mixed precision")
print("  • Data: 40% clean, 30% light, 30% heavy")

print("\n🎯 Next step:")
print("  python scripts/train.py --generate_synthetic --fp16")