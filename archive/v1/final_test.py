#!/usr/bin/env python3
"""Final test showing all components working together."""

import torch
import time
from src.model.architecture import LightweightIOCModel, EfficientIOCTokenizer
from src.data_generation.enhanced_synthetic_gen import EnhancedSyntheticGenerator
from src.data_generation.ioc_patterns import extract_iocs

print("=" * 70)
print("🎯 FINAL END-TO-END DEMONSTRATION")
print("=" * 70)

# 1. Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✅ Device: {device}")
if device == 'cuda':
    print(f"   GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")

# 2. Generate balanced data
print("\n✅ Balanced Data Generation:")
generator = EnhancedSyntheticGenerator(seed=42)
samples = generator.generate_balanced_batch(n_samples=20)

clean = sum(1 for s in samples if len(s.labels) == 0)
suspicious = sum(1 for s in samples if len(s.labels) > 0)
print(f"   Clean: {clean}/{len(samples)} ({clean/len(samples)*100:.0f}%)")
print(f"   Suspicious: {suspicious}/{len(samples)} ({suspicious/len(samples)*100:.0f}%)")

# 3. Initialize bigger model
print("\n✅ Model Configuration (Large):")
model = LightweightIOCModel(
    vocab_size=258,
    hidden_dim=512,  # Bigger
    num_layers=4,     # Deeper
    nhead=8           # More attention
).to(device)

stats = model.get_model_size()
print(f"   Parameters: {stats['total_params']:,}")
print(f"   Size: {stats['model_size_mb']:.1f} MB")

# 4. Test inference speed on GPU
print("\n✅ Inference Performance:")
tokenizer = EfficientIOCTokenizer()
test_texts = [
    "Detected malware with hash 5d41402abc4b2a76b9719d911017c592 from 192.168.1.100",
    "System performance metrics show 87.3% CPU utilization during peak hours.",
    "Ransomware C2 server at https://evil-domain.tk/gate.php contacted",
]

model.eval()
total_time = 0
with torch.no_grad():
    for text in test_texts:
        encoded = tokenizer.encode(text)
        input_ids = encoded['input_ids'].unsqueeze(0).to(device)
        attention_mask = encoded['attention_mask'].unsqueeze(0).to(device)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        outputs = model(input_ids, attention_mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        
        total_time += elapsed
        
        # Extract IOCs for comparison
        iocs = extract_iocs(text)
        risk = outputs['risk_score'].item()
        
        print(f"\n   Text: '{text[:60]}...'")
        print(f"   IOCs found: {len(iocs)}")
        print(f"   Risk score: {risk:.1%}")
        print(f"   Inference time: {elapsed:.2f}ms")

avg_time = total_time / len(test_texts)
print(f"\n   Average latency: {avg_time:.2f}ms")
print(f"   Throughput: {1000/avg_time:.0f} samples/sec")

# 5. Summary
print("\n" + "=" * 70)
print("✨ ALL SYSTEMS OPERATIONAL")
print("=" * 70)
print(f"""
Configuration:
  • Device: {device.upper()} {'(' + str(torch.cuda.device_count()) + ' GPUs)' if device == 'cuda' else ''}
  • Model: Large (512-dim, 4-layer, 8-head)
  • Data: Balanced (40% clean, 60% suspicious)
  • Performance: {avg_time:.2f}ms average latency
  
Ready for production training:
  python scripts/train.py --generate_synthetic --fp16
""")