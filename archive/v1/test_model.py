#!/usr/bin/env python3
"""Test model architecture and basic training."""

import torch
import torch.nn as nn
from src.model.architecture import LightweightIOCModel, EfficientIOCTokenizer
from src.data_generation.synthetic_gen import IOCSyntheticGenerator

print("=" * 60)
print("MODEL ARCHITECTURE TEST")
print("=" * 60)

# Initialize model
model = LightweightIOCModel(
    vocab_size=258,
    embed_dim=128,
    hidden_dim=256,
    num_ioc_types=18,
    nhead=4,
    num_layers=2
)

# Get model stats
stats = model.get_model_size()
print(f"\nModel Statistics:")
print(f"  Total Parameters: {stats['total_params']:,}")
print(f"  Trainable Parameters: {stats['trainable_params']:,}")
print(f"  Model Size: {stats['model_size_mb']:.2f} MB")

# Test tokenizer
print("\n" + "=" * 60)
print("TOKENIZER TEST")
print("=" * 60)
tokenizer = EfficientIOCTokenizer(max_length=128)

test_text = "Malicious IP 192.168.1.100 detected with hash 5d41402abc4b2a76b9719d911017c592"
encoded = tokenizer.encode(test_text)
print(f"\nOriginal text: {test_text}")
print(f"Text length: {len(test_text)}")
print(f"Encoded shape: {encoded['input_ids'].shape}")
print(f"First 20 tokens: {encoded['input_ids'][:20].tolist()}")

# Test forward pass
print("\n" + "=" * 60)
print("FORWARD PASS TEST")
print("=" * 60)

# Create batch
batch_texts = [
    "IP address 192.168.1.1 is suspicious",
    "Hash e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 found",
    "CVE-2021-44228 vulnerability detected"
]

batch_encoded = tokenizer.batch_encode(batch_texts)
print(f"\nBatch size: {len(batch_texts)}")
print(f"Input shape: {batch_encoded['input_ids'].shape}")

# Forward pass
model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=batch_encoded['input_ids'],
        attention_mask=batch_encoded['attention_mask']
    )

print(f"\nOutput shapes:")
print(f"  NER logits: {outputs['ner_logits'].shape}")
print(f"  IOC logits: {outputs['ioc_logits'].shape}")
print(f"  Risk scores: {outputs['risk_score'].shape}")

# Test mini training
print("\n" + "=" * 60)
print("MINI TRAINING TEST")
print("=" * 60)

# Generate small dataset
generator = IOCSyntheticGenerator(seed=42)
samples = generator.generate_batch(n_samples=10)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train for a few steps
model.train()
total_loss = 0.0

for i, sample in enumerate(samples[:5]):
    # Encode sample
    encoded = tokenizer.encode(sample.text)
    input_ids = encoded['input_ids'].unsqueeze(0)
    attention_mask = encoded['attention_mask'].unsqueeze(0)
    
    # Create dummy labels (for testing)
    labels = torch.zeros(1, tokenizer.max_length, dtype=torch.long)
    
    # Forward pass
    outputs = model(input_ids, attention_mask)
    
    # Calculate loss (simplified)
    loss = criterion(
        outputs['ner_logits'].view(-1, 18),
        labels.view(-1)
    )
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    print(f"  Step {i+1}: Loss = {loss.item():.4f}")

print(f"\nAverage Loss: {total_loss/5:.4f}")
print("\n✅ Model training test successful!")

# Test inference speed
print("\n" + "=" * 60)
print("INFERENCE SPEED TEST")
print("=" * 60)

import time

model.eval()
num_samples = 100
texts = [f"Test sample {i} with IP 192.168.1.{i}" for i in range(num_samples)]

start_time = time.time()
with torch.no_grad():
    for text in texts:
        encoded = tokenizer.encode(text)
        input_ids = encoded['input_ids'].unsqueeze(0)
        attention_mask = encoded['attention_mask'].unsqueeze(0)
        outputs = model(input_ids, attention_mask)

end_time = time.time()
total_time = end_time - start_time
avg_time = (total_time / num_samples) * 1000  # ms

print(f"\nProcessed {num_samples} samples in {total_time:.2f} seconds")
print(f"Average inference time: {avg_time:.2f} ms per sample")
print(f"Throughput: {num_samples/total_time:.1f} samples/sec")