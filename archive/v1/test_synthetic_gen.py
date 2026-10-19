#!/usr/bin/env python3
"""Test synthetic data generation."""

from src.data_generation.synthetic_gen import IOCSyntheticGenerator
import json

# Initialize generator
generator = IOCSyntheticGenerator(seed=42)

print("=" * 60)
print("SYNTHETIC DATA GENERATION TEST")
print("=" * 60)

# Generate different types of samples
print("\n1. MALICIOUS SAMPLE WITH IOCs:")
print("-" * 40)
sample1 = generator.generate_context_with_iocs(n_iocs=3, malicious_ratio=1.0)
print(f"Text: {sample1.text}")
print(f"Risk Score: {sample1.risk_score}")
print(f"Labels: {sample1.labels}")

print("\n2. BENIGN SAMPLE:")
print("-" * 40)
sample2 = generator.generate_benign_text()
print(f"Text: {sample2.text}")
print(f"Risk Score: {sample2.risk_score}")

print("\n3. MIXED SAMPLE:")
print("-" * 40)
sample3 = generator.generate_context_with_iocs(n_iocs=2, malicious_ratio=0.5)
print(f"Text: {sample3.text}")
print(f"Risk Score: {sample3.risk_score}")

# Generate batch and save
print("\n4. GENERATING BATCH:")
print("-" * 40)
samples = generator.generate_batch(n_samples=100, malicious_ratio=0.3, benign_ratio=0.3)
print(f"Generated {len(samples)} samples")
print(f"  - Malicious: {sum(1 for s in samples if s.context_type == 'malicious')}")
print(f"  - Benign: {sum(1 for s in samples if s.context_type == 'benign')}")
print(f"  - Mixed: {sum(1 for s in samples if s.context_type == 'mixed')}")

# Save to file
generator.export_to_jsonl(samples[:10], "data/test_samples.jsonl")
print("\nSaved 10 samples to data/test_samples.jsonl")

# Show sample distribution
print("\n5. SAMPLE IOC GENERATION:")
print("-" * 40)
print(f"IPv4 (malicious): {generator.generate_ipv4(malicious=True)}")
print(f"IPv4 (benign): {generator.generate_ipv4(malicious=False)}")
print(f"Domain (malicious): {generator.generate_domain(malicious=True)}")
print(f"Domain (benign): {generator.generate_domain(malicious=False)}")
print(f"URL (malicious): {generator.generate_url(malicious=True)}")
print(f"Hash (MD5): {generator.generate_hash('md5')}")
print(f"CVE: {generator.generate_cve()}")
print(f"File path (Windows): {generator.generate_filepath(windows=True, malicious=True)}")
print(f"Registry key: {generator.generate_registry_key(malicious=True)}")