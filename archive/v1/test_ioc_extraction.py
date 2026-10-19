#!/usr/bin/env python3
"""Test IOC pattern extraction."""

from src.data_generation.ioc_patterns import extract_iocs, calculate_risk_score

# Test samples
test_samples = [
    "Malicious activity from IP 192.168.1.100 detected",
    "File hash: 5d41402abc4b2a76b9719d911017c592 found in system",
    "Phishing attempt from evil@malicious-domain.tk",
    "CVE-2021-44228 vulnerability exploited via https://evil.com/payload",
    "Connection to C2 server at 185.220.101.45:8080",
    "Ransomware deployed SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
]

print("=" * 60)
print("IOC PATTERN EXTRACTION TEST")
print("=" * 60)

for text in test_samples:
    print(f"\nAnalyzing: {text}")
    print("-" * 40)
    
    iocs = extract_iocs(text)
    
    if iocs:
        print("Detected IOCs:")
        for ioc_type, matched_text, start, end in iocs:
            print(f"  [{ioc_type.value:15}] '{matched_text}' @ position {start}-{end}")
        
        risk_score = calculate_risk_score(iocs)
        print(f"Risk Score: {risk_score:.2f}")
    else:
        print("  No IOCs detected")