#!/usr/bin/env python3
"""Test script for the improved model."""

import sys
import torch
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.model.improved_architecture import ImprovedIOCModel

IOC_PATTERNS = {
    'ip_address': 'IP Address',
    'domain': 'Domain',
    'url': 'URL', 
    'email': 'Email',
    'hash_md5': 'MD5 Hash',
    'hash_sha1': 'SHA1 Hash',
    'hash_sha256': 'SHA256 Hash',
    'cve': 'CVE',
    'file_path': 'File Path',
    'registry_key': 'Registry Key',
    'bitcoin_address': 'Bitcoin Address',
    'user_agent': 'User Agent',
    'mutex': 'Mutex',
    'yara_rule': 'YARA Rule',
    'subnet': 'Subnet',
    'asn': 'ASN',
    'ssl_cert': 'SSL Certificate',
    'jwt_token': 'JWT Token'
}

def test_string(text: str, model, device):
    """Test a single string."""
    # Tokenize
    tokens = []
    for c in text[:512]:
        if ord(c) < 256:
            tokens.append(ord(c))
        else:
            tokens.append(257)  # UNK
    
    text_length = len(tokens)
    tokens = tokens + [256] * (512 - len(tokens))  # PAD = 256
    attention_mask = [1] * text_length + [0] * (512 - text_length)
    
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        risk_score = torch.sigmoid(outputs['risk_score']).item()
        
    print(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"Risk Score: {risk_score:.3f} {'⚠️' if risk_score > 0.5 else '✓'}")
    print()
    
    return risk_score

def main():
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ImprovedIOCModel(
        vocab_size=258,
        embed_dim=256,
        hidden_dim=768,
        num_ioc_types=18,
        nhead=12,
        num_layers=8,
        dropout=0.1
    )
    
    checkpoint_path = "models/best_improved_model.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded improved model\n")
    
    model.to(device)
    model.eval()
    
    # Test various strings
    print("="*60)
    print("TESTING BENIGN STRINGS")
    print("="*60)
    
    benign_texts = [
        "The weather today is sunny and beautiful",
        "Hello world, this is a test message", 
        "Meeting scheduled for 3pm tomorrow",
        "The quick brown fox jumps over the lazy dog",
        "Software version 1.2.3 installed successfully"
    ]
    
    benign_scores = []
    for text in benign_texts:
        score = test_string(text, model, device)
        benign_scores.append(score)
    
    print("="*60)
    print("TESTING MALICIOUS INDICATORS")
    print("="*60)
    
    malicious_texts = [
        "098f6bcd4621d373cade4e832627b4f6",  # MD5 from training
        "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f",  # SHA256
        "192.168.1.1",
        "evil.malware.com",
        "http://badsite.tk/payload.exe",
        "Detected malware hash: 44d88612fea8a8f36de82e1278abb02f"
    ]
    
    malicious_scores = []
    for text in malicious_texts:
        score = test_string(text, model, device)
        malicious_scores.append(score)
    
    print("="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Benign average: {sum(benign_scores)/len(benign_scores):.3f}")
    print(f"Benign range: {min(benign_scores):.3f} - {max(benign_scores):.3f}")
    print(f"Malicious average: {sum(malicious_scores)/len(malicious_scores):.3f}")
    print(f"Malicious range: {min(malicious_scores):.3f} - {max(malicious_scores):.3f}")
    print(f"Score separation: {abs(sum(malicious_scores)/len(malicious_scores) - sum(benign_scores)/len(benign_scores)):.3f}")

if __name__ == "__main__":
    main()