#!/usr/bin/env python3
"""
Simple script to test IOC detection on individual strings.

Usage:
    python scripts/test_string.py "test string here"
    python scripts/test_string.py "192.168.1.1"
    python scripts/test_string.py "evil.malware.com"
    echo "test" | python scripts/test_string.py
"""

import sys
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.architecture import LightweightIOCModel

# IOC types mapping
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


def load_model(checkpoint_path: str = "models/best_model.pth"):
    """Load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Match the trained model configuration
    model = LightweightIOCModel(
        vocab_size=258,  # Extended vocab
        hidden_dim=512,  # Scaled up
        num_layers=4,    # More layers
        nhead=8          # More heads
    )
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded model from {checkpoint_path}")
    else:
        print(f"⚠ No checkpoint found at {checkpoint_path}, using untrained model")
    
    model.to(device)
    model.eval()
    return model, device


def test_string(text: str, model, device, verbose: bool = True, debug: bool = False):
    """Test a single string for IOCs."""
    # Tokenize using the same scheme as training
    # 0-255: regular ASCII, 256: PAD, 257: UNK
    tokens = []
    for c in text[:512]:
        if ord(c) < 256:
            tokens.append(ord(c))
        else:
            tokens.append(257)  # UNK token
    
    # Pad with 256 (PAD token), not 0!
    text_length = len(tokens)
    tokens = tokens + [256] * (512 - len(tokens))  # PAD = 256
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = [1] * text_length + [0] * (512 - text_length)
    
    # Convert to tensors
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        ner_logits = outputs['ner_logits']
        risk_score = torch.sigmoid(outputs['risk_score']).item()
        
        if debug:
            print(f"\nDebug Info:")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Text length: {text_length}")
            print(f"  NER logits shape: {ner_logits.shape}")
            print(f"  Risk logit: {outputs['risk_score'].item():.4f}")
            print(f"  Risk score (sigmoid): {risk_score:.4f}")
            # Show predictions for first few characters
            print(f"  First 10 char predictions: {torch.argmax(ner_logits, dim=-1).squeeze()[:10].cpu().tolist()}")
            print(f"  Unique predictions in text: {set(torch.argmax(ner_logits, dim=-1).squeeze()[:text_length].cpu().tolist())}")
        
        # Get predictions
        ner_preds = torch.argmax(ner_logits, dim=-1).squeeze().cpu().numpy()
    
    # Map predictions to IOC types
    ioc_types = ['O'] + list(IOC_PATTERNS.keys())
    detected_iocs = []
    current_ioc = None
    current_start = None
    
    # Only look at predictions for actual text, not padding
    for i, pred in enumerate(ner_preds[:text_length]):
        if pred > 0:  # Not 'O'
            ioc_type = ioc_types[pred] if pred < len(ioc_types) else f"Unknown_{pred}"
            if current_ioc != ioc_type:
                if current_ioc:
                    detected_iocs.append((current_ioc, current_start, i))
                current_ioc = ioc_type
                current_start = i
        else:
            if current_ioc:
                detected_iocs.append((current_ioc, current_start, i))
                current_ioc = None
    
    # Don't forget the last IOC if it extends to the end
    if current_ioc:
        detected_iocs.append((current_ioc, current_start, text_length))
    
    # Results
    results = {
        'text': text,
        'risk_score': risk_score,
        'detected_iocs': detected_iocs,
        'ioc_strings': [(ioc_type, text[start:end]) for ioc_type, start, end in detected_iocs]
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Input: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"{'='*60}")
        print(f"Risk Score: {risk_score:.3f} {'⚠️ HIGH' if risk_score > 0.5 else '✓ LOW'}")
        
        if detected_iocs:
            print(f"\nDetected IOCs:")
            for ioc_type, ioc_text in results['ioc_strings']:
                print(f"  • {ioc_type}: {ioc_text}")
        else:
            print("\nNo IOCs detected")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test IOC detection on strings")
    parser.add_argument("text", nargs="?", help="Text to test (or pipe from stdin)")
    parser.add_argument("--model", default="models/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--batch", action="store_true", help="Process multiple lines from stdin")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model)
    
    # Get input text
    if args.text:
        # From command line
        test_string(args.text, model, device, debug=args.debug)
    else:
        # From stdin
        if args.batch:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    test_string(line, model, device, debug=args.debug)
                    print()
        else:
            text = sys.stdin.read().strip()
            if text:
                test_string(text, model, device, debug=args.debug)
            else:
                print("Usage: python scripts/test_string.py \"text to test\"")
                print("   or: echo \"text\" | python scripts/test_string.py")


if __name__ == "__main__":
    main()