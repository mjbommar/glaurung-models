#!/usr/bin/env python3
"""Simple inference script for IOC detection."""

import sys
from pathlib import Path
import argparse
import json
import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.architecture import LightweightIOCModel, EfficientIOCTokenizer
from src.data_generation.ioc_patterns import IOCType, extract_iocs


def load_model(checkpoint_path: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = LightweightIOCModel(
        vocab_size=checkpoint['args'].vocab_size if 'args' in checkpoint else 258,
        hidden_dim=checkpoint['args'].hidden_dim if 'args' in checkpoint else 256,
        num_ioc_types=checkpoint['args'].num_ioc_types if 'args' in checkpoint else 18,
        nhead=checkpoint['args'].num_heads if 'args' in checkpoint else 4,
        num_layers=checkpoint['args'].num_layers if 'args' in checkpoint else 2
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def detect_iocs(text: str, model: LightweightIOCModel, tokenizer: EfficientIOCTokenizer):
    """Detect IOCs in text."""
    # Tokenize
    encoded = tokenizer.encode(text)
    
    # Add batch dimension
    input_ids = encoded['input_ids'].unsqueeze(0)
    attention_mask = encoded['attention_mask'].unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    # Get predictions
    ner_predictions = torch.argmax(outputs['ner_logits'], dim=-1).squeeze().tolist()
    risk_score = outputs['risk_score'].item()
    
    # Extract IOCs using regex patterns (for comparison)
    regex_iocs = extract_iocs(text)
    
    # Map predictions to IOC types
    detected_iocs = []
    current_ioc = None
    current_start = None
    
    for i, pred in enumerate(ner_predictions):
        if pred > 0 and pred < len(IOCType):  # Valid IOC type
            if current_ioc is None:
                current_ioc = IOCType(pred)
                current_start = i
            elif IOCType(pred) != current_ioc:
                # End of previous IOC, start of new one
                detected_iocs.append({
                    'type': current_ioc.value,
                    'start': current_start,
                    'end': i,
                    'text': text[current_start:i]
                })
                current_ioc = IOCType(pred)
                current_start = i
        elif current_ioc is not None:
            # End of IOC
            detected_iocs.append({
                'type': current_ioc.value,
                'start': current_start,
                'end': i,
                'text': text[current_start:i]
            })
            current_ioc = None
            current_start = None
    
    # Handle last IOC if exists
    if current_ioc is not None:
        detected_iocs.append({
            'type': current_ioc.value,
            'start': current_start,
            'end': len(text),
            'text': text[current_start:]
        })
    
    return {
        'text': text,
        'risk_score': risk_score,
        'model_detected_iocs': detected_iocs,
        'regex_detected_iocs': [
            {
                'type': ioc_type.value,
                'text': matched_text,
                'start': start,
                'end': end
            }
            for ioc_type, matched_text, start, end in regex_iocs
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Run IOC detection inference')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='File containing text to analyze')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample texts')
    
    args = parser.parse_args()
    
    if args.demo:
        # Demo mode with sample texts
        demo_texts = [
            "Suspicious activity detected from IP 185.220.101.45 on port 445",
            "Malware hash found: 5d41402abc4b2a76b9719d911017c592",
            "Phishing email from hacker@evil-domain.tk containing https://malicious-site.com/payload.exe",
            "CVE-2021-44228 vulnerability exploited to download backdoor from 192.168.1.100",
            "Normal system update completed at 10:30 AM",
            "Registry key HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\EvilMalware created",
            "Command and control server contacted at http://45.142.213.99:8080/gate.php",
            "Bitcoin wallet address for ransom: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        ]
        
        print("=" * 60)
        print("IOC Detection Demo")
        print("=" * 60)
        
        # Create dummy model for demo (since we don't have trained model yet)
        print("\nNote: Using untrained model for demonstration")
        model = LightweightIOCModel()
        tokenizer = EfficientIOCTokenizer()
        
        for text in demo_texts:
            print(f"\nAnalyzing: {text}")
            print("-" * 40)
            
            # Use regex extraction for demo
            iocs = extract_iocs(text)
            
            if iocs:
                print("Detected IOCs:")
                for ioc_type, matched_text, start, end in iocs:
                    print(f"  - {ioc_type.value}: {matched_text} (pos {start}-{end})")
            else:
                print("  No IOCs detected")
                
    else:
        # Load model if provided
        if not args.model_path:
            print("Error: --model_path required (or use --demo for demonstration)")
            return
            
        model = load_model(args.model_path)
        tokenizer = EfficientIOCTokenizer()
        
        # Get text to analyze
        if args.text:
            text = args.text
        elif args.file:
            with open(args.file, 'r') as f:
                text = f.read()
        else:
            print("Error: Provide --text or --file to analyze")
            return
        
        # Detect IOCs
        results = detect_iocs(text, model, tokenizer)
        
        # Print results
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()