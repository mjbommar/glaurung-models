#!/usr/bin/env python3
"""
Evaluate NER classification performance - precision, recall, F1 per IOC type.
This directly addresses the question: "how is the CLASSIFICATION accuracy / precision / recall?"
"""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.model.improved_architecture import ImprovedIOCModel
from src.data_generation.ultra_hard_generator import UltraHardDataGenerator
from src.model.architecture import EfficientIOCTokenizer

# IOC types mapping - must match training
IOC_TYPES = [
    'O',  # Non-IOC
    'ip_address',
    'domain', 
    'url',
    'email',
    'hash_md5',
    'hash_sha1',
    'hash_sha256',
    'cve',
    'file_path',
    'registry_key',
    'bitcoin_address',
    'user_agent',
    'mutex',
    'yara_rule',
    'subnet',
    'asn',
    'ssl_cert'
]

def evaluate_model(model_path: str, num_samples: int = 1000):
    """Evaluate NER classification accuracy."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = ImprovedIOCModel(
        vocab_size=258,
        embed_dim=256,
        hidden_dim=768,
        num_ioc_types=18,
        nhead=12,
        num_layers=8,
        dropout=0.1
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Generate test data
    print(f"Generating {num_samples} test samples...")
    generator = UltraHardDataGenerator()
    test_samples = generator.generate_ultra_hard_batch(
        n_samples=num_samples,
        clean_ratio=0.25,
        easy_ratio=0.25,
        hard_ratio=0.50
    )
    
    # Tokenizer
    tokenizer = EfficientIOCTokenizer(max_length=512)
    
    # Track predictions vs ground truth
    all_true_labels = []
    all_pred_labels = []
    per_type_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    
    print("\n" + "="*80)
    print("EVALUATING NER CLASSIFICATION PERFORMANCE")
    print("="*80)
    
    # Process each sample
    for i, sample in enumerate(test_samples):
        if i % 100 == 0:
            print(f"Processing sample {i}/{num_samples}...")
        
        # Tokenize text
        tokens = tokenizer.encode(sample.text)
        # Count actual text length (non-padding tokens)
        text_length = (tokens['attention_mask'] == 1).sum().item()
        
        # Get model predictions
        with torch.no_grad():
            input_ids = tokens['input_ids'].unsqueeze(0).to(device)
            attention_mask = tokens['attention_mask'].unsqueeze(0).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Get NER predictions (shape: [batch, seq_len, num_classes])
            ner_logits = outputs['ner_logits']
            ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy()[0][:text_length]
        
        # Create ground truth labels for each token
        true_labels = np.zeros(text_length, dtype=int)  # Default to 'O' (0)
        
        # Map IOC positions to token labels  
        for start, end, ioc_type in sample.labels:
            # Find which tokens correspond to this IOC span
            char_to_token = []
            current_char = 0
            for j, token_id in enumerate(tokens['input_ids'].tolist()[:text_length]):
                if token_id < 256:  # Valid character
                    char_to_token.append(j)
                    current_char += 1
            
            # Map character positions to token positions
            if start < len(char_to_token) and end <= len(char_to_token):
                token_start = char_to_token[start] if start < len(char_to_token) else 0
                token_end = char_to_token[end-1] + 1 if end-1 < len(char_to_token) else token_start + 1
                
                # Find IOC type index
                if isinstance(ioc_type, str):
                    ioc_type_name = ioc_type
                else:
                    ioc_type_name = ioc_type.value if hasattr(ioc_type, 'value') else str(ioc_type)
                
                # Map to our IOC_TYPES list
                type_index = 0  # Default to 'O'
                for idx, t in enumerate(IOC_TYPES):
                    if t in ioc_type_name or ioc_type_name in t:
                        type_index = idx
                        break
                
                # Set labels for this span
                true_labels[token_start:token_end] = type_index
        
        # Collect for overall metrics
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(ner_preds)
        
        # Per-type accuracy tracking
        for j in range(text_length):
            true_type = IOC_TYPES[true_labels[j]]
            pred_type = IOC_TYPES[ner_preds[j]] if ner_preds[j] < len(IOC_TYPES) else 'O'
            
            if true_type == pred_type:
                if true_type != 'O':
                    per_type_results[true_type]["tp"] += 1
                else:
                    per_type_results[true_type]["tn"] += 1
            else:
                if pred_type != 'O':
                    per_type_results[pred_type]["fp"] += 1
                if true_type != 'O':
                    per_type_results[true_type]["fn"] += 1
    
    # Calculate metrics
    print("\n" + "="*80)
    print("PER-CLASS CLASSIFICATION METRICS")
    print("="*80)
    
    # Filter to valid range
    all_true_labels = [l for l in all_true_labels if l < len(IOC_TYPES)]
    all_pred_labels = [l for l in all_pred_labels if l < len(IOC_TYPES)]
    
    # Generate classification report with all possible labels
    report = classification_report(
        all_true_labels, 
        all_pred_labels,
        labels=list(range(len(IOC_TYPES))),
        target_names=IOC_TYPES,
        digits=3,
        zero_division=0
    )
    print(report)
    
    # Per-type precision/recall/F1
    print("\n" + "="*80)
    print("PER-IOC-TYPE PERFORMANCE")
    print("="*80)
    print(f"{'IOC Type':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-"*80)
    
    for ioc_type in IOC_TYPES[1:]:  # Skip 'O'
        stats = per_type_results[ioc_type]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        
        print(f"{ioc_type:<20} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {support:<12}")
    
    # Overall accuracy
    correct = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == p)
    total = len(all_true_labels)
    overall_acc = correct / total if total > 0 else 0
    
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"Overall Token-Level Accuracy: {overall_acc:.3f}")
    print(f"Total Tokens Evaluated: {total:,}")
    print(f"Correct Classifications: {correct:,}")
    
    # Confusion matrix for top IOC types
    print("\n" + "="*80)
    print("CONFUSION ANALYSIS (Top Misclassifications)")
    print("="*80)
    
    # Track misclassifications
    misclass_count = defaultdict(int)
    for t, p in zip(all_true_labels, all_pred_labels):
        if t != p and t < len(IOC_TYPES) and p < len(IOC_TYPES):
            true_type = IOC_TYPES[t]
            pred_type = IOC_TYPES[p]
            misclass_count[(true_type, pred_type)] += 1
    
    # Show top misclassifications
    top_misclass = sorted(misclass_count.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"{'True Type':<20} {'Predicted As':<20} {'Count':<10}")
    print("-"*50)
    for (true_type, pred_type), count in top_misclass:
        print(f"{true_type:<20} {pred_type:<20} {count:<10}")
    
    return overall_acc


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate NER classification performance')
    parser.add_argument('--model_path', type=str, default='models/best_improved_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of test samples to generate')
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"Error: Model not found at {args.model_path}")
        print("Available models:")
        for model_file in Path("models").glob("*.pth"):
            print(f"  - {model_file}")
        return
    
    accuracy = evaluate_model(args.model_path, args.num_samples)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"✓ Evaluated NER classification performance on {args.num_samples} samples")
    print(f"✓ Overall token-level accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()