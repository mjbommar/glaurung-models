#!/usr/bin/env python3
"""
Train optimized XXL model (242M params) with enhanced NER-focused data generation.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_improved import *
from src.model.xxl_architecture import XXLIOCModel
from src.data_generation.ultra_hard_generator import UltraHardDataGenerator
from src.data_generation.ioc_patterns import IOC_PATTERNS, IOCType

# Enhanced data generator focused on NER diversity
class NERFocusedDataGenerator(UltraHardDataGenerator):
    """Data generator that ensures balanced representation of all IOC types."""
    
    def _generate_hex_string(self, length: int) -> str:
        """Generate a hex string of specified length."""
        return ''.join(np.random.choice(list('0123456789abcdef')) for _ in range(length))
    
    def _generate_ioc_by_type(self, ioc_type: str) -> str:
        """Generate an IOC of a specific type."""
        if ioc_type == 'ip_address':
            return self.generate_ipv4(malicious=True)
        elif ioc_type == 'domain':
            return self.generate_domain(malicious=True)
        elif ioc_type == 'url':
            return self.generate_url(malicious=True)
        elif ioc_type == 'email':
            return self.generate_email(malicious=True)
        elif ioc_type == 'hash_md5':
            # Use real MD5 if available
            if self.real_iocs and self.real_iocs.get('hashes_md5'):
                return np.random.choice(self.real_iocs['hashes_md5'])
            return self.generate_hash('md5')
        elif ioc_type == 'hash_sha1':
            return self.generate_hash('sha1')
        elif ioc_type == 'hash_sha256':
            # Use real SHA256 if available
            if self.real_iocs and self.real_iocs.get('hashes_sha256'):
                return np.random.choice(self.real_iocs['hashes_sha256'])
            return self.generate_hash('sha256')
        elif ioc_type == 'cve':
            return self.generate_cve()
        elif ioc_type == 'file_path':
            return self.generate_filepath(windows=np.random.choice([True, False]), malicious=True)
        elif ioc_type == 'registry_key':
            return self.generate_registry_key(malicious=True)
        elif ioc_type == 'bitcoin_address':
            # Generate bitcoin-like address
            return '1' + ''.join(np.random.choice(list('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz')) for _ in range(33))
        elif ioc_type == 'user_agent':
            agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            ]
            return np.random.choice(agents)
        elif ioc_type == 'mutex':
            return 'Global\\' + ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')) for _ in range(16))
        elif ioc_type == 'yara_rule':
            return f"rule {self._generate_hex_string(8)} {{ condition: true }}"
        elif ioc_type == 'subnet':
            return f"{np.random.randint(10, 240)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.0/24"
        elif ioc_type == 'asn':
            return f"AS{np.random.randint(1, 65535)}"
        elif ioc_type == 'ssl_cert':
            return self._generate_hex_string(40)
        else:  # jwt_token or unknown
            # Generate JWT-like token
            header = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            payload = self._generate_hex_string(32)
            signature = self._generate_hex_string(32)
            return f"{header}.{payload}.{signature}"
    
    def generate_stratified_batch(self, n_samples: int = 10000) -> list:
        """Generate data with guaranteed representation of all IOC types."""
        samples = []
        
        # Ensure minimum samples per IOC type
        samples_per_type = max(n_samples // (len(IOC_PATTERNS) * 2), 100)
        
        # Generate pure samples for each IOC type
        for ioc_type, pattern_info in IOC_PATTERNS.items():
            type_samples = []
            
            # Pure malicious samples
            for _ in range(samples_per_type):
                ioc = self._generate_ioc_by_type(ioc_type)
                
                # Create different contexts
                contexts = [
                    f"Detected {ioc_type.replace('_', ' ')}: {ioc}",
                    f"Malicious {ioc_type.replace('_', ' ')} found: {ioc}",
                    f"Security alert for {ioc}",
                    f"Blocked {ioc}",
                    f"Threat intelligence match: {ioc}",
                    f"{ioc}",  # Pure IOC
                ]
                
                text = np.random.choice(contexts)
                
                # Find IOC position for labeling
                start = text.find(ioc)
                end = start + len(ioc) if start >= 0 else 0
                
                type_samples.append({
                    'text': text,
                    'labels': [[start, end, ioc_type]] if start >= 0 else [],
                    'risk_score': np.random.uniform(0.7, 1.0),
                    'context_type': 'malicious',
                    'primary_ioc_type': ioc_type
                })
            
            # Mixed samples with multiple IOCs
            for _ in range(samples_per_type // 2):
                text_parts = []
                labels = []
                current_pos = 0
                
                # Primary IOC
                ioc1 = self._generate_ioc_by_type(ioc_type)
                prefix = np.random.choice([
                    f"Found {ioc_type.replace('_', ' ')} ",
                    f"Detected ",
                    f"Alert: ",
                    ""
                ])
                text_parts.append(prefix + ioc1)
                
                if prefix:
                    start = len(prefix)
                    labels.append([start, start + len(ioc1), ioc_type])
                else:
                    labels.append([0, len(ioc1), ioc_type])
                
                current_pos = len(text_parts[0])
                
                # Add 1-2 other IOCs
                for _ in range(np.random.randint(1, 3)):
                    other_type = np.random.choice([t for t in IOC_PATTERNS.keys() if t != ioc_type])
                    ioc2 = self._generate_ioc_by_type(other_type)
                    
                    connector = np.random.choice([
                        " and ",
                        " with ",
                        " containing ",
                        ", also found ",
                        " communicating with "
                    ])
                    
                    text_parts.append(connector + ioc2)
                    start = current_pos + len(connector)
                    labels.append([start, start + len(ioc2), other_type])
                    current_pos = start + len(ioc2)
                
                type_samples.append({
                    'text': ''.join(text_parts),
                    'labels': labels,
                    'risk_score': np.random.uniform(0.8, 1.0),
                    'context_type': 'malicious_multi',
                    'primary_ioc_type': ioc_type
                })
            
            samples.extend(type_samples)
        
        # Add challenging edge cases
        edge_cases = []
        
        # Partial/truncated IOCs
        for ioc_type in ['hash_md5', 'hash_sha256', 'ip_address']:
            for _ in range(50):
                ioc = self._generate_ioc_by_type(ioc_type)
                # Truncate
                if ioc_type.startswith('hash'):
                    truncated = ioc[:np.random.randint(10, len(ioc)-5)]
                else:
                    truncated = ioc[:np.random.randint(5, len(ioc)-2)]
                
                edge_cases.append({
                    'text': f"Partial {ioc_type.replace('_', ' ')}: {truncated}...",
                    'labels': [],  # Partial = no label
                    'risk_score': np.random.uniform(0.3, 0.5),
                    'context_type': 'boundary',
                    'primary_ioc_type': 'partial_' + ioc_type
                })
        
        # Obfuscated IOCs
        for _ in range(200):
            ioc_type = np.random.choice(list(IOC_PATTERNS.keys()))
            ioc = self._generate_ioc_by_type(ioc_type)
            
            # Different obfuscation methods
            obfuscation = np.random.choice([
                lambda x: x.replace('.', '[.]'),  # Defanged
                lambda x: x.replace('http', 'hxxp'),  # Protocol obfuscation
                lambda x: ' '.join(x),  # Spaced
                lambda x: x.upper(),  # Case change
                lambda x: f"base64:{self._base64_encode(x)}" if len(x) < 50 else x
            ])
            
            obfuscated = obfuscation(ioc)
            
            edge_cases.append({
                'text': f"Obfuscated indicator: {obfuscated}",
                'labels': [[22, 22 + len(obfuscated), ioc_type]],
                'risk_score': np.random.uniform(0.85, 0.95),
                'context_type': 'obfuscated',
                'primary_ioc_type': ioc_type
            })
        
        samples.extend(edge_cases)
        
        # Add clean samples that look like IOCs but aren't
        clean_confusing = []
        for _ in range(n_samples // 10):
            confusing_patterns = [
                f"Version {'.'.join([str(np.random.randint(0, 255)) for _ in range(4)])}",
                f"Build {self._generate_hex_string(32)}",
                f"Session {''.join(np.random.choice('0123456789abcdef') for _ in range(32))}",
                f"ID: {self._generate_hex_string(64)}",
                f"Checksum validation: {''.join(np.random.choice('0123456789abcdef') for _ in range(32))}",
                f"Git commit: {self._generate_hex_string(40)}",
                f"Docker image sha256:{self._generate_hex_string(64)}",
            ]
            
            clean_confusing.append({
                'text': np.random.choice(confusing_patterns),
                'labels': [],
                'risk_score': np.random.uniform(0.0, 0.1),
                'context_type': 'clean_confusing',
                'primary_ioc_type': 'none'
            })
        
        samples.extend(clean_confusing)
        
        # Regular clean samples
        clean_samples = super().generate_clean_samples(n_samples // 4)
        samples.extend(clean_samples)
        
        # Shuffle and limit
        np.random.shuffle(samples)
        return samples[:n_samples]
    
    def _base64_encode(self, text: str) -> str:
        """Simple base64-like encoding."""
        import base64
        return base64.b64encode(text.encode()).decode()[:50]


# Training script
def main():
    parser = argparse.ArgumentParser(description='Train XXL model optimized for NER accuracy')
    
    # Model config - Optimized for ~242M params
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--num_experts', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training config
    parser.add_argument('--train_samples', type=int, default=100000)
    parser.add_argument('--val_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    # Other args
    parser.add_argument('--generate_synthetic', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--compile', action='store_true', default=True)
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead')
    parser.add_argument('--data_dir', type=str, default='data/xxl_processed')
    parser.add_argument('--output_dir', type=str, default='models')
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate stratified data
    if args.generate_synthetic:
        logger.info("Generating NER-focused stratified data...")
        generator = NERFocusedDataGenerator()
        
        # Generate with stratification
        train_samples = generator.generate_stratified_batch(args.train_samples)
        val_samples = generator.generate_stratified_batch(args.val_samples)
        
        # Log IOC type distribution
        train_types = Counter([s.get('primary_ioc_type', 'none') for s in train_samples])
        logger.info(f"Training IOC distribution: {dict(train_types.most_common(10))}")
        
        # Export
        generator.export_to_jsonl(train_samples, f"{args.data_dir}/train.jsonl")
        generator.export_to_jsonl(val_samples, f"{args.data_dir}/val.jsonl")
        
        logger.info("Data generation complete!")
    
    # Initialize XXL model
    logger.info("Initializing XXL model (242M parameters)...")
    model = XXLIOCModel(
        vocab_size=258,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_ioc_types=18,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        dropout=args.dropout
    )
    
    # Multi-GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    # Compile for speed
    if args.compile and hasattr(torch, 'compile'):
        logger.info(f"Compiling model with {args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)
    
    # Model info
    if hasattr(model, 'get_model_size'):
        info = model.get_model_size()
    elif hasattr(model, 'module'):
        info = model.module.get_model_size()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        info = {'total_params': total_params, 'model_size_mb': total_params * 4 / 1024 / 1024}
    
    logger.info(f"Model: {info['total_params']:,} parameters ({info['model_size_mb']:.1f} MB)")
    
    # Per-class weighted loss for balanced NER
    class_weights = torch.ones(18).to(args.device)
    class_weights[0] = 0.1  # Lower weight for 'O' class
    # Higher weights for rare but important types
    rare_types = ['cve', 'bitcoin_address', 'mutex', 'yara_rule', 'jwt_token']
    ioc_types = ['ip_address', 'domain', 'url', 'email', 'hash_md5', 'hash_sha1', 
                 'hash_sha256', 'cve', 'file_path', 'registry_key', 'bitcoin_address',
                 'user_agent', 'mutex', 'yara_rule', 'subnet', 'asn', 'ssl_cert']
    for i, ioc_type in enumerate(['O'] + ioc_types):
        if ioc_type in rare_types:
            class_weights[i] = 2.0
    
    # Custom trainer with weighted focal loss
    class XXLTrainer(ImprovedTrainer):
        def __init__(self, model, args):
            super().__init__(model, args)
            # Override loss with class-weighted focal loss
            self.ner_loss_fn = FocalLoss(alpha=class_weights, gamma=2.5)
    
    # Load data
    tokenizer = EfficientIOCTokenizer(max_length=512)
    train_dataset = IOCDataset(f"{args.data_dir}/train.jsonl", tokenizer)
    val_dataset = IOCDataset(f"{args.data_dir}/val.jsonl", tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    args.steps_per_epoch = len(train_loader)
    
    # Train
    trainer = XXLTrainer(model, args)
    trainer.train(train_loader, val_loader)
    
    # Save final model
    final_path = Path(args.output_dir) / 'xxl_model_final.pth'
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save({
        'model_state_dict': state_dict,
        'model_config': {
            'hidden_dim': args.hidden_dim,
            'embed_dim': args.embed_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'num_experts': args.num_experts
        }
    }, final_path)
    
    logger.info(f"XXL model saved to {final_path}")


if __name__ == "__main__":
    main()