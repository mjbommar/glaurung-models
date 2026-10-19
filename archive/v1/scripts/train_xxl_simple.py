#!/usr/bin/env python3
"""
Train XXL model using existing infrastructure.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_improved import *
from src.model.xxl_architecture import XXLIOCModel

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
    
    # Generate ultra-hard data
    if args.generate_synthetic:
        logger.info("Generating ultra-hard training data...")
        generator = UltraHardDataGenerator()
        
        # Generate with more hard examples
        train_samples = generator.generate_ultra_hard_batch(
            n_samples=args.train_samples,
            clean_ratio=0.20,  # 20% clean
            easy_ratio=0.20,   # 20% easy  
            hard_ratio=0.60    # 60% hard
        )
        val_samples = generator.generate_ultra_hard_batch(
            n_samples=args.val_samples,
            clean_ratio=0.20,
            easy_ratio=0.20,
            hard_ratio=0.60
        )
        
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
    
    # Train with focal loss for better class balance
    trainer = ImprovedTrainer(model, args)
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