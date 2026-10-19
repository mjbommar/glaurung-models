#!/usr/bin/env python3
"""
Train XL model - 157M parameters for maximum accuracy.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Just import and modify the existing training script
from train_improved import *

def main():
    parser = argparse.ArgumentParser(description='Train XL IOC detection model (157M params)')
    
    # Model arguments - LARGER DEFAULTS
    parser.add_argument('--hidden_dim', type=int, default=1024,
                       help='Hidden dimension size (default: 1024)')
    parser.add_argument('--embed_dim', type=int, default=384,
                       help='Embedding dimension (default: 384)')
    parser.add_argument('--num_layers', type=int, default=12,
                       help='Number of transformer layers (default: 12)')
    parser.add_argument('--num_heads', type=int, default=16,
                       help='Number of attention heads (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.15,
                       help='Dropout rate (default: 0.15)')
    
    # Training arguments - MORE DATA
    parser.add_argument('--train_samples', type=int, default=200000,
                       help='Number of training samples (default: 200K)')
    parser.add_argument('--val_samples', type=int, default=20000,
                       help='Number of validation samples (default: 20K)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (default: 16 for large model)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of epochs (default: 30)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                       help='Weight decay (default: 0.05)')
    
    # Data generation
    parser.add_argument('--generate_synthetic', action='store_true', default=True,
                       help='Generate synthetic training data')
    parser.add_argument('--ultra_hard', action='store_true', default=True,
                       help='Use ultra-hard data generation')
    parser.add_argument('--use_real_iocs', action='store_true', default=True,
                       help='Use real IOCs from threat feeds')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--multi_gpu', action='store_true', default=True,
                       help='Use multiple GPUs if available')
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--compile', action='store_true', default=True,
                       help='Use torch.compile')
    parser.add_argument('--compile_mode', type=str, default='max-autotune',
                       help='torch.compile mode (default: max-autotune for best perf)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                       help='Use gradient checkpointing to save memory')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='data/processed_xl',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data if requested
    if args.generate_synthetic:
        logger.info(f"Generating {args.train_samples:,} synthetic training samples for XL model...")
        logger.info("Using ULTRA-HARD data generator with real IOCs...")
        
        generator = UltraHardDataGenerator()
        
        # Generate training data with harder distribution for XL
        train_samples = generator.generate_ultra_hard_batch(
            n_samples=args.train_samples,
            clean_ratio=0.25,      # 25% clean (less than before)
            easy_ratio=0.15,       # 15% easy IOCs
            hard_ratio=0.60        # 60% hard/ambiguous IOCs (more challenging)
        )
        generator.export_to_jsonl(train_samples, f"{args.data_dir}/train.jsonl")
        
        # Generate validation data
        val_samples = generator.generate_ultra_hard_batch(
            n_samples=args.val_samples,
            clean_ratio=0.25,
            easy_ratio=0.15,
            hard_ratio=0.60
        )
        generator.export_to_jsonl(val_samples, f"{args.data_dir}/val.jsonl")
        
        logger.info("Synthetic data generation completed!")
    
    # Initialize XL model
    logger.info("Initializing XL model (157M parameters)...")
    model = ImprovedIOCModel(
        vocab_size=258,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_ioc_types=18,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for XL model")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    # Compile model for maximum performance
    if args.compile and hasattr(torch, 'compile'):
        logger.info(f"Compiling XL model with mode={args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode, fullgraph=True)
    
    # Log model info
    if hasattr(model, 'get_model_size'):
        model_info = model.get_model_size()
    elif hasattr(model, 'module') and hasattr(model.module, 'get_model_size'):
        model_info = model.module.get_model_size()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        model_info = {'total_params': total_params, 'model_size_mb': total_params * 4 / 1024 / 1024}
    
    logger.info(f"XL Model: {model_info['total_params']:,} parameters, {model_info['model_size_mb']:.2f} MB")
    
    # Estimate GPU memory usage
    batch_memory_mb = (args.batch_size * 512 * args.hidden_dim * 4) / (1024 * 1024)
    total_memory_estimate = model_info['model_size_mb'] + batch_memory_mb * 2  # x2 for gradients
    logger.info(f"Estimated GPU memory usage: {total_memory_estimate:.1f} MB per GPU")
    
    # Create datasets
    tokenizer = EfficientIOCTokenizer(max_length=512)
    train_dataset = IOCDataset(f"{args.data_dir}/train.jsonl", tokenizer)
    val_dataset = IOCDataset(f"{args.data_dir}/val.jsonl", tokenizer)
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,  # More workers for XL
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=4  # Prefetch more batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Calculate steps per epoch for scheduler
    args.steps_per_epoch = len(train_loader)
    
    # Initialize trainer with XL-specific settings
    trainer = ImprovedTrainer(model, args)
    
    # Train XL model
    logger.info("Starting XL model training...")
    trainer.train(train_loader, val_loader)
    
    # Save final model
    final_path = Path(args.output_dir) / 'xl_model_final.pth'
    torch.save({
        'model_state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
        'model_config': {
            'hidden_dim': args.hidden_dim,
            'embed_dim': args.embed_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'model_params': model_info['total_params']
        }
    }, final_path)
    
    logger.info(f"XL training completed! Final model saved to {final_path}")
    logger.info(f"Model size: {model_info['total_params']:,} parameters")


if __name__ == "__main__":
    main()