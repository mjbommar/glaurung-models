#!/usr/bin/env python3
"""
Memory-efficient training script for GPUs with limited VRAM.
Uses gradient accumulation, mixed precision, and smaller batch sizes.
"""

import os
import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import gc

sys.path.append(str(Path(__file__).parent.parent))

from src.model.improved_architecture import ImprovedIOCModel
from src.data_generation.ultra_hard_generator import UltraHardDataGenerator
from src.model.architecture import EfficientIOCTokenizer
from scripts.train_improved import IOCDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


class MemoryEfficientTrainer:
    """Trainer optimized for limited GPU memory."""
    
    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.args = args
        
        # Enable gradient checkpointing to save memory
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Calculate class weights
        self.setup_class_weights()
        
        # Loss functions with memory efficiency in mind
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=-100,
            label_smoothing=0.1
        )
        
        self.risk_loss = nn.BCEWithLogitsLoss()
        
        # Optimizer with lower memory footprint
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False  # Fused optimizer for memory efficiency
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler()
        
        # Learning rate scheduler
        total_steps = args.num_epochs * args.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Gradient accumulation steps
        self.accumulation_steps = args.gradient_accumulation_steps
    
    def setup_class_weights(self):
        """Setup class weights for imbalanced NER."""
        # Simplified weights - focus on getting O vs IOC right first
        weights = torch.ones(18, dtype=torch.float32)
        weights[0] = 0.1  # Downweight 'O' class
        weights[1:] = 2.0  # Upweight all IOC classes
        
        self.class_weights = weights.to(self.device)
        logger.info(f"Class weights: O={weights[0]:.2f}, IOCs={weights[1]:.2f}")
    
    def train_epoch(self, dataloader):
        """Train one epoch with gradient accumulation and mixed precision."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Clear cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            risk_scores = batch['risk_score'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(input_ids, attention_mask)
                
                # NER loss
                ner_loss = self.criterion(
                    outputs['ner_logits'].view(-1, 18),
                    labels.view(-1)
                )
                
                # Risk loss
                risk_loss = self.risk_loss(
                    outputs['risk_score'].squeeze(),
                    risk_scores
                )
                
                # Combined loss
                loss = ner_loss + 0.3 * risk_loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Unscale gradients and clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)
                
                # Scheduler step
                self.scheduler.step()
            
            # Track loss
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.accumulation_steps:.3f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Periodic memory cleanup
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        avg_loss = total_loss / num_batches
        logger.info(f"Train Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, dataloader):
        """Validate with minimal memory usage."""
        self.model.eval()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                risk_scores = batch['risk_score'].to(self.device)
                
                # Forward pass with mixed precision
                with autocast():
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Calculate loss
                    ner_loss = self.criterion(
                        outputs['ner_logits'].view(-1, 18),
                        labels.view(-1)
                    )
                    risk_loss = self.risk_loss(
                        outputs['risk_score'].squeeze(),
                        risk_scores
                    )
                    loss = ner_loss + 0.3 * risk_loss
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs['ner_logits'], dim=-1)
                mask = labels != -100
                correct_tokens += (predictions[mask] == labels[mask]).sum().item()
                total_tokens += mask.sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        
        logger.info(f"Val Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader):
        """Full training loop with memory management."""
        best_accuracy = 0
        
        for epoch in range(self.args.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.args.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_accuracy = self.validate(val_loader)
            
            # Save checkpoint if improved
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                
                # Save with memory efficiency
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'val_loss': val_loss
                }
                
                torch.save(checkpoint, f'{self.args.output_dir}/best_model_efficient.pth')
                logger.info(f"New best model saved! Accuracy: {best_accuracy:.3f}")
                
                # Clear the checkpoint from memory
                del checkpoint
                gc.collect()
            
            # Memory cleanup after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient training for limited GPU')
    
    # Model arguments - Use smaller model for memory efficiency
    parser.add_argument('--model_size', type=str, default='medium',
                        choices=['small', 'medium', 'large'],
                        help='Model size preset')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--train_samples', type=int, default=10000)
    parser.add_argument('--val_samples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Micro batch size (use small value for limited memory)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default='data/memory_efficient')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--generate_data', action='store_true', default=True)
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length (reduce for memory efficiency)')
    
    args = parser.parse_args()
    
    # Model size presets for memory efficiency
    model_configs = {
        'small': {'hidden_dim': 384, 'embed_dim': 128, 'num_layers': 4, 'num_heads': 6},
        'medium': {'hidden_dim': 512, 'embed_dim': 192, 'num_layers': 6, 'num_heads': 8},
        'large': {'hidden_dim': 1024, 'embed_dim': 512, 'num_layers': 16, 'num_heads': 16}
    }
    
    config = model_configs[args.model_size]
    
    # Log configuration
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Model config: {config}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data
    if args.generate_data:
        logger.info("Generating training data...")
        generator = UltraHardDataGenerator()
        
        train_samples = generator.generate_ultra_hard_batch(
            n_samples=args.train_samples,
            clean_ratio=0.30,
            easy_ratio=0.40,
            hard_ratio=0.30
        )
        
        val_samples = generator.generate_ultra_hard_batch(
            n_samples=args.val_samples,
            clean_ratio=0.30,
            easy_ratio=0.40,
            hard_ratio=0.30
        )
        
        generator.export_to_jsonl(train_samples, f"{args.data_dir}/train.jsonl")
        generator.export_to_jsonl(val_samples, f"{args.data_dir}/val.jsonl")
        logger.info("Data generation complete!")
    
    # Initialize model
    logger.info("Initializing model...")
    model = ImprovedIOCModel(
        vocab_size=258,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_ioc_types=18,
        nhead=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=args.dropout
    )
    
    # Enable memory efficient attention if available
    if hasattr(model, 'set_memory_efficient_attention'):
        model.set_memory_efficient_attention(True)
    
    model = model.to(args.device)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Model size: {model_size_mb:.1f} MB")
    
    # Create datasets with reduced max length
    tokenizer = EfficientIOCTokenizer(max_length=args.max_length)
    train_dataset = IOCDataset(f"{args.data_dir}/train.jsonl", tokenizer)
    val_dataset = IOCDataset(f"{args.data_dir}/val.jsonl", tokenizer)
    
    # Create dataloaders with memory-efficient settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers to save memory
        pin_memory=False  # Don't pin memory to save RAM
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    args.steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log GPU memory status
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory before training: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Train
    trainer = MemoryEfficientTrainer(model, args)
    trainer.train(train_loader, val_loader)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
