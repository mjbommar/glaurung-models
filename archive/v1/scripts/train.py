#!/usr/bin/env python3
"""Training script for lightweight IOC detection model."""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.architecture import LightweightIOCModel, EfficientIOCTokenizer
from src.data_generation.enhanced_synthetic_gen import EnhancedSyntheticGenerator
from src.data_generation.ultra_hard_generator import UltraHardDataGenerator
from src.data_generation.ioc_patterns import IOCType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IOCDataset(Dataset):
    """PyTorch dataset for IOC detection."""
    
    def __init__(self, data_path: str, tokenizer: EfficientIOCTokenizer):
        self.tokenizer = tokenizer
        self.samples = []
        
        # Load data from JSONL
        with open(data_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
                
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize text
        encoded = self.tokenizer.encode(sample['text'])
        
        # Create labels tensor
        labels = torch.zeros(self.tokenizer.max_length, dtype=torch.long)
        for start, end, ioc_type in sample['labels']:
            if end <= self.tokenizer.max_length:
                # Map IOC type string to index
                try:
                    # Get the enum member by name
                    ioc_enum = IOCType[ioc_type.upper()] if isinstance(ioc_type, str) else ioc_type
                    # Get the index in the enum (1-based to avoid 0 which is padding)
                    type_idx = list(IOCType).index(ioc_enum) + 1
                    labels[start:end] = type_idx
                except (KeyError, ValueError):
                    # Skip unknown IOC types
                    continue
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels,
            'risk_score': torch.tensor(sample['risk_score'], dtype=torch.float)
        }


class IOCTrainer:
    """Trainer for IOC detection model."""
    
    def __init__(self, model: LightweightIOCModel, args):
        self.model = model
        self.args = args
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.num_epochs
        )
        
        # Loss functions
        self.ner_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.risk_loss_fn = nn.MSELoss()
        self.ioc_loss_fn = nn.CrossEntropyLoss()
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if (args.fp16 and args.device == 'cuda') else None
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch['input_ids'].to(self.args.device)
            attention_mask = batch['attention_mask'].to(self.args.device)
            labels = batch['labels'].to(self.args.device)
            risk_scores = batch['risk_score'].to(self.args.device)
            
            # Forward pass
            if self.args.fp16 and self.scaler:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Calculate losses
                    ner_loss = self.ner_loss_fn(
                        outputs['ner_logits'].view(-1, outputs['ner_logits'].size(-1)),
                        labels.view(-1)
                    )
                    risk_loss = self.risk_loss_fn(
                        outputs['risk_score'].squeeze(),
                        risk_scores
                    )
                    
                    # Combine losses
                    loss = ner_loss + self.args.risk_weight * risk_loss
                    
                # Backward pass with mixed precision
                self.scaler.scale(loss).backward()
                
                if ((batch_idx + 1) % self.args.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate losses
                ner_loss = self.ner_loss_fn(
                    outputs['ner_logits'].view(-1, outputs['ner_logits'].size(-1)),
                    labels.view(-1)
                )
                risk_loss = self.risk_loss_fn(
                    outputs['risk_score'].squeeze(),
                    risk_scores
                )
                
                # Combine losses
                loss = ner_loss + self.args.risk_weight * risk_loss
                loss.backward()
                
                if ((batch_idx + 1) % self.args.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                labels = batch['labels'].to(self.args.device)
                risk_scores = batch['risk_score'].to(self.args.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate losses
                ner_loss = self.ner_loss_fn(
                    outputs['ner_logits'].view(-1, outputs['ner_logits'].size(-1)),
                    labels.view(-1)
                )
                risk_loss = self.risk_loss_fn(
                    outputs['risk_score'].squeeze(),
                    risk_scores
                )
                
                loss = ner_loss + self.args.risk_weight * risk_loss
                total_loss += loss.item()
                
                # Collect predictions for metrics
                predictions = torch.argmax(outputs['ner_logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_labels)
        
        return total_loss / len(dataloader), metrics
    
    def calculate_metrics(self, predictions, labels):
        """Calculate F1, precision, recall."""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # Ensure arrays are flattened
        predictions = np.array(predictions).flatten()
        labels = np.array(labels).flatten()
        
        # Filter out padding tokens (label=0)
        mask = labels != 0
        if mask.sum() == 0:
            # No valid labels, return zeros
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        predictions = predictions[mask]
        labels = labels[mask]
        
        # Calculate metrics
        try:
            metrics = {
                'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
                'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
                'recall': recall_score(labels, predictions, average='weighted', zero_division=0)
            }
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}")
            metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        logger.info(f"Starting training for {self.args.num_epochs} epochs")
        
        for epoch in range(self.args.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Metrics: {metrics}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, metrics)
            
            # Update scheduler
            self.scheduler.step()
            
        logger.info("Training completed!")
        
    def save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]):
        """Save model checkpoint."""
        # Get the actual model state dict, unwrapping DataParallel and torch.compile
        if hasattr(self.model, '_orig_mod'):
            # torch.compile wrapped
            if isinstance(self.model._orig_mod, nn.DataParallel):
                # DataParallel inside torch.compile
                state_dict = self.model._orig_mod.module.state_dict()
            else:
                # Just torch.compile
                state_dict = self.model._orig_mod.state_dict()
        elif isinstance(self.model, nn.DataParallel):
            # Just DataParallel
            state_dict = self.model.module.state_dict()
        else:
            # No wrapping
            state_dict = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'args': self.args
        }
        
        save_path = Path(self.args.output_dir) / f'best_model.pth'
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train IOC detection model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--generate_synthetic', action='store_true',
                       help='Generate synthetic training data')
    
    # Data generation arguments
    parser.add_argument('--train_samples', type=int, default=50000,
                       help='Number of training samples to generate')
    parser.add_argument('--val_samples', type=int, default=10000,
                       help='Number of validation samples to generate')
    parser.add_argument('--ultra_hard', action='store_true', default=True,
                       help='Use ultra-hard data generation with real IOCs')
    parser.add_argument('--use_real_iocs', action='store_true', default=True,
                       help='Fetch and use real IOCs from threat feeds')
    
    # Model arguments - Bigger defaults for better performance
    parser.add_argument('--vocab_size', type=int, default=258,
                       help='Vocabulary size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_ioc_types', type=int, default=18,
                       help='Number of IOC types')
    
    # Training arguments - Better defaults
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--risk_weight', type=float, default=0.3,
                       help='Weight for risk loss')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Device - Use GPU by default
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--multi_gpu', action='store_true', default=torch.cuda.device_count() > 1,
                       help='Use multiple GPUs if available')
    
    # torch.compile optimization (PyTorch 2.0+)
    parser.add_argument('--compile', action='store_true', default=True,
                       help='Use torch.compile for optimization')
    parser.add_argument('--compile_mode', type=str, default='default',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile mode')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Generate synthetic data if requested
    if args.generate_synthetic:
        logger.info("Generating synthetic training data...")
        
        # Use ultra-hard generator if requested
        if args.ultra_hard:
            logger.info("Using ULTRA-HARD data generator with real IOCs...")
            generator = UltraHardDataGenerator(seed=42, use_real_iocs=args.use_real_iocs)
            
            # Generate training data with ultra-hard distribution
            train_samples = generator.generate_ultra_hard_batch(
                n_samples=args.train_samples,
                clean_ratio=0.3,  # Less clean samples
                easy_ratio=0.1,   # Very few easy samples
                hard_ratio=0.6    # Mostly hard samples
            )
        else:
            generator = EnhancedSyntheticGenerator(seed=42)
            
            # Generate training data with balanced distribution
            train_samples = generator.generate_balanced_batch(
                n_samples=args.train_samples,
                clean_ratio=0.4,
                light_suspicious=0.3,
                heavy_suspicious=0.3
            )
        
        generator.export_to_jsonl(train_samples, f"{args.data_dir}/train.jsonl")
        
        # Generate validation data (use same generator type)
        if args.ultra_hard:
            val_samples = generator.generate_ultra_hard_batch(
                n_samples=args.val_samples,
                clean_ratio=0.3,
                easy_ratio=0.2,  # Slightly more easy for validation
                hard_ratio=0.5
            )
        else:
            val_samples = generator.generate_balanced_batch(
                n_samples=args.val_samples,
                clean_ratio=0.4,
                light_suspicious=0.3,
                heavy_suspicious=0.3
            )
        
        generator.export_to_jsonl(val_samples, f"{args.data_dir}/val.jsonl")
        
        logger.info("Synthetic data generation completed!")
    
    # Initialize tokenizer and model
    tokenizer = EfficientIOCTokenizer(max_length=512)
    
    model = LightweightIOCModel(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_ioc_types=args.num_ioc_types,
        nhead=args.num_heads,
        num_layers=args.num_layers
    )
    
    # Use DataParallel for multi-GPU if available
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    # Compile model for faster training (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model, mode=args.compile_mode)
    
    # Log GPU info
    if args.device == 'cuda':
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Log model size
    # Handle compiled models and DataParallel
    if hasattr(model, '_orig_mod'):
        # torch.compile wrapped model
        if isinstance(model._orig_mod, nn.DataParallel):
            model_stats = model._orig_mod.module.get_model_size()
        else:
            model_stats = model._orig_mod.get_model_size()
    elif isinstance(model, nn.DataParallel):
        model_stats = model.module.get_model_size()
    else:
        model_stats = model.get_model_size()
    logger.info(f"Model statistics: {model_stats}")
    
    # Load datasets
    train_dataset = IOCDataset(f"{args.data_dir}/train.jsonl", tokenizer)
    val_dataset = IOCDataset(f"{args.data_dir}/val.jsonl", tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Initialize trainer
    trainer = IOCTrainer(model, args)
    
    # Train model
    start_time = time.time()
    trainer.train(train_loader, val_loader)
    training_time = time.time() - start_time
    
    logger.info(f"Total training time: {training_time/60:.2f} minutes")
    

if __name__ == "__main__":
    main()