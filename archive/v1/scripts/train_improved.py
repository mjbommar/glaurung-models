#!/usr/bin/env python3
"""
Improved training script with better architecture and massive data generation.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.improved_architecture import ImprovedIOCModel, FocalLoss
from src.model.architecture import EfficientIOCTokenizer
from src.data_generation.ultra_hard_generator import UltraHardDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IOCDataset(Dataset):
    """PyTorch Dataset for IOC detection."""
    
    def __init__(self, data_path: str, tokenizer: EfficientIOCTokenizer):
        self.tokenizer = tokenizer
        self.data = []
        
        with open(data_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize text
        encoded = self.tokenizer.encode(sample['text'])
        
        # Convert labels to tensor (use class 0 for non-IOC tokens)
        labels = sample.get('labels', [])
        label_tensor = torch.zeros(self.tokenizer.max_length, dtype=torch.long)
        
        for label_info in labels:
            if isinstance(label_info, dict):
                start = label_info.get('start', 0)
                end = label_info.get('end', start + 1)
                label_type = label_info.get('type', 0)
                label_tensor[start:end] = label_type
        
        # Risk score
        risk_score = torch.tensor(sample.get('risk_score', 0.0), dtype=torch.float32)
        
        # Binary IOC labels (1 if IOC, 0 if not)
        ioc_binary = (label_tensor > 0).long()
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': label_tensor,
            'risk_score': risk_score,
            'ioc_binary': ioc_binary
        }


class ImprovedTrainer:
    """Improved trainer with better loss handling and metrics."""
    
    def __init__(self, model: nn.Module, args):
        self.model = model
        self.args = args
        self.device = args.device
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Warmup + Cosine annealing scheduler
        total_steps = args.num_epochs * args.steps_per_epoch
        warmup_steps = int(total_steps * 0.1)  # 10% warmup
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Loss functions with class weights for imbalance
        # Weight rare classes higher (inverse frequency weighting)
        class_weights = torch.tensor([0.1] + [1.0] * 17).to(self.device)  # Lower weight for O class
        self.ner_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
        
        self.risk_loss_fn = nn.BCEWithLogitsLoss()
        self.ioc_loss_fn = FocalLoss(gamma=2.0)
        
        # Mixed precision training
        self.scaler = GradScaler() if args.fp16 else None
        
        # Gradient clipping value
        self.max_grad_norm = 1.0
        
        # Best model tracking
        self.best_val_loss = float('inf')
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch with improved training loop."""
        self.model.train()
        total_loss = 0
        total_ner_loss = 0
        total_risk_loss = 0
        total_ioc_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            risk_scores = batch['risk_score'].to(self.device)
            ioc_binary = batch['ioc_binary'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast(device_type='cuda'):
                    outputs = self.model(input_ids, attention_mask)
                    
                    # Calculate losses
                    ner_loss = self.ner_loss_fn(
                        outputs['ner_logits'].reshape(-1, outputs['ner_logits'].size(-1)),
                        labels.reshape(-1)
                    )
                    
                    risk_loss = self.risk_loss_fn(
                        outputs['risk_score'].squeeze(),
                        risk_scores
                    )
                    
                    ioc_loss = self.ioc_loss_fn(
                        outputs['ioc_logits'].reshape(-1, 2),
                        ioc_binary.reshape(-1)
                    )
                    
                    # Weighted combination of losses
                    loss = ner_loss + 0.5 * risk_loss + 0.3 * ioc_loss
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate losses
                ner_loss = self.ner_loss_fn(
                    outputs['ner_logits'].reshape(-1, outputs['ner_logits'].size(-1)),
                    labels.reshape(-1)
                )
                
                risk_loss = self.risk_loss_fn(
                    outputs['risk_score'].squeeze(),
                    risk_scores
                )
                
                ioc_loss = self.ioc_loss_fn(
                    outputs['ioc_logits'].reshape(-1, 2),
                    ioc_binary.reshape(-1)
                )
                
                # Weighted combination
                loss = ner_loss + 0.5 * risk_loss + 0.3 * ioc_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Learning rate scheduler step
            self.scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            total_ner_loss += ner_loss.item()
            total_risk_loss += risk_loss.item()
            total_ioc_loss += ioc_loss.item()
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'ner': f'{ner_loss.item():.3f}',
                'risk': f'{risk_loss.item():.3f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Train Loss: {avg_loss:.4f} (NER: {total_ner_loss/len(dataloader):.4f}, "
                   f"Risk: {total_risk_loss/len(dataloader):.4f}, IOC: {total_ioc_loss/len(dataloader):.4f})")
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Validation with comprehensive metrics."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_risk_preds = []
        all_risk_true = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                risk_scores = batch['risk_score'].to(self.device)
                ioc_binary = batch['ioc_binary'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate losses
                ner_loss = self.ner_loss_fn(
                    outputs['ner_logits'].reshape(-1, outputs['ner_logits'].size(-1)),
                    labels.reshape(-1)
                )
                risk_loss = self.risk_loss_fn(
                    outputs['risk_score'].squeeze(),
                    risk_scores
                )
                ioc_loss = self.ioc_loss_fn(
                    outputs['ioc_logits'].reshape(-1, 2),
                    ioc_binary.reshape(-1)
                )
                
                loss = ner_loss + 0.5 * risk_loss + 0.3 * ioc_loss
                total_loss += loss.item()
                
                # Collect predictions for metrics
                ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)
                all_predictions.extend(ner_preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
                risk_preds = torch.sigmoid(outputs['risk_score'].squeeze())
                all_risk_preds.extend(risk_preds.cpu().numpy())
                all_risk_true.extend(risk_scores.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # Filter out padding tokens (assuming padding label is -100 or 256)
        mask = np.array(all_labels) < 18
        filtered_preds = np.array(all_predictions)[mask]
        filtered_labels = np.array(all_labels)[mask]
        
        # Calculate F1 for IOC detection (binary: IOC vs non-IOC)
        binary_preds = (filtered_preds > 0).astype(int)
        binary_labels = (filtered_labels > 0).astype(int)
        
        metrics = {
            'f1': f1_score(binary_labels, binary_preds, average='weighted', zero_division=0),
            'precision': precision_score(binary_labels, binary_preds, average='weighted', zero_division=0),
            'recall': recall_score(binary_labels, binary_preds, average='weighted', zero_division=0),
            'risk_mae': np.mean(np.abs(np.array(all_risk_preds) - np.array(all_risk_true)))
        }
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict):
        """Save model checkpoint."""
        # Get clean state dict
        if hasattr(self.model, '_orig_mod'):
            if isinstance(self.model._orig_mod, nn.DataParallel):
                state_dict = self.model._orig_mod.module.state_dict()
            else:
                state_dict = self.model._orig_mod.state_dict()
        elif isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
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
        
        save_path = Path(self.args.output_dir) / 'best_improved_model.pth'
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Main training loop."""
        logger.info(f"Starting training for {self.args.num_epochs} epochs")
        
        for epoch in range(self.args.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_dataloader)
            
            # Validation
            val_loss, metrics = self.validate(val_dataloader)
            
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Metrics: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, "
                       f"Recall={metrics['recall']:.4f}, Risk MAE={metrics['risk_mae']:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, metrics)
                logger.info(f"New best model! Val loss: {val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train improved IOC detection model')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=8,
                       help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--train_samples', type=int, default=100000,
                       help='Number of training samples to generate')
    parser.add_argument('--val_samples', type=int, default=10000,
                       help='Number of validation samples')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    # Data generation
    parser.add_argument('--generate_synthetic', action='store_true',
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
                       help='torch.compile mode')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory for models')
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data if requested
    if args.generate_synthetic:
        logger.info(f"Generating {args.train_samples} synthetic training samples...")
        logger.info("Using ULTRA-HARD data generator with real IOCs...")
        
        generator = UltraHardDataGenerator()
        
        # Generate training data with proper distribution
        train_samples = generator.generate_ultra_hard_batch(
            n_samples=args.train_samples,
            clean_ratio=0.3,      # 30% clean samples
            easy_ratio=0.2,       # 20% easy IOCs
            hard_ratio=0.5        # 50% hard/ambiguous IOCs
        )
        generator.export_to_jsonl(train_samples, f"{args.data_dir}/train.jsonl")
        
        # Generate validation data
        val_samples = generator.generate_ultra_hard_batch(
            n_samples=args.val_samples,
            clean_ratio=0.3,
            easy_ratio=0.2,
            hard_ratio=0.5
        )
        generator.export_to_jsonl(val_samples, f"{args.data_dir}/val.jsonl")
        
        logger.info("Synthetic data generation completed!")
    
    # Initialize model
    model = ImprovedIOCModel(
        vocab_size=258,
        embed_dim=256,
        hidden_dim=args.hidden_dim,
        num_ioc_types=18,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(args.device)
    
    # Compile model
    if args.compile and hasattr(torch, 'compile'):
        logger.info(f"Compiling model with mode={args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)
    
    # Log model info
    if hasattr(model, 'get_model_size'):
        model_info = model.get_model_size()
    elif hasattr(model, 'module') and hasattr(model.module, 'get_model_size'):
        model_info = model.module.get_model_size()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        model_info = {'total_params': total_params, 'model_size_mb': total_params * 4 / 1024 / 1024}
    
    logger.info(f"Model: {model_info['total_params']:,} parameters, {model_info['model_size_mb']:.2f} MB")
    
    # Create datasets
    tokenizer = EfficientIOCTokenizer(max_length=512)
    train_dataset = IOCDataset(f"{args.data_dir}/train.jsonl", tokenizer)
    val_dataset = IOCDataset(f"{args.data_dir}/val.jsonl", tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Calculate steps per epoch for scheduler
    args.steps_per_epoch = len(train_loader)
    
    # Initialize trainer
    trainer = ImprovedTrainer(model, args)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()