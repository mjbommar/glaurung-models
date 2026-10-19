#!/usr/bin/env python3
"""
Training script with proper loss functions for NER classification.
Uses PyTorch 2.8 best practices for handling class imbalance.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))

from src.model.improved_architecture import ImprovedIOCModel
from src.data_generation.ultra_hard_generator import UltraHardDataGenerator
from src.model.architecture import EfficientIOCTokenizer
from scripts.train_improved import IOCDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IOC types for NER classification
IOC_TYPES = ['O', 'ip_address', 'domain', 'url', 'email', 'hash_md5', 'hash_sha1', 
             'hash_sha256', 'cve', 'file_path', 'registry_key', 'bitcoin_address',
             'user_agent', 'mutex', 'yara_rule', 'subnet', 'asn', 'ssl_cert']


class ImprovedFocalLoss(nn.Module):
    """
    Improved Focal Loss for PyTorch 2.8 with proper alpha balancing.
    Based on 2025 best practices from the original paper and community improvements.
    """
    
    def __init__(self, num_classes: int, alpha: torch.Tensor = None, gamma: float = 2.0, 
                 ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        # Register alpha as buffer so it moves with the model
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: [N, C] or [N, C, ...] logits
        targets: [N] or [N, ...] labels
        """
        # Reshape for token classification
        if inputs.dim() > 2:
            inputs = inputs.view(-1, self.num_classes)
        targets = targets.view(-1)
        
        # Filter out ignore_index
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        # Calculate focal loss
        log_probs = F.log_softmax(inputs, dim=-1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # Get probability of true class
        p_t = torch.exp(-ce_loss)
        
        # Apply focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class NERTrainer:
    """Trainer optimized for NER classification with proper loss weighting."""
    
    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.args = args
        
        # Calculate class weights based on inverse frequency
        # For NER, we want to upweight rare IOC types
        self.calculate_class_weights()
        
        # Loss functions following PyTorch 2.8 best practices
        self.setup_loss_functions()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.learning_rate,
            total_steps=args.num_epochs * args.steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos'
        )
    
    def calculate_class_weights(self):
        """Calculate class weights based on expected distribution."""
        # Approximate distribution (O class is most common)
        approx_frequencies = {
            'O': 0.70,  # 70% of tokens are non-IOC
            'ip_address': 0.03,
            'domain': 0.04,
            'url': 0.05,
            'email': 0.02,
            'hash_md5': 0.03,
            'hash_sha1': 0.01,
            'hash_sha256': 0.03,
            'cve': 0.01,
            'file_path': 0.02,
            'registry_key': 0.01,
            'bitcoin_address': 0.005,
            'user_agent': 0.01,
            'mutex': 0.005,
            'yara_rule': 0.005,
            'subnet': 0.01,
            'asn': 0.005,
            'ssl_cert': 0.005
        }
        
        # Calculate inverse frequency weights
        weights = []
        for ioc_type in IOC_TYPES:
            freq = approx_frequencies.get(ioc_type, 0.01)
            # Inverse frequency with smoothing
            weight = 1.0 / (freq + 1e-5)
            weights.append(weight)
        
        # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum() * len(weights)  # Normalize to average 1
        
        # Apply additional scaling
        weights[0] *= 0.1  # Reduce weight for 'O' class even more
        
        self.class_weights = weights.to(self.device)
        logger.info(f"Class weights calculated: {self.class_weights.tolist()[:5]}...")
    
    def setup_loss_functions(self):
        """Setup loss functions with proper weighting."""
        
        # Option 1: Weighted CrossEntropyLoss (PyTorch 2.8 native)
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=-100,  # Ignore padding
            label_smoothing=0.1  # Add label smoothing for better generalization
        )
        
        # Option 2: Focal Loss with alpha balancing
        self.focal_loss = ImprovedFocalLoss(
            num_classes=18,
            alpha=self.class_weights,
            gamma=2.0,  # Standard gamma value
            ignore_index=-100
        )
        
        # Risk prediction loss (binary)
        self.risk_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(self.device))
        
        # Binary IOC detection loss
        self.ioc_binary_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(self.device))
    
    def train_epoch(self, dataloader):
        """Train one epoch with proper loss calculation."""
        self.model.train()
        total_loss = 0
        total_ner_loss = 0
        total_risk_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            risk_scores = batch['risk_score'].to(self.device)
            ioc_binary = batch['ioc_binary'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            
            # Calculate NER loss with focal loss
            ner_loss_focal = self.focal_loss(
                outputs['ner_logits'],
                labels
            )
            
            # Alternative: Use weighted CE loss
            ner_loss_ce = self.ce_loss(
                outputs['ner_logits'].view(-1, 18),
                labels.view(-1)
            )
            
            # Combine both losses for better gradient signal
            ner_loss = 0.7 * ner_loss_focal + 0.3 * ner_loss_ce
            
            # Risk prediction loss
            risk_loss = self.risk_loss(
                outputs['risk_score'].squeeze(),
                risk_scores
            )
            
            # Binary IOC detection loss (select the positive class logits)
            ioc_loss = self.ioc_binary_loss(
                outputs['ioc_logits'][:, :, 1].reshape(-1),  # Get logits for IOC class
                ioc_binary.float().reshape(-1)
            )
            
            # Combined loss with adjusted weights
            # Prioritize NER classification
            loss = ner_loss * 1.0 + risk_loss * 0.3 + ioc_loss * 0.2
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            total_ner_loss += ner_loss.item()
            total_risk_loss += risk_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'ner': f'{ner_loss.item():.3f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Train Loss: {avg_loss:.4f} (NER: {total_ner_loss/len(dataloader):.4f})")
        
        return avg_loss
    
    def validate(self, dataloader):
        """Validate with per-class metrics."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                risk_scores = batch['risk_score'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss
                ner_loss = self.focal_loss(outputs['ner_logits'], labels)
                risk_loss = self.risk_loss(outputs['risk_score'].squeeze(), risk_scores)
                loss = ner_loss + 0.3 * risk_loss
                total_loss += loss.item()
                
                # Collect predictions
                ner_preds = torch.argmax(outputs['ner_logits'], dim=-1)
                
                # Mask out padding
                valid_mask = labels != -100
                valid_preds = ner_preds[valid_mask].cpu().numpy()
                valid_labels = labels[valid_mask].cpu().numpy()
                
                all_predictions.extend(valid_preds)
                all_labels.extend(valid_labels)
        
        # Calculate per-class metrics
        from sklearn.metrics import classification_report
        
        # Filter to valid range
        mask = np.array(all_labels) < 18
        filtered_preds = np.array(all_predictions)[mask]
        filtered_labels = np.array(all_labels)[mask]
        
        report = classification_report(
            filtered_labels,
            filtered_preds,
            labels=list(range(18)),  # Ensure all 18 classes are included
            target_names=IOC_TYPES,
            output_dict=True,
            zero_division=0
        )
        
        # Log key metrics
        avg_loss = total_loss / len(dataloader)
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        
        logger.info(f"Val Loss: {avg_loss:.4f}, Macro F1: {macro_f1:.3f}, Weighted F1: {weighted_f1:.3f}")
        
        # Log per-class F1 for important IOC types
        important_types = ['ip_address', 'domain', 'url', 'hash_md5', 'hash_sha256']
        for ioc_type in important_types:
            if ioc_type in report:
                f1 = report[ioc_type]['f1-score']
                logger.info(f"  {ioc_type}: F1={f1:.3f}")
        
        return avg_loss, macro_f1
    
    def train(self, train_loader, val_loader):
        """Full training loop."""
        best_f1 = 0
        
        for epoch in range(self.args.num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.args.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_f1 = self.validate(val_loader)
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'best_f1': best_f1
                }, f'{self.args.output_dir}/best_ner_model.pth')
                logger.info(f"New best model saved! F1: {best_f1:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Train with proper NER loss functions')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--train_samples', type=int, default=50000)
    parser.add_argument('--val_samples', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default='data/ner_focused')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--generate_data', action='store_true', default=True)
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate balanced data for NER
    if args.generate_data:
        logger.info("Generating balanced NER training data...")
        generator = UltraHardDataGenerator()
        
        # Generate with focus on diverse IOC types
        train_samples = generator.generate_ultra_hard_batch(
            n_samples=args.train_samples,
            clean_ratio=0.30,  # 30% clean samples
            easy_ratio=0.35,   # 35% clear IOCs
            hard_ratio=0.35    # 35% challenging cases
        )
        
        val_samples = generator.generate_ultra_hard_batch(
            n_samples=args.val_samples,
            clean_ratio=0.30,
            easy_ratio=0.35,
            hard_ratio=0.35
        )
        
        generator.export_to_jsonl(train_samples, f"{args.data_dir}/train.jsonl")
        generator.export_to_jsonl(val_samples, f"{args.data_dir}/val.jsonl")
        logger.info("Data generation complete!")
    
    # Initialize model
    logger.info("Initializing model...")
    model = ImprovedIOCModel(
        vocab_size=258,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_ioc_types=18,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    model = model.to(args.device)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
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
    
    args.steps_per_epoch = len(train_loader)
    
    # Train
    trainer = NERTrainer(model, args)
    trainer.train(train_loader, val_loader)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()