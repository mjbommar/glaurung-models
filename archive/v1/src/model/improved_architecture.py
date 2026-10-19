"""
Improved IOC detection model with deeper architecture and better components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class MultiScaleCharCNN(nn.Module):
    """Multi-scale character-level CNN with residual connections."""
    
    def __init__(self, vocab_size: int = 258, embed_dim: int = 256, hidden_dim: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=256)
        
        # Multi-scale convolutions (different kernel sizes)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, hidden_dim // 4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, hidden_dim // 4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv1d(embed_dim, hidden_dim // 4, kernel_size=11, padding=5)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # GELU activation
        self.activation = nn.GELU()
        
        # Projection to match transformer dim
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed characters
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)
        
        # Transpose for conv1d
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        # Multi-scale convolutions
        conv1_out = self.activation(self.conv1(x))
        conv2_out = self.activation(self.conv2(x))
        conv3_out = self.activation(self.conv3(x))
        conv4_out = self.activation(self.conv4(x))
        
        # Concatenate multi-scale features
        x = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)
        
        # Transpose back
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        # Layer norm and projection
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        
        return x


class ImprovedPositionalEncoding(nn.Module):
    """Learnable positional encoding with sinusoidal initialization."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        # Initialize with sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Make it learnable
        self.pe = nn.Parameter(pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with pre-norm and better dropout."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention with dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network with GELU
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm multi-head attention with residual
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), 
                                  key_padding_mask=mask)[0]
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x


class DeepTransformerEncoder(nn.Module):
    """Deep transformer encoder with 8 layers."""
    
    def __init__(self, d_model: int = 768, nhead: int = 12, num_layers: int = 8, 
                 dim_feedforward: int = 3072, dropout: float = 0.1):
        super().__init__()
        
        # Positional encoding
        self.pos_encoding = ImprovedPositionalEncoding(d_model)
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            EnhancedTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class ImprovedDetectionHead(nn.Module):
    """Improved detection head with better regularization."""
    
    def __init__(self, hidden_dim: int = 768, num_ioc_types: int = 18, dropout: float = 0.2):
        super().__init__()
        
        # NER classifier with 2-layer MLP
        self.ner_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_ioc_types)
        )
        
        # Risk scorer with attention pooling
        self.risk_attention = nn.Linear(hidden_dim, 1)
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Binary IOC detector
        self.ioc_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Hidden states (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask for pooling (batch_size, seq_len)
            
        Returns:
            Dictionary with predictions
        """
        # NER predictions for each token
        ner_logits = self.ner_classifier(x)
        
        # Binary IOC detection for each token
        ioc_logits = self.ioc_detector(x)
        
        # Risk score using attention-weighted pooling
        if attention_mask is not None:
            # Compute attention weights
            attn_weights = self.risk_attention(x).squeeze(-1)  # (batch, seq_len)
            attn_weights = attn_weights.masked_fill(~attention_mask.bool(), float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
            
            # Weighted pooling
            pooled = torch.sum(x * attn_weights, dim=1)  # (batch, hidden_dim)
        else:
            # Simple mean pooling
            pooled = torch.mean(x, dim=1)
        
        risk_score = self.risk_scorer(pooled)
        
        return {
            'ner_logits': ner_logits,
            'ioc_logits': ioc_logits,
            'risk_score': risk_score
        }


class ImprovedIOCModel(nn.Module):
    """Improved deep IOC detection model."""
    
    def __init__(self, 
                 vocab_size: int = 258,
                 embed_dim: int = 256,
                 hidden_dim: int = 768,
                 num_ioc_types: int = 18,
                 nhead: int = 12,
                 num_layers: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # Multi-scale character CNN
        self.char_encoder = MultiScaleCharCNN(vocab_size, embed_dim, hidden_dim)
        
        # Deep transformer encoder
        self.transformer = DeepTransformerEncoder(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        
        # Detection head
        self.detection_head = ImprovedDetectionHead(
            hidden_dim=hidden_dim,
            num_ioc_types=num_ioc_types,
            dropout=dropout * 2  # Higher dropout in the head
        )
        
        # Initialize weights with xavier/he initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Character-level input IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Dictionary containing predictions
        """
        # Encode with multi-scale CNN
        x = self.char_encoder(input_ids)  # (batch_size, seq_len, hidden_dim)
        
        # Apply deep transformer
        if attention_mask is not None:
            # Convert to key padding mask (True for positions to ignore)
            mask = ~attention_mask.bool()
        else:
            mask = None
            
        x = self.transformer(x, mask=mask)  # (batch_size, seq_len, hidden_dim)
        
        # Get predictions from detection head
        outputs = self.detection_head(x, attention_mask)
        
        return outputs
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size in MB
        model_size_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb
        }


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1), reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            at = self.alpha.gather(0, targets.view(-1))
            focal_loss = at * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss