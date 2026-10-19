"""
XXL IOC detection model (300M+ params) optimized for accurate NER classification.
Designed to distinguish between all 18 IOC types with high precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class MultiResolutionCharCNN(nn.Module):
    """Enhanced CNN that captures patterns at multiple resolutions."""
    
    def __init__(self, vocab_size: int = 258, embed_dim: int = 512, hidden_dim: int = 1536):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=256)
        
        # Multi-resolution convolutions for different pattern sizes
        # Small patterns (3-5): IP octets, file extensions
        self.conv_small_1 = nn.Conv1d(embed_dim, hidden_dim // 8, kernel_size=3, padding=1)
        self.conv_small_2 = nn.Conv1d(embed_dim, hidden_dim // 8, kernel_size=5, padding=2)
        
        # Medium patterns (7-11): domains, email parts
        self.conv_med_1 = nn.Conv1d(embed_dim, hidden_dim // 8, kernel_size=7, padding=3)
        self.conv_med_2 = nn.Conv1d(embed_dim, hidden_dim // 8, kernel_size=11, padding=5)
        
        # Large patterns (15-21): URLs, file paths
        self.conv_large_1 = nn.Conv1d(embed_dim, hidden_dim // 8, kernel_size=15, padding=7)
        self.conv_large_2 = nn.Conv1d(embed_dim, hidden_dim // 8, kernel_size=21, padding=10)
        
        # Hash-specific patterns (32, 64): MD5, SHA256
        self.conv_hash_1 = nn.Conv1d(embed_dim, hidden_dim // 8, kernel_size=32, padding=16)
        self.conv_hash_2 = nn.Conv1d(embed_dim, hidden_dim // 8, kernel_size=64, padding=32)
        
        # Batch normalization for each conv
        self.bn_small_1 = nn.BatchNorm1d(hidden_dim // 8)
        self.bn_small_2 = nn.BatchNorm1d(hidden_dim // 8)
        self.bn_med_1 = nn.BatchNorm1d(hidden_dim // 8)
        self.bn_med_2 = nn.BatchNorm1d(hidden_dim // 8)
        self.bn_large_1 = nn.BatchNorm1d(hidden_dim // 8)
        self.bn_large_2 = nn.BatchNorm1d(hidden_dim // 8)
        self.bn_hash_1 = nn.BatchNorm1d(hidden_dim // 8)
        self.bn_hash_2 = nn.BatchNorm1d(hidden_dim // 8)
        
        # Final projection and normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
        
        # Gated fusion mechanism
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embed characters
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.dropout(x)
        
        # Transpose for conv1d
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        # Apply multi-resolution convolutions with batch norm
        conv_outputs = [
            self.bn_small_1(self.activation(self.conv_small_1(x))),
            self.bn_small_2(self.activation(self.conv_small_2(x))),
            self.bn_med_1(self.activation(self.conv_med_1(x))),
            self.bn_med_2(self.activation(self.conv_med_2(x))),
            self.bn_large_1(self.activation(self.conv_large_1(x))),
            self.bn_large_2(self.activation(self.conv_large_2(x))),
            self.bn_hash_1(self.activation(self.conv_hash_1(x))),
            self.bn_hash_2(self.activation(self.conv_hash_2(x)))
        ]
        
        # Concatenate all scales
        x = torch.cat(conv_outputs, dim=1)  # (batch, hidden_dim, seq_len)
        
        # Transpose back
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        
        # Gated fusion
        gate = torch.sigmoid(self.gate(x))
        x = gate * x + (1 - gate) * self.projection(x)
        
        # Final normalization
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x


class IOCSpecificAttention(nn.Module):
    """Attention mechanism specialized for different IOC types."""
    
    def __init__(self, hidden_dim: int, num_ioc_types: int = 18):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_ioc_types = num_ioc_types
        
        # Type-specific attention weights
        self.type_queries = nn.Parameter(torch.randn(num_ioc_types, hidden_dim))
        self.type_keys = nn.Linear(hidden_dim, hidden_dim)
        self.type_values = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute keys and values
        keys = self.type_keys(x)  # (batch, seq_len, hidden_dim)
        values = self.type_values(x)
        
        # Expand type queries for batch
        queries = self.type_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_types, hidden_dim)
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        type_representations = torch.matmul(attn_weights, values)  # (batch, num_types, hidden_dim)
        
        # Broadcast type representations back to sequence length
        type_reps_expanded = type_representations.unsqueeze(2).expand(-1, -1, seq_len, -1)
        x_expanded = x.unsqueeze(1).expand(-1, self.num_ioc_types, -1, -1)
        
        # Concatenate and project
        combined = torch.cat([x_expanded, type_reps_expanded], dim=-1)  # (batch, num_types, seq_len, hidden_dim*2)
        output = self.output_proj(combined)  # (batch, num_types, seq_len, hidden_dim)
        
        # Average across IOC types
        output = output.mean(dim=1)  # (batch, seq_len, hidden_dim)
        
        # Residual connection and normalization
        output = self.layer_norm(x + output)
        
        return output


class XXLTransformerBlock(nn.Module):
    """Enhanced transformer block with mixture of experts."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 num_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Mixture of Experts FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Router for expert selection
        self.router = nn.Linear(d_model, num_experts)
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention
        x_norm = self.norm1(x)
        attn_out = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=mask)[0]
        x = x + self.dropout(attn_out)
        
        # Pre-norm MoE FFN
        x_norm = self.norm2(x)
        
        # Router scores
        router_scores = F.softmax(self.router(x_norm), dim=-1)  # (batch, seq_len, num_experts)
        
        # Apply mixture of experts
        expert_outputs = torch.stack([expert(x_norm) for expert in self.experts], dim=-1)  # (batch, seq_len, d_model, num_experts)
        expert_weights = router_scores.unsqueeze(2)  # (batch, seq_len, 1, num_experts)
        moe_output = (expert_outputs * expert_weights).sum(dim=-1)  # (batch, seq_len, d_model)
        
        x = x + self.dropout(moe_output)
        
        return x


class XXLTransformerEncoder(nn.Module):
    """XXL transformer with 16 layers and specialized components."""
    
    def __init__(self, d_model: int = 1536, nhead: int = 24, num_layers: int = 16,
                 dim_feedforward: int = 6144, num_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        # Alternating standard and MoE layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:  # Even layers: standard
                self.layers.append(
                    XXLTransformerBlock(d_model, nhead, dim_feedforward, num_experts=1, dropout=dropout)
                )
            else:  # Odd layers: MoE
                self.layers.append(
                    XXLTransformerBlock(d_model, nhead, dim_feedforward, num_experts=num_experts, dropout=dropout)
                )
        
        # IOC-specific attention every 4 layers
        self.ioc_attention_layers = nn.ModuleList([
            IOCSpecificAttention(d_model) for _ in range(num_layers // 4)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1)]
        x = self.dropout(x)
        
        # Pass through transformer layers
        ioc_attn_idx = 0
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            
            # Apply IOC-specific attention every 4 layers
            if (i + 1) % 4 == 0 and ioc_attn_idx < len(self.ioc_attention_layers):
                x = self.ioc_attention_layers[ioc_attn_idx](x, mask)
                ioc_attn_idx += 1
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class XXLDetectionHead(nn.Module):
    """Enhanced detection head with per-type specialists."""
    
    def __init__(self, hidden_dim: int = 1536, num_ioc_types: int = 18, dropout: float = 0.2):
        super().__init__()
        
        # Shared feature extraction
        self.shared_features = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Per-type NER classifiers (specialist networks)
        self.type_specialists = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 2)  # Binary: is this type or not
            ) for _ in range(num_ioc_types)
        ])
        
        # Main NER classifier
        self.ner_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_ioc_types)
        )
        
        # Risk scorer with multi-scale pooling
        self.risk_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=dropout, batch_first=True)
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # *3 for max, mean, attention pooling
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Binary IOC detector
        self.ioc_detector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Extract shared features
        features = self.shared_features(x)
        
        # Get specialist predictions for each IOC type
        specialist_logits = []
        for specialist in self.type_specialists:
            logits = specialist(features)
            specialist_logits.append(logits[:, :, 1:2])  # Take "is this type" probability
        specialist_logits = torch.cat(specialist_logits, dim=-1)  # (batch, seq_len, num_types)
        
        # Main NER classification
        ner_logits = self.ner_classifier(features)
        
        # Combine specialist and main predictions
        combined_ner_logits = (ner_logits + specialist_logits) / 2
        
        # Binary IOC detection
        ioc_logits = self.ioc_detector(features)
        
        # Risk scoring with multiple pooling strategies
        if attention_mask is not None:
            # Max pooling
            masked_x = x.masked_fill(~attention_mask.unsqueeze(-1).bool(), float('-inf'))
            max_pool = torch.max(masked_x, dim=1)[0]
            max_pool = torch.where(torch.isinf(max_pool), torch.zeros_like(max_pool), max_pool)
            
            # Mean pooling
            sum_x = (x * attention_mask.unsqueeze(-1)).sum(dim=1)
            mean_pool = sum_x / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            
            # Attention pooling
            query = mean_pool.unsqueeze(1)  # (batch, 1, hidden_dim)
            attn_output = self.risk_attention(query, x, x, key_padding_mask=~attention_mask.bool())[0]
            attn_pool = attn_output.squeeze(1)
        else:
            max_pool = torch.max(x, dim=1)[0]
            mean_pool = torch.mean(x, dim=1)
            attn_pool = mean_pool  # Fallback to mean
        
        # Concatenate pooled representations
        pooled = torch.cat([max_pool, mean_pool, attn_pool], dim=-1)
        risk_score = self.risk_scorer(pooled)
        
        return {
            'ner_logits': combined_ner_logits,
            'specialist_logits': specialist_logits,
            'ioc_logits': ioc_logits,
            'risk_score': risk_score
        }


class XXLIOCModel(nn.Module):
    """XXL IOC detection model (300M+ params) for maximum NER accuracy."""
    
    def __init__(self,
                 vocab_size: int = 258,
                 embed_dim: int = 512,
                 hidden_dim: int = 1536,
                 num_ioc_types: int = 18,
                 nhead: int = 24,
                 num_layers: int = 16,
                 num_experts: int = 4,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # Multi-resolution character encoder
        self.char_encoder = MultiResolutionCharCNN(vocab_size, embed_dim, hidden_dim)
        
        # XXL transformer encoder
        self.transformer = XXLTransformerEncoder(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            num_experts=num_experts,
            dropout=dropout
        )
        
        # Enhanced detection head
        self.detection_head = XXLDetectionHead(
            hidden_dim=hidden_dim,
            num_ioc_types=num_ioc_types,
            dropout=dropout * 2
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'router' in name or 'gate' in name:
                    nn.init.normal_(module.weight, std=0.01)
                else:
                    nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module, mean=0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Encode with multi-resolution CNN
        x = self.char_encoder(input_ids)
        
        # Apply XXL transformer
        if attention_mask is not None:
            mask = ~attention_mask.bool()
        else:
            mask = None
        
        x = self.transformer(x, mask=mask)
        
        # Get predictions
        outputs = self.detection_head(x, attention_mask)
        
        return outputs
    
    def get_model_size(self) -> Dict[str, float]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size in MB
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb
        }