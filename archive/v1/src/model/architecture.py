"""Lightweight IOC detection model architecture optimized for CPU."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class CharacterCNN(nn.Module):
    """Character-level CNN for pattern detection."""
    
    def __init__(self, vocab_size: int = 256, embed_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Multiple kernel sizes for different n-gram patterns
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1),
            nn.Conv1d(embed_dim, 64, kernel_size=4, padding=2),
            nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2),
            nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3),
        ])
        
        self.projection = nn.Linear(256, out_dim)  # 4 * 64 = 256
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        
        # Apply different kernel sizes
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch_size, 64, conv_seq_len)
            # Ensure all conv outputs have same sequence length by padding/truncating
            if conv_out.size(2) < seq_len:
                # Pad if needed
                pad_size = seq_len - conv_out.size(2)
                conv_out = F.pad(conv_out, (0, pad_size))
            elif conv_out.size(2) > seq_len:
                # Truncate if needed
                conv_out = conv_out[:, :, :seq_len]
            conv_outputs.append(conv_out)
        
        # Concatenate all conv outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, 256, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, 256)
        
        # Project to output dimension
        x = self.dropout(x)
        x = self.projection(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LightweightTransformer(nn.Module):
    """Lightweight transformer optimized for CPU inference."""
    
    def __init__(self, d_model: int = 256, nhead: int = 4, 
                 num_layers: int = 2, dim_feedforward: int = 512):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Use smaller transformer for CPU efficiency
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.layer_norm(x)
        return x


class IOCDetectionHead(nn.Module):
    """Multi-task detection head for IOC classification and risk scoring."""
    
    def __init__(self, hidden_dim: int = 256, num_ioc_types: int = 18):
        super().__init__()
        
        # NER head for token classification
        self.ner_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_ioc_types)
        )
        
        # Risk scoring head (sequence-level)
        self.risk_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Binary IOC detection head (is this token an IOC?)
        self.ioc_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)
        )
        
    def forward(self, x, pool_for_risk: bool = True):
        # x shape: (batch_size, seq_len, hidden_dim)
        
        # Token-level predictions
        ner_logits = self.ner_classifier(x)  # (batch_size, seq_len, num_ioc_types)
        ioc_logits = self.ioc_detector(x)  # (batch_size, seq_len, 2)
        
        # Sequence-level risk score
        if pool_for_risk:
            # Use max pooling for risk (most suspicious token)
            pooled = torch.max(x, dim=1)[0]  # (batch_size, hidden_dim)
            risk_score = self.risk_scorer(pooled)  # (batch_size, 1)
        else:
            risk_score = self.risk_scorer(x)  # (batch_size, seq_len, 1)
        
        return {
            'ner_logits': ner_logits,
            'ioc_logits': ioc_logits,
            'risk_score': risk_score
        }


class LightweightIOCModel(nn.Module):
    """Complete lightweight IOC detection model for CPU inference."""
    
    def __init__(self, 
                 vocab_size: int = 256,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_ioc_types: int = 18,
                 nhead: int = 4,
                 num_layers: int = 2,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # Character-level CNN encoder
        self.char_cnn = CharacterCNN(vocab_size, embed_dim, hidden_dim)
        
        # Lightweight transformer
        self.transformer = LightweightTransformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=hidden_dim * 2
        )
        
        # Detection heads
        self.detection_head = IOCDetectionHead(hidden_dim, num_ioc_types)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Character-level input IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Dictionary containing:
                - ner_logits: IOC type predictions (batch_size, seq_len, num_types)
                - ioc_logits: Binary IOC detection (batch_size, seq_len, 2)
                - risk_score: Sequence risk score (batch_size, 1)
        """
        # Encode with character CNN
        x = self.char_cnn(input_ids)  # (batch_size, seq_len, hidden_dim)
        
        # Apply transformer
        if attention_mask is not None:
            # Convert to key padding mask (True for positions to ignore)
            mask = ~attention_mask.bool()
        else:
            mask = None
            
        x = self.transformer(x, mask=mask)  # (batch_size, seq_len, hidden_dim)
        
        # Get predictions from detection head
        outputs = self.detection_head(x, pool_for_risk=True)
        
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
    
    def prepare_for_quantization(self):
        """Prepare model for quantization."""
        # Add quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # Fuse operations where possible
        for module in self.modules():
            if hasattr(module, 'fuse_modules'):
                module.fuse_modules()
                
    def prepare_for_onnx_export(self):
        """Prepare model for ONNX export."""
        # Set to eval mode
        self.eval()
        
        # Remove any quantization stubs if present
        if hasattr(self, 'quant'):
            delattr(self, 'quant')
        if hasattr(self, 'dequant'):
            delattr(self, 'dequant')


class EfficientIOCTokenizer:
    """Efficient tokenizer for IOC detection."""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        # Use byte-level encoding (0-255)
        self.vocab = {chr(i): i for i in range(256)}
        self.vocab['<PAD>'] = 256
        self.vocab['<UNK>'] = 257
        self.vocab_size = 258
        
    def encode(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text to character IDs."""
        # Convert to bytes then to IDs
        input_ids = []
        for char in text[:self.max_length]:
            if ord(char) < 256:
                input_ids.append(ord(char))
            else:
                input_ids.append(self.vocab['<UNK>'])
        
        # Pad if necessary
        attention_mask = [1] * len(input_ids)
        
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids.extend([self.vocab['<PAD>']] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }
    
    def batch_encode(self, texts: list) -> Dict[str, torch.Tensor]:
        """Batch encode multiple texts."""
        batch_input_ids = []
        batch_attention_mask = []
        
        for text in texts:
            encoded = self.encode(text)
            batch_input_ids.append(encoded['input_ids'])
            batch_attention_mask.append(encoded['attention_mask'])
        
        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask)
        }