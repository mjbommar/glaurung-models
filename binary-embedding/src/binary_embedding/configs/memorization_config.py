"""Configuration optimized for memorization on small binary datasets."""

from binary_embedding.models import BinaryEmbeddingConfig


def get_memorization_config() -> BinaryEmbeddingConfig:
    """Get configuration optimized for memorizing binary patterns.

    Key changes from default:
    - Minimal dropout (0.01) to allow memorization
    - Reduced weight decay (0.0001) to preserve learned patterns
    - Higher MLM probability (0.35) for faster learning
    - Higher learning rate (1e-4) for faster convergence
    - Shorter warmup (100 steps) for small datasets
    """
    return BinaryEmbeddingConfig(
        model_type="roberta",
        vocab_size=65536,
        # Architecture - consider increasing for more capacity
        hidden_size=768,
        num_hidden_layers=12,  # Could increase to 16 or 24 for more capacity
        num_attention_heads=12,
        intermediate_size=3072,
        # Activation and initialization
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        # CRITICAL: Minimal dropout for memorization
        hidden_dropout_prob=0.01,  # Was 0.1
        attention_probs_dropout_prob=0.01,  # Was 0.1
        classifier_dropout=None,
        # Positional embeddings
        max_position_embeddings=520,
        type_vocab_size=1,
        position_embedding_type="absolute",
        # Special tokens
        pad_token_id=4,
        use_cache=True,
        # Training hyperparameters optimized for memorization
        learning_rate=1e-4,  # Was 5e-5 - higher for faster learning
        weight_decay=0.0001,  # Was 0.01 - much lower to preserve patterns
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        warmup_steps=100,  # Was 1000 - shorter for small datasets
        mlm_probability=0.35,  # Was 0.20 - higher for faster learning
    )


def get_zero_regularization_config() -> BinaryEmbeddingConfig:
    """Get configuration with zero regularization for pure memorization.

    Use this for testing memorization capacity without any regularization.
    """
    config = get_memorization_config()

    # Remove ALL regularization
    config.hidden_dropout_prob = 0.0
    config.attention_probs_dropout_prob = 0.0
    config.weight_decay = 0.0
    config.mlm_probability = 0.40  # Even higher for maximum learning

    return config


def get_large_memorization_config() -> BinaryEmbeddingConfig:
    """Get larger model configuration for increased memorization capacity.

    Uses 16 layers instead of 12 for more pattern storage capacity.
    """
    config = get_memorization_config()

    # Increase model size
    config.num_hidden_layers = 16  # More layers for more capacity
    config.hidden_size = 1024  # Larger hidden size
    config.num_attention_heads = 16  # More attention heads
    config.intermediate_size = 4096  # Larger FFN

    return config
