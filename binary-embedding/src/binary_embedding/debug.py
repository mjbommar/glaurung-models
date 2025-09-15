"""Debug utilities for binary embedding training."""

from pathlib import Path
from typing import Any

import torch
from transformers import BertForMaskedLM, RobertaForMaskedLM

from binary_embedding.models import ModelSize, get_model_config
from binary_embedding.tokenizer import BinaryTokenizer, load_tokenizer


def apply_mlm_mask(
    input_ids: torch.Tensor, tokenizer: BinaryTokenizer, mlm_probability: float = 0.20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MLM masking to input_ids.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        tokenizer: Tokenizer with special tokens
        mlm_probability: Probability of masking

    Returns:
        Tuple of (masked_input_ids, labels)
    """
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)

    # Don't mask special tokens
    special_tokens_mask = (
        (input_ids == tokenizer.pad_token_id)
        | (input_ids == tokenizer.start_token_id)
        | (input_ids == tokenizer.end_token_id)
    )
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Create mask
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # Replace masked tokens
    # 80% of the time, replace with [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, replace with random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # 10% of the time, keep original token
    # (this is implicit - we don't change the remaining masked positions)

    return input_ids, labels


def debug_single_forward(
    model_path: str | None = None,
    tokenizer_path: str = "/home/mjbommar/src/glaurung-models/tokenizers/tokenizer-001/tokenizers/binary-tokenizer-01/iteration-001/tokenizer.json",
    data: bytes | None = None,
    max_length: int = 64,
    mlm_probability: float = 0.20,
    model_type: str = "roberta",
    verbose: bool = True,
) -> dict[str, Any]:
    """Debug a single forward pass through the model.

    Args:
        model_path: Path to checkpoint (if None, creates new model)
        tokenizer_path: Path to tokenizer
        data: Raw bytes to test (if None, uses ELF header)
        max_length: Maximum sequence length
        mlm_probability: Masking probability
        model_type: Model type (bert or roberta)
        verbose: Print detailed output

    Returns:
        Dictionary with debug information
    """
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Default data: ELF header from ollama
    if data is None:
        data = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8
        data += b"\x03\x00\x3e\x00\x01\x00\x00\x00"
        data += b"\x00" * 40  # Pad to reasonable length

    # Convert to latin-1 string
    text = data.decode("latin-1", errors="replace")

    if verbose:
        print(f"Input data ({len(data)} bytes):")
        print(f"  Hex: {data[:32].hex()}")
        print(f"  Text repr: {repr(text[:32])}")

    # Tokenize
    tokens = tokenizer.encode(text)[:max_length]

    if verbose:
        print("\nTokenization:")
        print(f"  Length: {len(tokens)} tokens")
        print(f"  Token IDs: {tokens[:20]}...")

        # Decode first few tokens
        for i in range(min(10, len(tokens))):
            token_id = tokens[i]
            token_str = tokenizer.tokenizer.id_to_token(token_id)
            print(f"    [{i}] ID={token_id:5d} Token={repr(token_str)}")

    # Convert to tensor
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

    # Apply MLM masking
    masked_input_ids, labels = apply_mlm_mask(
        input_ids.clone(), tokenizer, mlm_probability=mlm_probability
    )

    if verbose:
        print(f"\nMLM Masking (prob={mlm_probability}):")
        mask_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
        print(f"  Masked positions: {mask_positions.tolist()}")
        print(
            f"  Number masked: {len(mask_positions)}/{len(tokens)} ({len(mask_positions) / len(tokens) * 100:.1f}%)"
        )

        # Show some masked examples
        for i in mask_positions[:5]:
            orig_id = input_ids[0, i].item()
            masked_id = masked_input_ids[0, i].item()
            label_id = labels[0, i].item()
            print(
                f"    Pos {i}: orig={orig_id} -> masked={masked_id}, label={label_id}"
            )

    # Create or load model
    if model_path and Path(model_path).exists():
        if verbose:
            print(f"\nLoading model from {model_path}")
        if model_type == "bert":
            model = BertForMaskedLM.from_pretrained(model_path)
        else:
            model = RobertaForMaskedLM.from_pretrained(model_path)
    else:
        if verbose:
            print(f"\nCreating new {model_type} model")
        config_data = get_model_config(ModelSize.BASE)
        if model_type == "bert":
            from transformers import BertConfig

            config = BertConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=config_data.hidden_size,
                num_hidden_layers=config_data.num_hidden_layers,
                num_attention_heads=config_data.num_attention_heads,
                intermediate_size=config_data.intermediate_size,
                hidden_dropout_prob=config_data.hidden_dropout_prob,
                attention_probs_dropout_prob=config_data.attention_probs_dropout_prob,
                max_position_embeddings=config_data.max_position_embeddings,
            )
        else:
            from transformers import RobertaConfig

            config = RobertaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=config_data.hidden_size,
                num_hidden_layers=config_data.num_hidden_layers,
                num_attention_heads=config_data.num_attention_heads,
                intermediate_size=config_data.intermediate_size,
                hidden_dropout_prob=config_data.hidden_dropout_prob,
                attention_probs_dropout_prob=config_data.attention_probs_dropout_prob,
                max_position_embeddings=config_data.max_position_embeddings,
            )
        if model_type == "bert":
            model = BertForMaskedLM(config)
        else:
            model = RobertaForMaskedLM(config)

    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=masked_input_ids, labels=labels, return_dict=True)

    loss = outputs.loss
    logits = outputs.logits

    if verbose:
        print("\nForward Pass Results:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Logits shape: {logits.shape}")

        # Check predictions for masked positions
        print("\nPredictions for masked positions:")
        for i, pos in enumerate(mask_positions[:10]):
            pos = pos.item()
            true_id = labels[0, pos].item()
            pred_logits = logits[0, pos]
            pred_id = pred_logits.argmax().item()
            pred_prob = torch.softmax(pred_logits, dim=0)[pred_id].item()

            true_token = tokenizer.tokenizer.id_to_token(true_id)
            pred_token = tokenizer.tokenizer.id_to_token(pred_id)

            correct = "✓" if pred_id == true_id else "✗"
            print(
                f"    Pos {pos}: true={true_id}({repr(true_token)}) pred={pred_id}({repr(pred_token)}) prob={pred_prob:.3f} {correct}"
            )

        # Calculate accuracy
        correct_preds = 0
        total_preds = 0
        for pos in mask_positions:
            pos = pos.item()
            true_id = labels[0, pos].item()
            pred_id = logits[0, pos].argmax().item()
            if pred_id == true_id:
                correct_preds += 1
            total_preds += 1

        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        print(f"\nAccuracy: {correct_preds}/{total_preds} = {accuracy:.1%}")

    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "num_masked": len(mask_positions),
        "input_ids": input_ids,
        "masked_input_ids": masked_input_ids,
        "labels": labels,
        "logits": logits,
    }


def debug_dataset_sample(
    data_dir: str = "/usr/local/bin",
    tokenizer_path: str = "/home/mjbommar/src/glaurung-models/tokenizers/tokenizer-001/tokenizers/binary-tokenizer-01/iteration-001/tokenizer.json",
    max_length: int = 512,
    chunk_size: int = 4096,
    mlm_probability: float = 0.20,
    verbose: bool = True,
) -> dict[str, Any]:
    """Debug a sample from the dataset.

    Args:
        data_dir: Directory containing binary files
        tokenizer_path: Path to tokenizer
        max_length: Maximum sequence length
        chunk_size: Chunk size for reading files
        mlm_probability: Masking probability
        verbose: Print detailed output

    Returns:
        Dictionary with debug information
    """
    from binary_embedding.data import BinaryDataset

    tokenizer = load_tokenizer(tokenizer_path)
    dataset = BinaryDataset(
        data_dir,
        tokenizer,
        max_length=max_length,
        chunk_size=chunk_size,
        enable_entropy_filtering=False,  # Disable for debugging
    )

    if verbose:
        print("Dataset info:")
        print(f"  Directory: {data_dir}")
        print(f"  Total chunks: {len(dataset)}")
        print(f"  Max length: {max_length}")
        print(f"  Chunk size: {chunk_size}")

    # Get first sample
    sample = dataset[0]

    if sample is None:
        print("ERROR: Dataset returned None!")
        return {}

    print(f"\nSample keys: {sample.keys()}")

    input_ids = sample["input_ids"]
    labels = sample.get(
        "labels", torch.full_like(input_ids, -100)
    )  # Default to no labels

    if verbose:
        print("\nFirst sample:")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Labels shape: {labels.shape}")

        # Handle batch dimension if present
        if len(input_ids.shape) == 2:
            input_ids = input_ids[0]  # Take first item in batch
            labels = labels[0]

        print(f"  First 20 tokens: {input_ids[:20].tolist()}")

        # Check masking
        mask_positions = (labels != -100).nonzero(as_tuple=True)[0]
        print(
            f"  Masked positions: {len(mask_positions)}/{len(input_ids)} ({len(mask_positions) / len(input_ids) * 100:.1f}%)"
        )

        # Show token details
        print("\nFirst 20 tokens decoded:")
        for i in range(min(20, len(input_ids))):
            token_id = input_ids[i].item()
            token_str = tokenizer.tokenizer.id_to_token(token_id)
            if labels[i] != -100:
                print(f"    [{i}] ID={token_id:5d} Token={repr(token_str)} [MASKED]")
            else:
                print(f"    [{i}] ID={token_id:5d} Token={repr(token_str)}")

    return {
        "input_ids": input_ids,
        "labels": labels,
        "num_masked": len(mask_positions),
        "dataset_size": len(dataset),
    }


if __name__ == "__main__":
    # Run debug tests
    print("=" * 60)
    print("DEBUG: Single Forward Pass")
    print("=" * 60)
    debug_single_forward(verbose=True)

    print("\n" + "=" * 60)
    print("DEBUG: Dataset Sample")
    print("=" * 60)
    debug_dataset_sample(verbose=True)
