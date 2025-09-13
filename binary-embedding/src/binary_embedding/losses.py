"""Contrastive losses and embedding utilities.

This module defines two named contrastive objectives used in training:

- view_contrastive: InfoNCE over duplicate-view pairs (two masked views of the
  same chunk). Encourages invariance to masking/dropout (SimCSE-style).
- same_file_contrastive: InfoNCE over pairs of different chunks from the same
  file. Encourages proximity for semantically-related content within a binary.

Both are implemented via a generic NT-Xent/InfoNCE routine operating on two
embedding tensors (z1, z2) with in-batch negatives. Embeddings are expected to
be L2-normalized.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool last hidden states with padding mask and L2-normalize.

    Args:
        last_hidden: [B, L, H]
        attention_mask: [B, L]
    Returns:
        embeddings: [B, H] L2-normalized
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, L, 1]
    summed = (last_hidden * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    emb = summed / denom
    emb = F.normalize(emb, p=2, dim=-1)
    return emb


def cls_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    """CLS pooling (first token) with L2-normalization.

    Args:
        last_hidden: [B, L, H]
    Returns:
        [B, H] normalized embeddings
    """
    cls = last_hidden[:, 0, :]
    return F.normalize(cls, p=2, dim=-1)


def pool_embeddings(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """Apply a pooling method to token embeddings.

    Supported methods: 'mean', 'cls'. Defaults to 'mean'.
    """
    if method == "cls":
        return cls_pool(last_hidden)
    # Fallback to mean
    return mean_pool(last_hidden, attention_mask)


def info_nce_pairwise(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute symmetric InfoNCE loss given two aligned views.

    Args:
        z1: [N, D] normalized embeddings
        z2: [N, D] normalized embeddings
        temperature: Softmax temperature
    Returns:
        Scalar loss tensor
    Notes:
        - No cross-process gathering here (keep simple and robust). If desired,
          handle gathering and label offsets at the call site.
    """
    if z1.numel() == 0 or z2.numel() == 0:
        return z1.new_tensor(0.0)

    # Similarity logits
    logits = (z1 @ z2.t()) / temperature  # [N, N]
    labels = torch.arange(z1.size(0), device=z1.device)

    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_a + loss_b)


def split_by_pair_type(
    embeddings: torch.Tensor,
    pair_types: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split flattened embeddings into per-type (duplicate vs same_file).

    Args:
        embeddings: [B, 2, D]
        pair_types: [B], 0=duplicate, 1=same_file
    Returns:
        (dup_a, dup_b, sf_a, sf_b) where each tensor has shape [N_type, D].
    """
    assert embeddings.dim() == 3 and embeddings.size(1) == 2
    dup_mask = pair_types == 0
    sf_mask = pair_types == 1

    if dup_mask.any():
        dup_emb = embeddings[dup_mask]  # [N_dup, 2, D]
        dup_a, dup_b = dup_emb[:, 0, :], dup_emb[:, 1, :]
    else:
        dup_a = dup_b = embeddings.new_zeros((0, embeddings.size(-1)))

    if sf_mask.any():
        sf_emb = embeddings[sf_mask]  # [N_sf, 2, D]
        sf_a, sf_b = sf_emb[:, 0, :], sf_emb[:, 1, :]
    else:
        sf_a = sf_b = embeddings.new_zeros((0, embeddings.size(-1)))

    return dup_a, dup_b, sf_a, sf_b
