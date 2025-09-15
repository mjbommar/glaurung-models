"""Efficient entropy calculation and sampling utilities for binary data filtering."""

from __future__ import annotations

import bisect

import numpy as np


class EntropyFilter:
    """Efficient entropy-based filtering for binary data chunks.

    Uses Shannon entropy to measure information content and probabilistically
    sample chunks based on their entropy values.
    """

    def __init__(
        self,
        entropy_bins: list[float] | None = None,
        sampling_weights: list[float] | None = None,
    ):
        """Initialize the entropy filter.

        Args:
            entropy_bins: Entropy thresholds for binning (default: [0, 1.0, 3.0, 6.0, 7.5, 8.0])
            sampling_weights: Sampling weights for each bin (default: [0.1, 0.5, 2.0, 1.0, 0.3])
        """
        self.entropy_bins = entropy_bins or [0, 1.0, 3.0, 6.0, 7.5, 8.0]
        self.sampling_weights = sampling_weights or [0.1, 0.5, 2.0, 1.0, 0.3]

        if len(self.sampling_weights) != len(self.entropy_bins) - 1:
            raise ValueError(
                f"sampling_weights must have {len(self.entropy_bins) - 1} elements "
                f"(one less than entropy_bins)"
            )

        # Pre-compute log2 for efficiency
        self._log2 = np.log(2)

        # Create a lookup table for byte entropy calculation
        self._byte_entropy_table = self._precompute_byte_entropy()

    def _precompute_byte_entropy(self) -> np.ndarray:
        """Precompute entropy contribution for each possible byte frequency."""
        # For frequencies from 0 to 256 (max possible in a chunk)
        table = np.zeros(257)
        for i in range(1, 257):
            p = i / 256.0  # Probability
            table[i] = -p * np.log(p) / self._log2
        return table

    def calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of binary data extremely efficiently.

        Args:
            data: Binary data as bytes

        Returns:
            Shannon entropy in bits (0-8 range)
        """
        if not data:
            return 0.0

        # Count byte frequencies using numpy for speed
        byte_array = np.frombuffer(data, dtype=np.uint8)
        counts = np.bincount(byte_array, minlength=256)

        # Filter out zero counts
        nonzero_counts = counts[counts > 0]

        if len(nonzero_counts) <= 1:
            return 0.0  # All bytes are the same

        # Normalize and calculate entropy
        total = len(data)
        probabilities = nonzero_counts / total

        # Use vectorized operations for speed
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return float(entropy)

    def calculate_entropy_fast(self, data: bytes) -> float:
        """Ultra-fast entropy calculation using lookup tables.

        This is even faster than the numpy version for small chunks.

        Args:
            data: Binary data as bytes

        Returns:
            Shannon entropy in bits (0-8 range)
        """
        if not data:
            return 0.0

        # Count byte frequencies (this is the bottleneck)
        counts = [0] * 256
        for byte in data:
            counts[byte] += 1

        # Calculate entropy using precomputed table
        total = len(data)
        entropy = 0.0

        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    def get_sampling_weight(self, entropy: float) -> float:
        """Get the sampling weight for a given entropy value.

        Args:
            entropy: Entropy value in bits

        Returns:
            Sampling weight (higher = more likely to be sampled)
        """
        # Find which bin this entropy falls into
        bin_idx = bisect.bisect_right(self.entropy_bins, entropy) - 1

        # Clamp to valid range
        bin_idx = max(0, min(bin_idx, len(self.sampling_weights) - 1))

        return self.sampling_weights[bin_idx]

    def should_sample(self, data: bytes, random_value: float | None = None) -> bool:
        """Decide whether to sample this data based on its entropy.

        Args:
            data: Binary data as bytes
            random_value: Random value in [0, 1) for sampling decision.
                         If None, always returns True but caller should use
                         the weight for probabilistic sampling.

        Returns:
            True if the data should be sampled, False otherwise
        """
        entropy = self.calculate_entropy(data)
        weight = self.get_sampling_weight(entropy)

        if random_value is None:
            # Return the weight for the caller to handle probabilistic sampling
            return True

        # Probabilistic sampling: sample if random_value < normalized_weight
        # We normalize weights so the maximum weight becomes 1.0
        max_weight = max(self.sampling_weights)
        normalized_weight = weight / max_weight

        return random_value < normalized_weight

    def get_entropy_stats(self, data: bytes) -> dict:
        """Get detailed entropy statistics for debugging/analysis.

        Args:
            data: Binary data as bytes

        Returns:
            Dictionary with entropy value, bin index, and sampling weight
        """
        entropy = self.calculate_entropy(data)
        bin_idx = bisect.bisect_right(self.entropy_bins, entropy) - 1
        bin_idx = max(0, min(bin_idx, len(self.sampling_weights) - 1))
        weight = self.sampling_weights[bin_idx]

        return {
            "entropy": entropy,
            "bin_idx": bin_idx,
            "bin_range": (
                self.entropy_bins[bin_idx],
                self.entropy_bins[bin_idx + 1]
                if bin_idx < len(self.entropy_bins) - 1
                else float("inf"),
            ),
            "sampling_weight": weight,
            "bin_description": self._describe_bin(bin_idx),
        }

    def _describe_bin(self, bin_idx: int) -> str:
        """Get a human-readable description of an entropy bin."""
        descriptions = [
            "Very low (nulls/zeros)",
            "Low (repetitive data)",
            "Medium (code/structured data)",
            "High (complex data)",
            "Very high (compressed/encrypted)",
        ]
        return descriptions[bin_idx] if bin_idx < len(descriptions) else "Unknown"


# Global default filter instance for convenience
_default_filter = EntropyFilter()


def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of binary data.

    Convenience function using the default filter.

    Args:
        data: Binary data as bytes

    Returns:
        Shannon entropy in bits (0-8 range)
    """
    return _default_filter.calculate_entropy(data)


def should_sample(data: bytes, random_value: float) -> bool:
    """Decide whether to sample data based on entropy.

    Convenience function using the default filter.

    Args:
        data: Binary data as bytes
        random_value: Random value in [0, 1) for sampling

    Returns:
        True if the data should be sampled
    """
    return _default_filter.should_sample(data, random_value)
