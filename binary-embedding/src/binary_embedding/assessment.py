"""Binary embedding model assessment framework.

This module provides comprehensive evaluation metrics for assessing how well
the model understands binary file formats, headers, embedded strings, and
structural patterns using latin-1 encoding with proper token-level masking.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM

from binary_embedding.tokenizer import BinaryTokenizer, load_tokenizer


@dataclass
class FileHeader:
    """File format header with context for proper tokenization."""

    name: str
    magic_bytes: bytes
    context_bytes: bytes  # Additional bytes to ensure multiple tokens
    description: str

    def to_full_sequence(self) -> bytes:
        """Get full sequence including magic bytes and context."""
        return self.magic_bytes + self.context_bytes

    def to_latin1_string(self) -> str:
        """Convert full sequence to latin-1 encoded string."""
        return self.to_full_sequence().decode("latin-1")


# File headers with enough context to create multiple tokens
FILE_HEADERS = [
    FileHeader(
        name="ELF",
        magic_bytes=b"\x7fELF",
        context_bytes=b"\x01\x01\x01\x00" + b"\x00" * 32,
        description="ELF executable with header",
    ),
    FileHeader(
        name="PE",
        magic_bytes=b"MZ",
        context_bytes=b"\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff"
        + b"\x00" * 24,
        description="PE/DOS executable with header",
    ),
    FileHeader(
        name="ZIP",
        magic_bytes=b"PK\x03\x04",
        context_bytes=b"\x14\x00\x00\x00\x08\x00" + b"\x00" * 30,
        description="ZIP archive with header",
    ),
    FileHeader(
        name="PNG",
        magic_bytes=b"\x89PNG\r\n\x1a\n",
        context_bytes=b"\x00\x00\x00\rIHDR" + b"\x00" * 24,
        description="PNG image with header",
    ),
    FileHeader(
        name="PDF",
        magic_bytes=b"%PDF-",
        context_bytes=b"1.4\n%\xaa\xbb\xcc\xdd" + b"\n" * 32,
        description="PDF document with header",
    ),
]


@dataclass
class AssessmentResult:
    """Result of a single assessment test."""

    test_name: str
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    predictions: list[Any] = field(default_factory=list)
    expected: list[Any] = field(default_factory=list)


@dataclass
class AssessmentSuite:
    """Collection of assessment results."""

    results: list[AssessmentResult] = field(default_factory=list)

    def add_result(self, result: AssessmentResult) -> None:
        """Add a test result to the suite."""
        self.results.append(result)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "average_score": 0.0}

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        avg_score = sum(r.score for r in self.results) / len(self.results)

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results),
            "average_score": avg_score,
        }

    def to_json(self, path: Path) -> None:
        """Save results to JSON file."""
        data = {
            "summary": self.get_summary(),
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "score": r.score,
                    "details": r.details,
                    "predictions": r.predictions[:10] if r.predictions else [],
                    "expected": r.expected[:10] if r.expected else [],
                }
                for r in self.results
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class BinaryAssessment:
    """Assessment framework for binary embedding models using latin-1 encoding."""

    def __init__(
        self,
        model: AutoModelForMaskedLM,
        tokenizer: BinaryTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the assessment framework.

        Args:
            model: Trained masked language model.
            tokenizer: Binary tokenizer.
            device: Device to run inference on.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def predict_masked_token(
        self,
        token_ids: list[int],
        mask_position: int,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Predict a single masked token.

        Args:
            token_ids: Full sequence of token IDs.
            mask_position: Position to mask (must be valid token position).
            top_k: Number of top predictions to return.

        Returns:
            List of (token_id, probability) tuples.
        """
        # Create masked version
        masked_ids = token_ids.copy()
        masked_ids[mask_position] = self.tokenizer.mask_token_id

        # Prepare input
        inputs = {
            "input_ids": torch.tensor([masked_ids], dtype=torch.long).to(self.device),
            "attention_mask": torch.ones([1, len(masked_ids)], dtype=torch.long).to(
                self.device
            ),
        }

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, mask_position, :]
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(probs)))

        return [
            (int(idx), float(prob))
            for idx, prob in zip(top_k_indices.cpu(), top_k_probs.cpu(), strict=False)
        ]

    def assess_file_header_recognition(self) -> AssessmentResult:
        """Test if model recognizes file headers in context."""
        correct = 0
        total = 0
        all_predictions = []
        all_expected = []

        for header in FILE_HEADERS:
            # Get full sequence with context
            full_sequence = header.to_latin1_string()

            # Tokenize to get structure
            encoding = self.tokenizer.tokenizer.encode(full_sequence)
            token_ids = encoding.ids

            # Find maskable positions (skip start/end tokens)
            maskable_positions = list(
                range(1, min(len(token_ids) - 1, 6))
            )  # Test first 5 content tokens

            for mask_pos in maskable_positions:
                original_token_id = token_ids[mask_pos]

                # Get predictions
                predictions = self.predict_masked_token(token_ids, mask_pos, top_k=10)

                if predictions:
                    top_pred_id = predictions[0][0]
                    all_predictions.append(top_pred_id)
                    all_expected.append(original_token_id)

                    # Check if correct
                    if top_pred_id == original_token_id:
                        correct += 1
                    # Partial credit for top-3
                    elif any(
                        pred_id == original_token_id for pred_id, _ in predictions[:3]
                    ):
                        correct += 0.5

                    total += 1

        accuracy = correct / total if total > 0 else 0

        return AssessmentResult(
            test_name="File Header Recognition",
            passed=accuracy > 0.3,  # Lower threshold since this is harder
            score=accuracy,
            details={
                "correct": correct,
                "total": total,
                "headers_tested": len(FILE_HEADERS),
                "avg_maskable_positions": total / len(FILE_HEADERS)
                if FILE_HEADERS
                else 0,
            },
            predictions=all_predictions[:10],
            expected=all_expected[:10],
        )

    def assess_binary_pattern_learning(self) -> AssessmentResult:
        """Test if model learns binary patterns with proper tokenization."""
        patterns = [
            # Patterns that tokenize into multiple tokens
            (b"\x00\x01" * 16, "alternating 00 01"),
            (b"\xff\xfe" * 16, "alternating ff fe"),
            (b"\x00\x01\x02\x03" * 8, "sequential bytes"),
            (b"ABCD" * 8, "repeated ASCII"),
            (struct.pack("<I", 0x12345678) * 8, "repeated little-endian int"),
            (b"\x90\xcc" * 16, "x86 NOP and INT3"),
        ]

        correct = 0
        total = 0

        for pattern_bytes, _description in patterns:
            pattern_str = pattern_bytes.decode("latin-1")

            # Tokenize
            encoding = self.tokenizer.tokenizer.encode(pattern_str)
            token_ids = encoding.ids

            # Only test if we have enough tokens
            if len(token_ids) > 4:  # Need at least start, 3 content tokens, end
                # Test masking in the middle
                maskable_positions = list(range(2, len(token_ids) - 2))[
                    :3
                ]  # Test up to 3 positions

                for mask_pos in maskable_positions:
                    original_token_id = token_ids[mask_pos]

                    predictions = self.predict_masked_token(
                        token_ids, mask_pos, top_k=5
                    )

                    if predictions:
                        # Check if original is in top predictions
                        if predictions[0][0] == original_token_id:
                            correct += 1
                        elif any(
                            pred_id == original_token_id
                            for pred_id, _ in predictions[:3]
                        ):
                            correct += 0.3  # Partial credit

                        total += 1

        accuracy = correct / total if total > 0 else 0

        return AssessmentResult(
            test_name="Binary Pattern Learning",
            passed=accuracy > 0.25,
            score=accuracy,
            details={
                "correct": correct,
                "total": total,
                "patterns_tested": len(patterns),
            },
        )

    def assess_context_understanding(self) -> AssessmentResult:
        """Test if model understands context in binary sequences."""
        # Create sequences where context matters
        test_cases = []

        # Sequential patterns
        for start in [0, 16, 32]:
            seq = bytes([start + i for i in range(32)])
            test_cases.append((seq, "sequential from " + str(start)))

        # Structured patterns
        test_cases.append(
            (
                b"\x00\x00\x00\x01"
                + b"\x00\x00\x00\x02"
                + b"\x00\x00\x00\x03"
                + b"\x00\x00\x00\x04" * 4,
                "little-endian sequence",
            )
        )

        # Text with binary
        test_cases.append(
            (
                b"START" + b"\x00" * 8 + b"MIDDLE" + b"\x00" * 8 + b"END" + b"\x00" * 4,
                "text markers with padding",
            )
        )

        scores = []

        for test_bytes, _description in test_cases:
            test_str = test_bytes.decode("latin-1")

            # Tokenize
            encoding = self.tokenizer.tokenizer.encode(test_str)
            token_ids = encoding.ids

            if len(token_ids) > 5:  # Need enough tokens
                # Mask a token in the middle
                mask_pos = len(token_ids) // 2
                original_token_id = token_ids[mask_pos]

                predictions = self.predict_masked_token(token_ids, mask_pos, top_k=10)

                if predictions:
                    # Score based on rank of correct prediction
                    for rank, (pred_id, _prob) in enumerate(predictions):
                        if pred_id == original_token_id:
                            # Higher score for higher rank
                            scores.append(1.0 - (rank * 0.1))
                            break
                    else:
                        scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0

        return AssessmentResult(
            test_name="Context Understanding",
            passed=avg_score > 0.4,
            score=avg_score,
            details={
                "test_cases": len(test_cases),
                "scores": scores,
            },
        )

    def assess_embedding_quality(self) -> AssessmentResult:
        """Test if similar binaries have similar embeddings."""
        # Create pairs of similar and dissimilar sequences
        similar_pairs = [
            # Similar file headers with context
            (
                b"\x7fELF\x01\x01\x01" + b"\x00" * 32,
                b"\x7fELF\x02\x01\x01" + b"\x00" * 32,
            ),
            # Similar PE headers
            (b"MZ\x90\x00\x03" + b"\x00" * 32, b"MZ\x90\x00\x04" + b"\x00" * 32),
            # Similar null patterns with small differences
            (b"\x00" * 32 + b"\x01", b"\x00" * 32 + b"\x02"),
        ]

        dissimilar_pairs = [
            # Different file types
            (b"\x7fELF" + b"\x00" * 36, b"MZ\x90\x00" + b"\xff" * 36),
            # Null vs high bytes
            (b"\x00" * 40, b"\xff" * 40),
            # Text vs binary
            (b"Hello World This is Text" + b" " * 16, b"\x90" * 20 + b"\xcc" * 20),
        ]

        def get_embedding(byte_seq: bytes) -> torch.Tensor:
            """Get embedding for a byte sequence."""
            text = byte_seq.decode("latin-1")
            inputs = self.tokenizer(
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use mean pooling of last hidden states
                hidden_states = outputs.hidden_states[-1]
                mask = (
                    inputs["attention_mask"].unsqueeze(-1).expand(hidden_states.size())
                )
                masked_hidden = hidden_states * mask
                summed = torch.sum(masked_hidden, dim=1)
                count = torch.clamp(mask.sum(dim=1), min=1e-9)
                embedding = summed / count

            return embedding[0]

        # Calculate similarities
        similar_scores = []
        for seq1, seq2 in similar_pairs:
            emb1 = get_embedding(seq1)
            emb2 = get_embedding(seq2)
            similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
            similar_scores.append(similarity)

        dissimilar_scores = []
        for seq1, seq2 in dissimilar_pairs:
            emb1 = get_embedding(seq1)
            emb2 = get_embedding(seq2)
            similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
            dissimilar_scores.append(similarity)

        # Good embeddings should have higher similarity for similar pairs
        avg_similar = sum(similar_scores) / len(similar_scores) if similar_scores else 0
        avg_dissimilar = (
            sum(dissimilar_scores) / len(dissimilar_scores) if dissimilar_scores else 0
        )

        # Score based on separation
        separation = avg_similar - avg_dissimilar
        score = max(0, min(1, separation))

        return AssessmentResult(
            test_name="Embedding Quality",
            passed=separation > 0.1,
            score=score,
            details={
                "avg_similar_similarity": avg_similar,
                "avg_dissimilar_similarity": avg_dissimilar,
                "separation": separation,
                "similar_pairs": len(similar_pairs),
                "dissimilar_pairs": len(dissimilar_pairs),
            },
        )

    def run_full_assessment(self) -> AssessmentSuite:
        """Run all assessment tests."""
        suite = AssessmentSuite()

        # Run all assessments
        assessments = [
            self.assess_file_header_recognition,
            self.assess_binary_pattern_learning,
            self.assess_context_understanding,
            self.assess_embedding_quality,
        ]

        for assess_func in assessments:
            try:
                result = assess_func()
                suite.add_result(result)
                print(
                    f"✓ {result.test_name}: {'PASS' if result.passed else 'FAIL'} (score: {result.score:.2%})"
                )
            except Exception as e:
                print(f"✗ {assess_func.__name__} failed: {e}")
                suite.add_result(
                    AssessmentResult(
                        test_name=assess_func.__name__.replace("assess_", "")
                        .replace("_", " ")
                        .title(),
                        passed=False,
                        score=0.0,
                        details={"error": str(e)},
                    )
                )

        return suite


def load_and_assess_model(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path | None = None,
    output_path: Path | None = None,
) -> AssessmentSuite:
    """Load a model checkpoint and run full assessment.

    Args:
        checkpoint_path: Path to model checkpoint.
        tokenizer_path: Path to tokenizer (uses default if None).
        output_path: Path to save results JSON (optional).

    Returns:
        AssessmentSuite with all test results.
    """
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)

    # Load model
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)

    # Create assessment
    assessment = BinaryAssessment(model, tokenizer)

    # Run assessment
    suite = assessment.run_full_assessment()

    # Save results if requested
    if output_path:
        suite.to_json(output_path)
        print(f"\nResults saved to {output_path}")

    # Print summary
    summary = suite.get_summary()
    print("\n" + "=" * 50)
    print("ASSESSMENT SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Average Score: {summary['average_score']:.1%}")

    return suite
