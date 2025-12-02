"""
Phase 3 Anti-Theater Validation

Validates that generated thoughts are genuine reasoning, not "theater"
(fake reasoning that doesn't actually help).

Three Critical Tests:
1. Divergence Test: Thoughts diverge from direct continuation (>0.30)
2. Ablation Test: Accuracy improves WITH thoughts vs WITHOUT (>2%)
3. Correlation Test: Coherence scores correlate with utility (>0.5)

All 3 tests must pass or Phase 3 fails.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

from .config import AntiTheaterConfig


class AntiTheaterValidator:
    """
    Validates genuine reasoning vs theater.

    Theater = Model generates "thinking" text that looks like reasoning
    but doesn't actually improve predictions (memorized patterns).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: AntiTheaterConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    def divergence_test(self, input_ids: torch.Tensor, num_samples: int = 100) -> float:
        """
        Test 1: Divergence from Direct Continuation

        Genuine thoughts should diverge from direct greedy continuation.

        Args:
            input_ids: (batch, seq_len)
            num_samples: Number of samples to test

        Returns:
            Divergence score (0-1, higher = more divergent)
        """
        self.model.eval()
        total_divergence = 0.0

        with torch.no_grad():
            for i in range(min(num_samples, input_ids.size(0))):
                sample = input_ids[i : i + 1]

                # Generate direct continuation (greedy)
                direct_output = self.model.base_model.generate(
                    sample,
                    max_new_tokens=20,
                    do_sample=False,
                    num_return_sequences=1,
                )

                # Generate with thoughts (sampled)
                thought_output = self.model(sample, use_thoughts=True)

                # Extract generated tokens
                direct_tokens = direct_output[0, sample.size(1) :]
                thought_logits = thought_output["logits"][0, -20:, :]
                thought_tokens = thought_logits.argmax(dim=-1)

                # Compute edit distance
                divergence = self._edit_distance(direct_tokens, thought_tokens) / max(
                    len(direct_tokens), len(thought_tokens)
                )

                total_divergence += divergence

        avg_divergence = total_divergence / min(num_samples, input_ids.size(0))

        return avg_divergence

    def ablation_test(self, dataloader, max_batches: int = 50) -> float:
        """
        Test 2: Ablation Study

        Accuracy should be higher WITH thoughts than WITHOUT.

        Args:
            dataloader: Validation data
            max_batches: Max batches to test

        Returns:
            Accuracy improvement (positive = thoughts help)
        """
        self.model.eval()

        acc_with_thoughts = 0.0
        acc_without_thoughts = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # WITH thoughts
                outputs_with = self.model(input_ids, labels=labels, use_thoughts=True)
                predictions_with = outputs_with["logits"].argmax(dim=-1)
                correct_with = (predictions_with == labels).float().mean()

                # WITHOUT thoughts
                outputs_without = self.model(input_ids, labels=labels, use_thoughts=False)
                predictions_without = outputs_without["logits"].argmax(dim=-1)
                correct_without = (predictions_without == labels).float().mean()

                acc_with_thoughts += correct_with.item()
                acc_without_thoughts += correct_without.item()
                num_batches += 1

        avg_acc_with = acc_with_thoughts / num_batches
        avg_acc_without = acc_without_thoughts / num_batches

        improvement = avg_acc_with - avg_acc_without

        return improvement

    def correlation_test(self, dataloader, max_batches: int = 50) -> float:
        """
        Test 3: Coherence-Utility Correlation

        Coherence scores should correlate with prediction accuracy.
        High coherence thoughts should lead to better predictions.

        Args:
            dataloader: Validation data
            max_batches: Max batches to test

        Returns:
            Pearson correlation coefficient
        """
        self.model.eval()

        coherence_scores = []
        utilities = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward with thoughts
                outputs = self.model(input_ids, labels=labels, use_thoughts=True)

                # Coherence score (from model output)
                coherence = outputs.get("avg_coherence", 0.0)

                # Utility (prediction accuracy)
                predictions = outputs["logits"].argmax(dim=-1)
                utility = (predictions == labels).float().mean().item()

                coherence_scores.append(coherence)
                utilities.append(utility)

        # Pearson correlation
        if len(coherence_scores) > 1:
            correlation, _ = pearsonr(coherence_scores, utilities)
        else:
            correlation = 0.0

        return correlation

    def validate_all(self, dataloader) -> Dict[str, float]:
        """
        Run all 3 anti-theater tests.

        Returns:
            Results dict with all test scores and pass/fail
        """
        print("\n🎭 Running Anti-Theater Validation...")

        # Test 1: Divergence
        print("  Test 1/3: Divergence from direct continuation...")
        input_ids = next(iter(dataloader))["input_ids"].to(self.device)
        divergence = self.divergence_test(input_ids)
        divergence_passed = divergence > self.config.divergence_threshold
        print(
            f"    Divergence: {divergence:.3f} "
            f"({'✅ PASS' if divergence_passed else '❌ FAIL'} - need >{self.config.divergence_threshold})"
        )

        # Test 2: Ablation
        print("  Test 2/3: Ablation study (WITH vs WITHOUT thoughts)...")
        ablation = self.ablation_test(dataloader)
        ablation_passed = ablation > self.config.ablation_threshold
        print(
            f"    Ablation improvement: {ablation:.4f} "
            f"({'✅ PASS' if ablation_passed else '❌ FAIL'} - need >{self.config.ablation_threshold})"
        )

        # Test 3: Correlation
        print("  Test 3/3: Coherence-utility correlation...")
        correlation = self.correlation_test(dataloader)
        correlation_passed = correlation > self.config.correlation_threshold
        print(
            f"    Correlation: {correlation:.3f} "
            f"({'✅ PASS' if correlation_passed else '❌ FAIL'} - need >{self.config.correlation_threshold})"
        )

        # Overall result
        all_passed = divergence_passed and ablation_passed and correlation_passed

        if all_passed:
            print("\n✅ All anti-theater tests PASSED - Genuine reasoning validated!")
        else:
            print("\n❌ Anti-theater tests FAILED - Theater detected, consider rollback")

        return {
            "divergence": divergence,
            "divergence_passed": divergence_passed,
            "ablation": ablation,
            "ablation_passed": ablation_passed,
            "correlation": correlation,
            "correlation_passed": correlation_passed,
            "all_passed": all_passed,
        }

    def _edit_distance(self, seq1: torch.Tensor, seq2: torch.Tensor) -> int:
        """
        Compute Levenshtein edit distance between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Edit distance (number of edits)
        """
        seq1 = seq1.cpu().numpy()
        seq2 = seq2.cpu().numpy()

        m, n = len(seq1), len(seq2)

        # DP table
        dp = np.zeros((m + 1, n + 1), dtype=int)

        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],  # Delete
                        dp[i][j - 1],  # Insert
                        dp[i - 1][j - 1],  # Replace
                    )

        return dp[m][n]


def validate_anti_theater(
    model: nn.Module,
    tokenizer,
    dataloader,
    config: Optional[AntiTheaterConfig] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Convenience function to run anti-theater validation.

    Args:
        model: Trained Quiet-STaR model
        tokenizer: Tokenizer
        dataloader: Validation data
        config: Anti-theater configuration
        device: Device

    Returns:
        Validation results
    """
    if config is None:
        config = AntiTheaterConfig()

    validator = AntiTheaterValidator(model, tokenizer, config, device)

    return validator.validate_all(dataloader)
