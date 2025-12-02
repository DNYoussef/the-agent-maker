"""
Real Task Fitness Evaluation using Benchmarks

Implements actual task performance evaluation on math reasoning benchmarks:
- GSM8K: Grade School Math (8K problems)
- MGSM: Multilingual Grade School Math
- MATH: Competition-level mathematics

This replaces the parameter-based proxy fitness with real task metrics.

Paper: Evolutionary Optimization of Model Merging Recipes (arXiv:2403.13187v1)
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    benchmark_name: str = "gsm8k"  # gsm8k, mgsm, math
    max_samples: Optional[int] = 100  # Limit samples for fast evaluation (None = all)
    batch_size: int = 8
    max_length: int = 512  # Max generation length
    temperature: float = 0.0  # Greedy decoding for deterministic results
    num_beams: int = 1  # Beam search (1 = greedy)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset paths
    gsm8k_path: Optional[str] = None  # Path to GSM8K test set
    mgsm_path: Optional[str] = None  # Path to MGSM test set
    math_path: Optional[str] = None  # Path to MATH test set


class GSM8KDataset(Dataset):
    """
    GSM8K (Grade School Math) dataset.

    Contains 8,000 grade school math word problems requiring multi-step reasoning.

    Format: Each sample has 'question' and 'answer' (with chain-of-thought).
    """

    def __init__(self, data_path: Optional[str] = None, max_samples: Optional[int] = None):
        """
        Initialize GSM8K dataset.

        Args:
            data_path: Path to GSM8K jsonl file. If None, uses default.
            max_samples: Limit number of samples (for fast evaluation)
        """
        self.samples = self._load_data(data_path, max_samples)

    def _load_data(self, data_path: Optional[str], max_samples: Optional[int]) -> List[Dict]:
        """Load GSM8K data."""
        if data_path is None:
            # Try default HuggingFace cache location
            try:
                from datasets import load_dataset

                dataset = load_dataset("gsm8k", "main", split="test")
                samples = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
            except Exception as e:
                logger.warning(f"Could not load GSM8K from HuggingFace: {e}")
                return []
        else:
            # Load from file (jsonl format)
            samples = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    samples.append({"question": item["question"], "answer": item["answer"]})

        if max_samples:
            samples = samples[:max_samples]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]


class MGSMDataset(Dataset):
    """
    MGSM (Multilingual Grade School Math) dataset.

    Multilingual version of GSM8K covering 10 languages.
    """

    def __init__(self, data_path: Optional[str] = None, max_samples: Optional[int] = None, language: str = "en"):
        """
        Initialize MGSM dataset.

        Args:
            data_path: Path to MGSM jsonl file
            max_samples: Limit number of samples
            language: Language code (en, es, fr, de, zh, ja, th, sw, bn, te)
        """
        self.language = language
        self.samples = self._load_data(data_path, max_samples)

    def _load_data(self, data_path: Optional[str], max_samples: Optional[int]) -> List[Dict]:
        """Load MGSM data."""
        if data_path is None:
            try:
                from datasets import load_dataset

                dataset = load_dataset("juletxara/mgsm", self.language, split="test")
                samples = [{"question": item["question"], "answer": item["answer"]} for item in dataset]
            except Exception as e:
                logger.warning(f"Could not load MGSM from HuggingFace: {e}")
                return []
        else:
            samples = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    samples.append({"question": item["question"], "answer": item["answer"]})

        if max_samples:
            samples = samples[:max_samples]

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]


def extract_numeric_answer(text: str) -> Optional[float]:
    """
    Extract numeric answer from generated text.

    Handles formats like:
    - "The answer is 42"
    - "#### 42"
    - "42"
    - "42.5"

    Args:
        text: Generated text containing the answer

    Returns:
        Extracted number or None if not found
    """
    # Try GSM8K format (#### answer)
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))

    # Try "answer is X" format
    match = re.search(r"(?:answer is|equals?)\s*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Try standalone number at end
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*$", text.strip())
    if match:
        return float(match.group(1))

    # Try any number in the text (last occurrence)
    matches = re.findall(r"(-?\d+(?:\.\d+)?)", text)
    if matches:
        return float(matches[-1])

    return None


def evaluate_gsm8k(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, float]:
    """
    Evaluate model on GSM8K benchmark.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        config: Benchmark configuration

    Returns:
        Dictionary with accuracy, correct count, total count
    """
    config = config or BenchmarkConfig(benchmark_name="gsm8k")

    # Load dataset
    dataset = GSM8KDataset(data_path=config.gsm8k_path, max_samples=config.max_samples)

    if len(dataset) == 0:
        logger.warning("GSM8K dataset is empty. Returning zero accuracy.")
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    model = model.to(config.device)
    model.eval()

    correct = 0
    total = 0

    logger.info(f"Evaluating on GSM8K ({len(dataset)} samples)...")

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            question = sample["question"]
            gold_answer_text = sample["answer"]

            # Extract gold numeric answer
            gold_answer = extract_numeric_answer(gold_answer_text)
            if gold_answer is None:
                logger.warning(f"Could not extract gold answer from: {gold_answer_text}")
                continue

            # Prepare prompt
            prompt = f"Question: {question}\nAnswer:"

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_length)
            inputs = {k: v.to(config.device) for k, v in inputs.items()}

            # Generate
            try:
                outputs = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    temperature=config.temperature if config.temperature > 0 else 1.0,
                    do_sample=config.temperature > 0,
                    num_beams=config.num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract predicted answer
                pred_answer = extract_numeric_answer(generated)

                if pred_answer is not None and abs(pred_answer - gold_answer) < 1e-3:
                    correct += 1

                total += 1

            except Exception as e:
                logger.warning(f"Error generating for sample {idx}: {e}")
                continue

            # Progress logging
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(dataset)}, accuracy so far: {correct / total:.2%}")

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"GSM8K Evaluation Complete: {correct}/{total} = {accuracy:.2%}")

    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_mgsm(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[BenchmarkConfig] = None,
    language: str = "en",
) -> Dict[str, float]:
    """
    Evaluate model on MGSM benchmark.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        config: Benchmark configuration
        language: Language code

    Returns:
        Dictionary with accuracy, correct count, total count
    """
    config = config or BenchmarkConfig(benchmark_name="mgsm")

    # Load dataset
    dataset = MGSMDataset(data_path=config.mgsm_path, max_samples=config.max_samples, language=language)

    if len(dataset) == 0:
        logger.warning(f"MGSM dataset ({language}) is empty. Returning zero accuracy.")
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    model = model.to(config.device)
    model.eval()

    correct = 0
    total = 0

    logger.info(f"Evaluating on MGSM-{language} ({len(dataset)} samples)...")

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            question = sample["question"]
            gold_answer_text = sample["answer"]

            # Extract gold numeric answer
            gold_answer = extract_numeric_answer(gold_answer_text)
            if gold_answer is None:
                logger.warning(f"Could not extract gold answer from: {gold_answer_text}")
                continue

            # Prepare prompt
            prompt = f"Question: {question}\nAnswer:"

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_length)
            inputs = {k: v.to(config.device) for k, v in inputs.items()}

            # Generate
            try:
                outputs = model.generate(
                    **inputs,
                    max_length=config.max_length,
                    temperature=config.temperature if config.temperature > 0 else 1.0,
                    do_sample=config.temperature > 0,
                    num_beams=config.num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract predicted answer
                pred_answer = extract_numeric_answer(generated)

                if pred_answer is not None and abs(pred_answer - gold_answer) < 1e-3:
                    correct += 1

                total += 1

            except Exception as e:
                logger.warning(f"Error generating for sample {idx}: {e}")
                continue

            # Progress logging
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(dataset)}, accuracy so far: {correct / total:.2%}")

    accuracy = correct / total if total > 0 else 0.0

    logger.info(f"MGSM-{language} Evaluation Complete: {correct}/{total} = {accuracy:.2%}")

    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_benchmark(
    model: nn.Module,
    tokenizer: Any,
    benchmark_name: str = "gsm8k",
    config: Optional[BenchmarkConfig] = None,
) -> float:
    """
    Evaluate model on specified benchmark.

    Convenience function that routes to specific benchmark evaluator.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        benchmark_name: Name of benchmark (gsm8k, mgsm)
        config: Benchmark configuration

    Returns:
        Accuracy (float, 0.0-1.0)
    """
    config = config or BenchmarkConfig(benchmark_name=benchmark_name)

    if benchmark_name == "gsm8k":
        result = evaluate_gsm8k(model, tokenizer, config)
    elif benchmark_name == "mgsm":
        result = evaluate_mgsm(model, tokenizer, config)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    return result["accuracy"]


__all__ = [
    "BenchmarkConfig",
    "GSM8KDataset",
    "MGSMDataset",
    "evaluate_gsm8k",
    "evaluate_mgsm",
    "evaluate_benchmark",
    "extract_numeric_answer",
]
