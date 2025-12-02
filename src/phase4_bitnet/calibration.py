"""
Calibration Dataset System
Dataset loaders for activation-aware quantization
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict
from pathlib import Path
from transformers import PreTrainedTokenizer
from src.phase4_bitnet.config import Phase4Config


class CalibrationDataset(Dataset):
    """
    Calibration dataset for BitNet quantization

    Loads representative text samples for activation statistics.
    Supports multiple dataset sources: OpenWebText, C4, WikiText.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: Phase4Config,
        dataset_name: Optional[str] = None
    ):
        """
        Initialize calibration dataset

        Args:
            tokenizer: HuggingFace tokenizer
            config: Phase 4 configuration
            dataset_name: Dataset to use (overrides config)
        """
        self.tokenizer = tokenizer
        self.config = config
        self.dataset_name = dataset_name or config.calibration_dataset

        self.samples: List[str] = []
        self._load_dataset()

    def _load_dataset(self):
        """Load calibration samples from specified dataset"""
        if self.dataset_name == "openwebtext":
            self._load_openwebtext()
        elif self.dataset_name == "c4":
            self._load_c4()
        elif self.dataset_name == "wikitext":
            self._load_wikitext()
        elif self.dataset_name == "custom":
            # Custom dataset, samples set externally
            pass
        else:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}. "
                f"Supported: openwebtext, c4, wikitext, custom"
            )

        # Limit to configured number of samples
        if len(self.samples) > self.config.calibration_samples:
            self.samples = self.samples[:self.config.calibration_samples]

    def _load_openwebtext(self):
        """Load OpenWebText samples"""
        try:
            from datasets import load_dataset

            # Load streaming to avoid full download
            dataset = load_dataset(
                "openwebtext",
                split="train",
                streaming=True
            )

            # Take first N samples
            for i, example in enumerate(dataset):
                if i >= self.config.calibration_samples:
                    break

                text = example.get("text", "")
                if len(text.strip()) > 100:  # Minimum length
                    self.samples.append(text)

        except Exception as e:
            print(f"Warning: OpenWebText loading failed: {e}")
            print("Falling back to WikiText")
            self.dataset_name = "wikitext"
            self._load_wikitext()

    def _load_c4(self):
        """Load C4 samples"""
        try:
            from datasets import load_dataset

            # Load C4 (smaller en subset)
            dataset = load_dataset(
                "c4",
                "en",
                split="train",
                streaming=True
            )

            # Take first N samples
            for i, example in enumerate(dataset):
                if i >= self.config.calibration_samples:
                    break

                text = example.get("text", "")
                if len(text.strip()) > 100:
                    self.samples.append(text)

        except Exception as e:
            print(f"Warning: C4 loading failed: {e}")
            print("Falling back to WikiText")
            self.dataset_name = "wikitext"
            self._load_wikitext()

    def _load_wikitext(self):
        """Load WikiText samples"""
        try:
            from datasets import load_dataset

            # Load WikiText-103
            dataset = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split="train"
            )

            # Take first N samples
            for i, example in enumerate(dataset):
                if i >= self.config.calibration_samples:
                    break

                text = example.get("text", "")
                if len(text.strip()) > 100:
                    self.samples.append(text)

            # If not enough samples, create synthetic ones
            if len(self.samples) < self.config.calibration_samples:
                self._add_synthetic_samples()

        except Exception as e:
            print(f"Warning: WikiText loading failed: {e}")
            print("Using synthetic samples")
            self._add_synthetic_samples()

    def _add_synthetic_samples(self):
        """Add synthetic calibration samples as fallback"""
        synthetic_templates = [
            "The quick brown fox jumps over the lazy dog. "
            "This is a calibration sample for quantization.",

            "In a world of artificial intelligence, models need "
            "to be optimized for efficient inference.",

            "Quantization reduces model size by representing "
            "weights with fewer bits while maintaining accuracy.",

            "Natural language processing has made significant "
            "progress in recent years with transformer models.",

            "Machine learning systems require careful tuning "
            "and calibration for optimal performance.",
        ]

        # Repeat and extend synthetic samples
        while len(self.samples) < self.config.calibration_samples:
            for template in synthetic_templates:
                if len(self.samples) >= self.config.calibration_samples:
                    break
                # Add variation by repeating
                self.samples.append(template * (len(self.samples) % 3 + 1))

    def set_custom_samples(self, samples: List[str]):
        """
        Set custom calibration samples

        Args:
            samples: List of text samples
        """
        self.samples = samples[:self.config.calibration_samples]

    def __len__(self) -> int:
        """Get number of samples"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get tokenized sample

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids and attention_mask
        """
        text = self.samples[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.calibration_sequence_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def create_calibration_dataloader(
    tokenizer: PreTrainedTokenizer,
    config: Phase4Config,
    dataset_name: Optional[str] = None
) -> DataLoader:
    """
    Create calibration dataloader

    Args:
        tokenizer: HuggingFace tokenizer
        config: Phase 4 configuration
        dataset_name: Dataset to use (overrides config)

    Returns:
        DataLoader instance
    """
    dataset = CalibrationDataset(
        tokenizer=tokenizer,
        config=config,
        dataset_name=dataset_name
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.calibration_batch_size,
        shuffle=False,  # Calibration doesn't need shuffling
        num_workers=0,  # Single worker for simplicity
        pin_memory=True if config.device == "cuda" else False
    )

    return dataloader


def collect_activation_statistics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collect activation statistics from model

    Args:
        model: PyTorch model
        dataloader: Calibration dataloader
        device: Device for computation

    Returns:
        Dictionary mapping layer names to statistics
    """
    model.eval()
    model.to(device)

    activation_stats = {}
    hooks = []

    def create_hook(name: str):
        """Create activation collection hook"""
        def hook_fn(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = {
                    'mean': [],
                    'std': [],
                    'max': [],
                    'min': [],
                }

            # Collect statistics
            if isinstance(output, torch.Tensor):
                activation_stats[name]['mean'].append(
                    output.detach().mean().cpu()
                )
                activation_stats[name]['std'].append(
                    output.detach().std().cpu()
                )
                activation_stats[name]['max'].append(
                    output.detach().max().cpu()
                )
                activation_stats[name]['min'].append(
                    output.detach().min().cpu()
                )

        return hook_fn

    # Register hooks on all modules
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)

    # Run calibration
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)

            # Forward pass
            try:
                model(input_ids=input_ids)
            except Exception as e:
                print(f"Warning: Calibration forward pass failed: {e}")
                break

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Aggregate statistics
    aggregated_stats = {}
    for name, stats in activation_stats.items():
        aggregated_stats[name] = {
            'mean': torch.stack(stats['mean']).mean().item(),
            'std': torch.stack(stats['std']).mean().item(),
            'max': torch.stack(stats['max']).max().item(),
            'min': torch.stack(stats['min']).min().item(),
        }

    return aggregated_stats
