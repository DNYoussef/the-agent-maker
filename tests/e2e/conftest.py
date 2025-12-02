"""
Shared fixtures for E2E tests.

Provides mock models, tokenizers, and data loaders for testing
the complete pipeline without requiring GPU or large models.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import torch
import torch.nn as nn

# Add src to path


class MockLanguageModel(nn.Module):
    """Mock language model for E2E testing."""

    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2):
        super().__init__()
        self.config = Mock()
        self.config.vocab_size = vocab_size
        self.config.hidden_size = hidden_size
        self.config.num_layers = num_layers

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Mock(logits=logits, loss=loss, hidden_states=x)

    def get_input_embeddings(self):
        return self.embeddings


class MockTokenizer:
    """Mock tokenizer for E2E testing."""

    vocab_size = 1000
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            tokens = [hash(word) % self.vocab_size for word in text.split()]
        else:
            tokens = [[hash(word) % self.vocab_size for word in t.split()] for t in text]

        max_len = kwargs.get("max_length", 32)
        if isinstance(tokens[0], list):
            tokens = [t[:max_len] + [self.pad_token_id] * (max_len - len(t)) for t in tokens]
        else:
            tokens = tokens[:max_len] + [self.pad_token_id] * (max_len - len(tokens))

        return {
            "input_ids": torch.tensor([tokens] if isinstance(tokens[0], int) else tokens),
            "attention_mask": torch.ones_like(
                torch.tensor([tokens] if isinstance(tokens[0], int) else tokens)
            ),
        }

    def decode(self, tokens, **kwargs):
        return f"decoded_{len(tokens)}_tokens"


@pytest.fixture
def mock_model():
    """Create a mock language model."""
    return MockLanguageModel()


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader with sample batches."""

    def create_batch(batch_size=2, seq_len=32, vocab_size=1000):
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        }

    class MockDataLoader:
        def __init__(self, num_batches=5):
            self.num_batches = num_batches
            self.batch_size = 2
            self.seq_len = 32

        def __iter__(self):
            for _ in range(self.num_batches):
                yield create_batch(self.batch_size, self.seq_len)

        def __len__(self):
            return self.num_batches

    return MockDataLoader()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "e2e_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create temporary directory for checkpoints."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir
