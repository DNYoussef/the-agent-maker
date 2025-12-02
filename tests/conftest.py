"""
pytest configuration and fixtures
Shared fixtures for all tests
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "wandb": {"enabled": True, "mode": "offline", "project": "test-project"},
        "hardware": {"device_vram_gb": 6, "max_batch_size": 32},
        "phases": {"phase1": {"num_models": 3, "epochs": 10}},
    }


@pytest.fixture
def mock_model():
    """Create mock PyTorch model for testing"""
    try:
        import torch
        import torch.nn as nn

        class MockModel(nn.Module):
            def __init__(self, hidden_size=256):
                super().__init__()
                self.layer1 = nn.Linear(128, hidden_size)
                self.layer2 = nn.Linear(hidden_size, 64)

            def forward(self, x):
                x = self.layer1(x)
                x = torch.relu(x)
                x = self.layer2(x)
                return x

        return MockModel()
    except ImportError:
        pytest.skip("PyTorch not available")


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing"""
    from cross_phase.utils import MockTokenizer

    return MockTokenizer()
