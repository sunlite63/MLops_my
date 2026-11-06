"""Pytest fixtures for the test suite."""

import pytest
import torch

from src.dataset import build_dataloader, SyntheticDataset
from src.models import SimpleClassifier


@pytest.fixture
def sample_dataset():
    """Create a small synthetic dataset for testing."""
    return SyntheticDataset(n_samples=50, n_features=4, random_seed=123)


@pytest.fixture
def sample_dataloader(sample_dataset):
    """Create a dataloader from the sample dataset."""
    return build_dataloader(sample_dataset, batch_size=8, shuffle=False)


@pytest.fixture
def classifier():
    """Create a simple classifier for testing."""
    return SimpleClassifier(input_features=4, hidden_size=8, num_classes=2)


@pytest.fixture
def random_batch():
    """Generate a random batch of data."""
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    return x, y