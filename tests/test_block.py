"""Tests for the DenseBlock neural network component."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from src.models import DenseBlock


@pytest.mark.unit
def test_dense_block_output_shape():
    """Test that DenseBlock produces correct output dimensions."""
    block = DenseBlock(input_dim=10, output_dim=5)
    x = torch.randn(32, 10)
    output = block(x)
    
    assert output.shape == (32, 5)


@pytest.mark.unit
def test_dense_block_with_dropout():
    """Test DenseBlock with dropout enabled."""
    block = DenseBlock(input_dim=10, output_dim=5, dropout=0.5)
    block.train()
    x = torch.randn(32, 10)
    output = block(x)
    
    assert output.shape == (32, 5)


@pytest.mark.property
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    input_dim=st.integers(min_value=1, max_value=128),
    output_dim=st.integers(min_value=1, max_value=128)
)
def test_dense_block_shape_invariance(batch_size, input_dim, output_dim):
    """Property-based test: output shape should always match expected dimensions."""
    block = DenseBlock(input_dim=input_dim, output_dim=output_dim)
    x = torch.randn(batch_size, input_dim)
    output = block(x)
    
    assert output.shape == (batch_size, output_dim)