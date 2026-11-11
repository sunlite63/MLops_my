"""Tests for the package initialization module."""

import pytest

from src import calculate_sum


@pytest.mark.unit
def test_calculate_sum_integers():
    """Test sum calculation with integers."""
    assert calculate_sum(2, 3) == 5
    assert calculate_sum(-1, 1) == 0
    assert calculate_sum(0, 0) == 0


@pytest.mark.unit
def test_calculate_sum_floats():
    """Test sum calculation with floats."""
    assert calculate_sum(1.5, 2.5) == 4.0
    assert calculate_sum(-0.5, 0.5) == 0.0