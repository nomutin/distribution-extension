"""Tests for `distribution_extension/utils.py`."""

import torch
from distribution_extension.continuous import Normal
from distribution_extension.utils import cat_distribution, stack_distribution


def test__cat_distribution() -> None:
    """Test `cat()`."""
    batch_size, dim = 8, 16
    num_dist = 4
    dist_list = [
        Normal(
            loc=torch.zeros([batch_size, dim]),
            scale=torch.zeros([batch_size, dim]).add(1e-7),
        )
        for _ in range(num_dist)
    ]
    cat_dist = cat_distribution(dist_list, dim=1)
    result = torch.testing.assert_close(
        actual=cat_dist.rsample(),
        expected=torch.zeros([batch_size, dim * num_dist]),
        atol=1e-5,
        rtol=1,
    )
    assert isinstance(cat_dist, Normal)
    assert result is None


def test__stack_distribution() -> None:
    """Test `stack_distribution()`."""
    batch_size, dim = 8, 16
    num_dist = 4
    dist_list = [
        Normal(
            loc=torch.zeros([batch_size, dim]),
            scale=torch.zeros([batch_size, dim]).add(1e-7),
        )
        for _ in range(num_dist)
    ]
    stack_dist = stack_distribution(dist_list, dim=1)
    result = torch.testing.assert_close(
        actual=stack_dist.rsample(),
        expected=torch.zeros([batch_size, num_dist, dim]),
        atol=1e-5,
        rtol=1,
    )
    assert isinstance(stack_dist, Normal)
    assert result is None
