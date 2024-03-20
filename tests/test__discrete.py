"""Tests for `distribution_extension/discrete.py`,."""

import pytest
import torch
from torch import Tensor

from distribution_extension.discrete import (
    Categorical,
    MultiDimentionalOneHotCategorical,
    OneHotCategorical,
)


class TestMultiDimentionalOneHotCategorical:
    """Tests for `Categorical`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        batch_size = 8
        seq_len = 16
        class_size = 4
        category_size = 3
        self.sample_shape = torch.Size(
            [batch_size, seq_len, class_size * category_size],
        )
        return torch.rand([batch_size, seq_len, class_size, category_size])

    def test_rsample(self, init_tensor: Tensor) -> None:
        """Test `rsample()`."""
        dist = MultiDimentionalOneHotCategorical(init_tensor)
        sample = dist.rsample()
        assert sample.shape == self.sample_shape

    def test_parameters(self, init_tensor: Tensor) -> None:
        """Test `parameters`."""
        dist = MultiDimentionalOneHotCategorical(init_tensor)
        assert "probs" in dist.parameters


class TestOneHotCategorical:
    """Tests for `OneHotCategorical`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        batch_size = 8
        seq_len = 16
        class_size = 4
        self.sample_shape = torch.Size([batch_size, seq_len, class_size])
        return torch.rand([batch_size, seq_len, class_size])

    def test_rsample(self, init_tensor: Tensor) -> None:
        """Test `rsample()`."""
        dist = OneHotCategorical(init_tensor)
        sample = dist.rsample()
        assert sample.shape == self.sample_shape

    def test_parameters(self, init_tensor: Tensor) -> None:
        """Test `parameters`."""
        dist = OneHotCategorical(init_tensor)
        assert "probs" in dist.parameters


class TestCategorical:
    """Tests for `Categorical`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        batch_size = 8
        seq_len = 16
        self.sample_shape = torch.Size([batch_size, seq_len, 1])
        zeros = torch.zeros([batch_size, seq_len, 3])
        ones = torch.ones([batch_size, seq_len, 1])
        return torch.cat([zeros, ones], dim=-1)

    def test_rsample(self, init_tensor: Tensor) -> None:
        """Test `rsample()`."""
        dist = Categorical(probs=init_tensor)
        sample = dist.rsample()
        assert sample.shape == self.sample_shape
        assert torch.equal(sample, torch.ones_like(sample) * 3)

    def test_parameters(self, init_tensor: Tensor) -> None:
        """Test `parameters`."""
        dist = Categorical(init_tensor)
        assert "probs" in dist.parameters
