"""Tests for `distribution_extension/discrete.py`,."""

import pytest
import torch
from distribution_extension.discrete import (
    Categorical,
    MultiOneHot,
    OneHotCategorical,
)
from torch import Tensor


class TestMultiOneHot:
    """Tests for `MultiOneHot`."""

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
        dist = MultiOneHot(init_tensor)
        sample = dist.rsample()
        assert sample.shape == self.sample_shape

    def test_parameters(self, init_tensor: Tensor) -> None:
        """Test `parameters`."""
        dist = MultiOneHot(init_tensor)
        assert "logits" in dist.parameters


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
        assert "logits" in dist.parameters


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
        dist = Categorical(logits=init_tensor)
        sample = dist.rsample()
        assert sample.shape == self.sample_shape

    def test_parameters(self, init_tensor: Tensor) -> None:
        """Test `parameters`."""
        dist = Categorical(init_tensor)
        assert "logits" in dist.parameters
