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

    def test_unsqueeze(self, init_tensor: Tensor) -> None:
        """Test `unsqueeze()`."""
        dist = MultiOneHot(init_tensor)
        unsqueezed = dist.unsqueeze(1)
        sample = unsqueezed.rsample()
        assert sample.shape == torch.Size([8, 1, 16, 4, 3])

    def test__detach(self, init_tensor: Tensor) -> None:
        """Test `detach()`."""
        dist = MultiOneHot(init_tensor)
        detached = dist.detach()
        sample = detached.rsample()
        assert sample.requires_grad is False


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

    def test_unsqueeze(self, init_tensor: Tensor) -> None:
        """Test `unsqueeze()`."""
        dist = OneHotCategorical(init_tensor)
        unsqueezed = dist.unsqueeze(1)
        sample = unsqueezed.rsample()
        assert sample.shape == torch.Size([8, 1, 16, 4])

    def test__detach(self, init_tensor: Tensor) -> None:
        """Test `detach()`."""
        dist = OneHotCategorical(init_tensor)
        detached = dist.detach()
        sample = detached.rsample()
        assert sample.requires_grad is False


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

    def test_unsqueeze(self, init_tensor: Tensor) -> None:
        """Test `unsqueeze()`."""
        dist = Categorical(init_tensor)
        unsqueezed = dist.unsqueeze(1)
        sample = unsqueezed.rsample()
        assert sample.shape == torch.Size([8, 1, 16, 1])

    def test_detach(self, init_tensor: Tensor) -> None:
        """Test `detach()`."""
        dist = Categorical(init_tensor)
        detached = dist.detach()
        sample = detached.rsample()
        assert sample.requires_grad is False
