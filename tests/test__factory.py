"""Tests for `distribution_extension/factory.py`."""

import pytest
import torch
from distribution_extension.base import Independent
from distribution_extension.factory import (
    CategoricalFactory,
    GMMFactory,
    IndependentFactory,
    MultiOneHotFactory,
    NormalFactory,
    OneHotCategoricalFactory,
)
from torch import Tensor


class TestGMMFactory:
    """Tests for `GMMFactory`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        batch_size = 8
        seq_len = 16
        dim = 4
        self.num_mix = 3
        self.sample_shape = torch.Size([batch_size, seq_len, dim])
        return torch.rand([batch_size, seq_len, dim * self.num_mix * 3])

    def test_forward(self, init_tensor: Tensor) -> None:
        """Test `forward()`."""
        factory = GMMFactory(num_mixture=self.num_mix)
        dist = factory(init_tensor)
        assert dist.rsample().shape == self.sample_shape


class TestNormalFactory:
    """Tests for `NormalFactory`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        batch_size = 8
        seq_len = 16
        dim = 4
        self.sample_shape = torch.Size([batch_size, seq_len, dim])
        return torch.rand([batch_size, seq_len, dim * 2])

    def test_forward(self, init_tensor: Tensor) -> None:
        """Test `forward()`."""
        factory = NormalFactory()
        dist = factory(init_tensor)
        assert dist.rsample().shape == self.sample_shape


class TestMultiOneHotFactory:
    """Tests for `MultiOneHotFactory`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        batch_size = 8
        seq_len = 16
        self.class_size = 4
        self.category_size = 3
        self.state_size = self.class_size * self.category_size
        self.sample_shape = torch.Size([batch_size, seq_len, self.state_size])
        return torch.rand([batch_size, seq_len, self.state_size])

    def test_forward(self, init_tensor: Tensor) -> None:
        """Test `forward()`."""
        factory = MultiOneHotFactory(
            category_size=self.category_size,
            class_size=self.class_size,
        )
        dist = factory(init_tensor)
        assert dist.rsample().shape == self.sample_shape


class TestOneHotCategoricalFactory:
    """Tests for `OneHotCategoricalFactory`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        batch_size = 8
        seq_len = 16
        self.class_size = 4
        self.sample_shape = torch.Size([batch_size, seq_len, self.class_size])
        return torch.rand([batch_size, seq_len, self.class_size])

    def test_forward(self, init_tensor: Tensor) -> None:
        """Test `forward()`."""
        factory = OneHotCategoricalFactory()
        dist = factory(init_tensor)
        assert dist.rsample().shape == self.sample_shape


class TestCategoricalFactory:
    """Tests for `CategoricalFactory`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        batch_size = 8
        seq_len = 16
        self.sample_shape = torch.Size([batch_size, seq_len, 1])
        zeros = torch.zeros([batch_size, seq_len, 3]) * -10
        ones = torch.ones([batch_size, seq_len, 1]) * 10
        return torch.cat([zeros, ones], dim=-1)

    def test_forward(self, init_tensor: Tensor) -> None:
        """Test `forward()`."""
        factory = CategoricalFactory()
        dist = factory(init_tensor)
        sample = dist.sample()
        assert sample.shape == self.sample_shape
        assert torch.equal(sample, torch.ones_like(sample) * 3)


class TestIndependentFactory:
    """Tests for `IndependentFactory`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        self.batch_size = 8
        self.seq_len = 16
        self.dim = 4
        self.batch_shape = torch.Size([self.batch_size, self.seq_len])
        self.event_shape = torch.Size([self.dim])
        return torch.rand([self.batch_size, self.seq_len, self.dim * 2])

    def test_forward(self, init_tensor: Tensor) -> None:
        """Test `forward()`."""
        normal_dist = NormalFactory()(init_tensor)
        dist = IndependentFactory(dim=1)(normal_dist)
        assert isinstance(dist, Independent)
        assert dist.batch_shape == self.batch_shape
        assert dist.event_shape == self.event_shape
