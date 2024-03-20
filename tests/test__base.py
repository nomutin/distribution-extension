"""Tests for `distribution_extension/base.py`."""

import pytest
import torch

from distribution_extension.base import Distribution, Independent
from distribution_extension.continuous import Normal


class TestDistribution:
    """Tests for `Distribution`."""

    def test_parameters(self) -> None:
        """Test `DistributionBase.parameters()`."""
        dist = Distribution()
        with pytest.raises(NotImplementedError) as e:
            _ = dist.parameters
        assert e.type == NotImplementedError


class TestIndependent:
    """Tests for `Independent`."""

    @pytest.fixture()
    def init_dist(self) -> Normal:
        """Initialize distribution."""
        self.batch = 1
        self.seq_len = 32
        self.dim = 4
        return Normal(
            loc=torch.zeros(self.batch, self.seq_len, self.dim),
            scale=torch.ones(self.batch, self.seq_len, self.dim),
        )

    def test_init(self, init_dist: Normal) -> None:
        """Test `Independent.__init__()`."""
        reinterpreted_batch_ndims = 1
        dist = Independent(
            base_distribution=init_dist,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        assert dist.reinterpreted_batch_ndims == reinterpreted_batch_ndims

    def test_detach(self, init_dist: Normal) -> None:
        """Test `Independent.detach()`."""
        reinterpreted_batch_ndims = 1
        dist = Independent(
            base_distribution=init_dist,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        detached_dist = dist.detach()
        assert isinstance(detached_dist, Independent)
        assert detached_dist.mean.requires_grad is False
        assert detached_dist.stddev.requires_grad is False

    def test_to(self, init_dist: Normal) -> None:
        """Test `Independent.to()`."""
        reinterpreted_batch_ndims = 1
        dist = Independent(
            base_distribution=init_dist,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        device = torch.device("cpu")
        dist = dist.to(device)
        assert dist.mean.device == device
        assert dist.stddev.device == device

    def test__getitem__(self, init_dist: Normal) -> None:
        """Test `Independent.__getitem__()`."""
        reinterpreted_batch_ndims = 1
        dist = Independent(
            base_distribution=init_dist,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )[:, 0, :]
        sample = dist.sample()
        sample_shape = torch.Size([self.batch, self.dim])
        assert sample.shape == sample_shape

    def test_squeeze(self, init_dist: Normal) -> None:
        """Test `Independent.squeeze()`."""
        reinterpreted_batch_ndims = 1
        dist = Independent(
            base_distribution=init_dist,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        dist = dist.squeeze(0)
        expected = torch.Size([self.seq_len, self.dim])
        assert dist.sample().shape == expected

    def test_unsqueeze(self, init_dist: Normal) -> None:
        """Test `Independent.unsqueeze()`."""
        reinterpreted_batch_ndims = 1
        dist = Independent(
            base_distribution=init_dist,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
        )
        dist = dist.unsqueeze(0)
        expected = torch.Size([1, self.batch, self.seq_len, self.dim])
        assert dist.sample().shape == expected
