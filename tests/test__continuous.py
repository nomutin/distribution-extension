"""Tests for `distribution_extention/continuous.py`."""


import pytest
import torch
from torch import Tensor

from distribution_extention.continuous import GMM, Normal


class TestNormal:
    """Tests for `Normal`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        self.batch_size = 8
        self.seq_len = 16
        self.dim = 4
        return torch.rand([self.batch_size, self.seq_len, self.dim * 2])

    def test_kl_divergence_starndard_normal(self, init_tensor: Tensor) -> None:
        """Test `kl_divergence_standard_normal`."""
        loc, scale = torch.chunk(init_tensor, 2, dim=-1)
        dist = Normal(loc, scale)
        kld = dist.kl_divergence_starndard_normal()
        assert kld.shape == torch.Size([])

    def test_parameters(self, init_tensor: Tensor) -> None:
        """Test `parameters`."""
        loc, scale = torch.chunk(init_tensor, 2, dim=-1)
        dist = Normal(loc, scale)
        assert "loc" in dist.parameters
        assert "scale" in dist.parameters


class TestGMM:
    """Tests for `GMM`."""

    @pytest.fixture()
    def init_tensor(self) -> Tensor:
        """Initialize tensor."""
        batch_size = 8
        seq_len = 16
        dim = 4
        num_mix = 3
        self.sample_shape = torch.Size([batch_size, seq_len, dim])
        return torch.rand([batch_size, seq_len, dim, num_mix * 3])

    def test_mean(self, init_tensor: Tensor) -> None:
        """Test `mean`."""
        probs, loc, scale = torch.chunk(init_tensor, 3, dim=-1)
        probs = torch.softmax(probs, dim=-1)
        dist = GMM(probs, loc, scale)
        assert torch.equal(dist.mean, loc)

    def test_sample(self, init_tensor: Tensor) -> None:
        """Test `sample()`."""
        probs, loc, scale = torch.chunk(init_tensor, 3, dim=-1)
        probs = torch.softmax(probs, dim=-1)
        dist = GMM(probs, loc, scale)
        sample = dist.sample()
        assert sample.shape == self.sample_shape

    def test_rsample(self, init_tensor: Tensor) -> None:
        """Test `rsample()`."""
        probs, loc, scale = torch.chunk(init_tensor, 3, dim=-1)
        probs = torch.softmax(probs, dim=-1)
        dist = GMM(probs, loc, scale)
        sample = dist.rsample()
        assert sample.shape == self.sample_shape

    def test_log_prob(self, init_tensor: Tensor) -> None:
        """Test `log_prob()`."""
        probs, loc, scale = torch.chunk(init_tensor, 3, dim=-1)
        probs = torch.softmax(probs, dim=-1)
        dist = GMM(probs, loc, scale)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == self.sample_shape[:-1]
