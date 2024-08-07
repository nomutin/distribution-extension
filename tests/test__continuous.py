"""Tests for `distribution_extension/continuous.py`."""

import pytest
import torch
from distribution_extension.continuous import GMM, Normal
from torch import Tensor


class TestNormal:
    """Tests for `Normal`."""

    @pytest.fixture
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

    def test_rsample(self, init_tensor: Tensor) -> None:
        """Test `rsample()`."""
        loc, scale = torch.chunk(init_tensor, 2, dim=-1)
        dist = Normal(loc, scale)
        sample = dist.rsample()
        expected_shape = torch.Size([self.batch_size, self.seq_len, self.dim])
        assert sample.shape == expected_shape

    def test_unsqueeze(self, init_tensor: Tensor) -> None:
        """Test `unsqueeze()`."""
        loc, scale = torch.chunk(init_tensor, 2, dim=-1)
        dist = Normal(loc, scale)
        unsqueezed = dist.unsqueeze(1)
        sample = unsqueezed.rsample()
        expected_shape = torch.Size([8, 1, 16, 4])
        assert sample.shape == expected_shape

    def test_detach(self, init_tensor: Tensor) -> None:
        """Test `detach()`."""
        loc, scale = torch.chunk(init_tensor, 2, dim=-1)
        dist = Normal(loc, scale)
        detached = dist.detach()
        sample = detached.rsample()
        assert sample.requires_grad is False

    def test_clone(self, init_tensor: Tensor) -> None:
        """Test `clone()`."""
        loc, scale = torch.chunk(init_tensor, 2, dim=-1)
        dist = Normal(loc, scale)
        cloned = dist.clone()
        # 1. Parameter
        assert torch.equal(dist.mean, cloned.mean)
        # 2. Memory Independency
        assert dist.mean.data_ptr() != cloned.mean.data_ptr()
        # 3. Parameter Independency
        cloned.mean[0, 0, 0] = 100
        assert not torch.equal(dist.mean, cloned.mean)


class TestGMM:
    """Tests for `GMM`."""

    @pytest.fixture
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
