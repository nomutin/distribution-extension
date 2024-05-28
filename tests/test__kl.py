"""Tests for `distribution_extension/kl.py`."""

import torch
from distribution_extension import GMMFactory, NormalFactory
from distribution_extension.kl import mc_kl_divergence


def test_mc_kl_divergence() -> None:
    """Test for `mc_kl_divergence`."""
    batch_size, seq_len, dim, num_mixture = 32, 16, 8, 5
    normal_source = torch.rand([batch_size, seq_len, dim * 2])
    normal = NormalFactory().forward(normal_source)
    gmm_source = torch.rand([batch_size, seq_len, dim * num_mixture * 3])
    gmm = GMMFactory(num_mixture=num_mixture).forward(gmm_source)
    kl = mc_kl_divergence(q=normal, p=gmm)
    assert kl.shape == torch.Size([])
