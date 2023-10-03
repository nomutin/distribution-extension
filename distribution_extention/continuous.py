"""Custom continuous distributions."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributions as td
import torch.nn.functional as tf
from einops import pack, rearrange, unpack
from torch import Tensor
from torch.distributions import constraints

from .base import DistributionBase

_zero_size = torch.Size([])


class Normal(td.Normal, DistributionBase):
    """Extension of `torch.distributions.Normal` ."""

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        **kwargs: dict,  # noqa: ARG003
    ) -> Normal:
        """Build Normal from tensor."""
        mean, scale = torch.chunk(tensor, chunks=2, dim=-1)
        scale = tf.softplus(scale) + 0.1
        return cls(loc=mean, scale=scale)

    def kl_divergence_starndard_normal(self) -> Tensor:
        """Calculate KL divergence between self and standard normal."""
        kld = 1 + self.variance.log() - self.mean.pow(2) - self.variance
        return kld.sum().mul(-0.5)

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Define `loc` and `scale` as parameters of self."""
        return {"loc": self.loc, "scale": self.scale}


class GMM(DistributionBase, td.Distribution):
    """
    Gaussian Mixture Model Implementation.

    References
    ----------
    * https://geostatisticslessons.com/lessons/gmm
    """

    def __init__(
        self,
        probs: Tensor,
        loc: Tensor,
        scale: Tensor,
        validate_args: None | bool = None,
    ) -> None:
        """
        Gaussian Mixture Model.

        `probs`, `loc`, `scale` must have the same shape:
        `[batch_size*, dim, num_mix]`.
        """
        self.probs = probs
        self.loc = loc
        self.scale = scale
        super().__init__(validate_args=validate_args)
        self.num_mixture = probs.shape[-1]
        self.dist = self._mixture_of_gaussian()
        self.has_rsample = True

    @property
    def mean(self) -> Tensor:
        """Mean of GMM(Same as `self.loc`)."""
        return self.loc

    @property
    def arg_constraints(self) -> dict[str, constraints.Constraint]:
        """Define mathmatical constraints of each argments."""
        return {
            "probs": constraints.simplex,
            "loc": constraints.real,
            "scale": constraints.positive,
        }

    def _mixture_of_gaussian(self) -> td.Independent:
        """Create Mixture of Gaussian Distribution."""
        gaussian = td.Normal(self.loc, self.scale)
        categorical_dist = td.Categorical(probs=self.probs)
        mixture_dist = td.MixtureSameFamily(
            mixture_distribution=categorical_dist,
            component_distribution=gaussian,
        )
        return td.Independent(
            base_distribution=mixture_dist,
            reinterpreted_batch_ndims=1,
        )

    def sample(self, _: torch.Size = _zero_size) -> Tensor:
        """Sample from GMM."""
        return self.dist.sample()

    def rsample(self, _: torch.Size = _zero_size) -> Tensor:
        """Sample from GMM."""
        return self.dist.sample()

    def log_prob(self, value: Tensor) -> Tensor:
        """Calculate log-likelihood."""
        return self.dist.log_prob(value)

    @classmethod
    def from_tensor(cls, tensor: Tensor, **kwargs: Any) -> GMM:
        """Build GMM from feature."""
        num_mixture = kwargs["num_mixture"]

        # feat[batch_size*,dim*3*mix] -> feat[batch_size,dim*3*mix]
        feature, ps = pack([tensor], "* d")

        # feat[batch_size,dim*3*mix] -> π,μ,o[batch_size,dim,mix]
        weighting, mean, log_var = torch.chunk(feature, chunks=3, dim=-1)
        weighting = rearrange(weighting, "b (d n) -> b d n", n=num_mixture)
        mean = rearrange(mean, "b (d n) -> b d n", n=num_mixture)
        log_var = rearrange(log_var, "b (d n) -> b d n", n=num_mixture)

        # weightings -> probs, log_var -> scale
        probs = tf.softmax(weighting, dim=-1)
        scale = torch.clamp(log_var, min=-7.0, max=7.0).exp().sqrt()

        # π,μ,o[batch_size,dim,mix] -> π,μ,o[batch_size*,dim,mix]
        probs = unpack(probs, ps, "* d n")[0]
        mean = unpack(mean, ps, "* d n")[0]
        scale = unpack(scale, ps, "* d n")[0]

        return cls(probs=probs, loc=mean, scale=scale)
