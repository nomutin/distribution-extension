"""Custom continuous distributions."""

from __future__ import annotations

import torch
import torch.distributions as td
from torch import Tensor
from torch.distributions import constraints

from .base import Distribution

_zero_size = torch.Size([])


class Normal(td.Normal, Distribution):
    """Extension of `torch.distributions.Normal` ."""

    def kl_divergence_starndard_normal(self) -> Tensor:
        """Calculate KL divergence between self and standard normal."""
        kld = 1 + self.variance.log() - self.mean.pow(2) - self.variance
        return kld.sum().mul(-0.5)


class GMM(Distribution, td.Distribution):
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
