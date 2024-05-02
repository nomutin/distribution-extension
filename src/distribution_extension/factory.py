"""Facrory method for custom distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as tf
from einops import pack, rearrange, unpack
from torch import Tensor, nn

from .continuous import GMM, Normal
from .discrete import (
    Categorical,
    MultiOneHot,
    OneHotCategorical,
)

if TYPE_CHECKING:
    from .base import Distribution, Independent


class GMMFactory(nn.Module):
    """Factory method for `GMM`."""

    def __init__(self, num_mixture: int) -> None:
        """Initialize."""
        super().__init__()
        self.num_mixture = num_mixture

    def forward(self, tensor: Tensor) -> GMM:
        """Build GMM from tensor."""
        # feat[batch_size*,dim*3*mix] -> feat[batch_size,dim*3*mix]
        feature, ps = pack([tensor], "* d")

        # feat[batch_size,dim*3*mix] -> π,μ,o[batch_size,dim,mix]
        weighting, mean, log_var = torch.chunk(feature, chunks=3, dim=-1)
        weighting = rearrange(
            weighting,
            "b (d n) -> b d n",
            n=self.num_mixture,
        )
        mean = rearrange(mean, "b (d n) -> b d n", n=self.num_mixture)
        log_var = rearrange(log_var, "b (d n) -> b d n", n=self.num_mixture)

        # weightings -> probs, log_var -> scale
        probs = tf.softmax(weighting, dim=-1)
        scale = torch.clamp(log_var, min=-7.0, max=7.0).exp().sqrt()

        # π,μ,o[batch_size,dim,mix] -> π,μ,o[batch_size*,dim,mix]
        probs = unpack(probs, ps, "* d n")[0]
        mean = unpack(mean, ps, "* d n")[0]
        scale = unpack(scale, ps, "* d n")[0]

        return GMM(probs=probs, loc=mean, scale=scale)


class NormalFactory(nn.Module):
    """Factory method for `Normal`."""

    def forward(self, tensor: Tensor) -> Normal:
        """Build Normal from tensor."""
        mean, scale = torch.chunk(tensor, chunks=2, dim=-1)
        scale = tf.softplus(scale) + 0.1
        return Normal(loc=mean, scale=scale)


class MultiOneHotFactory(nn.Module):
    """Factory method for `MultiDimentionalOneHotCategorical`."""

    def __init__(self, category_size: int, class_size: int) -> None:
        """Initialize."""
        super().__init__()
        self.category_size = category_size
        self.class_size = class_size

    def forward(self, tensor: Tensor) -> MultiOneHot:
        """Generate `MultiDimentionalOneHotCategorical` from tensor."""
        logit, ps = pack([tensor], "* dim")
        logit = rearrange(
            tensor=logit,
            pattern="batch (c s) -> batch c s",
            c=self.category_size,
            s=self.class_size,
        )
        logit = unpack(logit, ps, "* c s")[0]
        return MultiOneHot(logits=logit)


class OneHotCategoricalFactory(nn.Module):
    """Factory method for `OneHotCategorical`."""

    def forward(self, tensor: Tensor) -> OneHotCategorical:
        """Generate OneHotCategorical from tensor."""
        return OneHotCategorical(logits=tensor)


class CategoricalFactory(nn.Module):
    """Factory method for `Categorical`."""

    def forward(self, tensor: Tensor) -> Categorical:
        """Generate Categorical from tensor."""
        return Categorical(logits=tensor)


class IndependentFactory(nn.Module):
    """Factory method for `Independent`."""

    def __init__(self, dim: int) -> None:
        """Initialize."""
        super().__init__()
        self.dim = dim

    def forward(self, dist: Distribution) -> Independent:
        """Generate Independent from `Distribution`."""
        return dist.independent(dim=self.dim)
