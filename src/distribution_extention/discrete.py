"""Custom discrete distributions."""

from __future__ import annotations

import torch
import torch.distributions as td
from einops import pack, rearrange, unpack
from torch import Tensor

from .base import DistributionBase

_zero_size = torch.Size([])


class MultiDimentionalOneHotCategorical(
    td.OneHotCategoricalStraightThrough,
    DistributionBase,
):
    """
    Extension of `torch.distributions.OneHotCategorical`.

    This class is used to represent a 2D categorical distribution
    with arguments `class_size` and `category_size`.
    To use a 1D categorical distribution, set the argument `class_size` to 1.
    A simpler usage is to use `.OneHotCategorical`.
    """

    def rsample(self, sample_shape: torch.Size = _zero_size) -> Tensor:
        """Sample multi-dimentional categorical value."""
        sample = super().rsample(sample_shape=sample_shape)
        sample, ps = pack([sample], "* c s")
        sample = rearrange(sample, "batch c s -> batch (c s)")
        return unpack(sample, ps, "* dim")[0]

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Set `probs`(not `logits`) as parameter."""
        return {"probs": self.probs}


class OneHotCategorical(td.OneHotCategoricalStraightThrough, DistributionBase):
    """
    Extension of `torch.distributions.OneHotCategorical`.

    This class is used to represent a 1D categorical distribution.
    """

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Set `probs` (not `logits`) as parameter."""
        return {"probs": self.probs}


class Categorical(td.Categorical, DistributionBase):
    """Extension of `torch.distributions.Categorical`."""

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Set `probs` (not `logits`) as parameter."""
        return {"probs": self.probs}

    def rsample(self, sample_shape: torch.Size = _zero_size) -> Tensor:
        """Sample categorical value."""
        return super().sample(sample_shape)
