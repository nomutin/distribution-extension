"""Custom discrete distributions."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributions as td
from einops import pack, rearrange, unpack
from torch import Tensor

from .base import DistributionBase

_zero_size = torch.Size([])


class Categorical(td.OneHotCategoricalStraightThrough, DistributionBase):
    """
    Extension of `torch.distributions.OneHotCategorical`.

    This class is used to represent a 2D categorical distribution
    with arguments `class_size` and `category_size`.
    To use a 1D categorical distribution, set the argument `class_size` to 1.
    A simpler usage is to use `.OneHotCategorical`.
    """

    @classmethod
    def from_tensor(cls, tensor: Tensor, **kwargs: Any) -> Categorical:
        """Build Categorical from tensor."""
        temperature = kwargs.get("temperature", 1.0)
        category_size = kwargs["category_size"]
        class_size = kwargs["class_size"]

        logit, ps = pack([tensor], "* dim")
        logit = rearrange(
            tensor=logit,
            pattern="batch (c s) -> batch c s",
            c=category_size,
            s=class_size,
        )
        logit = unpack(logit, ps, "* c s")[0]
        probs = torch.softmax(logit / temperature, dim=-1)
        return cls(probs=probs)

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

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        **kwargs: Any,
    ) -> OneHotCategorical:
        """Create `OneHotCategorical` from given logits tensor."""
        temperature = kwargs.get("temperature", 1.0)
        probs = torch.softmax(tensor / temperature, dim=-1)
        return cls(probs=probs)

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Set `probs` (not `logits`) as parameter."""
        return {"probs": self.probs}
