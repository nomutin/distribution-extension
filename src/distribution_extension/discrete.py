"""Custom discrete distributions."""

from __future__ import annotations

import torch
import torch.distributions as td
from einops import pack, rearrange, unpack
from torch import Tensor

from .base import Distribution

_zero_size = torch.Size([])


class MultiOneHot(td.OneHotCategoricalStraightThrough, Distribution):
    """Extension of `torch.distributions.OneHotCategorical`.

    This class is used to represent a 2D categorical distribution
    with arguments `class_size` and `category_size`.
    To use a 1D categorical distribution, set the argument `class_size` to 1.
    A simpler usage is to use `.OneHotCategorical`.
    """

    def __init__(self, logits: Tensor) -> None:
        """Initialize."""
        super().__init__(logits=logits)

    def sample(self, sample_shape: torch.Size = _zero_size) -> Tensor:
        """Sample multi-dimentional categorical value."""
        return super().sample(sample_shape=sample_shape)

    def rsample(self, sample_shape: torch.Size = _zero_size) -> Tensor:
        """Sample multi-dimentional categorical value."""
        sample = super().rsample(sample_shape=sample_shape)
        sample, ps = pack([sample], "* c s")
        sample = rearrange(sample, "batch c s -> batch (c s)")
        return unpack(sample, ps, "* dim")[0]

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Set `logits`(not `probs`) as parameter."""
        return {"logits": self.logits}


class OneHotCategorical(td.OneHotCategoricalStraightThrough, Distribution):
    """Extension of `torch.distributions.OneHotCategorical`.

    This class is used to represent a 1D categorical distribution.
    """

    def __init__(self, logits: Tensor) -> None:
        """Initialize."""
        super().__init__(logits=logits)

    def sample(self, sample_shape: torch.Size = _zero_size) -> Tensor:
        """Sample categorical value."""
        return super().sample(sample_shape=sample_shape)

    def rsample(self, sample_shape: torch.Size = _zero_size) -> Tensor:
        """Sample categorical value."""
        return super().rsample(sample_shape=sample_shape)

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Set `logits` (not `probs`) as parameter."""
        return {"logits": self.logits}


class Categorical(td.Categorical, Distribution):
    """Extension of `torch.distributions.Categorical`."""

    def __init__(self, logits: Tensor) -> None:
        """Initialize."""
        super().__init__(logits=logits)

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Set `logits` (not `probs`) as parameter."""
        return {"logits": self.logits}

    def rsample(self, sample_shape: torch.Size = _zero_size) -> Tensor:
        """Sample categorical value."""
        return super().sample(sample_shape).unsqueeze(-1)

    def sample(self, sample_shape: torch.Size = _zero_size) -> Tensor:
        """Sample categorical value."""
        return super().sample(sample_shape).unsqueeze(-1)

    def log_prob(self, value: Tensor) -> Tensor:
        """Calculate log probability of categorical value."""
        return super().log_prob(value.squeeze(-1))
