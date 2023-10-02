"""Distribution Base Classes."""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.distributions as td
from torch import Tensor

_slicelike = Union[slice, int, Tuple[Union[slice, int], ...]]


class Independent(td.Independent):
    """
    Extension of `torch.distributions.Independent`.

    This class is not instantiated directly,
    but by `DistributionBase.independent()`.
    """

    def __init__(
        self,
        base_distribution: DistributionBase,
        reinterpreted_batch_ndims: int,
    ) -> None:
        """Initialize."""
        super().__init__(base_distribution, reinterpreted_batch_ndims)

    def detach(self) -> Independent:
        """Detach computational graph."""
        return Independent(
            base_distribution=self.base_dist.detach(),
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
        )

    def to(self, device: torch.device) -> Independent:
        """Convert device."""
        return Independent(
            base_distribution=self.base_dist.to(device),
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
        )

    def __getitem__(self, loc: _slicelike) -> Independent:
        """Slice distribution."""
        return Independent(
            base_distribution=self.base_dist[loc],
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
        )

    def squeeze(self, dim: int) -> Independent:
        """Squeeze distribution."""
        return Independent(
            base_distribution=self.base_dist.squeeze(dim),
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
        )

    def unsqueeze(self, dim: int) -> Independent:
        """Unsqueeze distribution."""
        return Independent(
            base_distribution=self.base_dist.unsqueeze(dim),
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
        )


class DistributionBase(td.Distribution):
    """Abstract class for Custom Distribution."""

    def __init__(self) -> None:
        super().__init__(validate_args=self._validate_args)

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Define distribion parapers as dict."""
        raise NotImplementedError

    def independent(self, dim: int) -> Independent:
        """Create `Independent` that has `self` as `base_distribution`."""
        return Independent(self, dim)

    def to(self, device: torch.device) -> DistributionBase:
        """Convert device."""
        params = {k: v.to(device) for k, v in self.parameters.items()}
        return type(self)(**params)

    def squeeze(self, dim: int) -> DistributionBase:
        """Squeeze distribution."""
        params = {k: v.squeeze(dim) for k, v in self.parameters.items()}
        return type(self)(**params)

    def unsqueeze(self, dim: int) -> DistributionBase:
        """Unsqueeze distribution."""
        params = {k: v.unsqueeze(dim) for k, v in self.parameters.items()}
        return type(self)(**params)

    def __getitem__(self, loc: _slicelike) -> DistributionBase:
        """Silice distribution."""
        params = {k: v[loc] for k, v in self.parameters.items()}
        return type(self)(**params)

    def detach(self) -> DistributionBase:
        """Detach computational graph."""
        params = {k: v.detach() for k, v in self.parameters.items()}
        return type(self)(**params)
