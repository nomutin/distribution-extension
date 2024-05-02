"""Distribution Base Classes."""

from __future__ import annotations

from typing import Any, Tuple, Union

import torch
import torch.distributions as td
from torch import Tensor

_slicelike = Union[slice, int, Tuple[Union[slice, int], ...]]


class Distribution(td.Distribution):
    """Abstract class for Custom Distribution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize."""
        super().__init__(*args, **kwargs)

    @property
    def parameters(self) -> dict[str, Tensor]:
        """Define distribion parapers as dict."""
        return {k: getattr(self, k) for k in self.arg_constraints}

    def independent(self, dim: int) -> Independent:
        """Create `Independent` that has `self` as `base_distribution`."""
        return Independent(self, dim)

    def to(self, device: torch.device) -> Distribution:
        """Convert device."""
        params = {k: v.to(device) for k, v in self.parameters.items()}
        return type(self)(**params, validate_args=self._validate_args)

    def squeeze(self, dim: int) -> Distribution:
        """Squeeze distribution."""
        params = {k: v.squeeze(dim) for k, v in self.parameters.items()}
        return type(self)(**params, validate_args=self._validate_args)

    def unsqueeze(self, dim: int) -> Distribution:
        """Unsqueeze distribution."""
        params = {k: v.unsqueeze(dim) for k, v in self.parameters.items()}
        return type(self)(**params, validate_args=self._validate_args)

    def __getitem__(self, loc: _slicelike) -> Distribution:
        """Silice distribution."""
        params = {k: v[loc] for k, v in self.parameters.items()}
        return type(self)(**params, validate_args=self._validate_args)

    def detach(self) -> Distribution:
        """Detach computational graph."""
        params = {k: v.detach() for k, v in self.parameters.items()}
        return type(self)(**params, validate_args=self._validate_args)


class Independent(td.Independent, Distribution):
    """Extension of `torch.distributions.Independent`.

    This class is not instantiated directly,
    but by `DistributionBase.independent()`.
    """

    def __init__(
        self,
        base_distribution: Distribution,
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
