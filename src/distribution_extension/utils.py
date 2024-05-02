"""Utility functions for distribution_extension."""

from typing import Sequence, TypeVar

import torch

from distribution_extension.base import Distribution

T = TypeVar("T", bound=Distribution)


def stack(distribution_list: Sequence[T], dim: int) -> T:
    """Stack distributions along given dimentions."""
    parameters = {}
    for parameter_name in distribution_list[0].parameters:
        params = [getattr(d, parameter_name) for d in distribution_list]
        parameters[parameter_name] = torch.stack(params, dim=dim)
    return distribution_list[0].__class__(**parameters)


def cat(distribution_list: Sequence[T], dim: int) -> T:
    """Concatenate distributions along given dimentions."""
    parameters = {}
    for parameter_name in distribution_list[0].parameters:
        params = [getattr(d, parameter_name) for d in distribution_list]
        parameters[parameter_name] = torch.cat(params, dim=dim)
    return distribution_list[0].__class__(**parameters)
