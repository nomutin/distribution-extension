"""Custom Distribution classes."""

from .base import DistributionBase, Independent
from .continuous import GMM, Normal
from .discrete import (
    Categorical,
    MultiDimentionalOneHotCategorical,
    OneHotCategorical,
)
from .factory import (
    CategoricalFactory,
    GMMFactory,
    IndependentFactory,
    MultiDimentionalOneHotCategoricalFactory,
    NormalFactory,
    OneHotCategoricalFactory,
)
from .kl import kl_divergence

__all__ = [
    "DistributionBase",
    "Independent",
    "GMM",
    "Normal",
    "Categorical",
    "MultiDimentionalOneHotCategorical",
    "OneHotCategorical",
    "kl_divergence",
    "MultiDimentionalOneHotCategoricalFactory",
    "GMMFactory",
    "IndependentFactory",
    "NormalFactory",
    "OneHotCategoricalFactory",
    "CategoricalFactory",
]
