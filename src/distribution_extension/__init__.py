"""Custom Distribution classes."""

from .base import Distribution, Independent
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
    "Distribution",
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
