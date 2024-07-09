"""Custom Distribution classes."""

from .base import Distribution, Independent
from .continuous import GMM, Normal
from .discrete import (
    BernoulliStraightThrough,
    Categorical,
    MultiOneHot,
    OneHotCategorical,
)
from .factory import (
    CategoricalFactory,
    GMMFactory,
    IndependentFactory,
    MultiOneHotFactory,
    NormalFactory,
    OneHotCategoricalFactory,
)
from .kl import kl_divergence

__all__ = [
    "GMM",
    "BernoulliStraightThrough",
    "Categorical",
    "CategoricalFactory",
    "Distribution",
    "GMMFactory",
    "Independent",
    "IndependentFactory",
    "MultiOneHot",
    "MultiOneHotFactory",
    "Normal",
    "NormalFactory",
    "OneHotCategorical",
    "OneHotCategoricalFactory",
    "kl_divergence",
]
