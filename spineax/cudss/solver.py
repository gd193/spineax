# pyright: reportMissingImports=false
"""Public compatibility facade for the modular cuDSS implementation."""

from ._api import (
    analyze,
    cache_capacity,
    factorize,
    inertia,
    query,
    rebuild_count,
    refactorize,
    registry_size,
    release,
    solve,
)
from ._core import FactorToken
from ._lineax import CSROperator, CuDSS
from ._transpose import _transpose_csr as _transpose_csr

# Preserve historical class paths for introspection and pickling.
FactorToken.__module__ = __name__
CSROperator.__module__ = __name__
CuDSS.__module__ = __name__

__all__ = [
    "CSROperator",
    "CuDSS",
    "FactorToken",
    "analyze",
    "cache_capacity",
    "factorize",
    "inertia",
    "query",
    "rebuild_count",
    "refactorize",
    "registry_size",
    "release",
    "solve",
]
