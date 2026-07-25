# pyright: reportGeneralTypeIssues=false, reportMissingImports=false
"""Convenience solver for one immutable CSR matrix."""

from contextlib import suppress
from dataclasses import dataclass, field

import jax.numpy as jnp

from spineax.cudss.solver import FactorToken
from spineax.cudss.solver import analyze as _analyze
from spineax.cudss.solver import factorize as _factorize
from spineax.cudss.solver import query as _query
from spineax.cudss.solver import release as _release
from spineax.cudss.solver import solve as _solve


@dataclass(frozen=True, init=False, eq=False)
class ConstantCSRCuDSSSolver:
    """Factor one immutable CSR matrix and solve repeatedly.

    Construction eagerly copies, analyzes, and factorizes the matrix. The
    solver may be captured by ``jax.jit`` and used by ``vmap``. It is
    intentionally not a pytree argument: keeping its token private and frozen
    prevents transforms from replacing the supposedly constant matrix values.
    Calls accept only right-hand sides; use the explicit token API when matrix
    values must change.

    Call :meth:`release` outside transformations to free the resident factors.
    A later call remains valid because token states self-heal from their owned
    CSR arrays.
    """

    _token: FactorToken = field(repr=False)

    def __init__(
        self,
        csr_values,
        csr_offsets,
        csr_columns,
        *,
        mtype_id: str | int = "symmetric",
        mview_id: str | int = "upper",
        device_id=0,
        reordering: str | int = "default",
        memory: str | int = "device",
    ):
        # Own the complete rebuild recipe. In particular, release/LRU healing
        # must never observe caller-controlled buffers that changed later.
        values = jnp.array(csr_values, copy=True)
        offsets = jnp.array(csr_offsets, dtype=jnp.int32, copy=True)
        columns = jnp.array(csr_columns, dtype=jnp.int32, copy=True)
        token = _analyze(
            values,
            offsets,
            columns,
            mtype_id=mtype_id,
            mview_id=mview_id,
            device_id=device_id,
            reordering=reordering,
            memory=memory,
        )
        try:
            token = _factorize(token, values)
            token.id.block_until_ready()
        except Exception:
            with suppress(Exception):
                _release(token)
            raise
        object.__setattr__(self, "_token", token)

    @property
    def n(self) -> int:
        return self._token.n

    @property
    def nnz(self) -> int:
        return self._token.nnz

    @property
    def dtype(self):
        return self._token.dtype

    @property
    def batch_size(self) -> int:
        return self._token.batch_size

    def __call__(self, rhs, *, ir_nsteps=None):
        return _solve(self._token, rhs, ir_nsteps=ir_nsteps)

    def query(self) -> dict:
        """Return cuDSS diagnostics for the resident factorization."""
        return _query(self._token)

    def release(self) -> bool:
        """Free resident factors; the next solve transparently rebuilds."""
        return _release(self._token)
