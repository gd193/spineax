import hashlib
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from . import _batching, _dispatch, _primitives, _validation

# Public and compatibility aliases. These reference the canonical objects in
# the implementation modules; no replacement primitives are constructed here.
_solve = _dispatch._solve
solve = _dispatch.solve
batch_solve = _dispatch.batch_solve
pbatch_solve = _dispatch.pbatch_solve
PBATCH_AVAILABLE = _primitives.PBATCH_AVAILABLE
vmap_using_pseudo_batch = PBATCH_AVAILABLE

KIND_SINGLE = _primitives.KIND_SINGLE
KIND_BATCH = _primitives.KIND_BATCH
KIND_PBATCH = _primitives.KIND_PBATCH
KIND_SINGLE_CONST = _primitives.KIND_SINGLE_CONST
KIND_MULTI_RHS_CONST = _primitives.KIND_MULTI_RHS_CONST
_PRIMITIVES = _primitives._PRIMITIVES
_primitive_name = _primitives._primitive_name
_primitive_for = _primitives._primitive_for
_batched_primitive_for = _primitives._batched_primitive_for
_const_xonly_solver_for_dtype = _primitives._const_xonly_solver_for_dtype
compute_inertia_from_diag_perm = _primitives.compute_inertia_from_diag_perm
register_ffi = _primitives.register_ffi
solve_aval = _primitives.solve_aval
solve_xonly_aval = _primitives.solve_xonly_aval
solve_batch_aval = _primitives.solve_batch_aval
solve_batch_xonly_aval = _primitives.solve_batch_xonly_aval
_DTYPE_SUFFIXES = _validation._DTYPE_SUFFIXES
_DTYPE_BY_SUFFIX = _validation._DTYPE_BY_SUFFIX
_dtype_suffix = _validation._dtype_suffix
_cudss_debug_enabled = _validation._cudss_debug_enabled
_log_dtype = _validation._log_dtype
_validate_solver_configuration = _validation._validate_solver_configuration
_validate_return_diagnostics = _validation._validate_return_diagnostics
_validate_values_and_rhs = _validation._validate_values_and_rhs

for _name, _primitive in _PRIMITIVES.items():
    globals()[f"{_name}_p"] = _primitive

if not PBATCH_AVAILABLE:
    warnings.warn(
        "pbatch_solve not available (CUDA version mismatch?). "
        "Falling back to batch_solve. Batched inertia will not be computed correctly.",
        RuntimeWarning,
    )


def general_solve_vmap(vector_arg_values, batch_axes, **kwargs):
    return _batching.general_solve_vmap(
        vector_arg_values, batch_axes, prefer_pbatch=vmap_using_pseudo_batch, **kwargs
    )


def solve_batch_vmap(vector_arg_values, batch_axes, **kwargs):
    return _batching.solve_batch_vmap(
        vector_arg_values, batch_axes, prefer_pbatch=vmap_using_pseudo_batch, **kwargs
    )


def solve_batch_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return solve_batch_vmap(
        vector_arg_values, batch_axes, return_diagnostics=False, **kwargs
    )


def solve_single_f32_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)


def solve_single_f64_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)


def solve_single_c64_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)


def solve_single_c128_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)


def solve_single_f32_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(
        vector_arg_values, batch_axes, return_diagnostics=False, **kwargs
    )


def solve_single_f64_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(
        vector_arg_values, batch_axes, return_diagnostics=False, **kwargs
    )


def solve_single_c64_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(
        vector_arg_values, batch_axes, return_diagnostics=False, **kwargs
    )


def solve_single_c128_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(
        vector_arg_values, batch_axes, return_diagnostics=False, **kwargs
    )


solve_single_const_xonly_vmap = _batching.solve_single_const_xonly_vmap
solve_multi_rhs_const_xonly_vmap = _batching.solve_multi_rhs_const_xonly_vmap

_batching.install_batching_rules(
    general_solve_vmap,
    lambda values, axes, **kwargs: general_solve_vmap(
        values, axes, return_diagnostics=False, **kwargs
    ),
    solve_batch_vmap,
    solve_batch_xonly_vmap,
)


class CuDSSSolver(eqx.Module):
    """Sparse linear solver wrapper with dynamic CSR values.

    ``__call__(b, csr_values)`` always uses the CSR values supplied for that
    call.  Use :class:`ConstantCSRCuDSSSolver` when the matrix values are known
    to be fixed for the lifetime of the compiled executable; that API removes
    ``csr_values`` from the call signature so changing values cannot be passed
    accidentally and silently ignored by the constant cuDSS state.
    """

    csr_offsets: Array = eqx.field(static=True)
    csr_columns: Array = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)
    return_diagnostics: bool = eqx.field(static=True)
    _n: int = eqx.field(static=True)
    _nnz: int = eqx.field(static=True)

    def __init__(
        self,
        csr_offsets,
        csr_columns,
        device_id,
        mtype_id,
        mview_id,
        return_diagnostics: bool = True,
    ):
        self._n, self._nnz = _validate_solver_configuration(
            csr_offsets, csr_columns, device_id, mtype_id, mview_id
        )
        self.csr_offsets = csr_offsets
        self.csr_columns = csr_columns
        try:
            self.device_id = int(device_id)
            self.mtype_id = int(mtype_id)
            self.mview_id = int(mview_id)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as error:  # validated above; defensive normalization
            raise TypeError(
                "cuDSS IDs must be representable as Python integers"
            ) from error
        self.return_diagnostics = _validate_return_diagnostics(return_diagnostics)

    def __call__(self, b, csr_values):
        _validate_values_and_rhs(csr_values, b, self._n, self._nnz)
        if b.ndim != 1:
            raise ValueError(
                "dynamic cuDSS direct solves require a one-dimensional RHS; use vmap for batching"
            )
        return solve(
            b,
            csr_values,
            csr_offsets=self.csr_offsets,
            csr_columns=self.csr_columns,
            device_id=self.device_id,
            mtype_id=self.mtype_id,
            mview_id=self.mview_id,
            return_diagnostics=self.return_diagnostics,
        )


class ConstantCSRCuDSSSolver(eqx.Module):
    """cuDSS solver for a CSR matrix whose values are fixed.

    The fixed ``csr_values`` are provided at construction time and are not part
    of ``__call__``.  The underlying cuDSS FFI may cache/factor those values in
    mutable native state and reuse them on later calls, so this class must only
    be used when the CSR values are genuinely invariant (for example a
    Helmholtz density-filter matrix).  For dynamic matrices, use
    :class:`CuDSSSolver` instead.
    """

    csr_offsets: Array = eqx.field(static=True)
    csr_columns: Array = eqx.field(static=True)
    _csr_values_bytes: bytes = eqx.field(static=True)
    _csr_values_dtype: str = eqx.field(static=True)
    matrix_token: str = eqx.field(static=True)
    _n: int = eqx.field(static=True)
    _nnz: int = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)

    def __init__(
        self,
        csr_offsets,
        csr_columns,
        csr_values,
        device_id,
        mtype_id,
        mview_id,
    ):
        n, nnz = _validate_solver_configuration(
            csr_offsets, csr_columns, device_id, mtype_id, mview_id
        )
        values_host = np.asarray(jax.device_get(csr_values))
        if values_host.ndim != 1 or values_host.size != nnz:
            raise ValueError(
                "CSR values must be one-dimensional and match column indices"
            )
        _dtype_suffix(values_host.dtype)
        digest = hashlib.sha256()
        for array in (
            np.asarray(jax.device_get(csr_offsets)),
            np.asarray(jax.device_get(csr_columns)),
            values_host,
        ):
            digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        self.csr_offsets = csr_offsets
        self.csr_columns = csr_columns
        self._csr_values_bytes = values_host.tobytes()
        self._csr_values_dtype = values_host.dtype.str
        self.matrix_token = digest.hexdigest()
        self._n = n
        self._nnz = nnz
        try:
            self.device_id = int(device_id)
            self.mtype_id = int(mtype_id)
            self.mview_id = int(mview_id)
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as error:  # validated above; defensive normalization
            raise TypeError(
                "cuDSS IDs must be representable as Python integers"
            ) from error

    @property
    def csr_values(self):
        """The fixed values reconstructed from immutable, hashable metadata."""
        return jnp.asarray(
            np.frombuffer(
                self._csr_values_bytes, dtype=np.dtype(self._csr_values_dtype)
            )
        )

    def __call__(self, b):
        csr_values = self.csr_values
        _validate_values_and_rhs(csr_values, b, self._n, self._nnz)
        if b.ndim not in (1, 2):
            raise ValueError("constant cuDSS supports one or multiple RHS vectors")
        return _solve(
            b,
            csr_values,
            csr_offsets=self.csr_offsets,
            csr_columns=self.csr_columns,
            device_id=self.device_id,
            mtype_id=self.mtype_id,
            mview_id=self.mview_id,
            return_diagnostics=False,
            constant_values=True,
            matrix_token=self.matrix_token,
        )
