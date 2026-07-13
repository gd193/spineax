import numbers
import os

import jax
import jax.numpy as jnp
import numpy as np


def _cudss_debug_enabled() -> bool:
    return os.environ.get("SPINEAX_CUDSS_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_DTYPE_SUFFIXES = {
    jnp.dtype(jnp.float32): "f32",
    jnp.dtype(jnp.float64): "f64",
    jnp.dtype(jnp.complex64): "c64",
    jnp.dtype(jnp.complex128): "c128",
}
_DTYPE_BY_SUFFIX = {suffix: dtype for dtype, suffix in _DTYPE_SUFFIXES.items()}


def _dtype_suffix(dtype) -> str:
    normalized = jnp.dtype(dtype)
    try:
        return _DTYPE_SUFFIXES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def _log_dtype(dtype) -> None:
    if _cudss_debug_enabled() and jnp.dtype(dtype) in {
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float64),
    }:
        print(f"solving with {_dtype_suffix(dtype).replace('f', 'float')}")


def _validate_solver_configuration(
    csr_offsets, csr_columns, device_id, mtype_id, mview_id
):
    offsets = np.asarray(jax.device_get(csr_offsets))
    columns = np.asarray(jax.device_get(csr_columns))
    if offsets.ndim != 1 or columns.ndim != 1:
        raise ValueError("CSR offsets and columns must be one-dimensional")
    if offsets.dtype != np.int32 or columns.dtype != np.int32:
        raise TypeError("cuDSS CSR offsets and columns must use int32")
    if offsets.size < 2 or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("CSR offsets must start at zero and be nondecreasing")
    n = offsets.size - 1
    if offsets[-1] != columns.size:
        raise ValueError("final CSR offset must equal the number of columns/values")
    if np.any(columns < 0) or np.any(columns >= n):
        raise ValueError("CSR column indices are outside the square matrix")
    ids = {
        "device_id": (device_id, 2**63 - 1),
        "mtype_id": (mtype_id, 4),
        "mview_id": (mview_id, 2),
    }
    for name, (value, upper) in ids.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, numbers.Integral
        ):
            raise TypeError(f"{name} must be a non-boolean integer")
        if not 0 <= int(value) <= upper:
            if name == "device_id":
                raise ValueError("device_id must be a nonnegative integer")
            raise ValueError(f"{name} must be in [0, {upper}]")
    return n, columns.size


def _validate_return_diagnostics(return_diagnostics):
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise TypeError("return_diagnostics must be a boolean")
    return bool(return_diagnostics)


def _validate_values_and_rhs(csr_values, b, n, nnz):
    if csr_values.ndim != 1 or csr_values.shape[0] != nnz:
        raise ValueError(
            "CSR values must be one-dimensional with one value per column index"
        )
    _dtype_suffix(csr_values.dtype)
    if b.ndim < 1 or b.shape[-1] != n:
        raise ValueError("RHS trailing dimension must equal the CSR matrix dimension")
    if b.dtype != csr_values.dtype:
        raise TypeError("RHS and CSR values must have the same dtype")
