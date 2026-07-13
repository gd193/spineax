import functools as ft

import jax
import jax.numpy as jnp

from ._primitives import (
    KIND_BATCH,
    KIND_PBATCH,
    KIND_SINGLE,
    _const_xonly_solver_for_dtype,
    _primitive_for,
)
from ._validation import _log_dtype


# single solve interface =======================================================
@ft.partial(
    jax.jit,
    static_argnames=[
        "device_id",
        "mtype_id",
        "mview_id",
        "return_diagnostics",
        "constant_values",
        "matrix_token",
    ],
)
def _solve(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    device_id,
    mtype_id,
    mview_id,
    return_diagnostics=True,
    constant_values=False,
    matrix_token=None,
):
    if b_values.ndim < 1 or b_values.shape[-1] != csr_offsets.shape[0] - 1:
        raise ValueError("RHS trailing dimension must equal the CSR matrix dimension")
    if not constant_values and b_values.ndim != 1:
        raise ValueError(
            "dynamic cuDSS direct solves require a one-dimensional RHS; use vmap for batching"
        )
    if b_values.dtype != csr_values.dtype:
        raise TypeError("RHS and CSR values must have the same dtype")
    if constant_values and matrix_token is None:
        raise ValueError("constant cuDSS solve requires an immutable matrix token")
    if constant_values and return_diagnostics:
        raise ValueError("constant_values=True requires return_diagnostics=False")
    if constant_values:
        if b_values.ndim not in (1, 2):
            raise ValueError(
                "constant_values cuDSS supports 1D RHS or 2D (batch, n) RHS"
            )
        if b_values.ndim == 2:
            if b_values.shape[0] == 1:
                # Preserve the fast single-RHS constant path for singleton
                # batches. The multi-RHS cuDSS FFI path has extra setup cost
                # even when nrhs=1, which regresses batch-1 filtered rendering.
                solver = _const_xonly_solver_for_dtype(
                    csr_values.dtype, multi_rhs=False
                )
                out = solver.bind(
                    jnp.reshape(b_values, (-1,)),
                    csr_values,
                    csr_offsets,
                    csr_columns,
                    device_id=device_id,
                    mtype_id=mtype_id,
                    mview_id=mview_id,
                    matrix_token=matrix_token,
                )[0]
                return [jnp.reshape(out, b_values.shape)]
            # Pass a flattened contiguous row-major [batch, n] buffer to C++.
            # The C++ handler interprets each contiguous row as one column-major
            # RHS vector with n rows and nrhs=batch, avoiding layout-sensitive
            # assumptions about a rank-2 custom-call operand/result.
            solver = _const_xonly_solver_for_dtype(csr_values.dtype, multi_rhs=True)
            flat = jnp.reshape(b_values, (-1,))
            out = solver.bind(
                flat,
                csr_values,
                csr_offsets,
                csr_columns,
                device_id=device_id,
                mtype_id=mtype_id,
                mview_id=mview_id,
                matrix_token=matrix_token,
            )[0]
            return [jnp.reshape(out, b_values.shape)]
        solver = _const_xonly_solver_for_dtype(csr_values.dtype, multi_rhs=False)
    else:
        _log_dtype(csr_values.dtype)
        solver = _primitive_for(
            KIND_SINGLE,
            csr_values.dtype,
            return_diagnostics=return_diagnostics,
        )

    return solver.bind(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
        **({"matrix_token": matrix_token} if constant_values else {}),
    )


def solve(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    device_id,
    mtype_id,
    mview_id,
    return_diagnostics=True,
):
    """Solve with CSR values supplied for this call.

    This public dynamic API always uses the ``csr_values`` argument.  Constant
    matrix/factor-reuse mode is intentionally exposed only through
    :class:`ConstantCSRCuDSSSolver`, whose call signature accepts RHS only.
    """
    return _solve(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        device_id,
        mtype_id,
        mview_id,
        return_diagnostics=return_diagnostics,
        constant_values=False,
    )


# manual batch solve interface =================================================
@ft.partial(
    jax.jit,
    static_argnames=[
        "batch_size",
        "device_id",
        "mtype_id",
        "mview_id",
        "return_diagnostics",
    ],
)
def batch_solve(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    batch_size,
    device_id,
    mtype_id,
    mview_id,
    return_diagnostics=True,
):
    if b_values.ndim != 2 or b_values.shape[-1] != csr_offsets.shape[-1] - 1:
        raise ValueError("RHS trailing dimension must equal the CSR matrix dimension")
    if csr_values.ndim != 2 or csr_values.shape[0] != batch_size:
        raise ValueError("batched CSR values must have shape (batch_size, nnz)")
    if b_values.shape[0] != batch_size or b_values.dtype != csr_values.dtype:
        raise ValueError("RHS batch size and dtype must match CSR values")
    _log_dtype(csr_values.dtype)
    solver = _primitive_for(
        KIND_BATCH,
        csr_values.dtype,
        return_diagnostics=return_diagnostics,
    )

    return solver.bind(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        batch_size=batch_size,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )


# manual psuedo batch solve interface =================================================
@ft.partial(
    jax.jit,
    static_argnames=[
        "batch_size",
        "device_id",
        "mtype_id",
        "mview_id",
        "return_diagnostics",
    ],
)
def pbatch_solve(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    batch_size,
    device_id,
    mtype_id,
    mview_id,
    return_diagnostics=True,
):
    if b_values.ndim != 2 or b_values.shape[-1] != csr_offsets.shape[-1] - 1:
        raise ValueError("RHS trailing dimension must equal the CSR matrix dimension")
    if csr_values.ndim != 2 or csr_values.shape[0] != batch_size:
        raise ValueError("pseudo-batched CSR values must have shape (batch_size, nnz)")
    if b_values.shape[0] != batch_size or b_values.dtype != csr_values.dtype:
        raise ValueError("RHS batch size and dtype must match CSR values")
    _log_dtype(csr_values.dtype)
    solver = _primitive_for(
        KIND_PBATCH,
        csr_values.dtype,
        return_diagnostics=return_diagnostics,
    )

    return solver.bind(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        batch_size=batch_size,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )
