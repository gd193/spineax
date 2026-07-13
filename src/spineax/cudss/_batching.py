import jax
import jax.numpy as jnp
from jax.interpreters import batching

from ._dispatch import _solve, batch_solve, solve
from ._primitives import (
    KIND_BATCH,
    KIND_PBATCH,
    KIND_SINGLE,
    _DTYPE_BY_SUFFIX,
    _batched_primitive_for,
    _const_xonly_solver_for_dtype,
    _primitive_for,
)
from ._validation import _cudss_debug_enabled


def solve_multi_rhs_const_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    """Flatten another mapped RHS axis into the native multi-RHS solve."""
    b_values, csr_values, csr_offsets, csr_columns = vector_arg_values
    a_b, a_val, a_off, a_col = batch_axes
    if a_off is not None:
        csr_offsets = jax.lax.index_in_dim(csr_offsets, 0, axis=a_off, keepdims=False)
        a_off = None
    if a_col is not None:
        csr_columns = jax.lax.index_in_dim(csr_columns, 0, axis=a_col, keepdims=False)
        a_col = None
    if a_val is not None or a_off is not None or a_col is not None:
        raise NotImplementedError(
            "constant_values cuDSS vmap supports only batched RHS with constant matrix"
        )
    if a_b is None:
        return _const_xonly_solver_for_dtype(csr_values.dtype, multi_rhs=True).bind(
            b_values, csr_values, csr_offsets, csr_columns, **kwargs
        ), (None,)
    if a_b != 0:
        b_values = jnp.moveaxis(b_values, a_b, 0)
    original_shape = b_values.shape
    out = _const_xonly_solver_for_dtype(csr_values.dtype, multi_rhs=True).bind(
        jnp.reshape(b_values, (-1,)),
        csr_values,
        csr_offsets,
        csr_columns,
        **kwargs,
    )[0]
    return (jnp.reshape(out, original_shape),), (0,)


def solve_single_const_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    b_values, csr_values, csr_offsets, csr_columns = vector_arg_values
    a_b, a_val, a_off, a_col = batch_axes
    if a_off is not None:
        csr_offsets = jax.lax.index_in_dim(csr_offsets, 0, axis=a_off, keepdims=False)
        a_off = None
    if a_col is not None:
        csr_columns = jax.lax.index_in_dim(csr_columns, 0, axis=a_col, keepdims=False)
        a_col = None
    if a_val is not None or a_off is not None or a_col is not None:
        raise NotImplementedError(
            "constant_values cuDSS vmap supports only batched RHS with constant matrix"
        )
    if a_b is None:
        return _solve(
            b_values,
            csr_values,
            csr_offsets,
            csr_columns,
            return_diagnostics=False,
            constant_values=True,
            **kwargs,
        ), (None,)
    if a_b != 0:
        b_values = jnp.moveaxis(b_values, a_b, 0)

    return _solve(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        return_diagnostics=False,
        constant_values=True,
        **kwargs,
    ), (0,)


def general_solve_vmap(
    vector_arg_values,  # [b_values, csr_values, csr_offsets, csr_columns]
    batch_axes,  # [b_values, csr_values, csr_offsets, csr_columns]
    *,
    prefer_pbatch,
    **kwargs,  # static params
):

    b_values, csr_values, csr_offsets, csr_columns = vector_arg_values
    a_b, a_val, a_off, a_col = batch_axes
    return_diagnostics = kwargs.get("return_diagnostics", True)
    bind_kwargs = dict(kwargs)
    bind_kwargs.pop("return_diagnostics", None)
    out_axes = (0, 0) if return_diagnostics else (0,)

    if _cudss_debug_enabled():
        jax.debug.print(
            "vmap batch_axes: a_b={}, a_val={}, a_off={}, a_col={}",
            a_b,
            a_val,
            a_off,
            a_col,
        )

    # Handle spurious batch axes on sparsity patterns.
    # This happens when the solve is inside a jax.lax.switch that gets vmapped -
    # JAX broadcasts all branch inputs to have batch dimensions, even constants.
    # Since sparsity patterns are the same across all batch elements, extract first.
    if a_off is not None:
        csr_offsets = jax.lax.index_in_dim(csr_offsets, 0, axis=a_off, keepdims=False)
        a_off = None
    if a_col is not None:
        csr_columns = jax.lax.index_in_dim(csr_columns, 0, axis=a_col, keepdims=False)
        a_col = None

    # Update vector_arg_values with the corrected sparsity patterns
    vector_arg_values = (b_values, csr_values, csr_offsets, csr_columns)

    # guards (these should never trigger now since we handle spurious batch axes above)
    if any(ax is not None for ax in (a_off, a_col)):
        raise NotImplementedError(
            "don't support batches of heterogeneous sparsity patterns yet (its coming tho...)"
        )

    if all(ax is None for ax in (a_val, a_b)):
        raise NotImplementedError(
            "Only batched csr_values and b_values are supported right now"
        )

    # the non-batched path
    if a_val is None and a_b is None:
        return solve(*vector_arg_values, **kwargs), (
            (None, None) if return_diagnostics else (None,)
        )

    # if only one of the sets of values are batched
    elif (a_val is None) != (a_b is None):
        if a_b is not None and a_val is None:
            # Only b is batched - broadcast csr_values to match batch dimension
            csr_values_batched = jnp.broadcast_to(
                csr_values[None, :], (b_values.shape[0],) + csr_values.shape
            )
            vector_arg_values = (b_values, csr_values_batched, csr_offsets, csr_columns)

            solver = _batched_primitive_for(
                csr_values.dtype,
                return_diagnostics=return_diagnostics,
                prefer_pbatch=prefer_pbatch,
            )

            return solver.bind(
                *vector_arg_values, batch_size=b_values.shape[0], **bind_kwargs
            ), out_axes
        else:
            # Only csr_values is batched (not b) - not supported
            raise NotImplementedError(
                "Only csr_values batched (not b_values) is not supported"
            )

    # the batched path binding
    elif a_val is not None and a_b is not None:
        solver = _batched_primitive_for(
            csr_values.dtype,
            return_diagnostics=return_diagnostics,
            prefer_pbatch=prefer_pbatch,
        )
        return solver.bind(
            *vector_arg_values, batch_size=b_values.shape[0], **bind_kwargs
        ), out_axes

    else:
        raise NotImplementedError("This path should not be possible")


def solve_batch_vmap(vector_arg_values, batch_axes, *, prefer_pbatch, **kwargs):
    """Handle vmap of already-batched solve"""
    b_values, csr_values, csr_offsets, csr_columns = vector_arg_values
    a_b, a_val, a_off, a_col = batch_axes
    return_diagnostics = kwargs.get("return_diagnostics", True)
    bind_kwargs = dict(kwargs)
    bind_kwargs.pop("return_diagnostics", None)
    out_axes = (0, 0) if return_diagnostics else (0,)

    # Handle spurious batch axes on sparsity patterns (same fix as general_solve_vmap)
    if a_off is not None:
        csr_offsets = jax.lax.index_in_dim(csr_offsets, 0, axis=a_off, keepdims=False)
        a_off = None
    if a_col is not None:
        csr_columns = jax.lax.index_in_dim(csr_columns, 0, axis=a_col, keepdims=False)
        a_col = None
    vector_arg_values = (b_values, csr_values, csr_offsets, csr_columns)

    if any(ax is not None for ax in (a_off, a_col)):
        raise NotImplementedError(
            "don't support batches of heterogeneous sparsity patterns yet (its coming tho...)"
        )

    if a_b is None and a_val is None:
        # Not actually batching
        return batch_solve(*vector_arg_values, **kwargs), (
            (None, None) if return_diagnostics else (None,)
        )

    # Flatten nested batches
    batch_size1 = b_values.shape[0]
    batch_size2 = b_values.shape[1]
    total_batch = batch_size1 * batch_size2

    b_flat = b_values.reshape(total_batch, -1)
    csr_flat = csr_values.reshape(total_batch, -1)

    # the non-batched path
    if a_val is None and a_b is None:
        return solve(*vector_arg_values, **kwargs), (
            (None, None) if return_diagnostics else (None,)
        )

    # if only one of the sets of values are batched
    elif (a_val is None) != (a_b is None):
        if a_b is not None and a_val is None:
            # Only b is batched in nested vmap - broadcast csr_values
            csr_values_batched = jnp.broadcast_to(
                csr_values[None, :, :], (b_values.shape[0],) + csr_values.shape
            )
            b_flat = b_values.reshape(-1, b_values.shape[-1])
            csr_flat = csr_values_batched.reshape(-1, csr_values.shape[-1])

            solver = _batched_primitive_for(
                csr_values.dtype,
                return_diagnostics=return_diagnostics,
                prefer_pbatch=prefer_pbatch,
            )

            total_batch = b_flat.shape[0]
            # Remove old batch_size from kwargs
            kwargs_copy = dict(bind_kwargs)
            kwargs_copy.pop("batch_size", None)
            outputs = solver.bind(
                b_flat,
                csr_flat,
                csr_offsets,
                csr_columns,
                batch_size=total_batch,
                **kwargs_copy,
            )

            # Reshape back
            x = outputs[0].reshape(b_values.shape[0], b_values.shape[1], -1)
            if not return_diagnostics:
                return (x,), out_axes
            inertia = outputs[1].reshape(b_values.shape[0], b_values.shape[1], 2)

            return (x, inertia), out_axes
        else:
            # Only csr_values is batched (not b) - not supported
            raise NotImplementedError(
                "Only csr_values batched (not b_values) is not supported"
            )

    elif a_val is not None and a_b is not None:
        solver = _batched_primitive_for(
            csr_values.dtype,
            return_diagnostics=return_diagnostics,
            prefer_pbatch=prefer_pbatch,
        )
    else:
        raise NotImplementedError("This path should not be possible")

    # Remove batch_size from kwargs if present (happens with nested vmap)
    kwargs_copy = dict(bind_kwargs)
    kwargs_copy.pop("batch_size", None)

    outputs = solver.bind(
        b_flat,
        csr_flat,
        csr_offsets,
        csr_columns,
        batch_size=total_batch,
        **kwargs_copy,
    )

    # Reshape back
    x = outputs[0].reshape(batch_size1, batch_size2, -1)
    if not return_diagnostics:
        return (x,), out_axes
    inertia = outputs[1].reshape(batch_size1, batch_size2, 2)

    return (x, inertia), out_axes


def install_batching_rules(
    single_rule,
    single_xonly_rule,
    batch_rule,
    batch_xonly_rule,
):
    """Install all cuDSS primitive batching rules explicitly and idempotently."""
    for dtype in _DTYPE_BY_SUFFIX.values():
        batching.primitive_batchers[
            _primitive_for(KIND_SINGLE, dtype, return_diagnostics=True)
        ] = single_rule
        batching.primitive_batchers[
            _primitive_for(KIND_SINGLE, dtype, return_diagnostics=False)
        ] = single_xonly_rule
        batching.primitive_batchers[
            _const_xonly_solver_for_dtype(dtype, multi_rhs=False)
        ] = solve_single_const_xonly_vmap
        batching.primitive_batchers[
            _const_xonly_solver_for_dtype(dtype, multi_rhs=True)
        ] = solve_multi_rhs_const_xonly_vmap
        for kind in (KIND_BATCH, KIND_PBATCH):
            batching.primitive_batchers[
                _primitive_for(kind, dtype, return_diagnostics=True)
            ] = batch_rule
            batching.primitive_batchers[
                _primitive_for(kind, dtype, return_diagnostics=False)
            ] = batch_xonly_rule
