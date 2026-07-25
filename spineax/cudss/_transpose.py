# pyright: reportMissingImports=false, reportPrivateImportUsage=false
"""CSR algebra and differentiable cuSPARSE transpose operations."""

import jax
import jax.numpy as jnp

from ._core import _batch_of, _expand_structure, _suffix
from ._ffi import _ffi_call4

# autodiff for solve ===========================================================


def _pattern_rows(offsets, nnz):
    """Row index of each stored CSR entry, from the (possibly per-block
    batched) offsets pattern."""

    def one(o):
        return jnp.repeat(
            jnp.arange(o.shape[-1] - 1, dtype=jnp.int32),
            jnp.diff(o),
            total_repeat_length=nnz,
        )

    return jax.vmap(one)(offsets) if offsets.ndim == 2 else one(offsets)


def _gather_last(a, idx):
    """a[..., idx] with idx either shared ``(nnz,)`` or per-block
    ``(B, nnz)`` (matching a's ``(..., B, n)`` block axis)."""
    if idx.ndim == 1:
        return jnp.take(a, idx, axis=-1)
    idx_b = jnp.broadcast_to(idx, a.shape[:-1] + (idx.shape[-1],))
    return jnp.take_along_axis(a, idx_b, axis=-1)


def _matvec(token, x):
    B = _batch_of(token)
    offs_bd, cols_bd, _ = _expand_structure(token.offsets, token.columns, B)
    rows_bd = _pattern_rows(offs_bd, B * token.nnz)
    vals = token.values.reshape(-1)
    xf = x.reshape(x.shape[:-2] + (-1,)) if B > 1 else x
    y = jnp.zeros_like(xf).at[..., rows_bd].add(vals * _gather_last(xf, cols_bd))
    if token.mview_id in (1, 2):
        mvals = jnp.conj(vals) if token.mtype_id in (2, 4) else vals
        mvals = jnp.where(rows_bd == cols_bd, 0, mvals)  # diag stored once
        y = y.at[..., cols_bd].add(mvals * _gather_last(xf, rows_bd))
    return y.reshape(x.shape)


@jax.custom_batching.custom_vmap
def _raw_transpose_csr(values, offsets, columns):
    n = offsets.shape[0] - 1
    nnz = values.shape[0]
    return tuple(
        _ffi_call4(
            f"spineax_csr_transpose_{_suffix(values.dtype)}",
            (
                jax.ShapeDtypeStruct((nnz,), values.dtype),
                jax.ShapeDtypeStruct((n + 1,), jnp.int32),
                jax.ShapeDtypeStruct((nnz,), jnp.int32),
            ),
            values,
            offsets,
            columns,
            has_side_effect=False,
        )
    )


@_raw_transpose_csr.def_vmap
def _(axis_size, in_batched, values, offsets, columns):
    values_batched, offsets_batched, columns_batched = in_batched

    if not (offsets_batched or columns_batched):
        # shared pattern: the transposed structure is batch-invariant and
        # the values just permute — one unbatched order FFI + batched gather
        if not values_batched:
            values = jnp.broadcast_to(values, (axis_size,) + values.shape)
        n = offsets.shape[0] - 1
        order = _transpose_csr_order(offsets, columns)
        rows = _pattern_rows(offsets, columns.shape[0])
        counts = jnp.zeros((n,), dtype=jnp.int32).at[columns].add(1)
        transposed_offsets = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.cumsum(counts, dtype=jnp.int32),
            ]
        )
        outputs = (
            jnp.take(values, order, axis=-1),
            transposed_offsets,
            jnp.take(rows, order),
        )
        return outputs, (True, False, False)

    # per-example patterns: a batch of systems IS one block-diagonal system,
    # and blockdiag(A_k)^T = blockdiag(A_k^T) — ONE flat transpose, split
    # back per block. Recursing into the wrapped function (never the raw
    # FFI) lets any outer vmap levels peel through this rule again. vmap
    # guarantees uniform shapes, so each block holds exactly nnz entries,
    # contiguous in the transposed block-diagonal CSR.
    vals = (
        values
        if values_batched
        else jnp.broadcast_to(values, (axis_size,) + values.shape)
    )
    n = offsets.shape[-1] - 1
    nnz = columns.shape[-1]
    offs_bd, cols_bd, _ = _expand_structure(offsets, columns, axis_size)
    t_vals, t_offs_bd, t_cols_bd = _raw_transpose_csr(
        vals.reshape(-1), offs_bd, cols_bd
    )
    k = jnp.arange(axis_size, dtype=jnp.int32)[:, None]
    t_offs = t_offs_bd[k * n + jnp.arange(n + 1, dtype=jnp.int32)] - k * jnp.int32(nnz)
    outputs = (
        t_vals.reshape(axis_size, nnz),
        t_offs,
        t_cols_bd.reshape(axis_size, nnz) - k * jnp.int32(n),
    )
    return outputs, (True, True, True)


@jax.custom_batching.custom_vmap
def _transpose_csr_order(offsets, columns):
    nnz = columns.shape[0]
    identity = jnp.arange(nnz, dtype=jnp.int32)
    (order,) = _ffi_call4(
        "spineax_csr_transpose_order",
        (jax.ShapeDtypeStruct((nnz,), jnp.int32),),
        offsets,
        columns,
        identity,
        has_side_effect=False,
    )
    return order


@_transpose_csr_order.def_vmap
def _(axis_size, in_batched, offsets, columns):
    # only reached with a batched pattern (a fully-shared call is hoisted
    # out of vmap): block-diagonalize, then split — block k's entries occupy
    # flat slots [k*nnz, (k+1)*nnz) on both sides of the permutation
    del in_batched
    nnz = columns.shape[-1]
    offs_bd, cols_bd, _ = _expand_structure(offsets, columns, axis_size)
    flat = _transpose_csr_order(offs_bd, cols_bd)
    k = jnp.arange(axis_size, dtype=jnp.int32)[:, None]
    return flat.reshape(axis_size, nnz) - k * jnp.int32(nnz), True


@jax.custom_jvp
def _transpose_csr(values, offsets, columns):
    """Return the CSR arrays of a square matrix transpose.

    Column indices may be unsorted but must be unique within each row.
    """
    values = jnp.asarray(values)
    offsets = jnp.asarray(offsets, dtype=jnp.int32)
    columns = jnp.asarray(columns, dtype=jnp.int32)
    if values.ndim != 1 or offsets.ndim != 1 or columns.ndim != 1:
        raise ValueError("CSR values, offsets, and columns must be one-dimensional")
    if values.shape != columns.shape:
        raise ValueError("CSR values and columns must have equal lengths")
    if offsets.shape[0] < 2:
        raise ValueError("CSR offsets must describe at least one row")
    return _raw_transpose_csr(values, offsets, columns)


@_transpose_csr.defjvp
def _transpose_csr_jvp(primals, tangents):
    values, offsets, columns = primals
    values_dot, _offsets_dot, _columns_dot = tangents
    primal = _transpose_csr(values, offsets, columns)
    offsets = jnp.asarray(offsets, dtype=jnp.int32)
    columns = jnp.asarray(columns, dtype=jnp.int32)
    order = _transpose_csr_order(offsets, columns)
    tangent = (
        jnp.take(values_dot, order),
        jnp.zeros(offsets.shape, dtype=jax.dtypes.float0),
        jnp.zeros(columns.shape, dtype=jax.dtypes.float0),
    )
    return primal, tangent
