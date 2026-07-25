# pyright: reportMissingImports=false, reportPrivateImportUsage=false
"""Custom-vmap wrappers for token phases and block-diagonal batching."""

from functools import cache

import jax
import jax.numpy as jnp

from ._core import _expand_structure
from ._ffi import (
    _config_attrs,
    _ffi_analyze,
    _ffi_call4,
    _ffi_numeric,
    _ffi_solve,
)

# vmap-aware wrappers over the single (B=1) view ===============================
# Static config is baked via cached closures so custom_vmap only ever sees
# array arguments. The rules are where vmap becomes block-diagonal batching.


@cache
def _make_analyze(suffix, cfg):
    del suffix  # dispatch is by values dtype; suffix only keys the cache

    @jax.custom_batching.custom_vmap
    def analyze_id(csr_values, csr_offsets, csr_columns):
        offs_bd, cols_bd, fp = _expand_structure(csr_offsets, csr_columns, 1)
        return _ffi_analyze(csr_values, offs_bd, cols_bd, fp, batch_size=1, cfg=cfg)

    @analyze_id.def_vmap
    def _(axis_size, in_batched, csr_values, csr_offsets, csr_columns):
        vb, _, _ = in_batched
        vals = (
            csr_values
            if vb
            else jnp.broadcast_to(csr_values, (axis_size,) + csr_values.shape)
        )
        offs_bd, cols_bd, _ = _expand_structure(csr_offsets, csr_columns, axis_size)
        token_id = analyze_id(vals.reshape(-1), offs_bd, cols_bd)
        return jnp.broadcast_to(token_id, (axis_size,) + token_id.shape), True

    return analyze_id


@cache
def _make_numeric(suffix, refactor, cfg):
    op = "refactorize" if refactor else "factorize"
    del suffix

    @jax.custom_batching.custom_vmap
    def numeric_id(token_id, offsets, columns, csr_values):
        offs_bd, cols_bd, fp = _expand_structure(offsets, columns, 1)
        return _ffi_numeric(token_id, offs_bd, cols_bd, fp, csr_values, op=op, cfg=cfg)

    @numeric_id.def_vmap
    def _(axis_size, in_batched, token_id, offsets, columns, csr_values):
        tb, _, _, vb = in_batched
        if not tb:
            raise ValueError(
                f"spineax tokens: vmap({op}) with an unbatched token and "
                "batched values — one entry cannot hold a batch of "
                "factorizations. vmap(analyze) over the batch first."
            )
        vals = (
            csr_values
            if vb
            else jnp.broadcast_to(csr_values, (axis_size,) + csr_values.shape)
        )
        # Collapse one axis and recurse (see the analyze rule): the batched
        # ids are equal copies of the one block entry, so the first id plus
        # the expanded pattern describe the whole level.
        offs_bd, cols_bd, _ = _expand_structure(offsets, columns, axis_size)
        out_id = numeric_id(token_id[0], offs_bd, cols_bd, vals.reshape(-1))
        return jnp.broadcast_to(out_id, (axis_size,) + out_id.shape), True

    return numeric_id


@cache
def _make_solve(suffix, cfg):
    del suffix

    @jax.custom_batching.custom_vmap
    def solve_id(token_id, offsets, columns, values, b_values):
        # The base case already absorbs any leading rhs axes as one
        # multi-RHS SOLVE (nrhs is derived from element counts).
        offs_bd, cols_bd, fp = _expand_structure(offsets, columns, 1)
        return _ffi_solve(token_id, offs_bd, cols_bd, fp, values, b_values, cfg=cfg)

    @solve_id.def_vmap
    def _(axis_size, in_batched, token_id, offsets, columns, values, b_values):
        tb, _, _, vb, bb = in_batched
        b = (
            b_values
            if bb
            else jnp.broadcast_to(b_values, (axis_size,) + b_values.shape)
        )
        if not tb:
            # unbatched token: the batch axis is just more rhs columns
            return solve_id(token_id, offsets, columns, values, b), True
        # batched ids (one block entry): flatten the batch axis into the ONE
        # block-diagonal system of dimension B*n and solve it single-style
        offs_bd, cols_bd, _ = _expand_structure(offsets, columns, axis_size)
        vals = values if vb else jnp.broadcast_to(values, (axis_size,) + values.shape)
        b2 = jnp.moveaxis(b, 0, -2)  # (B, ..., n) -> (..., B, n)
        bf = b2.reshape(b2.shape[:-2] + (-1,))
        out = solve_id(token_id[0], offs_bd, cols_bd, vals.reshape(-1), bf)
        return jnp.moveaxis(out.reshape(b2.shape), -2, 0), True

    return solve_id


_QUERY_FIELDS = (
    "lu_nnz",
    "npivots",
    "inertia",
    "perm_reorder_row",
    "perm_reorder_col",
    "perm_row",
    "perm_col",
    "perm_matching",
    "diag",
    "scale_row",
    "scale_col",
    "nd_partition_tree",
    "nsuperpanels",
    "schur_shape",
)


# Fields whose (N,)-sized values are in INPUT ORDER, so a block-diagonal
# system splits them cleanly into per-block (B, n) slices. Everything else is
# block-global: under vmap it is broadcast unchanged to every batch element
# (perm VALUES index the whole block system; lu_nnz/npivots/inertia/
# nd_partition_tree/nsuperpanels/schur_shape describe the one factorization).
_QUERY_SPLIT_FIELDS = frozenset({"diag", "scale_row", "scale_col"})


@cache
def _make_query(suffix, dtype, n, tree, cfg):
    """token -> tuple of the 14 query outputs for a system of dimension n.

    The CSR arrays ride along like every other phase so a non-resident id
    self-heals (and a resident one gets the fingerprint tamper check)."""

    @jax.custom_batching.custom_vmap
    def query_id(token_id, offsets, columns, values):
        offs_bd, cols_bd, fp = _expand_structure(offsets, columns, 1)
        outs = _ffi_call4(
            f"spineax_token_query_{suffix}",
            (
                jax.ShapeDtypeStruct((1,), jnp.int64),  # lu_nnz
                jax.ShapeDtypeStruct((1,), jnp.int32),  # npivots
                jax.ShapeDtypeStruct((2,), jnp.int32),  # inertia (cuDSS native)
                jax.ShapeDtypeStruct((n,), jnp.int32),  # perm_reorder_row
                jax.ShapeDtypeStruct((n,), jnp.int32),  # perm_reorder_col
                jax.ShapeDtypeStruct((n,), jnp.int32),  # perm_row
                jax.ShapeDtypeStruct((n,), jnp.int32),  # perm_col
                jax.ShapeDtypeStruct((n,), jnp.int32),  # perm_matching
                jax.ShapeDtypeStruct((n,), dtype),  # diag
                jax.ShapeDtypeStruct((n,), jnp.float32),  # scale_row
                jax.ShapeDtypeStruct((n,), jnp.float32),  # scale_col
                jax.ShapeDtypeStruct((tree,), jnp.int32),  # nd_partition_tree
                jax.ShapeDtypeStruct((1,), jnp.int32),  # nsuperpanels
                jax.ShapeDtypeStruct((2,), jnp.int64),  # schur_shape
            ),
            token_id,
            offs_bd,
            cols_bd,
            fp,
            values,
            **_config_attrs(cfg),
        )
        return tuple(outs)

    @query_id.def_vmap
    def _(axis_size, in_batched, token_id, offsets, columns, values):
        # Batched ids are B equal copies of ONE block entry: run a single
        # query on the whole B*n block system, then split the input-ordered
        # per-block fields to (B, n) and broadcast the block-global rest.
        _, _, _, vb = in_batched
        vals = values if vb else jnp.broadcast_to(values, (axis_size,) + values.shape)
        offs_bd, cols_bd, _ = _expand_structure(offsets, columns, axis_size)
        outs = _make_query(suffix, dtype, axis_size * n, tree, cfg)(
            token_id[0], offs_bd, cols_bd, vals.reshape(-1)
        )
        batched = []
        for index, field in enumerate(_QUERY_FIELDS):
            output = outs[index]
            if field in _QUERY_SPLIT_FIELDS:
                batched.append(output.reshape(axis_size, n))
            else:
                batched.append(jnp.broadcast_to(output, (axis_size,) + output.shape))
        return tuple(batched), (True,) * len(_QUERY_FIELDS)

    return query_id
