# pyright: reportMissingImports=false, reportGeneralTypeIssues=false
"""Public token-phase API and its autodiff rules."""

import dataclasses
from functools import cache
from math import prod

import jax
import jax.numpy as jnp
import numpy as np

from ._batching import (
    _QUERY_FIELDS,
    _make_analyze,
    _make_numeric,
    _make_query,
    _make_solve,
)
from ._core import FactorToken, _batch_of, _expand_structure, _suffix
from ._ffi import (
    _CFG_KEYS,
    _cfg_of,
    _nd_partition_tree_size,
    _token_cache_capacity,
    _token_rebuild_count,
    _token_registry_size,
    _token_release,
)
from ._transpose import _matvec, _transpose_csr

# free functions ===============================================================
# name -> cuDSS enum value (header order); every knob also takes the raw int
_MTYPE_IDS = {
    "general": 0,
    "symmetric": 1,
    "hermitian": 2,
    "spd": 3,
    "symmetric_positive_definite": 3,
    "hpd": 4,
    "hermitian_positive_definite": 4,
}
_MVIEW_IDS = {"full": 0, "upper": 1, "lower": 2}
_REORDERING_IDS = {
    "default": 0,
    "btf_colamd": 1,
    "colamd": 2,
    "amd": 3,
    "nested_dissection": 4,
    "none": 5,
}
_MEMORY_IDS = {"device": 0, "hybrid": 1}


def _integer_id(value, what):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"spineax tokens: invalid {what} enum value {value!r}"
        ) from error


def _knob_id(table, value, what):
    if not isinstance(value, str):
        return _integer_id(value, what)  # raw cuDSS enum value
    try:
        return table[value]
    except KeyError:
        raise ValueError(
            f"spineax tokens: unknown {what} {value!r} (one of: {', '.join(table)})"
        ) from None


def analyze(
    csr_values,
    csr_offsets,
    csr_columns,
    *,
    mtype_id: str | int = "symmetric",
    mview_id: str | int = "upper",
    device_id=0,
    reordering: str | int = "default",
    memory: str | int = "device",
) -> FactorToken:
    """``mtype_id``: "general", "symmetric", "hermitian", "spd" or "hpd".
    ``mview_id``: "full", "upper" or "lower".
    ``reordering``: "default", "btf_colamd", "colamd", "amd",
    "nested_dissection" or "none" (natural order).
    ``memory``: "device" (default) or "hybrid" host+device factors.
    Every knob also accepts the raw cuDSS enum int.

    Column indices may be unsorted but must be unique within each row.
    """
    cfg = (
        _integer_id(device_id, "device"),
        _knob_id(_MTYPE_IDS, mtype_id, "mtype"),
        _knob_id(_MVIEW_IDS, mview_id, "mview"),
        _knob_id(_REORDERING_IDS, reordering, "reordering"),
        _knob_id(_MEMORY_IDS, memory, "memory"),
    )
    core = _make_analyze_ad(_suffix(jnp.dtype(csr_values.dtype)), cfg)
    return core(
        csr_values, csr_offsets.astype(jnp.int32), csr_columns.astype(jnp.int32)
    )


@cache
def _make_analyze_ad(suffix, cfg):
    """The analyze implementation wrapped for autodiff (identity rule)."""
    statics = {key: cfg[index] for index, key in enumerate(_CFG_KEYS)}

    def impl(csr_values, csr_offsets, csr_columns):
        dtype = jnp.dtype(csr_values.dtype)
        n = csr_offsets.shape[-1] - 1
        nnz = csr_columns.shape[-1]
        if csr_values.ndim == 2:
            batch_size = csr_values.shape[0]
            offs_bd, cols_bd, _ = _expand_structure(
                csr_offsets, csr_columns, batch_size
            )
            token_id = _make_analyze(suffix, cfg)(
                csr_values.reshape(-1), offs_bd, cols_bd
            )
            return FactorToken(
                id=token_id,
                values=csr_values,
                offsets=csr_offsets,
                columns=csr_columns,
                phase="analyzed",
                kind="pbatch",
                dtype=dtype,
                n=n,
                nnz=nnz,
                batch_size=batch_size,
                **statics,
            )
        token_id = _make_analyze(suffix, cfg)(csr_values, csr_offsets, csr_columns)
        return FactorToken(
            id=token_id,
            values=csr_values,
            offsets=csr_offsets,
            columns=csr_columns,
            phase="analyzed",
            kind="single",
            dtype=dtype,
            n=n,
            nnz=nnz,
            batch_size=1,
            **statics,
        )

    core = jax.custom_jvp(impl)

    @core.defjvp
    def _(primals, tangents):
        # Recursive call (not impl): under higher-order differentiation the
        # primals themselves carry tangents, which must hit this rule again
        # rather than the raw ffi_call inside impl.
        dvals, _, _ = tangents
        token = core(*primals)

        # tangent = values tangent on the values leaf, float0 zeros on the
        # non-differentiable integer leaves
        def f0(x):
            return np.zeros(jnp.shape(x), jax.dtypes.float0)

        dtoken = dataclasses.replace(
            token,
            id=f0(token.id),
            values=dvals,
            offsets=f0(token.offsets),
            columns=f0(token.columns),
        )
        return token, dtoken

    return core


def _require_factorized(token, op):
    # phase is static, so misuse fails at trace time, not on the GPU
    if token.phase != "factorized":
        raise ValueError(
            f"spineax tokens: {op} requires a factorized token (call factorize first)"
        )


def _numeric(token, csr_values, refactor):
    op = "refactorize" if refactor else "factorize"
    if refactor:
        _require_factorized(token, op)
    if jnp.dtype(csr_values.dtype) != token.dtype:
        raise ValueError(
            f"spineax tokens: {op} values dtype {csr_values.dtype} != "
            f"token dtype {token.dtype}"
        )
    if csr_values.shape[-1] != token.nnz:
        raise ValueError(
            f"spineax tokens: {op} values size {csr_values.shape[-1]} != "
            f"token nnz {token.nnz}"
        )
    B = _batch_of(token)
    if (B > 1 or token.id.ndim >= 2) and (
        csr_values.ndim < 2 or prod(csr_values.shape[:-1]) != B
    ):
        raise ValueError(
            f"spineax tokens: {op} on a batch token expects values "
            f"({B}, {token.nnz}) (leading axes flattening to {B}), "
            f"got {csr_values.shape}"
        )
    return _make_numeric_ad(refactor)(token, csr_values)


def _numeric_impl(token, csr_values, refactor):
    B = _batch_of(token)
    if B > 1 or token.id.ndim >= 2:
        # batch entry used eagerly (explicit door, or vmap-minted token used
        # outside vmap): one block-diagonal numeric phase, through the same
        # custom_vmap wrapper so transform-added axes stay collapsible
        offs_bd, cols_bd, _ = _expand_structure(token.offsets, token.columns, B)
        tid = token.id.reshape(-1)[:1]
        out_id = _make_numeric(_suffix(token.dtype), refactor, _cfg_of(token))(
            tid, offs_bd, cols_bd, csr_values.reshape(-1)
        )
        token_id = jnp.broadcast_to(out_id, token.id.shape)
    else:
        token_id = _make_numeric(_suffix(token.dtype), refactor, _cfg_of(token))(
            token.id, token.offsets, token.columns, csr_values
        )
    # FRESH id: the numeric phase consumed the input state's name; values
    # leaf swapped to the just-factorized values
    return dataclasses.replace(
        token, id=token_id, values=csr_values, phase="factorized"
    )


@cache
def _make_numeric_ad(refactor):
    def impl(token, csr_values):
        return _numeric_impl(token, csr_values, refactor)

    core = jax.custom_jvp(impl)

    @core.defjvp
    def _(primals, tangents):
        dtoken, dvals = tangents
        out = core(*primals)
        # phase is a static: the tangent pytree must match the output's
        return out, dataclasses.replace(dtoken, values=dvals, phase="factorized")

    return core


def factorize(token: FactorToken, csr_values) -> FactorToken:
    return _numeric(token, csr_values, refactor=False)


def refactorize(token: FactorToken, csr_values) -> FactorToken:
    return _numeric(token, csr_values, refactor=True)


def solve(token: FactorToken, b, ir_nsteps=None):
    _require_factorized(token, "solve")
    if jnp.dtype(b.dtype) != token.dtype:
        raise ValueError(
            f"spineax tokens: rhs dtype {b.dtype} != token dtype {token.dtype}"
        )
    if b.shape[-1] != token.n:
        raise ValueError(
            f"spineax tokens: rhs trailing dim {b.shape[-1]} != token n {token.n}"
        )
    B = _batch_of(token)
    if (B > 1 or token.id.ndim >= 2) and (b.ndim < 2 or b.shape[-2] != B):
        raise ValueError(
            f"spineax tokens: solve on a batch token expects rhs "
            f"(..., {B}, {token.n}), got {b.shape}"
        )
    nsteps = 0 if ir_nsteps is None else ir_nsteps  # int OR traced scalar
    # The solver callables below are pure numerical inverses: all derivative
    # information enters through the matvec closure (implicit function
    # theorem), so the token they use is gradient-stopped.
    tok_ng = jax.lax.stop_gradient(token)

    def mv(x):
        return _matvec(token, x)

    def solve_fn(_mv, rhs):
        return _refined_solve(tok_ng, rhs, nsteps)

    if token.mtype_id in (1, 3):  # A^T = A (incl. complex symmetric): same factors
        return jax.lax.custom_linear_solve(mv, b, solve_fn, solve_fn, symmetric=True)
    if token.mtype_id in (2, 4):  # hermitian: A^T = conj(A), same factors

        def hermitian_t_solve(_mv, rhs):
            return jnp.conj(_refined_solve(tok_ng, jnp.conj(rhs), nsteps))

        return jax.lax.custom_linear_solve(mv, b, solve_fn, hermitian_t_solve)

    def general_t_solve(_mv, rhs):  # general: cuDSS has no transpose solve
        return _transpose_solve_general(tok_ng, rhs, nsteps)

    return jax.lax.custom_linear_solve(mv, b, solve_fn, general_t_solve)


def _solve_impl(token, b):
    B = _batch_of(token)
    fn = _make_solve(_suffix(token.dtype), _cfg_of(token))
    if B > 1 or token.id.ndim >= 2:
        # batch entry used eagerly: solve the ONE expanded block-diagonal
        # system single-style, through the same custom_vmap wrapper so any
        # transform-added batch axes stay collapsible (multi-RHS)
        offs_bd, cols_bd, _ = _expand_structure(token.offsets, token.columns, B)
        tid = token.id.reshape(-1)[:1]
        bf = b.reshape(b.shape[:-2] + (-1,))
        out = fn(tid, offs_bd, cols_bd, token.values.reshape(-1), bf)
        return out.reshape(b.shape)
    return fn(token.id, token.offsets, token.columns, token.values, b)


def _refined_solve(token, b, nsteps):
    """``ir_nsteps`` rounds of Richardson refinement, JAX-side.

    cuDSS-internal IR is permanently OFF: its refinement SpMV dereferences
    CSR pointers captured at earlier phase calls (compute-sanitizer:
    out-of-bounds atomics in cudss::spmv_ker once the expanded batch
    structure is a freed XLA temporary). This same-precision loop is what
    cuDSS runs internally anyway, and its residual SpMV consumes its
    expanded indices inside the executable that builds them.

    ``nsteps`` may be TRACED (runtime-varying under one jit trace): the loop
    becomes a fori_loop then, which is legal here because the solve
    callables sit inside custom_linear_solve and are only ever re-traced
    and re-applied by its IFT rules, never differentiated through.
    """
    x = _solve_impl(token, b)

    def refine(_, current):
        return current + _solve_impl(token, b - _matvec(token, current))

    if isinstance(nsteps, jax.Array):
        return jax.lax.fori_loop(0, nsteps, refine, x)
    for i in range(nsteps):  # static: unrolled
        x = refine(i, x)
    return x


def _transpose_solve_general(token, rhs, nsteps):
    """``A^-T rhs`` for a general (mtype 0) token.

    cuDSS cannot solve against the transpose of existing factors
    (CUDSS_CONFIG_SOLVE_MODE is "not supported right now" as of 0.8), so
    reverse mode through a general solve transposes the CSR system on device
    and runs a fresh analyze+factorize+solve. That is a full factorization
    AND a new LRU registry entry per backward execution.
    """
    B = _batch_of(token)
    offs_bd, cols_bd, _ = _expand_structure(token.offsets, token.columns, B)
    t_vals, t_offs, t_cols = _transpose_csr(token.values.reshape(-1), offs_bd, cols_bd)
    # TODO: remove the explicit transpose and second factorization when cuDSS
    # implements nonzero CUDSS_CONFIG_SOLVE_MODE values.
    t_token = analyze(
        t_vals,
        t_offs,
        t_cols,
        mtype_id=0,
        mview_id=0,
        device_id=token.device_id,
        reordering=token.reordering_id,
        memory=token.memory_id,
    )
    t_token = factorize(t_token, t_token.values)
    rf = rhs.reshape(rhs.shape[:-2] + (-1,)) if B > 1 else rhs
    return _refined_solve(t_token, rf, nsteps).reshape(rhs.shape)


def query(token: FactorToken) -> dict:
    _require_factorized(token, "query")
    B = _batch_of(token)
    tree = _nd_partition_tree_size()
    fn = _make_query(
        _suffix(token.dtype), token.dtype, B * token.n, tree, _cfg_of(token)
    )
    if B > 1 or token.id.ndim >= 2:
        # batch entry used eagerly: one query of the expanded block system,
        # through the same custom_vmap wrapper (see _solve_impl)
        offs_bd, cols_bd, _ = _expand_structure(token.offsets, token.columns, B)
        tid = token.id.reshape(-1)[:1]
        outs = fn(tid, offs_bd, cols_bd, token.values.reshape(-1))
    else:
        outs = fn(token.id, token.offsets, token.columns, token.values)
    return {field: outs[index] for index, field in enumerate(_QUERY_FIELDS)}


def inertia(data: dict, batch_size: int = 1):
    """Per-block [positive, negative] LDL^T inertia from ``query`` output. CuDSS
    doesnt yet reliably return 0 eigenvalues as of 0.8.
    """
    diag = data["diag"]
    diag_real = diag.real if jnp.iscomplexobj(diag) else diag
    n = diag.shape[0] // batch_size

    out = diag_real.reshape([batch_size, n])

    # cuDSS pivoting threshold seems to be 1e-13. everything above this on
    # plus or minus side seems to reliably indicate that particular inertia value.
    threshold = 1e-13
    positive = jnp.sum(out >= threshold, axis=1)
    negative = jnp.sum(out <= -threshold, axis=1)

    result = jnp.stack([positive, negative], axis=1, dtype=jnp.int32)

    return result if batch_size > 1 else result[0]


# registry escape hatches ======================================================
def release(token: FactorToken) -> bool:
    """manually free registry (only outside jit - otherwise whenever LRU overflows
    according to SPINEAX_FACTOR_CACHE). Using the token again after release
    self-heals by rebuilding, so this frees memory, not the token."""
    return _token_release(token.id)


def registry_size() -> int:
    """Number of live factorizations in the registry."""
    return _token_registry_size()


def cache_capacity() -> int:
    """LRU capacity (``SPINEAX_FACTOR_CACHE``, default 8)."""
    return _token_cache_capacity()


def rebuild_count() -> int:
    """Times a phase call rebuilt a non-resident factorization (self-heal).

    A rising count means live tokens are being evicted and re-factorized —
    raise ``SPINEAX_FACTOR_CACHE`` to fit the working set."""
    return _token_rebuild_count()
