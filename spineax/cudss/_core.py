# pyright: reportMissingImports=false, reportUnhashable=false
"""Token data model and side-effect-free CSR/configuration helpers."""

from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

_SUFFIXES = ("f32", "f64", "c64", "c128")
_DTYPE_SUFFIX = {
    jnp.dtype(jnp.float32): "f32",
    jnp.dtype(jnp.float64): "f64",
    jnp.dtype(jnp.complex64): "c64",
    jnp.dtype(jnp.complex128): "c128",
}


def _suffix(dtype) -> str:
    try:
        return _DTYPE_SUFFIX[jnp.dtype(dtype)]
    except KeyError:
        raise ValueError(
            f"spineax tokens: unsupported dtype {dtype} "
            f"(supported: f32, f64, c64, c128)"
        ) from None


# the token ====================================================================
class FactorToken(eqx.Module):
    """Handle to a cached cuDSS (block-diagonal) factorization.

    ``id`` is the traced dispatch leaf: dataflow ordering, vjp residuals and
    vmap batching all operate on it. It is ``int32[1]`` for a token minted
    outside vmap, or ``int32[B1, ..., Bk, 1]`` (equal copies of one entry id,
    one leading axis per vmap level) for a token minted under (possibly
    nested) ``vmap(analyze)``.

    An id names ONE immutable numeric state: ``factorize``/``refactorize``
    consume their input's id and return a fresh one. Using an id that is no
    longer resident (superseded, or LRU-evicted) transparently REBUILDS that
    state from the token's own arrays — every phase, ``query`` included.
    ``rebuild_count()`` counts the heals; a rising count means
    SPINEAX_FACTOR_CACHE is too small for the working set.

    ``values``/``offsets``/``columns`` are zero-copy references to the
    caller's CSR arrays (the block pattern exactly as handed to ``analyze``,
    NOT the expanded block-diagonal form). They keep the CSR data alive as
    long as the factorization and are passed to every phase call so cuDSS
    always reads live buffers; ``values`` is the values of the last numeric
    phase. The static fields resolve dispatch at trace time and make the
    token self-describing — a FactorToken is directly a lineax solver state.
    """

    id: Array  # int32[1] (+1 axis per vmap level) — traced
    values: Array  # (nnz,) or (B, nnz)
    offsets: Array  # int32 (n+1,) or (B, n+1)
    columns: Array  # int32 (nnz,) or (B, nnz)
    phase: str = eqx.field(static=True)  # "analyzed" | "factorized"
    kind: str = eqx.field(static=True)  # "single" | "pbatch" (descriptive)
    dtype: jnp.dtype = eqx.field(static=True)
    n: int = eqx.field(static=True)  # BLOCK dimension (one system)
    nnz: int = eqx.field(static=True)  # BLOCK nnz (one system)
    batch_size: int = eqx.field(static=True)  # 1 unless minted by the explicit door
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    reordering_id: int = eqx.field(static=True)  # cudssReorderingAlg_t 0-5
    memory_id: int = eqx.field(static=True)  # 0 device, 1 hybrid host+device


def _structure_fingerprint(offsets_bd, columns_bd):
    """Position-weighted checksum (uint32[2]) of the expanded structure.

    The token's offsets/columns leaves are IMMUTABLE by contract — cuDSS's
    analysis and pivot order are tied to the pattern that was analyzed, so a
    swapped same-sized pattern would silently produce garbage factors. The
    fingerprint is computed on-device in the same pass that reads the
    structure anyway, and every phase handler compares it (8 bytes on the
    host) against the value stored at analysis — full content verification
    at zero-copy cost. Position weights make permuted contents distinct.
    """
    w_off = jnp.arange(offsets_bd.shape[0], dtype=jnp.uint32) * jnp.uint32(
        2654435761
    ) + jnp.uint32(0x9E3779B9)
    w_col = jnp.arange(columns_bd.shape[0], dtype=jnp.uint32) * jnp.uint32(
        2246822519
    ) + jnp.uint32(0x85EBCA6B)
    h_off = jnp.sum(
        (offsets_bd.astype(jnp.uint32) + jnp.uint32(1)) * w_off, dtype=jnp.uint32
    )
    h_col = jnp.sum(
        (columns_bd.astype(jnp.uint32) + jnp.uint32(1)) * w_col, dtype=jnp.uint32
    )
    return jnp.stack([h_off, h_col])


def _expand_structure(offsets, columns, batch_size):
    """Block-diagonal CSR structure + fingerprint from a block pattern.

    ``offsets``/``columns`` are ``(n+1,)``/``(nnz,)`` for one shared pattern
    or ``(B, n+1)``/``(B, nnz)`` for per-block patterns; the result is the
    expanded ``(B*n + 1,)``/``(B*nnz,)`` int32 structure of the one big
    block-diagonal system plus its fingerprint. B=1 passes the arrays
    through untouched (zero-copy). The expansion is an elementwise int add —
    bandwidth-trivial, XLA-temporary — recomputed per phase call instead of
    persisting in device memory; the fingerprint rides the same pass.
    """
    offsets = offsets.astype(jnp.int32)
    columns = columns.astype(jnp.int32)
    if batch_size == 1:
        offsets_bd = offsets.reshape(-1)
        columns_bd = columns.reshape(-1)
        return offsets_bd, columns_bd, _structure_fingerprint(offsets_bd, columns_bd)
    n = offsets.shape[-1] - 1
    nnz = columns.shape[-1]
    shift = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    offs_2d = offsets.reshape((-1, offsets.shape[-1]))  # any leading axes
    cols_2d = columns.reshape((-1, columns.shape[-1]))
    body = (offs_2d[:, 1:] + shift * jnp.int32(nnz)).reshape(-1)
    offsets_bd = jnp.concatenate([jnp.zeros((1,), jnp.int32), body])
    columns_bd = (cols_2d + shift * jnp.int32(n)).reshape(-1)
    return offsets_bd, columns_bd, _structure_fingerprint(offsets_bd, columns_bd)


def _batch_of(token: FactorToken) -> int:
    """Total block count: one id axis per vmap level times the explicit-door
    static — the two doors compose (e.g. vmap over an explicit batch)."""
    return prod(token.id.shape[:-1]) * token.batch_size
