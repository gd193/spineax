# pyright: reportMissingImports=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""Native extension bootstrap and the single canonical typed-FFI primitive."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.mlir import ir as _ir
from jax.interpreters import mlir as _jmlir
from jaxlib.mlir.dialects import stablehlo as _stablehlo

from ._core import _SUFFIXES, _suffix


# Force JAX to initialize the CUDA context before importing the native module.
jax.devices()

try:
    from spineax import pbatch_solve as _ps  # native nanobind module
except ImportError as e:
    raise ImportError(
        "spineax.cudss.solver requires the pbatch_solve native module: the "
        "token API treats a single solve as the batch_size=1 case of the "
        "block-diagonal batch construction, which lives there. Was the "
        "package built with its CUDA extension (pip install with CUDA "
        "toolkit + cuDSS >= 0.8 available)?"
    ) from e


# Plain FFI handlers (no instantiate/State): register_ffi_target only.
for _s in _SUFFIXES:
    _handlers = getattr(_ps, f"token_handlers_{_s}")()
    for _op in ("analyze", "factorize", "refactorize", "solve", "query"):
        jax.ffi.register_ffi_target(
            f"spineax_token_{_op}_{_s}", _handlers[_op], platform="CUDA"
        )
    jax.ffi.register_ffi_target(
        f"spineax_csr_transpose_{_s}",
        getattr(_ps, f"handler_{_s}")(),
        platform="CUDA",
    )
jax.ffi.register_ffi_target(
    "spineax_csr_transpose_order",
    _ps.order_handler(),
    platform="CUDA",
)

# raw FFI calls ================================================================
_FFI_API_VERSION = 4  # typed FFI: the contract solver.cpp's handlers compile


def _ffi_probe_jaxpr():
    """Trace (never run) a jax.ffi call to harvest jax's own FFI machinery:
    the effect instance (already in jax's control-flow allowlists, so our
    calls are while/cond-legal exactly like jax.ffi ones) and the concrete
    aval class — by value, not by private-module name."""
    fn = jax.ffi.ffi_call(
        "__spineax_probe", jax.ShapeDtypeStruct((1,), jnp.int32), has_side_effect=True
    )
    return jax.make_jaxpr(lambda: fn())()


_probe = _ffi_probe_jaxpr()
(_SPINEAX_FFI_EFFECT,) = tuple(_probe.effects)
_ShapedArray = type(_probe.out_avals[0])

_ffi_p = jax.extend.core.Primitive("spineax_ffi")
_ffi_p.multiple_results = True


@cache
def _ffi_eager(name, out_avals, attrs, has_side_effect):
    # eager binds execute through jit (one compiled call per unique config)
    return jax.jit(
        lambda *a: _ffi_p.bind(
            *a,
            name=name,
            out_avals=out_avals,
            attrs=attrs,
            has_side_effect=has_side_effect,
        )
    )


_ffi_p.def_impl(
    lambda *args, name, out_avals, attrs, has_side_effect: _ffi_eager(
        name, out_avals, attrs, has_side_effect
    )(*args)
)


# issue #18 fixed by Igor Kuszczak - no need for effectful things after token rework.
@_ffi_p.def_abstract_eval
def _ffi_abstract_eval(*avals_in, name, out_avals, attrs, has_side_effect):
    del avals_in, name, attrs, has_side_effect
    return list(out_avals)


def _ir_tensor_type(aval):
    dt = jnp.dtype(aval.dtype)
    if dt == jnp.float32:
        element_type = _ir.F32Type.get()
    elif dt == jnp.float64:
        element_type = _ir.F64Type.get()
    elif dt == jnp.complex64:
        element_type = _ir.ComplexType.get(_ir.F32Type.get())
    elif dt == jnp.complex128:
        element_type = _ir.ComplexType.get(_ir.F64Type.get())
    elif dt == jnp.int32:
        element_type = _ir.IntegerType.get_signless(32)
    elif dt == jnp.int64:
        element_type = _ir.IntegerType.get_signless(64)
    elif dt == jnp.uint32:
        element_type = _ir.IntegerType.get_unsigned(32)
    else:
        raise ValueError(f"spineax ffi: unhandled dtype {dt}")
    return _ir.RankedTensorType.get(aval.shape, element_type)


def _ffi_lowering(ctx, *operands, name, out_avals, attrs, has_side_effect):
    del out_avals  # ctx.avals_out is authoritative
    # Build the stablehlo.custom_call directly: the TYPED_FFI form is
    # api_version=4 with a DICTIONARY backend_config (the stablehlo verifier
    # enforces this), which jax's mlir.custom_call helper cannot express —
    # its dict branch is what rewrites to api_version=1 + mhlo.backend_config.
    i64 = _ir.IntegerType.get_signless(64)

    def _layout(a):  # row-major, minor-to-major order (as the FFI expects)
        return _ir.DenseIntElementsAttr.get(
            np.atleast_1d(np.asarray(tuple(reversed(range(a.ndim))), np.int64)),
            type=_ir.IndexType.get(),
        )

    def _integer_attr(value):
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"spineax ffi: invalid integer attribute {value!r}"
            ) from error
        return _ir.IntegerAttr.get(i64, integer)

    attributes = {
        "call_target_name": _ir.StringAttr.get(name),
        "has_side_effect": _ir.BoolAttr.get(has_side_effect),
        "backend_config": _ir.DictAttr.get(
            {key: _integer_attr(value) for key, value in attrs}
        ),
        "api_version": _ir.IntegerAttr.get(
            _ir.IntegerType.get_signless(32), _FFI_API_VERSION
        ),
        "called_computations": _ir.ArrayAttr.get([]),
        "operand_layouts": _ir.ArrayAttr.get([_layout(aval) for aval in ctx.avals_in]),
        "result_layouts": _ir.ArrayAttr.get([_layout(aval) for aval in ctx.avals_out]),
    }
    op = _stablehlo.CustomCallOp.build_generic(
        results=[_ir_tensor_type(a) for a in ctx.avals_out],
        operands=list(operands),
        attributes=attributes,
    )
    return op.results


_jmlir.register_lowering(_ffi_p, _ffi_lowering)


def _ffi_call4(name, out_specs, *operands, has_side_effect=True, **attrs):
    """Emit one spineax typed-FFI custom call (explicit api_version 4).

    ``out_specs`` is a tuple of ShapeDtypeStructs; returns a list of arrays.
    Integer ``attrs`` become i64 FFI attributes (solver.cpp Attr<int64_t>).
    """
    out_avals = tuple(_ShapedArray(s.shape, jnp.dtype(s.dtype)) for s in out_specs)
    return _ffi_p.bind(
        *operands,
        name=name,
        out_avals=out_avals,
        attrs=tuple(sorted(attrs.items())),
        has_side_effect=has_side_effect,
    )


# The per-entry cuDSS config, threaded as ONE tuple of i64 FFI attrs. It
# rides every phase call (not just analyze) so a non-resident id (renamed
# away by a later numeric phase, or LRU-evicted) can be rebuilt in the
# handler from the call's own operands (self-healing; see solver.cpp).
_CFG_KEYS = ("device_id", "mtype_id", "mview_id", "reordering_id", "memory_id")


def _cfg_of(token):
    return tuple(getattr(token, key) for key in _CFG_KEYS)


def _config_attrs(cfg):
    return {key: cfg[index] for index, key in enumerate(_CFG_KEYS)}


def _ffi_analyze(values, offsets_bd, columns_bd, fingerprint, *, batch_size, cfg):
    (token_id,) = _ffi_call4(
        f"spineax_token_analyze_{_suffix(values.dtype)}",
        (jax.ShapeDtypeStruct((1,), jnp.int32),),
        values,
        offsets_bd,
        columns_bd,
        fingerprint,
        batch_size=batch_size,
        **_config_attrs(cfg),
    )
    return token_id


def _ffi_numeric(token_id, offsets_bd, columns_bd, fingerprint, values, *, op, cfg):
    (out_id,) = _ffi_call4(
        f"spineax_token_{op}_{_suffix(values.dtype)}",
        (jax.ShapeDtypeStruct(token_id.shape, jnp.int32),),  # token (FRESH id)
        token_id,
        offsets_bd,
        columns_bd,
        fingerprint,
        values,
        **_config_attrs(cfg),
    )
    return out_id


def _ffi_solve(token_id, offsets_bd, columns_bd, fingerprint, values, b, *, cfg):
    (x,) = _ffi_call4(
        f"spineax_token_solve_{_suffix(b.dtype)}",
        (jax.ShapeDtypeStruct(b.shape, b.dtype),),
        token_id,
        offsets_bd,
        columns_bd,
        fingerprint,
        values,
        b,
        **_config_attrs(cfg),
    )
    return x


def _nd_partition_tree_size() -> int:
    return _ps.nd_partition_tree_size()


def _token_release(token_id) -> bool:
    native_id = jax.device_get(token_id).ravel()[0].item()
    return _ps.token_release(native_id)


def _token_registry_size() -> int:
    return _ps.token_registry_size()


def _token_cache_capacity() -> int:
    return _ps.token_cache_capacity()


def _token_rebuild_count() -> int:
    return _ps.token_rebuild_count()
