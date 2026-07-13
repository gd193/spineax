import jax
import jax.core
import jax.extend.core
from jax.interpreters import mlir
import jax.numpy as jnp

from ._validation import (
    _DTYPE_BY_SUFFIX,
    _cudss_debug_enabled,
    _dtype_suffix,
)

# Force JAX to initialize CUDA context BEFORE importing my C++ functions!!!!!!!!
jax.devices()

# Import the functions that return pointers from our compiled C++
from spineax import single_solve as _single_solve_mod, batch_solve as _batch_solve_mod  # type: ignore[attr-defined]  # noqa: E402

try:
    from spineax import pbatch_solve as _pbatch_solve_mod  # type: ignore[attr-defined]

    PBATCH_AVAILABLE = True
except ImportError:
    _pbatch_solve_mod = None
    PBATCH_AVAILABLE = False

KIND_SINGLE = "single"
KIND_BATCH = "batch"
KIND_PBATCH = "pbatch"
KIND_SINGLE_CONST = "single_const"
KIND_MULTI_RHS_CONST = "multi_rhs_const"


def _primitive_name(kind: str, suffix: str, *, return_diagnostics: bool = False) -> str:
    xonly = not return_diagnostics
    if kind == KIND_SINGLE:
        return f"solve_single_{suffix}" + ("_xonly" if xonly else "")
    if kind == KIND_BATCH:
        return f"solve_batch_{suffix}" + ("_xonly" if xonly else "")
    if kind == KIND_PBATCH:
        return f"solve_pbatch_{suffix}" + ("_xonly" if xonly else "")
    if kind == KIND_SINGLE_CONST:
        if return_diagnostics:
            raise ValueError("constant cuDSS primitives are solution-only")
        return f"solve_single_{suffix}_const_xonly"
    if kind == KIND_MULTI_RHS_CONST:
        if return_diagnostics:
            raise ValueError("constant cuDSS primitives are solution-only")
        return f"solve_multi_rhs_{suffix}_const_xonly"
    raise ValueError(f"Unsupported primitive kind: {kind}")


_PRIMITIVES: dict[str, jax.extend.core.Primitive] = {}
for _kind in (KIND_SINGLE, KIND_BATCH, KIND_PBATCH):
    for _suffix in _DTYPE_BY_SUFFIX:
        for _return_diagnostics in (True, False):
            _name = _primitive_name(
                _kind, _suffix, return_diagnostics=_return_diagnostics
            )
            _primitive = jax.extend.core.Primitive(_name)
            _primitive.multiple_results = True
            _PRIMITIVES[_name] = _primitive
            globals()[f"{_name}_p"] = _primitive
for _kind in (KIND_SINGLE_CONST, KIND_MULTI_RHS_CONST):
    for _suffix in _DTYPE_BY_SUFFIX:
        _name = _primitive_name(_kind, _suffix, return_diagnostics=False)
        _primitive = jax.extend.core.Primitive(_name)
        _primitive.multiple_results = True
        _PRIMITIVES[_name] = _primitive
        globals()[f"{_name}_p"] = _primitive


def _primitive_for(kind: str, dtype, *, return_diagnostics: bool):
    return _PRIMITIVES[
        _primitive_name(
            kind, _dtype_suffix(dtype), return_diagnostics=return_diagnostics
        )
    ]


def _batched_primitive_for(dtype, *, return_diagnostics: bool, prefer_pbatch: bool):
    return _primitive_for(
        KIND_PBATCH if prefer_pbatch else KIND_BATCH,
        dtype,
        return_diagnostics=return_diagnostics,
    )


def _const_xonly_solver_for_dtype(dtype, *, multi_rhs: bool):
    return _primitive_for(
        KIND_MULTI_RHS_CONST if multi_rhs else KIND_SINGLE_CONST,
        dtype,
        return_diagnostics=False,
    )


def compute_inertia_from_diag_perm(diag, perm, batch_size, matrix_dim):
    # Reorder diagonal according to permutation
    inv_perm = jnp.argsort(perm)
    diag_original_order = diag[inv_perm]
    out = diag_original_order.reshape([batch_size, matrix_dim])

    # cuDSS pivoting threshold seems to be 1e-13. everything above this on
    # plus or minus side seems to reliably indicate that particular inertia value.
    threshold = 1e-13
    positive = jnp.sum(out > threshold, axis=1)
    negative = jnp.sum(out < -threshold, axis=1)

    return jnp.stack([positive, negative], axis=1, dtype=jnp.int32)


# attempts at figuring out better estimates of inertia from diag and perm
# def compute_inertia_from_diag_perm(diag, perm, batch_size, matrix_dim, pivot_tol=1e-8, static_pivot_value=1e-8):
#     """Compute inertia (number of positive, negative, zero eigenvalues) from LDL^T factorization.

#     When cuDSS performs LDL^T factorization with static pivoting, huge negative values
#     (< -1e10) can appear in the diagonal for rank-deficient KKT matrices with H≈0.

#     Mathematical basis:
#         For KKT matrices [H A^T; A 0] with H≈0 (zero or tiny regularization):
#         - The huge negatives encode m = number of constraints
#         - Eigenvalue structure follows the saddle-point: m positive, m negative, (n-m) zero
#         - This is mathematically consistent with the rank-deficient structure

#         For standard matrices (no huge negatives):
#         - Use threshold-based counting with threshold = 1e-13 (cuDSS's static pivot value)

#         For PSD H matrices (detected post-hoc):
#         - Static pivots present, standard counting shows zero=0, small next value
#         - Mark as singular with zero=1

#     Args:
#         diag: Diagonal values from LDL^T factorization
#         perm: Permutation array from factorization
#         batch_size: Number of matrices in batch
#         matrix_dim: Dimension of each matrix (n+m for KKT systems)

#     Returns:
#         Array of shape (batch_size, 3) containing [positive, negative, zero] counts
#     """
#     # Reorder diagonal according to permutation
#     inv_perm = jnp.argsort(perm)
#     diag_original_order = diag[inv_perm]
#     out = diag_original_order.reshape([batch_size, matrix_dim])

#     # Count huge negative values in reordered diagonal
#     huge_neg_count = jnp.sum(out < -1e10, axis=1)

#     # Use standard threshold
#     threshold = 1e-13
#     positive = jnp.sum(out > threshold, axis=1)
#     negative = jnp.sum(out < -threshold, axis=1)
#     zero = matrix_dim - positive - negative

#     # Correction: For every 2 huge negative values, we need to add 1 to positive
#     # This accounts for the special encoding cuDSS uses
#     correction = (huge_neg_count + 1) // 2  # Round up division

#     positive = positive + correction
#     zero = zero - correction

#     return jnp.stack([positive, negative, zero], axis=1, dtype=jnp.int32)


# implementations ==============================================================
def general_single_solve_impl(
    name, b_values, csr_values, csr_offsets, csr_columns, device_id, mtype_id, mview_id
):

    call = jax.ffi.ffi_call(
        name,
        (
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),  # x
            jax.ShapeDtypeStruct((2,), jnp.int32),  # inertia [positive, negative]
        ),
        has_side_effect=True,
    )

    x, inertia = call(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )

    return [x, inertia]


def general_single_solve_xonly_impl(
    name, b_values, csr_values, csr_offsets, csr_columns, device_id, mtype_id, mview_id
):
    call = jax.ffi.ffi_call(
        name,
        (jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),),
        has_side_effect=True,
    )
    (x,) = call(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )
    return [x]


def general_batch_solve_impl(
    name,
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    batch_size,
    device_id,
    mtype_id,
    mview_id,
):

    call = jax.ffi.ffi_call(
        name,
        (
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),  # x
            jax.ShapeDtypeStruct((batch_size, 2), jnp.int32),  # inertia
        ),
        has_side_effect=True,
    )

    x, inertia = call(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        batch_size=batch_size,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )

    return [x, inertia]


def general_batch_solve_xonly_impl(
    name,
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    batch_size,
    device_id,
    mtype_id,
    mview_id,
):
    call = jax.ffi.ffi_call(
        name,
        (jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),),
        has_side_effect=True,
    )
    (x,) = call(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        batch_size=batch_size,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )
    return [x]


def general_pbatch_solve_impl(
    name,
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    batch_size,
    device_id,
    mtype_id,
    mview_id,
):

    call = jax.ffi.ffi_call(
        name,
        (
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),  # x
            jax.ShapeDtypeStruct((b_values.size,), b_values.dtype),  # diag
            jax.ShapeDtypeStruct((b_values.size,), jnp.int32),  # perm_reorder_row
        ),
        has_side_effect=True,
    )

    x, diag, perm = call(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        batch_size=batch_size,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )

    # Compute inertia instead of returning diag and perm
    matrix_dim = b_values.shape[1]  # Assuming b_values shape is (batch_size, n)
    inertia = compute_inertia_from_diag_perm(diag, perm, batch_size, matrix_dim)
    if _cudss_debug_enabled():
        jax.debug.print("inertia: {}", inertia)
    return [x, inertia]


def general_pbatch_solve_xonly_impl(
    name,
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    batch_size,
    device_id,
    mtype_id,
    mview_id,
):
    call = jax.ffi.ffi_call(
        name,
        (jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),),
        has_side_effect=True,
    )
    (x,) = call(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        batch_size=batch_size,
        device_id=device_id,
        mtype_id=mtype_id,
        mview_id=mview_id,
    )
    return [x]


def _impl_for(kind: str, *, return_diagnostics: bool):
    def _impl(*args, _kind=kind, _return_diagnostics=return_diagnostics, **kwargs):
        suffix = _dtype_suffix(args[1].dtype)
        name = _primitive_name(_kind, suffix, return_diagnostics=_return_diagnostics)
        if _kind == KIND_SINGLE:
            impl = (
                general_single_solve_impl
                if _return_diagnostics
                else general_single_solve_xonly_impl
            )
        elif _kind == KIND_BATCH:
            impl = (
                general_batch_solve_impl
                if _return_diagnostics
                else general_batch_solve_xonly_impl
            )
        elif _kind == KIND_PBATCH:
            impl = (
                general_pbatch_solve_impl
                if _return_diagnostics
                else general_pbatch_solve_xonly_impl
            )
        else:
            raise ValueError(f"Unsupported primitive kind: {_kind}")
        return impl(name, *args, **kwargs)

    return _impl


for _kind in (KIND_SINGLE, KIND_BATCH, KIND_PBATCH):
    for _suffix in _DTYPE_BY_SUFFIX:
        for _return_diagnostics in (True, False):
            _name = _primitive_name(
                _kind, _suffix, return_diagnostics=_return_diagnostics
            )
            _PRIMITIVES[_name].def_impl(
                _impl_for(_kind, return_diagnostics=_return_diagnostics)
            )
for _kind in (KIND_SINGLE_CONST, KIND_MULTI_RHS_CONST):
    for _suffix in _DTYPE_BY_SUFFIX:
        _name = _primitive_name(_kind, _suffix, return_diagnostics=False)

        def _const_impl(*args, _name=_name, **kwargs):
            # This immutable fingerprint is a primitive cache key only. Native
            # code receives and owns a snapshot of all CSR buffers.
            kwargs.pop("matrix_token")
            return general_single_solve_xonly_impl(_name, *args, **kwargs)

        _PRIMITIVES[_name].def_impl(_const_impl)


# registrations and lowerings ==================================================
try:
    from jax._src.lib import jaxlib_extension_version

    _NEW_FFI_API = jaxlib_extension_version >= 381
except ImportError:
    _NEW_FFI_API = False


def register_ffi(name: str, func, *, type: str, platform: str = "CUDA"):
    handler = getattr(func, f"handler_{type}")()
    state_type = (
        type.removeprefix("const_multi_rhs_xonly_")
        .removeprefix("const_xonly_")
        .removeprefix("xonly_")
    )
    state_dict = getattr(func, f"state_dict_{state_type}")()
    type_id = getattr(func, f"type_id_{state_type}")()
    if _NEW_FFI_API:
        jax.ffi.register_ffi_type(name, state_dict, platform=platform)
    else:
        jax.ffi.register_ffi_type_id(name, type_id, platform=platform)
    # order matters, ffi_target needs to be registered after type
    jax.ffi.register_ffi_target(name, handler, platform=platform)


def _ffi_type(kind: str, suffix: str, *, return_diagnostics: bool) -> str:
    if kind == KIND_SINGLE_CONST:
        return f"const_xonly_{suffix}"
    if kind == KIND_MULTI_RHS_CONST:
        return f"const_multi_rhs_xonly_{suffix}"
    return suffix if return_diagnostics else f"xonly_{suffix}"


def _backend_module(kind: str):
    if kind in (KIND_SINGLE, KIND_SINGLE_CONST, KIND_MULTI_RHS_CONST):
        return _single_solve_mod
    if kind == KIND_BATCH:
        return _batch_solve_mod
    if kind == KIND_PBATCH:
        return _pbatch_solve_mod
    raise ValueError(f"Unsupported primitive kind: {kind}")


def _register_primitive(
    name: str, kind: str, suffix: str, *, return_diagnostics: bool
) -> None:
    if kind == KIND_PBATCH and not PBATCH_AVAILABLE:
        return
    register_ffi(
        name,
        _backend_module(kind),
        type=_ffi_type(kind, suffix, return_diagnostics=return_diagnostics),
    )
    lowered = mlir.lower_fun(_PRIMITIVES[name].impl, multiple_results=True)
    mlir.register_lowering(_PRIMITIVES[name], lowered)


for _kind in (KIND_SINGLE, KIND_BATCH, KIND_PBATCH):
    for _suffix in _DTYPE_BY_SUFFIX:
        for _return_diagnostics in (True, False):
            _name = _primitive_name(
                _kind, _suffix, return_diagnostics=_return_diagnostics
            )
            _register_primitive(
                _name, _kind, _suffix, return_diagnostics=_return_diagnostics
            )
for _kind in (KIND_SINGLE_CONST, KIND_MULTI_RHS_CONST):
    for _suffix in _DTYPE_BY_SUFFIX:
        _name = _primitive_name(_kind, _suffix, return_diagnostics=False)
        _register_primitive(_name, _kind, _suffix, return_diagnostics=False)


# abstract evaluations =========================================================
def solve_aval(
    b_values, csr_values, csr_offsets, csr_columns, device_id, mtype_id, mview_id
):
    return [
        jax.core.ShapedArray(b_values.shape, b_values.dtype),  # x
        jax.core.ShapedArray((2,), jnp.int32),  # inertia [positive, negative]
    ]


def solve_xonly_aval(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    device_id,
    mtype_id,
    mview_id,
    matrix_token=None,
):
    return [jax.core.ShapedArray(b_values.shape, b_values.dtype)]


def solve_batch_aval(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    batch_size,
    device_id,
    mtype_id,
    mview_id,
):
    return [
        jax.core.ShapedArray(b_values.shape, b_values.dtype),  # x
        jax.core.ShapedArray(
            (batch_size, 2), jnp.int32
        ),  # inertia [positive, negative]
    ]


def solve_batch_xonly_aval(
    b_values,
    csr_values,
    csr_offsets,
    csr_columns,
    batch_size,
    device_id,
    mtype_id,
    mview_id,
):
    return [jax.core.ShapedArray(b_values.shape, b_values.dtype)]


def _abstract_eval_for(kind: str, *, return_diagnostics: bool):
    if kind in (KIND_SINGLE_CONST, KIND_MULTI_RHS_CONST):
        return solve_xonly_aval
    if kind == KIND_SINGLE:
        return solve_aval if return_diagnostics else solve_xonly_aval
    return solve_batch_aval if return_diagnostics else solve_batch_xonly_aval


for _kind in (KIND_SINGLE, KIND_BATCH, KIND_PBATCH):
    for _suffix in _DTYPE_BY_SUFFIX:
        for _return_diagnostics in (True, False):
            _name = _primitive_name(
                _kind, _suffix, return_diagnostics=_return_diagnostics
            )
            _PRIMITIVES[_name].def_abstract_eval(
                _abstract_eval_for(_kind, return_diagnostics=_return_diagnostics)
            )
for _kind in (KIND_SINGLE_CONST, KIND_MULTI_RHS_CONST):
    for _suffix in _DTYPE_BY_SUFFIX:
        _name = _primitive_name(_kind, _suffix, return_diagnostics=False)
        _PRIMITIVES[_name].def_abstract_eval(solve_xonly_aval)
