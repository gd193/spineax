import functools as ft
import os

import jax
import jax.core
import jax.extend.core
from jax.interpreters import mlir, batching
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
import equinox as eqx


def _cudss_debug_enabled() -> bool:
    return os.environ.get("SPINEAX_CUDSS_DEBUG", "").lower() in {"1", "true", "yes", "on"}


# Force JAX to initialize CUDA context BEFORE importing my C++ functions!!!!!!!!
jax.devices()

# Import the functions that return pointers from our compiled C++
from spineax import single_solve as _single_solve_mod, batch_solve as _batch_solve_mod  # type: ignore[attr-defined]
try:
    from spineax import pbatch_solve as _pbatch_solve_mod  # type: ignore[attr-defined]
    PBATCH_AVAILABLE = True
except ImportError:
    _pbatch_solve_mod = None
    PBATCH_AVAILABLE = False

# primitives ===================================================================
_DTYPE_SUFFIXES = {
    jnp.dtype(jnp.float32): "f32",
    jnp.dtype(jnp.float64): "f64",
    jnp.dtype(jnp.complex64): "c64",
    jnp.dtype(jnp.complex128): "c128",
}
_DTYPE_BY_SUFFIX = {suffix: dtype for dtype, suffix in _DTYPE_SUFFIXES.items()}

KIND_SINGLE = "single"
KIND_BATCH = "batch"
KIND_PBATCH = "pbatch"
KIND_SINGLE_CONST = "single_const"
KIND_MULTI_RHS_CONST = "multi_rhs_const"


def _dtype_suffix(dtype) -> str:
    normalized = jnp.dtype(dtype)
    try:
        return _DTYPE_SUFFIXES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


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
            _name = _primitive_name(_kind, _suffix, return_diagnostics=_return_diagnostics)
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
    return _PRIMITIVES[_primitive_name(kind, _dtype_suffix(dtype), return_diagnostics=return_diagnostics)]


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


def _log_dtype(dtype) -> None:
    if _cudss_debug_enabled() and jnp.dtype(dtype) in {jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)}:
        print(f"solving with {_dtype_suffix(dtype).replace('f', 'float')}")

# Helper function to compute inertia from diag and perm
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
        name, 
        b_values, 
        csr_values, 
        csr_offsets,
        csr_columns,
        device_id, 
        mtype_id, 
        mview_id
    ):

    call = jax.ffi.ffi_call(
        name,
        (
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),   # x
            jax.ShapeDtypeStruct((2,), jnp.int32),                  # inertia [positive, negative]
        ),
        has_side_effect=True
    )

    x, inertia = call(
        b_values, 
        csr_values, 
        csr_offsets,
        csr_columns,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

    return [x, inertia]


def general_single_solve_xonly_impl(
        name,
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        device_id,
        mtype_id,
        mview_id
    ):
    call = jax.ffi.ffi_call(
        name,
        (jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),),
        has_side_effect=True
    )
    x, = call(
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
        mview_id
    ):

    call = jax.ffi.ffi_call(
        name,
        (
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),   # x
            jax.ShapeDtypeStruct((batch_size, 2), jnp.int32),       # inertia
        ),
        has_side_effect=True
    )

    x, inertia = call(
        b_values, 
        csr_values, 
        csr_offsets,
        csr_columns,
        batch_size = batch_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
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
        mview_id
    ):
    call = jax.ffi.ffi_call(
        name,
        (jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),),
        has_side_effect=True
    )
    x, = call(
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
        mview_id
    ):

    call = jax.ffi.ffi_call(
        name,
        (
            jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),   # x
            jax.ShapeDtypeStruct((b_values.size,), b_values.dtype),   # diag
            jax.ShapeDtypeStruct((b_values.size,), jnp.int32),        # perm_reorder_row
        ),
        has_side_effect=True
    )

    x, diag, perm = call(
        b_values, 
        csr_values, 
        csr_offsets,
        csr_columns,
        batch_size = batch_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
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
        mview_id
    ):
    call = jax.ffi.ffi_call(
        name,
        (jax.ShapeDtypeStruct(b_values.shape, b_values.dtype),),
        has_side_effect=True
    )
    x, = call(
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
            impl = general_single_solve_impl if _return_diagnostics else general_single_solve_xonly_impl
        elif _kind == KIND_BATCH:
            impl = general_batch_solve_impl if _return_diagnostics else general_batch_solve_xonly_impl
        elif _kind == KIND_PBATCH:
            impl = general_pbatch_solve_impl if _return_diagnostics else general_pbatch_solve_xonly_impl
        else:
            raise ValueError(f"Unsupported primitive kind: {_kind}")
        return impl(name, *args, **kwargs)
    return _impl


for _kind in (KIND_SINGLE, KIND_BATCH, KIND_PBATCH):
    for _suffix in _DTYPE_BY_SUFFIX:
        for _return_diagnostics in (True, False):
            _name = _primitive_name(_kind, _suffix, return_diagnostics=_return_diagnostics)
            _PRIMITIVES[_name].def_impl(_impl_for(_kind, return_diagnostics=_return_diagnostics))
for _kind in (KIND_SINGLE_CONST, KIND_MULTI_RHS_CONST):
    for _suffix in _DTYPE_BY_SUFFIX:
        _name = _primitive_name(_kind, _suffix, return_diagnostics=False)
        def _const_impl(*args, _name=_name, **kwargs):
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


def _register_primitive(name: str, kind: str, suffix: str, *, return_diagnostics: bool) -> None:
    if kind == KIND_PBATCH and not PBATCH_AVAILABLE:
        return
    register_ffi(name, _backend_module(kind), type=_ffi_type(kind, suffix, return_diagnostics=return_diagnostics))
    lowered = mlir.lower_fun(_PRIMITIVES[name].impl, multiple_results=True)
    mlir.register_lowering(_PRIMITIVES[name], lowered)


for _kind in (KIND_SINGLE, KIND_BATCH, KIND_PBATCH):
    for _suffix in _DTYPE_BY_SUFFIX:
        for _return_diagnostics in (True, False):
            _name = _primitive_name(_kind, _suffix, return_diagnostics=_return_diagnostics)
            _register_primitive(_name, _kind, _suffix, return_diagnostics=_return_diagnostics)
for _kind in (KIND_SINGLE_CONST, KIND_MULTI_RHS_CONST):
    for _suffix in _DTYPE_BY_SUFFIX:
        _name = _primitive_name(_kind, _suffix, return_diagnostics=False)
        _register_primitive(_name, _kind, _suffix, return_diagnostics=False)


# abstract evaluations =========================================================
def solve_aval(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        device_id,
        mtype_id,
        mview_id
    ):
    return [
            jax.core.ShapedArray(b_values.shape, b_values.dtype),       # x
            jax.core.ShapedArray((2,), jnp.int32),                      # inertia [positive, negative]
        ]


def solve_xonly_aval(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        device_id,
        mtype_id,
        mview_id
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
        mview_id
    ):
    return [
            jax.core.ShapedArray(b_values.shape, b_values.dtype),       # x
            jax.core.ShapedArray((batch_size, 2), jnp.int32),           # inertia [positive, negative]
        ]


def solve_batch_xonly_aval(
        b_values,
        csr_values,
        csr_offsets,
        csr_columns,
        batch_size,
        device_id,
        mtype_id,
        mview_id
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
            _name = _primitive_name(_kind, _suffix, return_diagnostics=_return_diagnostics)
            _PRIMITIVES[_name].def_abstract_eval(
                _abstract_eval_for(_kind, return_diagnostics=_return_diagnostics)
            )
for _kind in (KIND_SINGLE_CONST, KIND_MULTI_RHS_CONST):
    for _suffix in _DTYPE_BY_SUFFIX:
        _name = _primitive_name(_kind, _suffix, return_diagnostics=False)
        _PRIMITIVES[_name].def_abstract_eval(solve_xonly_aval)


# single solve interface =======================================================
@ft.partial(
    jax.jit, static_argnames=[
        "device_id",
        "mtype_id",
        "mview_id",
        "return_diagnostics",
        "constant_values",
    ]
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
    ):
    if constant_values and return_diagnostics:
        raise ValueError("constant_values=True requires return_diagnostics=False")
    if constant_values:
        if b_values.ndim not in (1, 2):
            raise ValueError("constant_values cuDSS supports 1D RHS or 2D (batch, n) RHS")
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
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
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
    jax.jit, static_argnames=[
        "batch_size",
        "device_id",
        "mtype_id",
        "mview_id",
        "return_diagnostics"
    ]
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
        return_diagnostics=True
    ):
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
        batch_size = batch_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )


# manual psuedo batch solve interface =================================================
@ft.partial(
    jax.jit, static_argnames=[
        "batch_size",
        "device_id",
        "mtype_id",
        "mview_id",
        "return_diagnostics"
    ]
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
        return_diagnostics=True
    ):
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
        batch_size = batch_size,
        device_id = device_id, 
        mtype_id = mtype_id,
        mview_id = mview_id,
    )

# vmap batch solve interface ===================================================

def solve_single_f32_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)
def solve_single_f64_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)
def solve_single_c64_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)
def solve_single_c128_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, **kwargs)
def solve_single_f32_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, return_diagnostics=False, **kwargs)
def solve_single_f64_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, return_diagnostics=False, **kwargs)
def solve_single_c64_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, return_diagnostics=False, **kwargs)
def solve_single_c128_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return general_solve_vmap(vector_arg_values, batch_axes, return_diagnostics=False, **kwargs)

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
        raise NotImplementedError("constant_values cuDSS vmap supports only batched RHS with constant matrix")
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

# Use pseudo_batch if available (provides correct inertia), fall back to batch_solve otherwise
vmap_using_pseudo_batch = PBATCH_AVAILABLE
if not PBATCH_AVAILABLE:
    import warnings
    warnings.warn(
        "pbatch_solve not available (CUDA version mismatch?). "
        "Falling back to batch_solve. Batched inertia will not be computed correctly.",
        RuntimeWarning
    )

def general_solve_vmap(
    vector_arg_values,  # [b_values, csr_values, csr_offsets, csr_columns]
    batch_axes,         # [b_values, csr_values, csr_offsets, csr_columns]
    **kwargs            # static params
):

    b_values, csr_values, csr_offsets, csr_columns = vector_arg_values
    a_b, a_val, a_off, a_col = batch_axes
    return_diagnostics = kwargs.get("return_diagnostics", True)
    bind_kwargs = dict(kwargs)
    bind_kwargs.pop("return_diagnostics", None)
    out_axes = (0, 0) if return_diagnostics else (0,)

    if _cudss_debug_enabled():
        jax.debug.print("vmap batch_axes: a_b={}, a_val={}, a_off={}, a_col={}", a_b, a_val, a_off, a_col)

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
        raise NotImplementedError("don't support batches of heterogeneous sparsity patterns yet (its coming tho...)")

    if all(ax is None for ax in (a_val, a_b)):
        raise NotImplementedError("Only batched csr_values and b_values are supported right now")
    
    # the non-batched path
    if a_val is None and a_b is None:
        return solve(*vector_arg_values, **kwargs), ((None, None) if return_diagnostics else (None,))

    # if only one of the sets of values are batched
    elif (a_val is None) != (a_b is None):
        if a_b is not None and a_val is None:
            # Only b is batched - broadcast csr_values to match batch dimension
            csr_values_batched = jnp.broadcast_to(csr_values[None, :], (b_values.shape[0],) + csr_values.shape)
            vector_arg_values = (b_values, csr_values_batched, csr_offsets, csr_columns)

            solver = _batched_primitive_for(
                csr_values.dtype,
                return_diagnostics=return_diagnostics,
                prefer_pbatch=vmap_using_pseudo_batch,
            )

            return solver.bind(*vector_arg_values, batch_size=b_values.shape[0], **bind_kwargs), out_axes
        else:
            # Only csr_values is batched (not b) - not supported
            raise NotImplementedError("Only csr_values batched (not b_values) is not supported")

    # the batched path binding
    elif a_val is not None and a_b is not None:
        solver = _batched_primitive_for(
            csr_values.dtype,
            return_diagnostics=return_diagnostics,
            prefer_pbatch=vmap_using_pseudo_batch,
        )
        return solver.bind(*vector_arg_values, batch_size=b_values.shape[0], **bind_kwargs), out_axes
    
    else:
        raise NotImplementedError("This path should not be possible")

for _suffix in _DTYPE_BY_SUFFIX:
    batching.primitive_batchers[
        _primitive_for(KIND_SINGLE, _DTYPE_BY_SUFFIX[_suffix], return_diagnostics=True)
    ] = general_solve_vmap
    batching.primitive_batchers[
        _primitive_for(KIND_SINGLE, _DTYPE_BY_SUFFIX[_suffix], return_diagnostics=False)
    ] = lambda vector_arg_values, batch_axes, **kwargs: general_solve_vmap(
        vector_arg_values, batch_axes, return_diagnostics=False, **kwargs
    )
    batching.primitive_batchers[
        _const_xonly_solver_for_dtype(_DTYPE_BY_SUFFIX[_suffix], multi_rhs=False)
    ] = solve_single_const_xonly_vmap

# vmap of vmap
def solve_batch_vmap(vector_arg_values, batch_axes, **kwargs):
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
        raise NotImplementedError("don't support batches of heterogeneous sparsity patterns yet (its coming tho...)")

    if a_b is None and a_val is None:
        # Not actually batching
        return batch_solve(*vector_arg_values, **kwargs), ((None, None) if return_diagnostics else (None,))
    
    # Flatten nested batches
    batch_size1 = b_values.shape[0]
    batch_size2 = b_values.shape[1]
    total_batch = batch_size1 * batch_size2
    
    b_flat = b_values.reshape(total_batch, -1)
    csr_flat = csr_values.reshape(total_batch, -1)
    
    # the non-batched path
    if a_val is None and a_b is None:
        return solve(*vector_arg_values, **kwargs), ((None, None) if return_diagnostics else (None,))

    # if only one of the sets of values are batched
    elif (a_val is None) != (a_b is None):
        if a_b is not None and a_val is None:
            # Only b is batched in nested vmap - broadcast csr_values
            csr_values_batched = jnp.broadcast_to(csr_values[None, :, :], (b_values.shape[0],) + csr_values.shape)
            b_flat = b_values.reshape(-1, b_values.shape[-1])
            csr_flat = csr_values_batched.reshape(-1, csr_values.shape[-1])

            solver = _batched_primitive_for(
                csr_values.dtype,
                return_diagnostics=return_diagnostics,
                prefer_pbatch=vmap_using_pseudo_batch,
            )

            total_batch = b_flat.shape[0]
            # Remove old batch_size from kwargs
            kwargs_copy = dict(bind_kwargs)
            kwargs_copy.pop("batch_size", None)
            outputs = solver.bind(
                b_flat, csr_flat, csr_offsets, csr_columns,
                batch_size=total_batch,
                **kwargs_copy
            )

            # Reshape back
            x = outputs[0].reshape(b_values.shape[0], b_values.shape[1], -1)
            if not return_diagnostics:
                return (x,), out_axes
            inertia = outputs[1].reshape(b_values.shape[0], b_values.shape[1], 2)

            return (x, inertia), out_axes
        else:
            # Only csr_values is batched (not b) - not supported
            raise NotImplementedError("Only csr_values batched (not b_values) is not supported")

    elif a_val is not None and a_b is not None:
        solver = _batched_primitive_for(
            csr_values.dtype,
            return_diagnostics=return_diagnostics,
            prefer_pbatch=vmap_using_pseudo_batch,
        )

    # Remove batch_size from kwargs if present (happens with nested vmap)
    kwargs_copy = dict(bind_kwargs)
    kwargs_copy.pop("batch_size", None)

    outputs = solver.bind(
        b_flat, csr_flat, csr_offsets, csr_columns,
        batch_size=total_batch,
        **kwargs_copy
    )
    
    # Reshape back
    x = outputs[0].reshape(batch_size1, batch_size2, -1)
    if not return_diagnostics:
        return (x,), out_axes
    inertia = outputs[1].reshape(batch_size1, batch_size2, 2)
    
    return (x, inertia), out_axes

def solve_batch_xonly_vmap(vector_arg_values, batch_axes, **kwargs):
    return solve_batch_vmap(vector_arg_values, batch_axes, return_diagnostics=False, **kwargs)

for _kind in (KIND_BATCH, KIND_PBATCH):
    for _suffix in _DTYPE_BY_SUFFIX:
        batching.primitive_batchers[
            _primitive_for(_kind, _DTYPE_BY_SUFFIX[_suffix], return_diagnostics=True)
        ] = solve_batch_vmap
        batching.primitive_batchers[
            _primitive_for(_kind, _DTYPE_BY_SUFFIX[_suffix], return_diagnostics=False)
        ] = solve_batch_xonly_vmap

# create python side composable class to ensure validity of the columns and offsets
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

    def __init__(
        self,
        csr_offsets,
        csr_columns,
        device_id,
        mtype_id,
        mview_id,
        return_diagnostics: bool = True,
    ):
        self.csr_offsets = csr_offsets
        self.csr_columns = csr_columns
        self.device_id = device_id
        self.mtype_id = mtype_id
        self.mview_id = mview_id
        self.return_diagnostics = bool(return_diagnostics)

    def __call__(self, b, csr_values):
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
    csr_values: Array
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
        self.csr_offsets = csr_offsets
        self.csr_columns = csr_columns
        self.csr_values = csr_values
        self.device_id = device_id
        self.mtype_id = mtype_id
        self.mview_id = mview_id

    def __call__(self, b):
        return _solve(
            b,
            self.csr_values,
            csr_offsets=self.csr_offsets,
            csr_columns=self.csr_columns,
            device_id=self.device_id,
            mtype_id=self.mtype_id,
            mview_id=self.mview_id,
            return_diagnostics=False,
            constant_values=True,
        )
