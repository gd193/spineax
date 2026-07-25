# pyright: reportMissingImports=false
"""Tests for the factor-once constant-CSR convenience API."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spineax import cudss
from spineax.cudss.constant import ConstantCSRCuDSSSolver

jax.config.update("jax_enable_x64", True)


def _require_gpu():
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("CUDA device required for cuDSS tests")


def _system(dtype=jnp.float64):
    values = np.asarray([4.0, 1.0, 3.0, 1.0, 2.0], dtype=dtype)
    offsets = np.asarray([0, 2, 4, 5], dtype=np.int32)
    columns = np.asarray([0, 1, 1, 2, 2], dtype=np.int32)
    matrix = np.asarray(
        [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
        dtype=dtype,
    )
    rhs = jnp.asarray([1.0, 2.0, 3.0], dtype=dtype)
    expected = jnp.asarray(np.linalg.solve(matrix, np.asarray(rhs)), dtype=dtype)
    return values, offsets, columns, matrix, rhs, expected


def _solver(values=None):
    base_values, offsets, columns, _, _, _ = _system()
    return ConstantCSRCuDSSSolver(
        base_values if values is None else values,
        offsets,
        columns,
        mtype_id="spd",
    )


def test_public_export_and_metadata():
    _require_gpu()
    solver = _solver()
    assert cudss.ConstantCSRCuDSSSolver is ConstantCSRCuDSSSolver
    assert solver.n == 3
    assert solver.nnz == 5
    assert solver.batch_size == 1
    assert solver.dtype == jnp.dtype(jnp.float64)
    assert jax.tree_util.tree_leaves(solver) == [solver]


def test_repeated_jit_and_multi_rhs_solve_reuse_factorization():
    _require_gpu()
    _, _, _, _, rhs, expected = _system()
    solver = _solver()

    @jax.jit
    def run(value):
        return solver(value)

    rebuilds = cudss.rebuild_count()
    np.testing.assert_allclose(np.asarray(run(rhs)), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(run(2.0 * rhs)), np.asarray(2.0 * expected))
    np.testing.assert_allclose(
        np.asarray(solver(jnp.stack([rhs, 3.0 * rhs]))),
        np.asarray(jnp.stack([expected, 3.0 * expected])),
    )
    assert cudss.rebuild_count() == rebuilds


def test_vmap_rhs_and_keyword_refinement():
    _require_gpu()
    _, _, _, _, rhs, expected = _system()
    solver = _solver()
    scales = jnp.asarray([0.5, 1.0, 2.0, 4.0])
    rhs_batch = scales[:, None] * rhs

    output = jax.jit(jax.vmap(solver))(rhs_batch)
    np.testing.assert_allclose(
        np.asarray(output), np.asarray(scales[:, None] * expected), rtol=1e-12
    )
    refined = solver(rhs, ir_nsteps=1)
    np.testing.assert_allclose(np.asarray(refined), np.asarray(expected), rtol=1e-12)


def test_constant_explicit_batch_and_query():
    _require_gpu()
    values, offsets, columns, _, rhs, expected = _system()
    values_batch = jnp.stack([jnp.asarray(values), 2.0 * jnp.asarray(values)])
    solver = ConstantCSRCuDSSSolver(values_batch, offsets, columns, mtype_id="spd")
    output = solver(jnp.stack([rhs, rhs]))

    np.testing.assert_allclose(np.asarray(output[0]), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(output[1]), np.asarray(expected / 2.0))
    inertia = cudss.inertia(solver.query(), batch_size=2)
    np.testing.assert_array_equal(np.asarray(inertia), [[3, 0], [3, 0]])


def test_owned_matrix_survives_release_and_caller_mutation():
    _require_gpu()
    values, _, _, _, rhs, expected = _system()
    solver = _solver(values)
    values[:] = 0.0

    assert solver.release()
    np.testing.assert_allclose(np.asarray(solver(rhs)), np.asarray(expected))


def test_rhs_validation_and_call_signature():
    _require_gpu()
    _, _, _, _, rhs, _ = _system()
    solver = _solver()

    with pytest.raises(ValueError, match="rhs trailing dim"):
        solver(rhs[:-1])
    with pytest.raises(ValueError, match="rhs trailing dim"):
        solver(jnp.ones((2, rhs.size - 1), dtype=rhs.dtype))
    signature = inspect.signature(solver)
    assert signature.parameters["ir_nsteps"].kind is inspect.Parameter.KEYWORD_ONLY
