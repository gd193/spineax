from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import equinox as eqx
import pytest
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp


def get_test_system(dtype=jnp.float32):
    a = jnp.array(
        [
            [4.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=dtype,
    )
    b = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0], dtype=dtype)
    a_sym = a + a.T - jnp.diag(jnp.diag(a))
    true_x = jnp.linalg.solve(a_sym, b)
    lhs = jsparse.BCSR.fromdense(a_sym)
    return lhs.indptr, lhs.indices, lhs.data, b, true_x


def make_constant_solver(dtype=jnp.float32):
    from spineax.cudss.solver import ConstantCSRCuDSSSolver  # type: ignore[import-not-found]

    csr_offsets, csr_columns, csr_values, b, true_x = get_test_system(dtype)
    solver = ConstantCSRCuDSSSolver(
        csr_offsets,
        csr_columns,
        csr_values,
        0,
        3,
        0,
    )
    return solver, b, true_x


def test_cudss_solver_no_longer_accepts_constant_values_flag():
    from spineax.cudss.solver import CuDSSSolver  # type: ignore[import-not-found]

    csr_offsets, csr_columns, _, _, _ = get_test_system()
    with pytest.raises(TypeError):
        CuDSSSolver(
            csr_offsets,
            csr_columns,
            0,
            3,
            0,
            return_diagnostics=False,
            constant_values=True,
        )


def test_constant_solver_call_signature_prevents_changed_values_misuse():
    solver, b, _ = make_constant_solver()
    call = getattr(solver, "__call__")
    with pytest.raises(TypeError):
        call(b, jnp.ones((1,), dtype=b.dtype))


def test_constant_values_solution_only_repeated_executable_calls():
    solver, b, true_x = make_constant_solver()

    @jax.jit
    def solve_once(rhs):
        return solver(rhs)[0]

    x1 = solve_once(b)
    x2 = solve_once(b * 2.0)
    assert jnp.allclose(x1, true_x, atol=1e-5)
    assert jnp.allclose(x2, true_x * 2.0, atol=2e-5)


def test_constant_values_solution_only_repeated_rhs():
    solver, b, true_x = make_constant_solver()

    @jax.jit
    def solve_two(rhs1, rhs2):
        x1 = solver(rhs1)[0]
        x2 = solver(rhs2)[0]
        return x1, x2

    x1, x2 = solve_two(b, b * 2.0)
    assert jnp.allclose(x1, true_x, atol=1e-5)
    assert jnp.allclose(x2, true_x * 2.0, atol=2e-5)


def test_constant_values_solution_only_direct_multi_rhs():
    solver, b, true_x = make_constant_solver()
    rhs = jnp.stack([b, b * 2.0, b * 0.5])

    @jax.jit
    def solve_batch(rhs_batch):
        return solver(rhs_batch)[0]

    x1 = solve_batch(rhs)
    x2 = solve_batch(rhs * 3.0)
    assert x1.shape == (3, b.shape[0])
    assert jnp.allclose(x1[0], true_x, atol=1e-5)
    assert jnp.allclose(x1[1], true_x * 2.0, atol=2e-5)
    assert jnp.allclose(x1[2], true_x * 0.5, atol=1e-5)
    assert jnp.allclose(x2, x1 * 3.0, atol=6e-5)


def test_dynamic_solution_only_single_rhs_uses_xonly_custom_call():
    from spineax.cudss.solver import CuDSSSolver  # type: ignore[import-not-found]

    csr_offsets, csr_columns, csr_values, b, _ = get_test_system()
    solver = CuDSSSolver(
        csr_offsets,
        csr_columns,
        0,
        3,
        0,
        return_diagnostics=False,
    )

    @jax.jit
    def solve_once(rhs, values):
        return solver(rhs, values)[0]

    lowered = solve_once.lower(b, csr_values).as_text()
    assert "solve_single_f32_xonly" in lowered
    assert 'solve_single_f32"' not in lowered
    assert "solve_single_f32_const_xonly" not in lowered


def test_constant_values_singleton_batch_uses_single_rhs_custom_call():
    solver, b, true_x = make_constant_solver()

    @jax.jit
    def solve_batch(rhs_batch):
        return solver(rhs_batch)[0]

    lowered = solve_batch.lower(b[None, :]).as_text()
    assert "solve_single_f32_const_xonly" in lowered
    assert "solve_multi_rhs_f32_const_xonly" not in lowered
    x = solve_batch(b[None, :])
    assert x.shape == (1, b.shape[0])
    assert jnp.allclose(x[0], true_x, atol=1e-5)


def test_constant_values_multi_batch_uses_multi_rhs_custom_call():
    solver, b, true_x = make_constant_solver()
    rhs = jnp.stack([b, b * 2.0])

    @jax.jit
    def solve_batch(rhs_batch):
        return solver(rhs_batch)[0]

    lowered = solve_batch.lower(rhs).as_text()
    assert "solve_multi_rhs_f32_const_xonly" in lowered
    x = solve_batch(rhs)
    assert x.shape == (2, b.shape[0])
    assert jnp.allclose(x[0], true_x, atol=1e-5)
    assert jnp.allclose(x[1], true_x * 2.0, atol=2e-5)


def test_constant_values_solution_only_nested_vmap_rhs():
    solver, b, true_x = make_constant_solver()
    rhs_2d = jnp.stack(
        [jnp.stack([b, b * 2.0, b * 0.5]), jnp.stack([b * 3.0, b, b * 4.0])]
    )
    rhs = jnp.stack([rhs_2d, rhs_2d * 1.5])

    @jax.jit
    def solve_nested(rhs_batch):
        return jax.vmap(jax.vmap(jax.vmap(lambda rhs_i: solver(rhs_i)[0])))(
            rhs_batch
        )

    x = solve_nested(rhs)
    assert x.shape == rhs.shape
    assert jnp.allclose(x, rhs / b * true_x, atol=8e-5)


def test_constant_values_solution_only_vmap_rhs():
    solver, b, true_x = make_constant_solver()
    rhs = jnp.stack([b, b * 2.0, b * 0.5])

    @jax.jit
    def solve_batch(rhs_batch):
        return jax.vmap(lambda rhs_i: solver(rhs_i)[0])(rhs_batch)

    x = solve_batch(rhs)
    assert x.shape == rhs.shape
    assert jnp.allclose(x[0], true_x, atol=1e-5)
    assert jnp.allclose(x[1], true_x * 2.0, atol=2e-5)
    assert jnp.allclose(x[2], true_x * 0.5, atol=1e-5)


def test_constant_solver_rejects_malformed_rhs_before_native_call():
    solver, b, _ = make_constant_solver()
    with pytest.raises(ValueError, match="RHS trailing dimension"):
        solver(b[:-1])
    with pytest.raises(ValueError, match="RHS trailing dimension"):
        solver(jnp.ones((3, b.size - 1), dtype=b.dtype))


def test_dynamic_solver_rejects_nonvector_direct_rhs_before_native_call():
    from spineax.cudss.solver import CuDSSSolver  # type: ignore[import-not-found]

    offsets, columns, values, b, _ = get_test_system()
    solver = CuDSSSolver(offsets, columns, 0, 3, 0, return_diagnostics=False)
    with pytest.raises(ValueError, match="one-dimensional RHS"):
        solver(jnp.stack([b, b]), values)


def test_dynamic_solver_rejects_malformed_rhs_before_native_call():
    from spineax.cudss.solver import CuDSSSolver  # type: ignore[import-not-found]

    offsets, columns, values, b, _ = get_test_system()
    solver = CuDSSSolver(offsets, columns, 0, 3, 0, return_diagnostics=False)
    with pytest.raises(ValueError, match="RHS trailing dimension"):
        solver(b[:-1], values)


def test_constant_values_and_token_are_not_replaceable_pytree_leaves():
    solver, _, _ = make_constant_solver()
    leaves = jax.tree_util.tree_leaves(solver)
    assert all(leaf is not solver.csr_values for leaf in leaves)
    with pytest.raises(TypeError, match="not a leaf"):
        eqx.tree_at(lambda item: item.csr_values, solver, solver.csr_values * 2)


def test_solver_configuration_rejects_coercible_or_boolean_ids_and_diagnostics():
    from spineax.cudss.solver import CuDSSSolver  # type: ignore[import-not-found]

    offsets, columns, _, _, _ = get_test_system()
    invalid_args = ((True, 3, 0), (0, 3.5, 0), (0, 3, "1"))
    for device_id, mtype_id, mview_id in invalid_args:
        with pytest.raises(TypeError, match="non-boolean integer"):
            CuDSSSolver(offsets, columns, device_id, mtype_id, mview_id)
    with pytest.raises(TypeError, match="return_diagnostics must be a boolean"):
        CuDSSSolver(offsets, columns, 0, 3, 0, return_diagnostics="false")


def test_same_shape_constant_matrices_do_not_alias_compiled_state():
    offsets, columns, values, b, _ = get_test_system()
    from spineax.cudss.solver import ConstantCSRCuDSSSolver  # type: ignore[import-not-found]

    other_values = values.at[0].set(values[0] + 2.0)
    first = ConstantCSRCuDSSSolver(offsets, columns, values, 0, 3, 0)
    second = ConstantCSRCuDSSSolver(offsets, columns, other_values, 0, 3, 0)

    @eqx.filter_jit
    def run(solver, rhs):
        return solver(rhs)[0]

    x_first = run(first, b)
    x_second = run(second, b)
    dense_first = jsparse.BCSR(
        (values, columns, offsets), shape=(b.size, b.size)
    ).todense()
    dense_second = jsparse.BCSR(
        (other_values, columns, offsets), shape=(b.size, b.size)
    ).todense()
    assert first.matrix_token != second.matrix_token
    assert jnp.allclose(x_first, jnp.linalg.solve(dense_first, b), atol=1e-5)
    assert jnp.allclose(x_second, jnp.linalg.solve(dense_second, b), atol=1e-5)
    assert not jnp.allclose(x_first, x_second)


def test_constant_compiled_state_serializes_concurrent_calls():
    solver, b, true_x = make_constant_solver()

    @jax.jit
    def run(rhs):
        return solver(rhs)[0]

    run(b).block_until_ready()
    scales = (0.5, 1.0, 2.0, 3.0)
    with ThreadPoolExecutor(max_workers=len(scales)) as pool:
        outputs = list(
            pool.map(lambda scale: run(b * scale).block_until_ready(), scales)
        )
    for index, scale in enumerate(scales):
        assert jnp.allclose(outputs[index], true_x * scale, atol=3e-5)


def test_dynamic_compiled_state_serializes_concurrent_calls():
    from spineax.cudss.solver import CuDSSSolver  # type: ignore[import-not-found]

    offsets, columns, values, b, true_x = get_test_system()
    solver = CuDSSSolver(offsets, columns, 0, 3, 0, return_diagnostics=False)

    @jax.jit
    def run(rhs, matrix_values):
        return solver(rhs, matrix_values)[0]

    run(b, values).block_until_ready()
    scales = (0.5, 1.0, 2.0, 3.0)
    with ThreadPoolExecutor(max_workers=len(scales)) as pool:
        outputs = list(
            pool.map(
                lambda scale: run(b * scale, values).block_until_ready(), scales
            )
        )
    for index, scale in enumerate(scales):
        assert jnp.allclose(outputs[index], true_x * scale, atol=3e-5)


@pytest.mark.parametrize("use_pbatch", [False, True])
def test_dynamic_batched_compiled_state_serializes_concurrent_calls(
    monkeypatch, use_pbatch
):
    import spineax.cudss.solver as solver_module  # type: ignore[import-not-found]
    from spineax.cudss.solver import CuDSSSolver  # type: ignore[import-not-found]

    if use_pbatch and not solver_module.PBATCH_AVAILABLE:
        pytest.skip("pseudo-batch backend is unavailable")
    monkeypatch.setattr(solver_module, "vmap_using_pseudo_batch", use_pbatch)
    offsets, columns, values, b, true_x = get_test_system()
    solver = CuDSSSolver(offsets, columns, 0, 3, 0, return_diagnostics=False)

    @jax.jit
    def run(rhs_batch, values_batch):
        return jax.vmap(solver)(rhs_batch, values_batch)[0]

    rhs = jnp.stack([b, b * 2.0])
    matrix_values = jnp.stack([values, values])
    run(rhs, matrix_values).block_until_ready()
    scales = (0.5, 1.0, 2.0)
    with ThreadPoolExecutor(max_workers=len(scales)) as pool:
        outputs = list(
            pool.map(
                lambda scale: run(rhs * scale, matrix_values).block_until_ready(),
                scales,
            )
        )
    expected = jnp.stack([true_x, true_x * 2.0])
    for index, scale in enumerate(scales):
        assert jnp.allclose(outputs[index], expected * scale, atol=8e-5)


@pytest.mark.parametrize("use_pbatch", [False, True])
def test_dynamic_batched_initialization_failure_can_retry(monkeypatch, use_pbatch):
    import spineax.cudss.solver as solver_module  # type: ignore[import-not-found]
    from spineax.cudss.solver import CuDSSSolver  # type: ignore[import-not-found]

    if use_pbatch and not solver_module.PBATCH_AVAILABLE:
        pytest.skip("pseudo-batch backend is unavailable")
    monkeypatch.setattr(solver_module, "vmap_using_pseudo_batch", use_pbatch)
    offsets, columns, values, b, true_x = get_test_system()
    solver = CuDSSSolver(offsets, columns, 0, 3, 0, return_diagnostics=False)

    @jax.jit
    def run(rhs_batch, values_batch):
        return jax.vmap(solver)(rhs_batch, values_batch)[0]

    rhs = jnp.stack([b, b * 2.0])
    matrix_values = jnp.stack([values, values])
    monkeypatch.setenv("SPINEAX_CUDSS_IR_N_STEPS", "invalid")
    with pytest.raises(Exception, match="SPINEAX_CUDSS_IR_N_STEPS"):
        run(rhs, matrix_values).block_until_ready()
    monkeypatch.delenv("SPINEAX_CUDSS_IR_N_STEPS")
    expected = jnp.stack([true_x, true_x * 2.0])
    assert jnp.allclose(run(rhs, matrix_values), expected, atol=5e-5)


def test_dynamic_initialization_failure_can_retry(monkeypatch):
    from spineax.cudss.solver import CuDSSSolver  # type: ignore[import-not-found]

    offsets, columns, values, b, true_x = get_test_system()
    solver = CuDSSSolver(offsets, columns, 0, 3, 0, return_diagnostics=False)

    @jax.jit
    def run(rhs, matrix_values):
        return solver(rhs, matrix_values)[0]

    monkeypatch.setenv("SPINEAX_CUDSS_IR_N_STEPS", "invalid")
    with pytest.raises(Exception, match="SPINEAX_CUDSS_IR_N_STEPS"):
        run(b, values).block_until_ready()
    monkeypatch.delenv("SPINEAX_CUDSS_IR_N_STEPS")
    assert jnp.allclose(run(b, values), true_x, atol=1e-5)


def test_constant_initialization_failure_can_retry(monkeypatch):
    solver, b, true_x = make_constant_solver()

    @jax.jit
    def run(rhs):
        return solver(rhs)[0]

    monkeypatch.setenv("SPINEAX_CUDSS_IR_N_STEPS", "invalid")
    with pytest.raises(Exception, match="SPINEAX_CUDSS_IR_N_STEPS"):
        run(b).block_until_ready()
    monkeypatch.delenv("SPINEAX_CUDSS_IR_N_STEPS")
    assert jnp.allclose(run(b), true_x, atol=1e-5)


def test_constant_values_solution_only_f64_when_enabled():
    if not bool(jax.config.read("jax_enable_x64")):
        pytest.skip("jax_enable_x64 is disabled")
    solver, b, true_x = make_constant_solver(jnp.float64)

    @jax.jit
    def solve_once(rhs):
        return solver(rhs)[0]

    x = solve_once(b)
    lowered = cast(Any, solve_once).lower(b).as_text()
    assert "solve_single_f64_const_xonly" in lowered
    assert jnp.allclose(x, true_x, atol=1e-10)
