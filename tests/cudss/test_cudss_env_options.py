import pytest
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse


def get_test_system():
    a = jnp.array(
        [
            [4.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    b = jnp.array([7.0, 12.0, 25.0, 4.0, 13.0], dtype=jnp.float32)
    a_sym = a + a.T - jnp.diag(jnp.diag(a))
    true_x = jnp.linalg.solve(a_sym, b)
    lhs = jsparse.BCSR.fromdense(a_sym)
    return lhs.indptr, lhs.indices, lhs.data, b, true_x


CUDSS_ENV_NAMES = (
    "SPINEAX_CUDSS_IR_N_STEPS",
    "SPINEAX_CUDSS_REORDERING_ALG",
    "SPINEAX_CUDSS_FACTORIZATION_ALG",
    "SPINEAX_CUDSS_DETERMINISTIC_MODE",
)


def clear_cudss_env(monkeypatch):
    for name in CUDSS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def solve_once():
    from spineax.cudss.solver import CuDSSSolver

    csr_offsets, csr_columns, csr_values, b, _ = get_test_system()
    solver = CuDSSSolver(csr_offsets, csr_columns, 0, 3, 0, return_diagnostics=False)

    @jax.jit
    def run(rhs, values):
        return solver(rhs, values)[0]

    return run(b, csr_values)


def solve_vmap_once(*, use_pbatch):
    from spineax.cudss import solver as solver_module
    from spineax.cudss.solver import CuDSSSolver

    csr_offsets, csr_columns, csr_values, b, _ = get_test_system()
    solver = CuDSSSolver(csr_offsets, csr_columns, 0, 3, 0, return_diagnostics=False)
    b_batch = jnp.stack([b, b * 2.0])
    csr_batch = jnp.stack([csr_values, csr_values])
    original_value = solver_module.vmap_using_pseudo_batch
    solver_module.vmap_using_pseudo_batch = use_pbatch
    try:
        return jax.vmap(solver)(b_batch, csr_batch)[0]
    finally:
        solver_module.vmap_using_pseudo_batch = original_value


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("SPINEAX_CUDSS_IR_N_STEPS", "invalid"),
        ("SPINEAX_CUDSS_IR_N_STEPS", "-1"),
        ("SPINEAX_CUDSS_REORDERING_ALG", "bogus"),
        ("SPINEAX_CUDSS_REORDERING_ALG", "6"),
        ("SPINEAX_CUDSS_FACTORIZATION_ALG", "alg_9"),
        ("SPINEAX_CUDSS_DETERMINISTIC_MODE", "maybe"),
    ],
)
def test_invalid_cudss_env_options_fail_fast(monkeypatch, env_name, env_value):
    clear_cudss_env(monkeypatch)
    monkeypatch.setenv(env_name, env_value)
    with pytest.raises(Exception, match=f"Invalid {env_name} value"):
        solve_once().block_until_ready()


@pytest.mark.parametrize("use_pbatch", [True, False])
def test_invalid_cudss_env_options_fail_fast_in_batched_paths(monkeypatch, use_pbatch):
    clear_cudss_env(monkeypatch)
    monkeypatch.setenv("SPINEAX_CUDSS_REORDERING_ALG", "not_an_alg")
    with pytest.raises(Exception, match="Invalid SPINEAX_CUDSS_REORDERING_ALG value"):
        solve_vmap_once(use_pbatch=use_pbatch).block_until_ready()


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("SPINEAX_CUDSS_REORDERING_ALG", ""),
        ("SPINEAX_CUDSS_REORDERING_ALG", "default"),
        ("SPINEAX_CUDSS_REORDERING_ALG", "0"),
        ("SPINEAX_CUDSS_REORDERING_ALG", "alg_1"),
        ("SPINEAX_CUDSS_FACTORIZATION_ALG", "1"),
        ("SPINEAX_CUDSS_DETERMINISTIC_MODE", "false"),
    ],
)
def test_valid_cudss_env_options_smoke(monkeypatch, env_name, env_value):
    csr_offsets, csr_columns, csr_values, b, true_x = get_test_system()
    clear_cudss_env(monkeypatch)
    monkeypatch.setenv(env_name, env_value)
    x = solve_once()
    assert jnp.allclose(x, true_x, atol=1e-5)
