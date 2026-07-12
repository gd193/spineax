"""Small reproducible cuDSS dynamic versus constant cold/warm benchmark."""

import json
import time

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

from spineax.cudss.solver import ConstantCSRCuDSSSolver, CuDSSSolver


def timed(call, *args, iterations=1):
    start = time.perf_counter()
    for _ in range(iterations):
        result = call(*args)
        result.block_until_ready()
    return (time.perf_counter() - start) / iterations


def main(iterations=50):
    n = 128
    diagonal = jnp.full((n,), 4.0, dtype=jnp.float32)
    off_diagonal = jnp.full((n - 1,), -1.0, dtype=jnp.float32)
    dense = jnp.diag(diagonal) + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
    csr = jsparse.BCSR.fromdense(dense)
    rhs = jnp.arange(1, n + 1, dtype=jnp.float32)

    dynamic = CuDSSSolver(csr.indptr, csr.indices, 0, 3, 0, return_diagnostics=False)
    constant = ConstantCSRCuDSSSolver(csr.indptr, csr.indices, csr.data, 0, 3, 0)

    dynamic_call = jax.jit(lambda b, values: dynamic(b, values)[0])
    constant_call = jax.jit(lambda b: constant(b)[0])
    results = {
        "matrix_n": n,
        "iterations": iterations,
        "dynamic_cold_seconds": timed(dynamic_call, rhs, csr.data),
        "dynamic_warm_seconds": timed(
            dynamic_call, rhs, csr.data, iterations=iterations
        ),
        "constant_cold_seconds": timed(constant_call, rhs),
        "constant_warm_seconds": timed(constant_call, rhs, iterations=iterations),
    }
    results["warm_speedup"] = (
        results["dynamic_warm_seconds"] / results["constant_warm_seconds"]
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
