#!/usr/bin/env python3
"""Benchmark ConstantCSRCuDSSSolver against an equivalent reused FactorToken.

Examples:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/cudss_constant_solver.py
    CUDA_VISIBLE_DEVICES=1 python benchmarks/cudss_constant_solver.py \
        --grid-sizes 32 64 128 --nrhs 1 8 --dtype float64 --output results.json

The two paths perform the same factor-once/solve-many work. This benchmark asks
whether the convenience wrapper adds setup or steady-state solve overhead; it
does not compare solve-only latency with factorization on every call.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from spineax import cudss


@dataclass(frozen=True)
class Timing:
    median_ms: float
    mean_ms: float
    p10_ms: float
    p90_ms: float
    minimum_ms: float


@dataclass(frozen=True)
class Result:
    grid_size: int
    n: int
    nnz: int
    nrhs: int
    constant_setup: Timing
    token_setup: Timing
    constant_solve: Timing
    token_solve: Timing
    setup_overhead_pct: float
    solve_overhead_pct: float
    relative_error: float


def _upper_laplacian(grid_size: int, dtype: np.dtype) -> tuple[np.ndarray, ...]:
    """Return the upper CSR triangle of a 2-D five-point SPD Laplacian."""
    n = grid_size * grid_size
    values: list[float] = []
    columns: list[int] = []
    offsets = [0]
    for row in range(n):
        values.append(4.0)
        columns.append(row)
        if row % grid_size + 1 < grid_size:
            values.append(-1.0)
            columns.append(row + 1)
        if row + grid_size < n:
            values.append(-1.0)
            columns.append(row + grid_size)
        offsets.append(len(values))
    return (
        np.asarray(values, dtype=dtype),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(columns, dtype=np.int32),
    )


def _symmetric_upper_matvec(
    values: np.ndarray,
    offsets: np.ndarray,
    columns: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(vectors)
    for row in range(offsets.size - 1):
        for index in range(offsets[row], offsets[row + 1]):
            column = columns[index]
            value = values[index]
            output[..., row] += value * vectors[..., column]
            if column != row:
                output[..., column] += value * vectors[..., row]
    return output


def _timing(samples_ns: list[int]) -> Timing:
    samples_ms = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return Timing(
        median_ms=np.median(samples_ms).item(),
        mean_ms=np.mean(samples_ms).item(),
        p10_ms=np.percentile(samples_ms, 10).item(),
        p90_ms=np.percentile(samples_ms, 90).item(),
        minimum_ms=np.min(samples_ms).item(),
    )


def _ordered_names(names: list[str], sample: int, order_index: int) -> list[str]:
    if (sample + order_index) % 2:
        return list(reversed(names))
    return names


def _measure_paired_calls(
    calls: dict[str, Callable],
    rhs,
    warmup: int,
    samples: int,
    order_index: int,
) -> dict[str, Timing]:
    names = list(calls)
    # One mandatory untimed call per path excludes JIT compilation.
    for name in _ordered_names(names, 0, order_index):
        calls[name](rhs).block_until_ready()
    for index in range(warmup):
        for name in _ordered_names(names, index + 1, order_index):
            calls[name](rhs).block_until_ready()

    elapsed: dict[str, list[int]] = {name: [] for name in names}
    for sample in range(samples):
        for name in _ordered_names(names, sample, order_index):
            start = time.perf_counter_ns()
            calls[name](rhs).block_until_ready()
            elapsed[name].append(time.perf_counter_ns() - start)
    return {name: _timing(values) for name, values in elapsed.items()}


def _measure_paired_setups(
    factories: dict[str, Callable],
    releasers: dict[str, Callable],
    samples: int,
    order_index: int,
) -> dict[str, Timing]:
    names = list(factories)
    # Compile/setup each path once. Release is intentionally outside setup time.
    for name in _ordered_names(names, 0, order_index):
        state = factories[name]()
        releasers[name](state)

    elapsed: dict[str, list[int]] = {name: [] for name in names}
    for sample in range(samples):
        for name in _ordered_names(names, sample, order_index):
            start = time.perf_counter_ns()
            state = factories[name]()
            elapsed[name].append(time.perf_counter_ns() - start)
            releasers[name](state)
    return {name: _timing(values) for name, values in elapsed.items()}


def _relative_error(actual, expected) -> float:
    numerator = jnp.linalg.norm(actual - expected)
    denominator = jnp.maximum(jnp.linalg.norm(expected), jnp.finfo(expected.dtype).eps)
    return (numerator / denominator).item()


def _benchmark_case(
    grid_size: int,
    nrhs: int,
    dtype: np.dtype,
    warmup: int,
    samples: int,
    construction_samples: int,
    order_index: int,
) -> Result:
    values_np, offsets_np, columns_np = _upper_laplacian(grid_size, dtype)
    n = grid_size * grid_size
    rng = np.random.default_rng(20260725 + n + nrhs)
    true_x_np = rng.standard_normal((nrhs, n)).astype(dtype)
    rhs_np = _symmetric_upper_matvec(values_np, offsets_np, columns_np, true_x_np)
    if nrhs == 1:
        true_x_np = true_x_np[0]
        rhs_np = rhs_np[0]

    values = jnp.asarray(values_np)
    offsets = jnp.asarray(offsets_np)
    columns = jnp.asarray(columns_np)
    rhs = jnp.asarray(rhs_np)
    expected = jnp.asarray(true_x_np)

    def make_constant():
        # The constructor blocks on its factor token before returning.
        return cudss.ConstantCSRCuDSSSolver(values, offsets, columns, mtype_id="spd")

    def make_token():
        token = cudss.factorize(
            cudss.analyze(values, offsets, columns, mtype_id="spd"), values
        )
        token.id.block_until_ready()
        return token

    setup = _measure_paired_setups(
        {"constant": make_constant, "token": make_token},
        {"constant": lambda solver: solver.release(), "token": cudss.release},
        construction_samples,
        order_index,
    )

    constant_solver = make_constant()
    token = make_token()
    constant_call = jax.jit(lambda value: constant_solver(value))
    token_call = jax.jit(lambda value: cudss.solve(token, value))
    try:
        solve = _measure_paired_calls(
            {"constant": constant_call, "token": token_call},
            rhs,
            warmup,
            samples,
            order_index,
        )
        errors = [
            _relative_error(constant_call(rhs), expected),
            _relative_error(token_call(rhs), expected),
        ]
        if not all(math.isfinite(value) for value in errors):
            raise RuntimeError(f"non-finite relative errors: {errors}")
        error = max(errors)
        tolerance = 5e-4 if dtype == np.dtype(np.float32) else 1e-10
        if error > tolerance:
            raise RuntimeError(
                f"relative error {error:.3e} exceeds tolerance {tolerance:.3e}"
            )
    finally:
        constant_solver.release()
        cudss.release(token)

    constant_setup = setup["constant"]
    token_setup = setup["token"]
    constant_solve = solve["constant"]
    token_solve = solve["token"]
    return Result(
        grid_size=grid_size,
        n=n,
        nnz=values.shape[0],
        nrhs=nrhs,
        constant_setup=constant_setup,
        token_setup=token_setup,
        constant_solve=constant_solve,
        token_solve=token_solve,
        setup_overhead_pct=(
            100.0 * (constant_setup.median_ms / token_setup.median_ms - 1.0)
        ),
        solve_overhead_pct=(
            100.0 * (constant_solve.median_ms / token_solve.median_ms - 1.0)
        ),
        relative_error=error,
    )


def _print_results(results: list[Result]) -> None:
    header = (
        " grid    n     nnz  rhs  const-setup  token-setup  setup-oh  "
        "const-solve  token-solve  solve-oh   relerr"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.grid_size:5d} {result.n:6d} {result.nnz:7d} "
            f"{result.nrhs:4d} {result.constant_setup.median_ms:10.3f}ms "
            f"{result.token_setup.median_ms:10.3f}ms "
            f"{result.setup_overhead_pct:+7.2f}% "
            f"{result.constant_solve.median_ms:10.3f}ms "
            f"{result.token_solve.median_ms:10.3f}ms "
            f"{result.solve_overhead_pct:+7.2f}% "
            f"{result.relative_error:.1e}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--nrhs", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--construction-samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(size < 2 for size in args.grid_sizes):
        parser.error("grid sizes must be at least 2")
    if any(nrhs < 1 for nrhs in args.nrhs):
        parser.error("nrhs values must be positive")
    if args.warmup < 0 or args.samples < 1 or args.construction_samples < 1:
        parser.error("warmup must be nonnegative and sample counts must be positive")
    return args


def main() -> None:
    args = _parse_args()
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    dtype = np.dtype(args.dtype)
    device = jax.devices("gpu")[0]
    print(f"JAX {jax.__version__} | {device} | {platform.platform()} | {args.dtype}")

    cases = [(grid_size, nrhs) for grid_size in args.grid_sizes for nrhs in args.nrhs]
    results = [
        _benchmark_case(
            grid_size,
            nrhs,
            dtype,
            args.warmup,
            args.samples,
            args.construction_samples,
            order_index,
        )
        for order_index, (grid_size, nrhs) in enumerate(cases)
    ]
    _print_results(results)

    if args.output is not None:
        payload = {
            "environment": {
                "jax": jax.__version__,
                "device": str(device),
                "platform": platform.platform(),
                "dtype": args.dtype,
                "warmup": args.warmup,
                "samples": args.samples,
                "construction_samples": args.construction_samples,
            },
            "results": [asdict(result) for result in results],
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
