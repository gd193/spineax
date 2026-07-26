# Benchmarks

## Constant CSR cuDSS solver

`cudss_constant_solver.py` measures the benefit and overhead of
`ConstantCSRCuDSSSolver` on symmetric positive-definite 2-D Laplacians.

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/cudss_constant_solver.py \
  --grid-sizes 32 64 128 \
  --nrhs 1 8 \
  --dtype float64 \
  --output cudss-constant.json
```

The benchmark reports synchronized median latency for:

- **construct**: copy, analyze, and factorize a constant solver;
- **constant**: solve through `ConstantCSRCuDSSSolver` with resident factors;
- **token**: solve through an equivalent pre-factorized `FactorToken`;
- **refactor**: analyze, factorize, solve, and release for every right-hand side;
- **speedup**: `refactor / constant` median latency;
- **wrapper**: constant-wrapper overhead relative to direct token solve.

JIT compilation and warmup are excluded, including when `--warmup=0`. Each
timed operation is synchronized, so results are observed operation latency
rather than asynchronous dispatch throughput. Transient refactor tokens are
released inside their timed lifecycle, and measurement order rotates between
cases to reduce systematic thermal/clock bias. The harness verifies all three
solve paths against a known solution and fails if the relative error exceeds
the dtype tolerance.

For stable results, use an otherwise idle GPU, pin `CUDA_VISIBLE_DEVICES`, and
record the JSON output alongside the JAX, CUDA, cuDSS, and GPU versions.
