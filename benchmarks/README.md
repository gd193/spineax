# Benchmarks

## Constant CSR cuDSS solver

`cudss_constant_solver.py` measures the overhead of
`ConstantCSRCuDSSSolver` against its equivalent explicit factor-once token API
on symmetric positive-definite 2-D Laplacians.

```bash
CUDA_VISIBLE_DEVICES=1 python benchmarks/cudss_constant_solver.py \
  --grid-sizes 32 64 128 \
  --nrhs 1 8 \
  --dtype float64 \
  --output cudss-constant.json
```

The benchmark reports paired, synchronized median latency for:

- **const-setup**: owned CSR copies, analyze, and factorize;
- **token-setup**: explicit analyze and factorize without wrapper-owned copies;
- **const-solve**: solve through `ConstantCSRCuDSSSolver`;
- **token-solve**: solve through the equivalent reused `FactorToken`;
- **setup-oh / solve-oh**: wrapper overhead relative to the explicit token path.

Both paths factor once and solve many times. The benchmark deliberately does
not report a “speedup” against factorizing on every call: that would measure
factorization avoidance already provided by the token API, not wrapper
performance.

JIT compilation and warmup are excluded, including when `--warmup=0`. Every
timed operation is synchronized; the constant solver constructor blocks on its
factor token internally. Paired method order alternates per sample, and all
solve paths are checked against a known solution. Setup teardown occurs outside
the timed interval.

For stable results, use an otherwise idle GPU, pin `CUDA_VISIBLE_DEVICES`, and
record the JSON output alongside the JAX, CUDA, cuDSS, and GPU versions.
