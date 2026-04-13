(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# Group MatMul Operator

## Overview

Benchmarks multiple independent matrix multiplications executed via
`group_matmul_direct`.  Each operation can have its own dimensions and
non-contiguous buffers.  Supports parallel execution with CCD-aware
scheduling and an optional MoE (Mixture of Experts) weighted-reduce
post-op that fuses expert outputs into token-major rows.

Use cases include MoE expert layers, multi-head attention projections,
parallel Q/K/V computation, and any workload requiring grouped GEMMs
with independent memory layouts.

## Usage

```sh
./install/benchdnn/bin/benchdnn --op=grp_matmul --input_file=<file>
```

### Environment variables

| Variable | Values | Description |
|----------|--------|-------------|
| `ZENDNNL_GRP_MATMUL_ALGO` | `0` (auto), `1` (sequential), `2` (per-expert), `3` (multilevel), `4` (flat CCD M-slice) | Parallel strategy. Default `0` auto-selects. |
| `ZENDNNL_MATMUL_ALGO` | `1`, `3`, `10`, `11`, ... | Backend GEMM kernel. Default from config. |
| `OMP_NUM_THREADS` | integer | Number of OpenMP threads. |

## Input file format

CSV, one configuration per line.  Lines starting with `#` are comments.

```
num_ops, M, K, N, iters, src_dt:wei_dt:dst_dt, is_weights_const, warmup[, moe_topk]
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `num_ops` | int | Number of parallel expert GEMMs. |
| `M` | int or colon-separated | Rows per expert.  Single int = uniform; `126:323:80:68` = per-expert. |
| `K` | int | Shared inner dimension. |
| `N` | int | Output columns (hidden dim D). |
| `iters` | int | Timed iterations. |
| `src_dt:wei_dt:dst_dt` | string | Data types (e.g. `bf16:bf16:bf16`, `f32:f32:f32`). |
| `is_weights_const` | bool | Weight caching hint (`true` / `false`). |
| `warmup` | int | Warmup iterations before timing. |
| `moe_topk` | int (optional) | MoE post-op topk.  `0` or omitted = disabled.  `>0` = enable fused weighted-reduce with that topk.  Requires `total_M % topk == 0`. |

### Examples

```
# 8 Mixtral experts, uniform M=4, no MoE post-op
8, 4, 4096, 14336, 200, bf16:bf16:bf16, true, 50

# 8 experts, uniform M=4, with MoE topk=2
8, 4, 4096, 14336, 200, bf16:bf16:bf16, true, 50, 2

# 8 experts, imbalanced M, with MoE topk=2
8, 126:323:80:68:256:37:15:119, 4096, 14336, 200, bf16:bf16:bf16, true, 50, 2
```

## MoE post-op

When `moe_topk > 0`, the benchmark builds:

- **`row_ptrs`**: sequential mapping of expert rows to flat slots
  (expert 0 rows first, then expert 1, etc.)
- **`topk_weights`**: uniform `1/topk` for all slots
- **Output buffer**: `[num_tokens, N]` where `num_tokens = total_M / topk`

The MoE weighted-reduce runs after all expert GEMMs complete and is
included in the timed iterations.  GFLOPS accounting includes both GEMM
FLOPs (`2 * M[i] * K * N` per expert) and MoE FLOPs
(`2 * num_tokens * N * topk`).

## Output

Results are printed to the console and written to a timestamped CSV file.

### Console columns

| Column | Description |
|--------|-------------|
| ops | Number of experts |
| M | Per-expert M (uniform or min-max(avg)) |
| K | Inner dimension |
| N | Output dimension |
| iters | Timed iterations |
| warmup | Warmup iterations |
| dtypes | src:wei:dst data types |
| moe | `off` or `topk=N` |
| avg_ms | Average iteration time (ms) |
| min_ms | Minimum iteration time (ms) |
| GFLOPS_a | GFLOPS based on average time |
| GFLOPS_p | GFLOPS based on minimum (peak) time |

### CSV columns

`num_ops, M, K, N, iters, warmup, dtypes, is_weights_const, moe_topk, total_ms, avg_ms, min_ms, GFLOPS_avg, GFLOPS_peak`

## Sweep script

Use `scripts/run_matmul_benchmark_sweep.sh` for automated benchmarking:

```sh
# Run all parallel strategies on imbalanced shapes, 128 threads
./scripts/run_matmul_benchmark_sweep.sh --op grp_matmul -v 0,1,2 -i imbalanced -t 128

# Run on uniform shapes with specific kernel
./scripts/run_matmul_benchmark_sweep.sh --op grp_matmul -v 2 -a 1 -i uniform -t 128
```

### Input shortcuts

| Shortcut | File |
|----------|------|
| `uniform` | `benchdnn/input/grp_matmul/moe_uniform.txt` |
| `imbalanced` | `benchdnn/input/grp_matmul/moe_imbalanced.txt` |

### Version flags (`-v`)

| Version | Strategy | Best for |
|---------|----------|----------|
| `0` | Auto-select | Default — picks V2 or V3 based on shape |
| `1` | Sequential | Experts serial, all threads per GEMM |
| `2` | Per-expert | Many experts (>= threads), 1 thread each |
| `3` | Multilevel CCD-aware | Few experts (< threads), standalone benchmarks |
| `4` | Flat CCD M-slice | Framework-safe: no nested OMP, proportional CCD + M-slicing |
