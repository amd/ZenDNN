(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# Group MatMul Operator

## Overview

Benchmarks multiple independent matrix multiplications executed via
`group_matmul_direct`.  Each operation can have its own dimensions and
non-contiguous buffers.  Supports parallel execution with CCD-aware
scheduling, an optional MoE (Mixture of Experts) weighted-reduce
post-op that fuses expert outputs into token-major rows, and an
optional gated activation post-op (silu_and_mul, gelu_and_mul,
swiglu_oai_mul) for fused gate+up projections.

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
| `ZENDNNL_GRP_MATMUL_ALGO` | `0` (auto), `1` (sequential), `2` (flat CCD M-tile), `3` (flat CCD N-tile), `4` (multilevel), `5` (per-expert) | Parallel strategy. Default `0` auto-selects. |
| `ZENDNNL_MATMUL_ALGO` | `1`, `3`, `10`, `11`, ... | Backend GEMM kernel. Default from config. |
| `OMP_NUM_THREADS` | integer | Number of OpenMP threads. |

## Input file format

CSV, one configuration per line.  Lines starting with `#` are comments.

```
num_ops, M, K, N, iters, src_dt:wei_dt:dst_dt, is_weights_const, warmup[, moe_topk[, gated_act[, N_down]]]
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
| `gated_act` | int (optional) | Gated activation.  `0` or omitted = disabled.  `1` = `silu_and_mul` (Mixtral/Llama/Qwen), `2` = `gelu_and_mul`, `3` = `swiglu_oai_mul` (GPT-OSS interleaved).  Requires `N` even. |
| `N_down` | int (optional) | Fused down_proj output columns.  `0` or omitted = disabled.  `>0` = fused Op1(gate+up) → activation → Op2(down_proj) with this output width.  K_down = N/2 (dim after activation). |

### Examples

```
# 8 Mixtral experts, uniform M=4, no MoE post-op, no activation
8, 4, 4096, 14336, 200, bf16:bf16:bf16, true, 50

# 8 experts, uniform M=4, with MoE topk=2, silu activation (fused gate+up)
8, 4, 4096, 28672, 200, bf16:bf16:bf16, true, 50, 2, 1

# 8 experts, imbalanced M, with MoE topk=2, no activation (down_proj)
8, 126:323:80:68:256:37:15:119, 4096, 14336, 200, bf16:bf16:bf16, true, 50, 2

# Full fused MoE block: gate+up → silu → down_proj (N_down=4096)
8, 4, 4096, 28672, 200, bf16:bf16:bf16, true, 50, 2, 1, 4096
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

## Fused MoE (Op1 → activation → Op2)

When `N_down > 0`, the benchmark allocates down_proj weights and output
buffers and passes `grp_matmul_fused_moe_params` to `group_matmul_direct`.
The entire MoE block — gate+up GEMM, gated activation, and down_proj GEMM —
runs as a single fused API call.

GFLOPS accounting includes both Op1 (`2 * M[i] * K * N` per expert) and
Op2 (`2 * M[i] * dim * N_down` per expert, where `dim = N/2`).

Input files for fused benchmarks are in `benchdnn/input/grp_matmul/moe_fused_gate_up_down/`.

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
| fused | `off` or `N_down=N` |
| avg_ms | Average iteration time (ms) |
| min_ms | Minimum iteration time (ms) |
| GFLOPS_a | GFLOPS based on average time |
| GFLOPS_p | GFLOPS based on minimum (peak) time |

### CSV columns

`num_ops, M, K, N, iters, warmup, dtypes, is_weights_const, moe_topk, gated_act, N_down, total_ms, avg_ms, min_ms, GFLOPS_avg, GFLOPS_peak`

## Sweep script

Use `scripts/run_matmul_benchmark_sweep.sh` for automated benchmarking:

```sh
# Run versions 0-3 on prompt shapes, 128 threads
./scripts/run_matmul_benchmark_sweep.sh --op grp_matmul -v 0,1,2,3 -i prompt -t 128

# Run version 2 on decode shapes with specific kernel
./scripts/run_matmul_benchmark_sweep.sh --op grp_matmul -v 2 -a 1 -i decode -t 128
```

### Input shortcuts

| Shortcut | File |
|----------|------|
| `prompt` | `benchdnn/input/grp_matmul/grp_matmul_prompt.txt` |
| `decode` | `benchdnn/input/grp_matmul/grp_matmul_decode.txt` |

### Version flags (`-v`)

| Version | Strategy | Best for |
|---------|----------|----------|
| `0` | Auto-select | Default — picks V1, V2, or V3 based on shape |
| `1` | Sequential | Experts serial, all threads per GEMM (default) |
| `2` | Flat CCD adaptive tile | Framework-safe: no nested OMP, hybrid M/N-tiling per expert |
| `3` | Flat CCD N-tile | Framework-safe: no nested OMP, proportional CCD + N-tiling |
| `4` | Multilevel CCD-aware | Nested OMP, inter-expert concurrency |
| `5` | Per-expert | Many experts (>= threads), 1 thread each |
