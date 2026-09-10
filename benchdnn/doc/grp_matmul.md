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
| `ZENDNNL_GRP_MATMUL_ALGO` | `0` (auto), `1` (sequential), `2` (flat CCD M-tile), `3` (flat CCD N-tile), `4` (W8A8 fused-MoE fast path), `5` (per-expert), `6` (multilevel) | Whole-call mode. Global 4 attempts either BF16-dynamic W8A8 or caller-prequantized S8 W8A8 in both phases. ALGO 4's direct-S8 fast path requires per-token BF16/F32 scales and caller-provided BF16 `dst_down` rows over the exact same BF16-sized backing as the tight S8 source prefix (`dst_down[i] == src[i]`, `ldc_down[i] == hidden`). Separate, offset, partial, padded, and cross-expert destinations decline the fast path. Any eligibility decline falls back to generic AUTO, whose merged prequantized-S8 support validates its own destination contract; ALGO 4 execution/allocation failures remain terminal. A global generic pin suppresses phase-local W8A8 requests. |
| `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO` | `0`, `1`, `2`, `3`, `4`, `5`, `6` | AUTO-only decode setting (`max active M <= 32`, default `3`). Value 4 attempts W8A8 for decode only. Any eligibility decline inherits the complete decode default policy. |
| `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO` | `0`, `1`, `2`, `3`, `4`, `5`, `6` | AUTO-only prompt setting (`max active M > 32`, default `2`). Value 4 attempts W8A8 for prompt only. Any eligibility decline inherits the complete prompt default policy. |
| `ZENDNNL_MATMUL_ALGO` | `1`, `3`, `10`, `11`, ... | Backend GEMM kernel. Default from config. |
| `ZENDNNL_GRP_MATMUL_PREPACK` | `0` / `1` | Ahead-of-time weight prepack.  Default `1` (ON) — eagerly warms the AOCL DLP / custom-kernel weight cache for all expert slots on the first call that observes a given configuration, so the timed iterations never pay an on-the-fly reorder cost.  Set `0` to fall back to the legacy lazy-on-first-touch behaviour (useful when comparing first-iter latency with and without prepack). |
| `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL` | `0` / `1` | Master switch for the in-house AVX-512 microkernels under ALGO 3 — the BF16 (`bf16:bf16:{bf16,f32}`), DQ-INT8 (`s8:s8:{bf16,f32}`, sym/asym), and native FP16 (`f16:f16:{f16,f32}`) families.  Default `1` (ON); set `0` to force the AOCL DLP path.  Per-call eligibility (dtype tuple, `alpha=1`, `beta=0`, `N % pack_nr == 0`, const weights, …) and the per-family ISA gate still apply, so ineligible calls transparently fall back to AOCL DLP regardless of this knob.  Exception: an `f16` call on a host without AVX-512-FP16 is rejected upstream by `group_matmul_direct` with `status_t::isa_unsupported` (the whole call fails) rather than falling back — no F16 GEMM backend can run without the ISA. |
| `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8` | `0` / `1` | Sub-switch for the DQ-INT8 CK family (requires the master knob ON and AVX-512 VNNI).  Default `1` (ON).  Set `0` to A/B just the int8 fast path against the AOCL DLP `s8s8s32obf16_sym_quant` reference, leaving the BF16 / FP16 CK paths untouched. |
| `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_F16` | `0` / `1` | Sub-switch for the native AVX-512-FP16 CK family (requires the master knob ON plus AVX-512-FP16 ISA and a toolchain that compiled the FP16 intrinsics).  Default `1` (ON).  Set `0` to route f16 calls to the AOCL DLP F16 reference, leaving the BF16 / INT8 CK paths untouched. |
| `ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE` | `0` / `1` | Pin ALGO 3's per-expert thread count to a `num_threads`-only formula so AOCL DLP cache keys stay stable under MoE routing variation (active-expert filtering, batch-size shifts).  Default `1` (ON). |
| `ZENDNNL_MATMUL_WEIGHT_CACHE` | `0` / `1` / `2` | Weight-pack policy shared with operator execution. Setting `0` disables AOCL lazy/prepack population, makes custom kernels use non-persistent per-call packs, and makes W8A8 ALGO 4 decline before writes; BF16 and caller-prequantized S8 may then use generic execution. Modes 1 and 2 keep ALGO 4's private out-of-place complete-tensor packs. Existing entries are released by the corresponding quiescent clear hooks, not merely by changing this runtime mode. |
| `OMP_NUM_THREADS` | integer | Number of OpenMP threads. |

For the full operator-side semantics of these knobs, see `docs/operator/low_overhead_operator/lowoha_group_matmul_operator.md` (sections **Environment variables** and **Weight caching, prepack, and memory**) and `docs/runtime_env.md` (section **Group MatMul Configuration**).

## Input file format

CSV, one configuration per line.  Lines starting with `#` are comments.

```
num_ops, M, K, N, iters, src_dt:wei_dt:dst_dt, is_weights_const, warmup
       [, moe_topk[, gated_act[, N_down[, use_internal_alloc[, total_experts
       [, dynamic_quant[, compute_dt[, group_size]]]]]]]]
```

The first eight fields are required.  The eight trailing fields are optional and forward-compatible — if absent each defaults to "off / disabled" so legacy input files keep working unchanged.

### Fields

| # | Field | Type | Description |
|---|-------|------|-------------|
| 1 | `num_ops` | int | Number of **firing** expert GEMMs in this call. |
| 2 | `M` | int or colon-separated | Rows per expert.  Single int = uniform across all `num_ops`; `126:323:80:68` = per-expert (length must be `num_ops`). |
| 3 | `K` | int | Shared inner dimension. |
| 4 | `N` | int | Output columns (hidden dim D for non-fused; gate+up width = 2 * `dim` for fused). |
| 5 | `iters` | int | Timed iterations. |
| 6 | `src_dt:wei_dt:dst_dt` | string | Data types (e.g. `bf16:bf16:bf16`, `f32:f32:f32`). |
| 7 | `is_weights_const` | bool | Weight caching hint (`true` / `false`). |
| 8 | `warmup` | int | Warmup iterations before timing. |
| 9 | `moe_topk` | int (optional) | MoE post-op topk.  `0` or omitted = disabled.  `>0` = enable fused weighted-reduce with that topk.  Requires `total_M % topk == 0`. |
| 10 | `gated_act` | int (optional) | Gated activation.  `0` or omitted = disabled.  `1` = `silu_and_mul` (Mixtral/Llama/Qwen), `2` = `gelu_and_mul`, `3` = `swiglu_oai_mul` (GPT-OSS interleaved).  Requires `N` even. |
| 11 | `N_down` | int (optional) | Fused down_proj output columns.  `0` or omitted = disabled.  `>0` = fused Op1(gate+up) → activation → Op2(down_proj) with this output width.  K_down = N/2 when a gated activation is present, K_down = N otherwise. |
| 12 | `use_internal_alloc` | int (optional) | Library-managed Op2 scratch + src reuse.  `1` = library allocates Op1 output in a thread-local arena and writes Op2 output back into the caller's `src` buffer (zero caller-side scratch).  `0` (default) or absent = caller allocates `dst_down`.  Requires `N_down > 0` and matched src/dst precision. |
| 13 | `total_experts` | int (optional) | Drives the **framework prepack-extras contract**: the total number of expert weight slots present in the call (`>= num_ops`).  When omitted or `0`, defaults to `num_ops` (legacy: every supplied weight is firing).  When `> num_ops`, the driver allocates `total_experts` weight buffers — the first `num_ops` are the firing experts, the remaining `(total_experts - num_ops)` are pre-pack extras whose weights are warmed by the prepack module but never computed in this call (mirrors the production MoE rotating-experts use case).  Rejected if `< num_ops`. |
| 14 | `dynamic_quant` | int / string (optional) | Selects the **DQ-INT8** (dynamic-quant int8) path.  `0` / `false` / `off` / omitted = bf16 (default).  `1` / `true` / `on` / `dq_int8` = drive `params[i].dynamic_quant=true`: the weight is s8 with an f32 `wei_scale`, and the library's pre-OMP hoist quantizes the bf16 `src` at runtime (per-token by default, per-group when `group_size>0`).  Requires `src:wei:dst == bf16:s8:bf16` and `K % 4 == 0`. |
| 15 | `compute_dt` | string (optional) | int8 compute discriminator on `params[i].dtypes.compute` (ignored when `dynamic_quant=0`).  `s8` / `sym` (default) = symmetric (`kS8_S8_BF16_SYM`, no src zero-point).  `u8` / `asym` = asymmetric (`kU8_S8_BF16_ASYM`, hoist allocates a per-token src_zp).  Per-group (`group_size>0`) is symmetric-only, so must be `s8`. |
| 16 | `group_size` | int (optional) | **Per-group** symmetric int8 quant group width in K-elements (only when `dynamic_quant=1`).  `0` / omitted = per-token / per-channel (`src_scale {M,1}`, `wei_scale {1,N}`).  `>0` = per-group: `G = K/group_size` groups; the driver fills a `G×N` f32 `wei_scale` (`{G,N}`) and requests a `{M,G}` `src_scale` (hoist-filled per-group at call time).  Runs the AOCL DLP sym-quant GEMM via the N-tile `do_tile` `{G,n_tile}` per-column repack.  Requires `compute_dt=s8`, `0 < group_size < K`, `K % group_size == 0`. |

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

# Same fused MoE block with library-managed Op2 scratch (use_internal_alloc=1)
8, 4, 4096, 28672, 200, bf16:bf16:bf16, true, 50, 2, 1, 4096, 1

# Production MoE: 4 firing experts in a 32-expert pool — exercises the
# prepack-extras contract end-to-end.  Driver allocates 32 weight buffers,
# computes only the first 4 GEMMs, prepack module pre-warms all 32.
4, 32, 2880, 5760, 200, bf16:bf16:bf16, true, 50, 4, 3, 2880, 1, 32

# DQ-INT8 MoE (per-token symmetric): bf16 activations, s8 weights, topk=2.
8, 4, 4096, 14336, 200, bf16:s8:bf16, true, 50, 2, 0, 0, 0, 0, 1, s8

# Per-group MoE (symmetric int8, group_size=128 → G=32 groups over K=4096),
# topk=2.  Weight carries a {G,N} f32 scale; the N-tile hoist quantizes the
# bf16 src per-group and the AOCL DLP sym-quant GEMM runs under ALGO 3.
8, 4, 4096, 14336, 200, bf16:s8:bf16, true, 50, 2, 0, 0, 0, 0, 1, s8, 128

# Per-group MoE with a gated gate+up projection (silu_and_mul, gated_act=1,
# N = 2*intermediate = 28672).  The activation collapses gate|up to
# [M, 14336] as the epilogue post-op, then the topk reduce runs.
8, 4, 4096, 28672, 200, bf16:s8:bf16, true, 50, 2, 1, 0, 0, 0, 1, s8, 128
```

A ready-made per-group int8 MoE sweep (plain expert GEMMs + a gated gate+up
section for Mixtral / Qwen3 / GPT-OSS) lives in
`benchdnn/input/grp_matmul/moe_fused_gate_up_down/moe_per_group_int8.txt`.

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
| quant | `bf16`, `dq8s` / `dq8u` (per-token sym / asym DQ-INT8), or `pg<group_size>` (per-group symmetric, e.g. `pg128`) |
| moe | `off` or `topk=N` |
| fused | `off` or `N_down=N` |
| avg_ms | Average iteration time (ms) |
| min_ms | Minimum iteration time (ms) |
| GFLOPS_a | GFLOPS based on average time |
| GFLOPS_p | GFLOPS based on minimum (peak) time |

### CSV columns

`num_ops, M, K, N, iters, warmup, dtypes, quant, is_weights_const, moe_topk, gated_act, N_down, use_internal_alloc, wall_ms, sum_iter_ms, avg_ms, min_ms, GFLOPS_avg, GFLOPS_peak`

The `quant` column is `bf16` (default), `dq8s` / `dq8u` (per-token symmetric / asymmetric DQ-INT8), or `pg<group_size>` (per-group symmetric int8).

| Column | Description |
|--------|-------------|
| `num_ops` | Number of firing expert GEMMs.  When the input file specifies `total_experts > num_ops`, the trailing `(total_experts - num_ops)` slots are pre-pack extras that are warmed but not computed; only `num_ops` shows up here. |
| `wall_ms` | Outer wall time `t1 - t0` enclosing the timed loop (includes any cold-cache flush overhead between iters when `--cache_mode=cold`). |
| `sum_iter_ms` | Cumulative per-call kernel time, excluding cache-flush overhead.  Use this for kernel-level perf analysis. |
| `avg_ms` / `min_ms` | Mean / minimum per-iter kernel time. |
| `GFLOPS_avg` / `GFLOPS_peak` | GFLOPS based on `avg_ms` / `min_ms`.  Includes Op1 + Op2 + MoE post-op FLOPs as applicable. |

## Hardware performance counters

Per-shape AMD Zen 4 / Zen 5 PMU counter collection is supported via the top-level
`--perf-counters[=<profile>]` flag, sharing the same infrastructure used by the
matmul driver (`benchdnn/utils/perf_counters.{hpp,cpp}`).

The grp_matmul driver opens the `perf_event_open()` file descriptors **once
per benchdnn process** at `bench()` entry — *before* any OpenMP parallel
region — and **reuses them across all configs**.  This guarantees the OMP
team that libgomp / libiomp5 spawns during the first config's warmup joins
the perf-event inheritance subtree (`attr.inherit=1`) and stays in it for
the whole sweep.  Each config then just does `reset/enable/disable/read`
around its own timed `iters` loop, so warmup, tensor init, and process
startup overhead are excluded from the totals.

Note that `restore_src()` (the in-place src refill used when `use_internal_alloc=1`)
and `flush_cache()` (used when `--cache_mode=cold`) sit **inside** the iter loop
but **outside** the per-iter `ti0/ti1` timing bracket — so they contribute to the
counter totals but not to `sum_iter_ms`.  When this matters (e.g. when correlating
`L2BW%m` with `sum_iter_ms`), prefer `--cache_mode=hot` and `use_internal_alloc=0`
so the counters and the kernel-time denominator both reflect pure GEMM work.

```bash
sudo benchdnn --op=grp_matmul --input_file=<file> --perf-counters             # cache (default)
sudo benchdnn --op=grp_matmul --input_file=<file> --perf-counters=tlb         # DTLB + IPC
sudo benchdnn --op=grp_matmul --input_file=<file> --perf-counters=stalls      # dispatch stalls + IPC
```

Requires `sudo` or `sysctl kernel.perf_event_paranoid <= 1`.  When unavailable,
benchdnn warns and continues without counters — the timing output is
unchanged.

Per-config output, printed immediately after the existing console row:

```
   8     M=4         4096   14336   200      50  bf16:bf16:bf16     topk=2  N_down=4096*    2.345     2.123    8127.1    8954.3
  [PERF]   18.2%    3.1%    0.0%   85.3%   12.4%    2.3%   34.7%
  [ARCH] Zen5 (Family 0x1A) profile=cache
        1234567890      L1-dcache-loads
          45678901      L1-dcache-load-misses
          ...
```

Two grp_matmul-specific notes on the derived metrics:

- **`elapsed_sec` basis** = `sum_iter_ms / 1000.0` (kernel-only wall time
  across all `cfg.iters`).  This matches the CSV's `sum_iter_ms` column, so
  `L2BW%m` and other per-time ratios line up 1:1 with the reported timing.
- **Per-thread normalization** uses `omp_get_max_threads()` for the
  `L2BW%m` denominator — i.e. the formula *assumes* the absolute
  counter sums reflect the whole OMP team.  In practice this assumption
  is broken by Linux perf-event read semantics + benchdnn's per-shape
  `open()` (the counters reflect the master thread only — see
  [perf_counters.md → What the counters actually measure](perf_counters.md#what-the-counters-actually-measure-important)).
  Ratio metrics (L1miss%, L2miss%, DTLB%, stall%, IPC) remain meaningful
  because numerator and denominator come from the same scope, but the
  absolute `L2BW%m` value will be roughly `1 / num_threads` of the true
  process-wide bandwidth.  Recommended workaround for multi-thread
  sweeps: `OMP_NUM_THREADS=1`, one config per invocation, or external
  `perf stat`.

The CSV is **not** modified — perf-counter values appear on stdout only.
See [perf_counters.md](perf_counters.md) for the full event tables, derived
metric formulas, and interpretation guide.

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

### Group selector flags (`-v`)

| Version | Strategy | Best for |
|---------|----------|----------|
| `0` | Auto-select | Default — picks V1, V2, or V3 based on shape |
| `1` | Sequential | Experts serial, all threads per GEMM |
| `2` | Flat CCD adaptive tile | Framework-safe: no nested OMP, hybrid M/N-tiling per expert |
| `3` | Flat CCD N-tile | Framework-safe: no nested OMP, proportional CCD + N-tiling |
| `4` | W8A8 fused-MoE fast path | Explicit BF16-dynamic/direct-S8 W8A8 attempt; ineligible calls fall back to generic AUTO |
| `5` | Per-expert | Many experts (>= threads), 1 thread each |
| `6` | Multilevel CCD-aware | Nested OMP, inter-expert concurrency |
