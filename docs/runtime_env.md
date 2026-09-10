
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNNL Runtime Environment Variables

This document lists all environment variables available for configuring ZenDNNL at runtime.

---

## Configuration File

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_CONFIG_FILE` | Path to a JSON configuration file. When set, configuration is loaded from this file instead of individual environment variables. | None | Valid file path to a JSON config file |

---

## MatMul Algorithm Configuration
| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_MATMUL_ALGO` | Selects the MatMul algorithm/kernel to use for 2D matrix multiplication | `none` (auto-select) | `-1` (none)<br>`auto` (auto_tuner)<br>`0` (dynamic_dispatch)<br>`1` (aocl_dlp_blocked)<br>`2` (onednn_blocked)<br>`3` (libxsmm_blocked)<br>`4` (aocl_dlp)<br>`5` (onednn)<br>`6` (libxsmm)<br>`8` (auto_tuner)<br>`9` (reference) |
| `ZENDNNL_BMM_ALGO` | Selects the Batch MatMul algorithm/kernel to use for 3D batch matrix multiplication | `-1` (none); see note below | Integer ID or name. `-1` (none / unset)<br>`0` (dynamic_dispatch)<br>`1` (aocl_dlp_blocked)<br>`2` (onednn_blocked)<br>`4` (aocl_dlp)<br>`5` (onednn)<br>`6` (libxsmm)<br>`7` (batched_sgemm) |
| `ZENDNNL_CACHE_OFF` | Process-wide cache kill switch for covered cache infrastructure. Equivalent JSON knob: top-level `global_cache_off`. | `0` (disabled) | `1`/`true`/`on`/`yes` (force covered caches off)<br>`0`/`false`/`off`/`no` (leave local cache knobs in control) |
| `ZENDNNL_MATMUL_WEIGHT_CACHE` | Weight-reorder caching mode for blocked / auto-tuner algorithms and grouped-matmul custom paths. W8A8 grouped-MoE ALGO 4 requires an enabled persistent pack and declines when this is `0`; BF16 and caller-prequantized S8 may then use generic fallback. | `1` (out-of-place, enabled) | `0` (disabled)<br>`1` (out-of-place, enabled)<br>`2` (in-place: reuse the user weight buffer as the reorder destination when the layout is safe; auto-downgraded to `1` under auto_tuner / dynamic_dispatch) |
| `ZENDNNL_ZP_COMP_CACHE` | Enable/disable zero-point compensation caching for quantized operations. Forced off when `ZENDNNL_CACHE_OFF=1` or JSON `global_cache_off=true`. | `1` (enabled) | `1`/`true`/`on`/`yes` (enabled)<br>`0`/`false`/`off`/`no` (disabled) |
| `ZENDNNL_ENABLE_POSTOP_CACHE` | Enable/disable AOCL DLP post-op metadata caching for the matmul fast path. When ON, the per-call build of post-op metadata (bias, binary_add, binary_mul, sum scales, src/wei/dst quant scales) is memoized and reused on subsequent calls with the same shape, dtypes, and post-op chain; mutable fields (data pointers, dynamic scale buffers) are refreshed on every hit. When OFF, the cache is cleared on every call and the cold path runs each time. Cached on first read after config resolution. Forced off when `ZENDNNL_CACHE_OFF=1` or JSON `global_cache_off=true`. Recognized spellings are case-insensitive; unrecognized or empty values leave the default untouched. | `1` (enabled) | `1`/`true`/`on`/`yes` (enabled)<br>`0`/`false`/`off`/`no` (disabled) |

> **`ZENDNNL_BMM_ALGO` default is path-dependent.** The stored config value when the variable is unset is `none` (`-1`). The effective fallback then depends on the dispatch path: the operator (`matmul`) path resolves `none` to `aocl_dlp` (4), while the LowOHA `matmul_direct` batched path resolves `none` to `libxsmm` (6). Set the variable explicitly to pin one backend across both paths.

### Algorithm Details

| Algorithm ID | Name | Description |
|--------------|------|-------------|
| -1 | `none` | No algorithm selected (uses default selection logic) |
| 0 | `dynamic_dispatch` | Dynamic kernel dispatch based on heuristics (supported by LOA) |
| 1 | `aocl_dlp_blocked` | Blocked AOCL DLP algorithm |
| 2 | `onednn_blocked` | Blocked OneDNN algorithm |
| 3 | `libxsmm_blocked` | Blocked LIBXSMM algorithm (supported by LOA) |
| 4 | `aocl_dlp` | AOCL DLP algorithm |
| 5 | `onednn` | OneDNN algorithm |
| 6 | `libxsmm` | LIBXSMM algorithm (supported by LOA) |
| auto | `auto_tuner` | Auto-tuner (automatically selects best algorithm, supported by LOA) |

---

## Group MatMul Configuration

User-facing knob for `group_matmul_direct` (the grouped GEMM dispatcher used by MoE expert layers and other parallel-GEMM workloads).  See `docs/operator/low_overhead_operator/lowoha_group_matmul_operator.md` for full operator semantics.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_GRP_MATMUL_ALGO` | Selects the grouped-matmul whole-call mode. `0` (AUTO) lets the library pick per call. `{1,2,3,5,6}` pin a generic scheduler globally and suppress phase-local W8A8 requests. `4` attempts the private symmetric per-channel W8A8 fused-MoE fast path in both phases, in either BF16-dynamic or caller-prequantized S8 mode. ALGO 4's direct-S8 fast path requires positive finite per-token BF16/F32 source scales, caller-owned BF16 `dst_down`, and `row_ptrs` that permute destination rows. `dst_down[i]` must be the tight BF16 view over the exact same BF16-sized backing as `src[i]` (`ldc_down[i] == hidden`), whose leading bytes hold the S8 source prefix; separate, offset, partial, padded, and cross-expert destinations decline ALGO 4. Any eligibility decline writes nothing and falls back to generic AUTO, whose merged prequantized-S8 path validates its own destination contract. ALGO 4 allocation/execution errors remain terminal. Read once and cached on first reference. | `0` (AUTO) | `0` (AUTO), `1` (sequential), `2` (flat M-tile), `3` (flat N-tile), `4` (two-mode W8A8 fused-MoE attempt), `5` (per-expert), `6` (multilevel CCD) |
| `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO` | AUTO-only decode setting (`max active M <= 32`). `{1,2,3,5,6}` pin a generic decode scheduler; `0` selects the legacy cascade; `4` attempts W8A8 only for decode. Any eligibility decline inherits the normal decode default policy (default 3 plus all existing refinements/safety clamps), never numeric 4. | `3` | `0`, `1`, `2`, `3`, `4`, `5`, `6` |
| `ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO` | AUTO-only prompt setting (`max active M > 32`). `{1,2,3,5,6}` pin a generic prompt scheduler; `0` selects the legacy cascade; `4` attempts W8A8 only for prompt. Any eligibility decline inherits the normal prompt default policy (default 2 plus all existing refinements/safety clamps), never numeric 4. | `2` | `0`, `1`, `2`, `3`, `4`, `5`, `6` |

Selector precedence is global `4` (attempt in either phase), then any global
generic pin, then the matching phase setting under global AUTO. The inactive
phase setting has no effect. All three variables are cached; set them before
the first grouped-matmul call.

> Internal tuning knobs (planner thresholds, custom-kernel sub-knobs,
> prepack-cache sizing, etc.) live inside the implementation and are
> not part of the user-facing contract.  They are documented in
> `zendnnl/src/lowoha_operators/matmul/group_matmul/
> group_matmul_parallel_common.hpp` for library developers; production
> deployments should not need them.  If a workload requires one, file
> an issue describing the problem and we will assess whether to
> promote the knob to user-visible status.

### Examples

```bash
# Default: auto-select (recommended for production MoE inference).
./your_app

# Pin ALGO 3 (N-tile) for the whole process (e.g. for A/B
# benchmarking against the auto choice).
export ZENDNNL_GRP_MATMUL_ALGO=3
```

---

## Auto-Tuner Configuration

The auto-tuner automatically selects the best-performing algorithm for matrix multiplication by benchmarking different kernels.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_AUTO_TUNER_TYPE` | Selects the auto-tuner version/strategy | `1` | `1` (v1 strategy), `2` (v2 strategy) |
| `ZENDNNL_MATMUL_SKIP_ITER` | Number of warmup iterations before algorithm evaluation begins | `2` | Positive integer |
| `ZENDNNL_MATMUL_EVAL_ITER` | Number of evaluation iterations to benchmark each algorithm | `3` | Positive integer |
| `ZENDNNL_MATMUL_AUTO_ALGO_CANDIDATES` | Comma-separated list of algorithm IDs to evaluate | `1,2,5` | Comma-separated list of algorithm IDs |

### Auto-Tuner Phases

1. **Skip Phase** (`ZENDNNL_MATMUL_SKIP_ITER`): Initial warmup iterations to stabilize caches before measurements.
2. **Evaluation Phase** (`ZENDNNL_MATMUL_EVAL_ITER`): Tests multiple algorithms and measures execution times.
3. **Execution Phase**: Uses the cached best-performing algorithm for subsequent calls.

---

## Embedding Bag Kernel Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_EMBAG_ALGO` | Selects the Embedding Bag kernel to use | `none` (auto-select) | `1` (native kernel), `2`  (FBGEMM)
---

## Embedding Bag Threading Algorithm Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_EMBAG_THREAD_ALGO` | Selects the Embedding Bag threading algorithm to use | `1` (table_threaded) | `0` (dynamic_dispatch), `1` (table_threaded), `2` (batch_threaded), `3` (ccd_threaded), `4` (hybrid_threaded), `5` or `auto` (auto_tuner). The integer maps directly to the `eb_thread_algo_t` enum; `dynamic_dispatch` and `auto_tuner` both fall back to `table_threaded` at execution time.
---

## Logging Configuration

Log levels control the verbosity of messages for different modules.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_COMMON_LOG_LEVEL` | Log level for common module | `2` (warning) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |
| `ZENDNNL_API_LOG_LEVEL` | Log level for API module | `2` (warning) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |
| `ZENDNNL_TEST_LOG_LEVEL` | Log level for test module | `2` (warning) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |
| `ZENDNNL_PROFILE_LOG_LEVEL` | Log level for profile module | `4` (verbose) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |
| `ZENDNNL_DEBUG_LOG_LEVEL` | Log level for debug module | `2` (warning) | `0` (disabled), `1` (error), `2` (warning), `3` (info), `4` (verbose) |

### Log Level Descriptions

| Level | Name | Description |
|-------|------|-------------|
| 0 | `disabled` | Print no messages |
| 1 | `error` | Print only error messages |
| 2 | `warning` | Print error and warning messages |
| 3 | `info` | Print error, warning, and info messages |
| 4 | `verbose` | Print all messages |

---

## Profiler Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_ENABLE_PROFILER` | Enable/disable the profiler functionality | `false` (disabled) | `1` (enabled), any other value (disabled) |

---

## Diagnostics Configuration

The diagnostics layer gates expensive input-validation paths (null-pointer checks, dimension checks, quantization-parameter checks, and fused-MoE / group-MatMul contract checks) inside the low-overhead operators (`matmul`, `group_matmul`, `flash_sdpa`, `normalization`).  See `src/lowoha_operators/common/operator_instrumentation.hpp` for the gate implementation.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_DIAGNOSTICS_ENABLE` | Master switch for runtime input validation in low-overhead operators.  When enabled, validators run and emit rich `log_error` diagnostics on contract violations.  When disabled, the gate collapses to a single predicted-taken branch with no validator body executed.  Cached on first read. | `1` (enabled) | `0` (disabled); unset or any other value (enabled) |

### Notes

- The gate covers diagnostic-only checks.  Memory-safety checks that are always required (e.g. empty-vector reject in `group_matmul_direct`, post-op `leading_dim` defaulting in `matmul`) execute unconditionally regardless of this flag.
- Profiling and logging are separate subsystems (`ZENDNNL_ENABLE_PROFILER`, `ZENDNNL_*_LOG_LEVEL`) and are NOT controlled by this flag.
- The value is captured once per process via `static const bool`, so in-process `setenv` / `unsetenv` calls after the first validator call have no effect.  Tests that need to deterministically toggle the gate must do so via `fork()` / `execve()` subprocesses (see `[15] TestDispatcherActiveTotalNegative` in `gtests/group_matmul/test_fused_moe.cpp`).

---

## LRU Cache Configuration

The LRU (Least Recently Used) cache stores precomputed data for reuse.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_LRU_CACHE_CAPACITY` | Maximum number of entries in the LRU cache | `UINT_MAX` (unlimited) | Positive integer (uint32_t) |

### Eviction semantics by cache layer

Three distinct caches are involved in MatMul / Group MatMul execution
with different eviction semantics — readers of `ZENDNNL_LRU_CACHE_CAPACITY`
should know which one is actually governed:

- **AOCL DLP reorder cache** — backed by the generic `lru_cache_t`
  layer.  Default capacity `UINT32_MAX`, so populated entries are
  held for process lifetime and any prior prepack / warm-up
  guarantee holds.  Deployments that lower the capacity via
  `ZENDNNL_LRU_CACHE_CAPACITY` (env or matmul-config JSON) shrink
  the upper bound; prepacked entries can then be evicted under
  steady-state pressure and a subsequent runtime reorder spike
  can return.  Set the LRU capacity high enough to fit your full
  warm working set if you need eager prepack to hold.

- **Custom-kernel pack arena** — process-wide singleton that
  intentionally ignores the LRU eviction path
  (`pack.cpp::clear_custom_kernel_pack_cache` is the only entry
  that frees entries; regular operation never evicts).
  `ZENDNNL_LRU_CACHE_CAPACITY` does NOT govern this cache.
  Populated entries are held for process lifetime regardless of
  the LRU knob.

- **W8A8 ALGO-4 complete-tensor cache** — a separate `lru_cache_t` holding
  complete W13/W2 packs. It honors `ZENDNNL_LRU_CACHE_CAPACITY` exactly;
  in-flight calls retain evicted entries through `shared_ptr`. Set
  `params[i].weight_cache_type=0` or the process-wide weight-cache mode to
  `0` to make ALGO 4 decline without publishing an entry.

For memory-bounded deployments (multi-tenant, container quotas):

- `ZENDNNL_MATMUL_WEIGHT_CACHE=0` disables AOCL lazy/eager population.
  The grouped custom-kernel path uses per-call caller-owned packs instead of
  publishing persistent entries, and W8A8 grouped-MoE ALGO 4 declines before
  writes; BF16 and caller-prequantized S8 may continue through generic
  execution.
  Entries created before a runtime mode change are
  not implicitly reclaimed: call `clear_custom_kernel_pack_cache()` and
  `clear_fused_moe_scratch()` from an outside-OMP quiescent window to release
  them. If weights may change during a `1 → 0 → 1` transition, that
  clear is REQUIRED before re-enabling; it closes the old raw-address cache
  generation before storage can be mutated, released, or reused.
  W8A8 weight scales are copied from the current call and may change without
  flushing the packed-weight generation.
- `ZENDNNL_LRU_CACHE_CAPACITY=<N>` caps the AOCL reorder cache at
  `N` entries.  Smaller `N` yields lower steady-state memory but
  reintroduces reorder spikes when prepacked entries are evicted
  under capacity pressure.  No effect on the custom-kernel pack
  arena.

---

## Testing/Debug Configuration

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `AI_GTEST_DEBUG` | Enable debug print output in AI gtests | `false` (disabled) | `1` or `true` (enabled) |

---

## Usage Examples

### Basic Configuration

```bash
# Use AOCL DLP blocked algorithm with weight caching
export ZENDNNL_MATMUL_ALGO=1
export ZENDNNL_MATMUL_WEIGHT_CACHE=1

# Enable verbose logging for debugging
export ZENDNNL_COMMON_LOG_LEVEL=4
export ZENDNNL_API_LOG_LEVEL=4
```

### Auto-Tuner Configuration

```bash
# Enable auto-tuner with custom iteration counts
export ZENDNNL_MATMUL_ALGO=auto
export ZENDNNL_MATMUL_SKIP_ITER=3
export ZENDNNL_MATMUL_EVAL_ITER=5
```

### Using Configuration File

```bash
# Use a JSON configuration file instead of individual env vars
export ZENDNNL_CONFIG_FILE=/path/to/zendnnl_config.json
```

### Performance Logs

```bash
# Enable profiler for performance analysis
export ZENDNNL_ENABLE_PROFILER=1
export ZENDNNL_PROFILE_LOG_LEVEL=4
```

---

## Configuration Priority

1. **JSON Configuration File**: If `ZENDNNL_CONFIG_FILE` is set and the file is valid, configuration is loaded from the file.
2. **Environment Variables**: If no config file is specified, individual environment variables are used.
3. **Default Values**: Built-in defaults are used for any unspecified settings.

---

## See Also

- `config/zendnnl_user_config.json` - Example JSON configuration file
- `zendnnl/src/common/config_manager.cpp` - Configuration manager implementation
- `zendnnl/src/common/op_config.cpp` - MatMul configuration implementation

