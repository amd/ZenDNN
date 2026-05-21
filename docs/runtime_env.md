
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
| `ZENDNNL_BMM_ALGO` | Selects the Batch MatMul algorithm/kernel to use for 3D batch matrix multiplication | `4` (aocl_dlp) | Integer ID or name. `-1` (none, falls back to default 4)<br>`0` (dynamic_dispatch)<br>`1` (aocl_dlp_blocked)<br>`2` (onednn_blocked)<br>`4` (aocl_dlp)<br>`5` (onednn)<br>`6` (libxsmm)<br>`7` (batched_sgemm) |
| `ZENDNNL_MATMUL_WEIGHT_CACHE` | Enable/disable weight caching for blocked algorithms | `0` (disabled), `1` for blocked/auto-tuner algos | `0` (disabled)<br>`1` (enabled) |
| `ZENDNNL_ZP_COMP_CACHE` | Enable/disable zero-point compensation caching for quantized operations | `0` (disabled) | `0` (disabled)<br>non-zero (enabled) |

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

User-facing knobs for `group_matmul_direct` (the grouped GEMM dispatcher used by MoE expert layers and other parallel-GEMM workloads).  See `docs/operator/lowoha_group_matmul_operator.md` for full operator semantics.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_GRP_MATMUL_ALGO` | Selects the parallel scheduling strategy.  `0` auto-selects based on expert count, M, and N.  Re-read on every call (production deployments may flip strategy between phases without restart). | `0` (auto) | `0` (auto), `1` (sequential), `2` (flat M-tile), `3` (flat N-tile), `4` (multilevel CCD), `5` (per-expert) |
| `ZENDNNL_GRP_MATMUL_PREPACK` | Master switch for ahead-of-time weight prepack.  When ON, the inner-kernel weight cache is eagerly populated for all expert weights on the first call that observes a given configuration, eliminating per-expert reorder spikes during steady-state inference.  Set `0` for lazy on-first-touch reorder (lower first-call latency at the cost of visible spikes when a fresh expert routes mid-stream).  Cached on first read. | `1` (ON) | `0` (lazy only), `1` (eager prepack) |
| `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL` | Enables the in-house BF16-only AVX-512 microkernel for ALGO 3 (N-tile).  Wins on single- and few-thread workloads; can lose to AOCL DLP at high thread counts on large MoE shapes.  Automatically bypasses to the standard backend when the contract fails (non-BF16 dtype, `transA`, `alpha != 1`, `beta != 0`, etc.).  Cached on first read. | `0` (OFF) | `0` (off), `1` (on) |
| `ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE` | When ON, pins ALGO 3's per-expert thread count to a `num_threads`-only formula so the AOCL DLP weight-reorder cache key stays stable across MoE routing variation (active-expert filtering, batch-size shifts, expert count drift).  Disable only for A/B comparison: the fully-relaxed planner can degrade cache hit-rate under churn.  Cached on first read. | `1` (ON) | `0` (relaxed), `1` (stable) |

### Memory note for `ZENDNNL_GRP_MATMUL_PREPACK=1`

Eager prepack populates the inner-kernel weight cache (AOCL DLP per-tile reorder cache and/or the custom-kernel pack arena) for every expert in the framework's `total_matmul` set.  Two distinct caches are involved with different eviction semantics:

- **AOCL DLP reorder cache** — backed by the generic `lru_cache_t` layer.  By default, capacity is `UINT32_MAX`, so populated entries are held for process lifetime and the prepack guarantee holds.  Deployments that override capacity via `ZENDNNL_LRU_CACHE_CAPACITY` (env or matmul-config JSON) shrink the upper bound, in which case prepacked AOCL entries can be evicted under steady-state pressure and a subsequent runtime reorder spike can return.  Set the LRU capacity high enough to fit your full `total_matmul` working set if you want eager prepack to hold.
- **Custom-kernel pack arena** — process-wide singleton that intentionally ignores the LRU eviction path (`pack.cpp::clear_custom_kernel_pack_cache` is the only entry that frees entries; regular operation never evicts).  `ZENDNNL_LRU_CACHE_CAPACITY` does NOT govern this cache.  Populated entries are held for process lifetime regardless of the LRU knob.

For a 32-expert gpt-oss-20B-class block at K=2880, N=5760, BF16, the resident footprint of the no-eviction default is roughly 2 GB.

For memory-bounded deployments (multi-tenant, container quotas), use one of:
- `ZENDNNL_GRP_MATMUL_PREPACK=0` — disables eager warm; lazy reorder fills the cache on demand, smaller working set during cold periods.
- `ZENDNNL_MATMUL_WEIGHT_CACHE=0` — disables the AOCL cache entirely (both lazy and eager paths for AOCL DLP).  Every call re-reorders; lowest memory, slowest steady-state.  The custom-kernel pack arena is unaffected — to bound it, call `clear_custom_kernel_pack_cache()` from a quiescent window between phases.
- `ZENDNNL_LRU_CACHE_CAPACITY=<N>` — caps the AOCL reorder cache at `N` entries.  Smaller `N` yields lower steady-state memory but reintroduces reorder spikes when prepacked entries are evicted under capacity pressure.  No effect on the custom-kernel pack arena.

### Examples

```bash
# Default: prepack ON, AOCL DLP backend, AOCL stable n-tile, custom kernel OFF.
# Best for production MoE inference with a warm-up phase.
./your_app

# Force ALGO 3 + custom kernel for low-thread (≤ 32t) MoE decode on Zen5.
export ZENDNNL_GRP_MATMUL_ALGO=3
export ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=1

# Latency-critical first-call workload (no warm-up window).
export ZENDNNL_GRP_MATMUL_PREPACK=0

# Multi-tenant container with strict memory cap.
export ZENDNNL_MATMUL_WEIGHT_CACHE=0
```

---

## Auto-Tuner Configuration

The auto-tuner automatically selects the best-performing algorithm for matrix multiplication by benchmarking different kernels.

| Variable | Description | Default | Valid Values |
|----------|-------------|---------|--------------|
| `ZENDNNL_AUTO_TUNER_TYPE` | Selects the auto-tuner version/strategy | `2` | `1` (v1 strategy), `2` (v2 strategy) |
| `ZENDNNL_MATMUL_SKIP_ITER` | Number of warmup iterations before algorithm evaluation begins | `2` | Positive integer |
| `ZENDNNL_MATMUL_EVAL_ITER` | Number of evaluation iterations to benchmark each algorithm | `3` | Positive integer |

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
| `ZENDNNL_EMBAG_THREAD_ALGO` | Selects the Embedding Bag threading algorithm to use | `1` (table_threaded) | `0` (batch_threaded), `1`  (table_threaded), `2` (ccd_threaded), `3` (hybrid_threaded)
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
- `src/common/config_manager.cpp` - Configuration manager implementation
- `src/operators/matmul/matmul_config.cpp` - MatMul configuration implementation

