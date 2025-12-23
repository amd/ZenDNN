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
| `ZENDNNL_MATMUL_ALGO` | Selects the MatMul algorithm/kernel to use | `none` (auto-select) | `auto`, or integer values: `0` (dynamic_dispatch), `1` (aocl_dlp_blocked), `2` (onednn_blocked), `3` (libxsmm_blocked), `4` (aocl_dlp), `5` (onednn), `6` (libxsmm), `7` (batched_sgemm), `8` (auto_tuner), `9` (reference) |
| `ZENDNNL_MATMUL_WEIGHT_CACHE` | Enable/disable weight caching for blocked algorithms | `0` (disabled), `1` for blocked/auto-tuner algos | `0` (disabled), `1` (enabled) |
| `ZENDNNL_ZP_COMP_CACHE` | Enable/disable zero-point compensation caching for quantized operations | `0` (disabled) | `0` (disabled), non-zero (enabled) |

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

