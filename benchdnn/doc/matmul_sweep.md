(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# Matmul / BMM Sweep

Sweep mode expands each eligible input row into a cross-product of **M**, **dtype profile**, and **cache mode** inside a single `benchdnn` process. Every emitted config is validated, printed in an expansion table, and then benchmarked. This avoids hand-editing many near-identical input rows for LLM shape sweeps and dtype comparisons.

For general matmul input syntax see [matmul.md](matmul.md)

---

## Two layers of sweeping

| Axis | Where it loops | CLI / script |
|------|----------------|--------------|
| **M**, **dtype**, **cache** | Inside one `benchdnn` process | `--sweep`, `--m_sweep`, `--dtype_sweep`, `--cache_sweep`, `--sweep_dedup` |
| **Core count**, **cache mode** | Bash outer loop in `scripts/run_matmul_benchmark_sweep.sh` | `-t`, `-C` |

Core count is fixed per process (`OMP_NUM_THREADS` / CPU bind at launch), so it cannot be swept in-binary. Cache mode can be swept at **either** layer.

**Supported operators:** 2D matmul and BMM (`--ndims=3`) only. BMM sweep
supports the `bf16` and `fp32` dtype profiles; quantized BMM profiles are
skipped with a warning because batched tensor creation does not yet attach
quantization metadata.

**Input modes:** `--input_file` or `--input_model_file`. Pipeline (multi-layer) rows are skipped individually; remaining single-layer rows still expand.

---

## Sweep CLI flags and defaults

| Flag | Values | Default | Behavior |
|------|--------|---------|----------|
| `--sweep` | `true\|false\|1\|0` | `false` | Master switch. When off, input rows run as-is. |
| `--m_sweep` | `M[:M:...]` | *(omitted)* | Colon-separated M list. When `--sweep=true` and `--m_sweep` is **omitted**, uses built-in default M list. When `--m_sweep=` is explicitly passed empty, that is an error. |
| `--dtype_sweep` | `all` or comma-separated names | *(omitted)* | When **omitted**, each row keeps its file dtype (**M-only sweep**). When set, catalog profiles are applied (see [Dtype sweep catalog](#dtype-sweep-catalog)). |
| `--cache_sweep` | `hot,cold,warm` (comma subset) | *(omitted)* | When omitted, each config inherits global `--cache_mode` (default `hot`). |
| `--sweep_dedup` | `true\|false\|1\|0` | `true` | When `true`, collapse identical expanded configs. When `false`, keep every expanded row (including DistilBERT/BERT repeats and catalog dtypes that reduce to the same config). |
| `--cache_mode` | `hot\|cold\|warm` | `hot` | Seeds every parsed config before expansion; overridden per config when `--cache_sweep` is set. |

**Default M list** (`kDefaultMSweep`, used when `--sweep=true` without `--m_sweep`):

`1, 4, 8, 16, 32, 64, 512, 1024, 2048`

**M values in legacy `benchmark_sweep` eval files but not in the default:**

`128, 256, 4096, 8192, 16384`

**Recommended extended sweep** to match all M in the legacy eval set:

```
--m_sweep=1:4:8:16:32:64:128:256:512:1024:2048:4096:8192:16384
```

**Other relevant defaults** (non-sweep; seeded into each config):

| Field | Default |
|-------|---------|
| `ndims` | `2` (use `--ndims=3` for BMM) |
| `iters` | `100` |
| `warmup_iters` | `-1` (unset) |
| `alpha`, `beta` | `1.0`, `0.0` |
| `num_weight_buffers` | `-1` (auto; warm mode only) |
| Per-group group size (catalog) | `32` |
| Float kernel | `aocl_dlp_blocked` |
| Quantized kernel | `aocl_dlp` (requires `--lowoha=true`) |

---

## Dtype sweep catalog

Each `--dtype_sweep` name selects a canonical quant profile. For 2D matmul,
`--dtype_sweep=all` considers all entries; entries **#8** and **#9** deduplicate
to the same effective config, so a compatible shape emits **11** unique dtype
configs. Per-group profiles are skipped with a warning when their group size
does not divide `K`. BMM emits only the two supported float profiles; each
dropped quantized catalog entry is logged once per input row (not once per M).
The same BMM and per-group filters apply to M-only sweeps (no `--dtype_sweep`),
so a quantized or non-divisible file dtype is dropped rather than executed.

| # | Name | dt (src:wei:dst) | Scheme | Weight scale | Src scale | Notes |
|---|------|------------------|--------|--------------|-----------|-------|
| 0 | `bf16` | bf16:bf16:bf16 | none | none | — | plain bf16 |
| 1 | `fp32` | f32:f32:f32 | none | none | — | plain fp32 |
| 2 | `int8_per_group` | bf16:s8:bf16 | dynamic | group=32, bf16 | per-group=32, bf16 | INT8 dynamic (s8 weights) |
| 3 | `int8_per_token` | bf16:s8:bf16 | dynamic | channel, bf16 | per-token, bf16 | per-token src ⇒ per-channel weight |
| 4 | `int4_per_group` | bf16:s4:bf16 | woq | group=32, bf16 | — | weight-only quant |
| 5 | `int4_per_token` | bf16:s4:bf16 | woq | channel, bf16 | — | weight-only quant |
| 6 | `int4_dyn_per_group` | bf16:s4:bf16 | dynamic | group=32, bf16 | per-group=32, bf16 | W4A8 |
| 7 | `int4_dyn_per_token` | bf16:s4:bf16 | dynamic | group=32, bf16 | per-token, bf16 | W4A8 (per-group weight + per-token src) |
| 8 | `int8_static_s8_per_group` | s8:s8:s8 | static | channel, bf16 | per-tensor | integer dst ⇒ per-channel weight + per-tensor src (folds onto #9) |
| 9 | `int8_static_s8_per_token` | s8:s8:s8 | static | channel, bf16 | per-tensor | integer dst |
| 10 | `int8_static_bf16_per_group` | s8:s8:bf16 | static | group=32, bf16 | per-group=32, bf16 | dequantized dst |
| 11 | `int8_static_bf16_per_token` | s8:s8:bf16 | static | channel, bf16 | per-token, bf16 | dequantized dst |

**Kernel granularity couplings** (normalized by validation with warnings):

- **INT8 dynamic:** weight granularity is authoritative — per-group weights ⇒ per-group src; per-channel weights ⇒ per-token src.
- **W4A8 dynamic:** weight and src granularity are independent.
- **Static int8, integer (s8) dst:** per-channel weight scale + per-tensor static src scale only.
- **Static int8, dequantized (bf16/f32) dst:** per-group/per-channel weights and per-group/per-token static src scales supported.

> **Note (u8 not supported):** Asymmetric `u8` src/dst is **not** currently supported in benchdnn matmul (including static-int8 sweep axes #8–#11). Rows specifying `u8` are normalized (`u8` src → `bf16`; `u8` dst → `bf16` with a warning).

---

## File-wins merge and `force_profile`

The input file has priority for dtype/quant fields the row explicitly sets (`MatmulConfig::provided` bitset).

- **M-only sweep** (`--sweep=true`, no `--dtype_sweep`): each row keeps its own `dt` and quant metadata; only `m` varies.
- **Explicit `--dtype_sweep`**: the catalog profile overrides file dtype/quant fields so the dtype axis genuinely varies. Non-quant fields (bias, transpose, alpha/beta, iters) still come from the file.
- **Model-file shape rows** (K, N, iters only): report nothing provided, so the catalog supplies all dtype/quant fields when `--dtype_sweep` is set.

If `ZENDNNL_MATMUL_ALGO` / `ZENDNNL_BMM_ALGO` is set (including runner `-a`), it
overrides the file/catalog kernel at runtime. The expansion table `kernel`
column, result CSV, and sweep-dedup signature use that executed algo, not the
placeholder in the input row (catalog files still list `aocl_dlp_blocked`).
When the env algo is unset, the table keeps the file/catalog kernel name.

---

## Requirements, guards, and expansion output

**Requirements:**

- `--sweep=true`
- Input from `--input_file` or `--input_model_file` (command-line-only input is ignored with a warning)
- Pipeline / multi-layer rows (`n` is more than one value) are skipped with a warning; remaining single-layer rows still expand
- `k > 0` and `n > 0`
- Quantized catalog entries require `--lowoha=true` (the default)

**Expansion:**

- Nested loops: `M × dtype × cache` (innermost = cache)
- Dedup on the full execution signature (shape + dtype/quant + kernel +
  weights-constant + transpose + alpha/beta + bias + post-ops + cache) unless
  `--sweep_dedup=false`
- Incompatible dtype/quant profiles (quantized BMM, per-group size that does
  not divide `K`, LOWOHA required while off) are skipped with a warning once
  per input row; the expansion summary reports how many were dropped
- Prints expansion table + one-line summary before benchmarking. The table
  includes `wconst`, `trA`/`trB`, `alpha`/`beta`, `bias`/`bdt`, and `postops`
  so rows that share `(K,N)` but differ in epilogue (e.g. BERT vs DLRM) are
  distinguishable.

**Example expansion output:**

```
Sweep expansion: 1 input row(s) -> 22 config(s) (M x dtype x cache cross-product).
Sweep expansion table (22 config(s)):
#    M    K     N      dt              kernel            ...  cache  wconst trA trB alpha  beta  bias bdt  postops
...
```

**Regression (BERT vs DLRM `(1024,1024)`):** `non_llm_matmul_shapes.txt` carries both GEMMs (BERT: no bias, `transB=false`; DLRMv2: bias+relu, `transB=true`). Expansion must keep both. The same pair is repeated at `iters=1` in that file's `REGRESSION` section. `iters` is not part of the signature, so with `--m_sweep=1` expect **26** configs from 30 rows (4 skipped: the two identical DistilBERT/BERT `(384,64)` / `(64,384)` rows plus the two regression repeats). Pass `--sweep_dedup=false` to run all **30**.

```sh
./install/benchdnn/bin/benchdnn --op=matmul --lowoha=true \
  --input_file=benchdnn/input/matmul/benchmark_sweep/non_llm_matmul_shapes.txt \
  --sweep=true --m_sweep=1 --sweep_dedup=false
```

---

## Input files (`benchmark_sweep/`)

Three compact shape files are intended for sweep-based benchmarking. They use `M=1` templates; M and dtype are expanded at runtime.

| File | Purpose |
|------|---------|
| `llm_matmul_shapes.txt` | Decode and generative prefill/eval LLM GEMMs (167 unique `(K,N)`) |
| `prefill_vision_kn_shapes.txt` | PyTorch vision/multimodal shapes (86 unique `(K,N)`) |
| `non_llm_matmul_shapes.txt` | Model-grouped MiniLM, BERT, DistilBERT, and DLRMv2 shapes, plus a 2-row `REGRESSION` section; BERT vs DLRM `(1024,1024)` differ by transpose/bias/relu — use the M list in that file's header |

LLM rows use `iters=1000` and `warmup_iters=1000`. PyTorch rows use
`warmup_iters=500`; each `(K,N)` uses the lowest iteration count found across
its source-file M values (`100`, `200`, or `1000`) to keep large-M sweeps
practical.

Input line format matches [matmul input file](matmul.md#1-input-file---input_file). Example from `llm_matmul_shapes.txt`:

```
1, 4096, 4096, 1000, bf16:bf16:bf16, false, , , , aocl_dlp_blocked, true, false, false, 1.0, 0.0, none, 0, f32, 1000
```

Lines starting with `#` and inline `#` comments are ignored.

---

## Runner script (`scripts/run_matmul_benchmark_sweep.sh`)

The runner adds process-level sweeps and forwards in-binary sweep flags:

| Flag | Meaning |
|------|---------|
| `-t, --threads N[,N,...]` | Core-count sweep (default: all cores via `nproc`) |
| `-C, --cache-mode m[,m,...]` | Cache-mode sweep (`hot`, `cold`, `warm`; default: benchdnn `hot`) |
| `-m, --m-sweep M[:M:...]` | Forwarded as `--sweep=true --m_sweep=...` |
| `-d, --dtype-sweep list\|all` | Forwarded as `--sweep=true --dtype_sweep=...` |
| `--keep-duplicates` | Forwarded as `--sweep_dedup=false` (keep every expanded row) |
| `-a, --algo N[,N,...]` | `ZENDNNL_MATMUL_ALGO` / `ZENDNNL_BMM_ALGO` |
| `-i, --input shortcut\|path` | Input file (shortcuts in script header) |

**Constraints:**

- `-d` without `-m` still uses the default M list → result is **M × dtype**
- Combining `-m`/`-d` with `-p/--perf` warns that external `perf stat` aggregates across all expanded configs. Internal counters (`-P`) are measured per expanded config.

---

## Usage examples

### In-binary (direct `benchdnn`)

```sh
# M-only sweep: default M list, preserve each row's dtype
./install/benchdnn/bin/benchdnn --op=matmul --lowoha=true \
  --input_file=benchdnn/input/matmul/benchmark_sweep/llm_matmul_shapes.txt --sweep=true

# M × full dtype catalog
./install/benchdnn/bin/benchdnn --op=matmul --lowoha=true \
  --input_file=benchdnn/input/matmul/benchmark_sweep/llm_matmul_shapes.txt \
  --sweep=true --dtype_sweep=all

# Explicit M + selected dtypes + cache sweep
./install/benchdnn/bin/benchdnn --op=matmul --lowoha=true \
  --input_file=benchdnn/input/matmul/benchmark_sweep/llm_matmul_shapes.txt \
  --sweep=true --m_sweep=1:16:512 \
  --dtype_sweep=bf16,int8_per_token --cache_sweep=hot,cold

# Keep duplicate input rows (no sweep-signature collapse)
./install/benchdnn/bin/benchdnn --op=matmul --lowoha=true \
  --input_file=benchdnn/input/matmul/benchmark_sweep/non_llm_matmul_shapes.txt \
  --sweep=true --m_sweep=1 --sweep_dedup=false

# Full benchmark_sweep coverage: extended M + all dtypes
./install/benchdnn/bin/benchdnn --op=matmul --lowoha=true \
  --input_file=benchdnn/input/matmul/benchmark_sweep/llm_matmul_shapes.txt \
  --sweep=true \
  --m_sweep=1:4:8:16:32:64:128:256:512:1024:2048:4096:8192:16384 \
  --dtype_sweep=all
```

### Via runner script

```sh
# Core sweep × cache sweep × in-binary M × dtype
./scripts/run_matmul_benchmark_sweep.sh -a 1 -i bf16 \
  -t 32,64,128 -C hot,cold -m 1:4:8 -d all

# Dtype sweep only (uses default M list)
./scripts/run_matmul_benchmark_sweep.sh -a 1 -i bf16 -d all

# LLM decode shapes
./scripts/run_matmul_benchmark_sweep.sh -a 1,11 \
  -i benchdnn/input/matmul/benchmark_sweep/llm_matmul_shapes.txt \
  -m 1:4:8:16:32:64:128:256:512:1024:2048 \
  -d all -t 128

# Full coverage: LLM + vision/multimodal (LLM M list)
for f in llm_matmul_shapes.txt prefill_vision_kn_shapes.txt; do
  ./scripts/run_matmul_benchmark_sweep.sh -a 1,11 \
    -i "benchdnn/input/matmul/benchmark_sweep/$f" \
    -m 1:4:8:16:32:64:128:256:512:1024:2048:4096:8192:16384 \
    -d all -t 128
done

# Full coverage: non-LLM (MiniLM / BERT / DistilBERT / DLRMv2 M list)
./scripts/run_matmul_benchmark_sweep.sh -a 1,11 \
  -i benchdnn/input/matmul/benchmark_sweep/non_llm_matmul_shapes.txt \
  -m 1:10:64:100:384:1000:1100:10000:10100 \
  -d all -t 128

# Same non-LLM file, keep duplicate DistilBERT/BERT rows
./scripts/run_matmul_benchmark_sweep.sh -a 1,11 \
  -i benchdnn/input/matmul/benchmark_sweep/non_llm_matmul_shapes.txt \
  -m 1:10:64:100:384:1000:1100:10000:10100 \
  -d all --keep-duplicates -t 128
```

## Limitations

- **Pipeline / multi-N rows** — skipped individually with a warning; other single-layer rows in the same file still expand.
- **Quantized dtypes** require `--lowoha=true`.
- **BMM** (`--ndims=3`) supports M/cache sweep and the `bf16`/`fp32` dtype
  profiles; quantized dtype profiles (catalog or file) are skipped with a
  warning.
- **Per-group dtype profiles** are skipped with a warning when the profile's
  group size does not divide `K` (catalog sweep and M-only file dtype).
- **External perf counters** (`-p`): with M/dtype sweep, counters aggregate across expanded configs; internal counters (`-P`) are measured per expanded config.
- **Duplicate input rows** collapse only when `--sweep_dedup=true` (default)
  and the full execution signature matches, including the executed kernel
  (env algo when `ZENDNNL_MATMUL_ALGO` / `ZENDNNL_BMM_ALGO` is set, otherwise
  the file/catalog name), `is_weights_const`, transpose, alpha/beta, bias,
  and post-ops. Same `(K,N)` with a different epilogue (e.g. BERT vs DLRM
  `(1024,1024)`) is kept. Identical DistilBERT/BERT `(384,64)` and `(64,384)`
  rows still collapse unless `--sweep_dedup=false`.

---

## Related documentation

- [matmul.md](matmul.md) — input file format, cache mode, dynamic quant
- [perf_counters.md](perf_counters.md) — hardware counter profiling with the sweep script
