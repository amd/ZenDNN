(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# SDPA (Scaled Dot-Product Attention) Operator

## Overview

Benchmarks Scaled Dot-Product Attention via the LOWOHA `sdpa_direct` API
(`zendnnl::lowoha::sdpa::sdpa_direct`). The driver invokes the Flash Attention
CPU backend (`flash_sdpa`) and times the call end-to-end per configuration.

For each line of input the driver:

1. Allocates Q/K/V/output tensors with logical axes `[B, H, S, D]` in the
   requested **physical layout** — either `bhsd` ( memory order
   `[B, H, S, D]`, the default) or `bshd` (memory order `[B, S, H, D]`,
   common in many LLM frameworks). The mask layout is independent.
2. Optionally allocates a `[S_q, S_kv]` (2D) or `[B, H, S_q, S_kv]` (4D)
   additive attention mask, and/or sets the causal flag.
3. Builds an `sdpa_params` struct with layout-derived per-axis strides and
   mask metadata.
4. Runs `warmup_iters` untimed warmup iterations followed by `iters` timed
   iterations of `sdpa_direct`.
5. Records the total wall-clock time, prints a results table to stdout, and
   appends a row to a timestamped CSV file.

> **Note:** Only the LOWOHA path is supported. `--lowoha=true` is the
> default; passing `--lowoha=false` with `--op=sdpa` is rejected with an
> error.

## Supported (qkv_dt, mask_ndims, mask_dt) combinations

The benchdnn driver mirrors the validation in
`zendnnl/src/lowoha_operators/sdpa/flash_sdpa/lowoha_flash_sdpa_utils.cpp`.

| `qkv_dt` | `mask_ndims` | Allowed `mask_dt` | Notes                                  |
|----------|--------------|-------------------|----------------------------------------|
| `f32`    | `0`          | _ignored_         | No additive mask (set `mask_dt=none`). |
| `f32`    | `2`          | `f32`             | `[S_q, S_kv]` mask, broadcast over BH. |
| `f32`    | `4`          | `f32`             | `[B, H, S_q, S_kv]` mask.              |
| `bf16`   | `0`          | _ignored_         | No additive mask.                      |
| `bf16`   | `2`          | `f32` or `bf16`   | `[S_q, S_kv]` mask.                    |
| `bf16`   | `4`          | `f32` or `bf16`   | `[B, H, S_q, S_kv]` mask.              |
| `f16`    | `0`          | _ignored_         | No additive mask. Requires AVX512-FP16.|
| `f16`    | `2`          | `f32` or `f16`    | `[S_q, S_kv]` mask. Requires AVX512-FP16. |
| `f16`    | `4`          | `f32` or `f16`    | `[B, H, S_q, S_kv]` mask. Requires AVX512-FP16. |

`is_causal` may be combined with any `mask_ndims` value (the operator applies
both the additive mask and an upper-triangular causal mask for future
positions).

`out_dt` must equal `qkv_dt` (or `none`, which defaults it to `qkv_dt`). The
operator does not currently support an output dtype that differs from the
QKV dtype.

> **`f16` runtime requirement:** the flash backend dispatches the FP16 path
> only when `zendnnl_platform_info().get_avx512_f16_status()` returns true
> (Zen5 / Granite Rapids and newer). On CPUs without AVX512-FP16 support
> the `sdpa_direct` call will fail at runtime; the benchdnn driver logs the
> failure and skips the config rather than aborting the whole sweep.

`dropout_p` must be `0.0` (only zero dropout is supported by the standalone
flash backend today).

## Usage

There are three ways to provide configurations: a CSV input file, the
command line, or a model-shapes file with CLI overrides.

### 1. FILE mode (`--input_file=...`) — primary

```sh
./install/benchdnn/bin/benchdnn --op=sdpa \
    --input_file=../benchdnn/input/sdpa/dtype_mask_coverage.txt
```

CSV format, one configuration per line. Lines starting with `#` and blank
lines are ignored.

```
batch, num_heads, seq_len, kv_seq_len, head_dim, qkv_dt,
is_causal, mask_ndims, mask_dt, iters
[, warmup_iters, scale, num_threads, out_dt, qkv_layout]
```

#### Required fields

| Field        | Type     | Description                                                                 |
|--------------|----------|-----------------------------------------------------------------------------|
| `batch`      | int      | Batch dimension B.                                                          |
| `num_heads`  | int      | Number of attention heads H.                                                |
| `seq_len`    | int      | Query sequence length S_q.                                                  |
| `kv_seq_len` | int      | Key/value sequence length. **Pass `0` for self-attention** (`S_kv = S_q`).  |
| `head_dim`   | int      | Per-head dimension D.                                                       |
| `qkv_dt`     | string   | `f32`, `bf16`, or `f16` (the `f16` path requires AVX512-FP16).              |
| `is_causal`  | bool     | `true` / `false` (or `1` / `0`). Apply causal upper-triangular mask.        |
| `mask_ndims` | int      | `0` (no additive mask), `2` (`[S_q,S_kv]`), or `4` (`[B,H,S_q,S_kv]`).      |
| `mask_dt`    | string   | Mask data type when `mask_ndims > 0`; `none` when `mask_ndims == 0`. Allowed values follow the (qkv_dt, mask_ndims) table above. |
| `iters`      | int      | Number of timed iterations.                                                 |

#### Optional trailing fields

| Field          | Default                | Description                                                |
|----------------|------------------------|------------------------------------------------------------|
| `warmup_iters` | `0.2 * iters`          | Untimed warmup iterations.                                 |
| `scale`        | `0.0`                  | Softmax scale. `0.0` means auto = `1 / sqrt(head_dim)`.    |
| `num_threads`  | `0`                    | OpenMP thread count. `0` = auto / `OMP_NUM_THREADS`.       |
| `out_dt`       | `none` (= `qkv_dt`)    | Output data type. Must equal `qkv_dt`.                     |
| `qkv_layout`   | `bhsd`                 | Physical Q/K/V layout. `bhsd` = `[B,H,S,D]` memory order (head-major); `bshd` = `[B,S,H,D]` memory order (token-major). Output follows the same layout as Q. |

#### Example file

```
# qkv=f32 with 4D causal + additive mask
2, 8, 384, 0, 64, f32, true, 4, f32, 100, 20, 0.0, 0, none

# qkv=bf16, self-attention, no additive mask
4, 16, 1024, 0, 64, bf16, false, 0, none, 100

# Cross-attention (S_q != S_kv) with bf16 mask
2, 12, 1, 512, 64, bf16, false, 4, bf16, 100, 20, 0.0, 0, none

# Same shape as the row above but in BSHD memory layout
2, 12, 1, 512, 64, bf16, false, 4, bf16, 100, 20, 0.0, 0, none, bshd

# qkv=f16, self-attention, causal-only (skipped on CPUs without AVX512-FP16)
4, 16, 1024, 0, 64, f16, true, 0, none, 100, 20, 0.0, 0, none

# qkv=f16 with 4D additive f16 mask
2, 12, 512, 0, 64, f16, false, 4, f16, 100, 20, 0.0, 0, none

# qkv=f16 with 2D additive f32 mask (mixed mask dtype is allowed)
2, 12, 512, 0, 64, f16, false, 2, f32, 100, 20, 0.0, 0, none
```

### 2. COMMAND_LINE mode

Pass shape and dtype directly on the CLI. A single configuration is
benchmarked.

```sh
./install/benchdnn/bin/benchdnn --op=sdpa \
    --bs=4 --num_heads=16 --seq_len=1024 --kv_seq_len=0 --head_dim=64 \
    --sdt=bf16 \
    --mask_ndims=4 --mask_dt=bf16 \
    --is_causal=false \
    --iters=100 --warmup_iters=20
```

For `f16` (requires AVX512-FP16):

```sh
./install/benchdnn/bin/benchdnn --op=sdpa \
    --bs=4 --num_heads=16 --seq_len=1024 --kv_seq_len=0 --head_dim=64 \
    --sdt=f16 \
    --mask_ndims=4 --mask_dt=f16 \
    --is_causal=true \
    --iters=100 --warmup_iters=20
```

#### SDPA-specific CLI flags

| Flag             | Maps to               | Description                                              |
|------------------|-----------------------|----------------------------------------------------------|
| `--bs=N`         | `batch`               | Batch dimension B (existing common flag).                |
| `--num_heads=N`  | `num_heads`           | Number of attention heads H.                             |
| `--seq_len=N`    | `seq_len`             | Query sequence length S_q.                               |
| `--kv_seq_len=N` | `kv_seq_len`          | Key/value sequence length (0 = use seq_len).             |
| `--head_dim=N`   | `head_dim`            | Per-head dimension D.                                    |
| `--sdt=DT`       | `qkv_dt`              | Data type for Q/K/V (`f32`, `bf16`, or `f16`).           |
| `--mask_ndims=N` | `mask_ndims`          | Mask rank: `0`, `2`, or `4`.                             |
| `--mask_dt=DT`   | `mask_dt`             | Mask data type (`none`, `f32`, `bf16`, or `f16`).        |
| `--is_causal=B`  | `is_causal`           | `true` / `false` (or `1` / `0`).                         |
| `--scale=F`      | `scale`               | Softmax scale (`0.0` = auto = `1/sqrt(head_dim)`).       |
| `--num_threads=N`| `num_threads`         | OpenMP thread count (`0` = auto).                        |
| `--out_dt=DT`    | `out_dt`              | Output data type (`none` = same as `qkv_dt`).            |
| `--qkv_layout=L` | `qkv_layout`          | Physical Q/K/V layout (`bhsd` default, or `bshd`).        |
| `--iters=N`      | `iters`               | Timed iterations (existing common flag).                 |
| `--warmup_iters=N`| `warmup_iters`       | Warmup iterations (existing common flag).                |

In COMMAND_LINE mode the four shape flags `--bs`, `--num_heads`, `--seq_len`,
and `--head_dim` are required (use `--kv_seq_len=0` for self-attention).
`--qkv_layout` defaults to `bhsd`.

### 3. MODEL mode (`--input_model_file=...`)

Useful for sweeping the same dtype/mask configuration across a list of
named model shapes. The model file supplies `(B, H, S_q, S_kv, D)` per
line; everything else (qkv_dt, mask config, iter counts, etc.) is taken
from the CLI flags.

```sh
./install/benchdnn/bin/benchdnn --op=sdpa \
    --input_model_file=../benchdnn/input/sdpa/sdpa_models.txt \
    --sdt=bf16 \
    --mask_ndims=4 --mask_dt=bf16 \
    --is_causal=false \
    --iters=100 --warmup_iters=20
```

#### Model file format

CSV, one model per line, `#` for comments:

```
ModelName, batch, num_heads, seq_len, kv_seq_len, head_dim
```

Example:

```
BERT_base_S384         , 1, 12,  384,    0,  64
Llama7B_prefill_S2048  , 1, 32, 2048,    0, 128
Llama7B_decode_K2048   , 1, 32,    1, 2048, 128
T5_base_xattn_S128_K512, 1, 12,  128,  512,  64
```

When `--input_model_file` is used, the benchmark output table includes a
`Model_Name` column.

## Provided input files

Located in `benchdnn/input/sdpa/`:

| File                          | Purpose                                                                 |
|-------------------------------|-------------------------------------------------------------------------|
| `dtype_mask_coverage.txt`     | Full `(qkv_dt × mask_ndims × mask_dt × is_causal)` matrix for the `f32`, `bf16`, and `f16` paths, plus a few cross-attention shapes and BSHD-layout variants. Use to verify all legal combinations work. The `f16` block is auto-skipped on CPUs without AVX512-FP16. |
| `bert_inputs.txt`             | Encoder self-attention shapes (BERT-base / BERT-large, S = 128/256/384/512, B = 1/8/32). |
| `llm_inputs.txt`              | Causal decoder shapes (Llama-7B / 13B, GPT-J 6B): both prefill (`S_q == S_kv`) and decode (`S_q = 1`, growing `S_kv`). |
| `cross_attention_inputs.txt`  | Encoder-decoder cross-attention shapes (T5-base / T5-large / Whisper). All `S_q != S_kv` with 4D padding masks. |
| `sdpa_models.txt`             | Shapes-only file for MODEL mode (BERT, ViT, LLM prefill/decode, T5, Whisper). |

## Output

For each configuration the benchmark prints a row with the following
columns to stdout, and writes the same data to a CSV file named
`timings_<YYYYMMDD>_<HHMMSS>_<ms>.csv` in the working directory.

| Column          | Description                                                       |
|-----------------|-------------------------------------------------------------------|
| `Model_Name`    | Only present in MODEL mode.                                       |
| `Batch`         | B.                                                                |
| `Num_Heads`     | H.                                                                |
| `Seq_Len`       | S_q.                                                              |
| `KV_Seq_Len`    | `0` means same as `Seq_Len`.                                      |
| `Head_Dim`      | D.                                                                |
| `QKV_DT`        | `f32`, `bf16`, or `f16`.                                          |
| `QKV_Layout`    | `bhsd` (head-major) or `bshd` (token-major).                      |
| `Is_Causal`     | `true` / `false`.                                                 |
| `Mask_Ndims`    | `none` / `2D` / `4D`.                                             |
| `Mask_DT`       | `none`, `f32`, `bf16`, or `f16`.                                  |
| `Out_DT`        | `none` (= same as `QKV_DT`), `f32`, `bf16`, or `f16`.             |
| `Scale`         | Softmax scale (`0.000000` indicates auto = `1/sqrt(D)`).          |
| `Num_Threads`   | `auto` (`0` requested) or the explicit thread count.              |
| `Iters`         | Timed iterations.                                                 |
| `Warmup_Iters`  | Untimed warmup iterations.                                        |
| `Total_time(ms)`| Sum of timed iteration latencies.                                 |
| `Avg_time(ms)`  | `Total_time(ms) / Iters`.                                         |

### Example console output

```
Batch  Num_Heads  Seq_Len  KV_Seq_Len  Head_Dim  QKV_DT  QKV_Layout  Is_Causal  Mask_Ndims  Mask_DT  Out_DT  Scale     Num_Threads  Iters  Warmup_Iters  Total_time(ms)  Avg_time(ms)
2      8          384      0           64        f32     bhsd        false      none        none     none    0.000000  auto         100    20            149.70          1.496988
4      16         1024     0           64        bf16    bhsd        true       none        none     none    0.000000  auto         100    20            334.36          3.343559
2      12         512      0           64        bf16    bshd        false      4D          bf16     none    0.000000  auto         100    20            63.41           0.634111
4      16         1024     0           64        f16     bhsd        true       none        none     none    0.000000  auto         100    20            175.92          1.759158
2      12         512      0           64        f16     bhsd        false      4D          f16      none    0.000000  auto         100    20            44.07           0.440720
```

> The two `f16` rows above are illustrative and were captured on a Zen5 CPU.
> The rows are skipped (with a logged error) on CPUs without AVX512-FP16
> support; see the runtime requirement note above.

## Notes & limitations

- The driver supports `bhsd` (default, head-major) and `bshd` (token-major)
  Q/K/V layouts; output always matches Q. Other layouts where `head_dim` is
  not the innermost contiguous axis are out of scope (the flash kernel
  requires `*_stride_d == 1`).
- The mask layout is independent of `qkv_layout`. Mask shape is always
  `[S_q, S_kv]` (2D) or `[B, H, S_q, S_kv]` (4D) with row-major strides.
- Only the standalone Flash backend (`sdpa_flash_cpu_standalone`) is
  exercised. The BMM-based path and the higher-level
  `sdpa_encoder_operator_t` API are not currently wired in.
- Correctness checks are not performed (perf-only, like the other benchdnn
  operators); use `gtest` for functional validation.
- `dropout_p` is fixed at `0.0`.
- Cache behavior is selected at runtime via `--cache_mode` (default `hot`).
  `--cache_mode=cold` calls `flush_cache()` before every timed iteration;
  `--cache_mode=hot` runs back-to-back iterations with no extra flush.
  `--cache_mode=warm` is matmul-only and is rejected by `main()` for SDPA.
