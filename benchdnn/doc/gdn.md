
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# GDN (Gated DeltaNet) Operator

## Overview

Benchmarks the Gated DeltaNet linear-attention core of Qwen3.5 / Qwen3.6 /
Qwen3.8 via the LOWOHA `gdn_direct` API (`zendnnl::lowoha::gdn::gdn_direct`,
see [lowoha_gdn_operator.md](../../docs/operator/low_overhead_operator/lowoha_gdn_operator.md)).
Each input line is one call over a vLLM-style non-speculative batch: decode
sequences first, then prefills.

For each line the driver:

1. Allocates `mixed_qkv`, `a`, `b`, `A_log`, `dt_bias`, the conv weight and
   bias, the output and (with `fused_norm`) `z` and the norm weight, plus a
   conv-state pool (`[slot][tap][channel]`, vLLM's SD layout) and an SSM
   state pool (`[slot][v_head][head_k_dim x head_v_dim]`) with one slot per
   sequence. Inputs and states hold uniform random values.
2. Builds the batch description as vLLM's GDN metadata does: `cu_seqlens`
   with `num_decodes` one-token decodes followed by `num_prefills` prefills
   of `prefill_len` tokens, shuffled slot ids, and `has_initial_state` for
   every sequence.
3. Runs `warmup_iters` untimed calls and `iters` timed calls of
   `gdn_direct`. The state pools advance in place from call to call, as they
   do across decode steps.
4. Prints a results table to stdout and writes the same rows to a
   timestamped CSV file.

> **Note:** Only the LOWOHA path and FILE input are supported
> (`--input_file=...`; `--input_model_file=...` reads the same format).
> `--lowoha=false` and COMMAND_LINE mode are rejected for `--op=gdn`.

## Cache mode

GDN times **cold by default**: unless `--cache_mode` is given, the caches are
flushed before every timed call. A decode step reads and writes the whole SSM
state of its sequences once (34 MB of state traffic per call for 8 Qwen3.6
sequences with an FP32 state), and in a model every GDN layer has its own
pool, so the state streams from memory. Back-to-back calls on one pool would keep it in cache
and overstate the speed. `--cache_mode=hot` runs the calls back to back
anyway (useful to isolate compute); `--cache_mode=warm` is matmul-only and is
rejected by `main()`.

## Input format

CSV, one configuration per line; lines starting with `#` and blank lines are
ignored.

```
model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_width,
num_decodes, num_prefills, prefill_len
[, iters, warmup_iters, src_dt, state_dt, has_initial_state, state_layout,
   fused_norm, prefill_algo, kernel, num_threads]
```

| Field | Default | Meaning |
|-------|---------|---------|
| `model` | required | Label printed with the results. |
| `num_k_heads`, `num_v_heads` | required | `linear_num_key_heads`, `linear_num_value_heads`; `num_v_heads` is a multiple of `num_k_heads`. |
| `head_k_dim`, `head_v_dim` | required | `linear_key_head_dim`, `linear_value_head_dim`. |
| `conv_width` | required | `linear_conv_kernel_dim` (0: `mixed_qkv` is already convolved). |
| `num_decodes` | required | Single-token decode sequences, leading the batch. |
| `num_prefills`, `prefill_len` | required | Prefill sequences after the decodes, `prefill_len` tokens each. |
| `iters` | `--iters` (100) | Timed calls. |
| `warmup_iters` | `--warmup_iters`, else `iters / 5` | Untimed calls. |
| `src_dt` | `--src_dt`, else `bf16` | Activations, conv weight / bias / state, `z` and output: `f32`, `bf16` or `f16`. |
| `state_dt` | `--state_dt`, else `f32` | SSM state pool: `f32`, `bf16` or `f16`. |
| `has_initial_state` | `true` | Prefills continue from their slots (`true`) or start from zero states (`false`); decodes always continue. |
| `state_layout` | `kv` | SSM state matrix layout: `kv` (vLLM CPU) or `vk` (FLA). |
| `fused_norm` | `false` | Fuse the gated RMSNorm (silu gate, rounded core output) on the output. |
| `prefill_algo` | `auto` | Native prefill: `auto` (chunked once prefills average 16 tokens with bf16 sources, 32 with f16 and 24 with f32), `recurrent` or `chunked`. |
| `kernel` | `auto` | `auto`, `native` or `reference`. |
| `num_threads` | `--num_threads` (0) | OpenMP threads of the call (0: all). |

Lines that `gdn_direct` would reject (for example a non-multiple
`num_v_heads`, an empty batch or a prefill of 0 tokens) are reported and
skipped before any allocation.

Two GDN-only options select the data types of lines that leave `src_dt` or
`state_dt` empty: `--src_dt=<f32|bf16|f16>` and `--state_dt=<f32|bf16|f16>`
(`none` keeps the defaults). For example `--src_dt=f16` runs every model row
with fp16 activations. Matmul's `--sdt` does not apply to `--op=gdn`.

[gdn_models.txt](../input/gdn/gdn_models.txt) holds the GDN shapes of
Qwen3.5-9B (also Qwen3.5-4B and -35B-A3B), Qwen3.6-35B-A3B and Qwen3.8-27B
(also Qwen3.5-27B), read from their `config.json`: decode at batch 1, 8, 16,
32 and 64, single prefills of 128, 1024 and 4096 tokens, and two mixed
batches (8 decodes with one 1024-token prefill, which runs chunked, and 16
decodes with eight 8-token prefills, which run recurrent). Qwen3.6-35B-A3B
also runs in fp16 (vLLM `--dtype float16`): the same rows with f16
activations and the default f32 state, then with an f16 state
(`--mamba-ssm-cache-dtype float16`) and with the fused norm.

## Usage

```sh
OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=0-31 --membind=0 \
./install/benchdnn/bin/benchdnn --op=gdn \
    --input_file=../benchdnn/input/gdn/gdn_models.txt
```

Add `--src_dt=f16` to time every row with fp16 activations.

Pin the threads to one NUMA node and keep its memory local: decode is
memory-bound and its time follows the bandwidth of the node it runs on.
`ZENDNNL_GDN_KERNEL` and `ZENDNNL_GDN_PREFILL` override the per-line
`kernel` and `prefill_algo` (see [runtime_env.md](../../docs/runtime_env.md));
with `ZENDNNL_API_LOG_LEVEL=4` every call logs the kernel and prefill
algorithm it ran.

## Output

| Column | Meaning |
|--------|---------|
| `Model` ... `Prefill_Len` | The line's label, shape and batch. |
| `Tokens` | `num_decodes + num_prefills * prefill_len`. |
| `Src_DT`, `State_DT`, `Init_State`, `Layout`, `Fused_Norm`, `Prefill_Algo`, `Kernel`, `Num_Threads` | The requested options (`auto` = automatic selection). |
| `Iters`, `Warmup_Iters` | Timed and untimed calls. |
| `Total_time(ms)`, `Avg_time(ms)` | Sum and mean of the timed calls. |
| `State_GB/s` | SSM state bytes a call moves (every slot written once, read once unless a prefill starts from zero) over `Avg_time`; for decode this is the figure to hold against the memory bandwidth. |

Sample (VM, AMD EPYC 9454, 32 threads on one node, cold cache):

```
Model            K_Heads  V_Heads  ...  Decodes  Prefills  Prefill_Len  Tokens  Src_DT  ...  Avg_time(ms)  State_GB/s
Qwen3.6-35B-A3B  16       32       ...  8        0         0            8       bf16    ...  0.092220      363.85
Qwen3.6-35B-A3B  16       32       ...  0        1         1024         1024    bf16    ...  1.799601      2.33
Qwen3.6-35B-A3B  16       32       ...  8        1         1024         1032    bf16    ...  1.892713      19.94
Qwen3.6-35B-A3B  16       32       ...  8        0         0            8       f16     ...  0.091653      366.10
Qwen3.6-35B-A3B  16       32       ...  0        1         1024         1024    f16     ...  2.649813      1.58
Qwen3.6-35B-A3B  16       32       ...  8        1         1024         1032    f16     ...  3.122983      12.09
```

Decode costs the same for f16 and bf16 activations (it streams the state);
a chunked f16 prefill costs about 1.3-1.6x a bf16 one, since its GEMMs run
FP32 FMA on FP32 operands where bf16 uses `vdpbf16ps` (see the operator
document).
