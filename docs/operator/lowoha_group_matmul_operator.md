
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA Group MatMul Operator

## Overview

The **Group MatMul API** (`group_matmul_direct`) executes multiple independent matrix multiplications in a single call.  Each operation can have its own dimensions, data buffers, and parameters — the buffers do **not** need to be contiguous in memory.

It supports two execution modes:

- **Sequential (linear)**: Chained operations where the output of one layer feeds the next (multi-layer perceptrons, transformer FFN stacks).
- **Parallel**: Independent GEMMs run concurrently with CCD-aware thread scheduling.  Use cases include multi-head attention projections, parallel Q/K/V computation, and MoE (Mixture of Experts) expert layers.

An optional **MoE post-op** can be attached to the parallel path to fuse expert outputs into a single token-major result via weighted-reduce, avoiding a separate kernel launch.

Include `lowoha_operators/matmul/lowoha_matmul.hpp` (namespace `zendnnl::lowoha::matmul`).

### Key benefits

- Single API call for multiple independent GEMMs — lower function-call overhead
- Non-contiguous buffers: each operation has its own src, weight, bias, and dst pointers
- Sequential chaining **or** parallel execution in one API
- CCD-aware adaptive parallelization (auto-selects best strategy)
- Optional MoE weighted-reduce post-op (last argument, `nullptr` to disable)
- Unified status: `status_t::success` only when everything succeeds

## API signature

```cpp
status_t group_matmul_direct(
  const std::vector<char> &layout,
  const std::vector<bool> &transA,
  const std::vector<bool> &transB,
  const std::vector<int> &M,
  const std::vector<int> &N,
  const std::vector<int> &K,
  const std::vector<float> &alpha,
  const std::vector<const void *> &src,
  const std::vector<int> &lda,
  const std::vector<const void *> &weight,
  const std::vector<int> &ldb,
  const std::vector<const void *> &bias,
  const std::vector<float> &beta,
  const std::vector<void *> &dst,
  const std::vector<int> &ldc,
  const std::vector<bool> &is_weights_const,
  std::vector<matmul_params> &params,
  const group_matmul_moe_postop_params *moe_postop = nullptr);
```

### Parameters

Per-op vectors must all have length `num_ops = M.size()`, except `src` whose length selects the execution mode.

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `vector<char>` | `'r'` row-major or `'c'` column-major per op |
| `transA` | `vector<bool>` | Transpose A per op |
| `transB` | `vector<bool>` | Transpose B (weights) per op |
| `M` | `vector<int>` | Rows of A / C per op |
| `N` | `vector<int>` | Columns of B / C per op |
| `K` | `vector<int>` | Inner dimension per op |
| `alpha` | `vector<float>` | Scale for A×B per op |
| `src` | `vector<const void*>` | A pointers; **length selects mode** |
| `lda` | `vector<int>` | Leading dimension of A |
| `weight` | `vector<const void*>` | B (weight) pointers |
| `ldb` | `vector<int>` | Leading dimension of B |
| `bias` | `vector<const void*>` | Bias per op (`nullptr` allowed) |
| `beta` | `vector<float>` | Scale for C accumulation |
| `dst` | `vector<void*>` | C pointers |
| `ldc` | `vector<int>` | Leading dimension of C |
| `is_weights_const` | `vector<bool>` | Weight-caching hint per op |
| `params` | `vector<matmul_params>&` | dtypes, threads, post-ops, etc. |
| `moe_postop` | `const group_matmul_moe_postop_params*` | Optional MoE post-op; **`nullptr`** disables (default) |

### `matmul_params`

```cpp
struct matmul_params {
  matmul_data_types dtypes;
  std::vector<matmul_post_op> postop_;
  matmul_quantization_params_t quant_params;
  char mem_format_a;
  char mem_format_b;
  matmul_algo_t lowoha_algo;
  int32_t num_threads;   // 0 = auto
  std::string plugin_op;
  bool dynamic_quant;
  pack_format packing;
};
```

### `matmul_data_types`

```cpp
struct matmul_data_types {
  data_type_t src;
  data_type_t wei;
  data_type_t dst;
  data_type_t bias;
  data_type_t compute;
};
```

### Return value

- `status_t::success` — all operations (and optional MoE post-op) completed
- `status_t::failure` — validation or kernel failure

## Execution modes

| Condition | Mode | Description |
|-----------|------|-------------|
| `src.size() == 1` | Sequential | Chain: `dst[i-1]` feeds op `i` |
| `src.size() > 1` | Parallel | Independent GEMMs + optional MoE post-op |

### Sequential (linear)

```
┌──────────────────────────────────────────────────────────┐
│           group_matmul_direct (sequential)               │
├──────────────────────────────────────────────────────────┤
│  src[0] ──► ┌──────────┐                                 │
│             │  Op 0    │ (all T threads)                 │
│             └────┬─────┘                                 │
│                  │ dst[0]                                │
│                  ▼                                       │
│             ┌──────────┐                                 │
│             │  Op 1    │                                 │
│             └────┬─────┘                                 │
│                  ▼  …                                    │
│              (output)                                    │
└──────────────────────────────────────────────────────────┘
```

- `K[i+1]` must equal `N[i]` (chaining).
- `moe_postop` must be `nullptr`.

### Parallel

```
┌────────────────────────────────────────────────────────────┐
│            group_matmul_direct (parallel)                  │
├────────────────────────────────────────────────────────────┤
│  OMP parallel for over ops                                 │
│  ┌─────────┐ ┌─────────┐       ┌─────────┐               │
│  │  Op 0   │ │  Op 1   │  ...  │ Op N-1  │               │
│  │ expert 0│ │ expert 1│       │expert E-1│               │
│  └────┬────┘ └────┬────┘       └────┬────┘               │
│       │           │                 │                      │
│       └───────────┴────── ▼ ────────┘                      │
│            (optional MoE post-op)                          │
│              weighted-reduce → output                      │
└────────────────────────────────────────────────────────────┘
```

## MoE post-op (parallel mode only)

When `moe_postop != nullptr`, a weighted-reduce runs after the parallel expert GEMMs:

### Weighted-reduce

For each token `t` and hidden dim `d`:

```
output[t, d] = Σ_{k=0}^{topk-1}  topk_weights[t, k] × row_ptrs[t·topk + k][d]
```

When `skip_weighted == true`, every weight is implicitly 1.0 (plain gather-sum).

The caller provides pre-gathered row pointers (`row_ptrs`) built during the token-to-expert scatter step on the frontend side. This keeps the library focused on compute and enables future GEMM fusion.

### `group_matmul_moe_postop_params`

```cpp
struct group_matmul_moe_postop_params {
  int num_tokens;
  int topk;
  void *output;
  int ldc_output;
  const float *topk_weights;
  bool skip_weighted;
  const void **row_ptrs;
};
```

| Field | Type | Description |
|-------|------|-------------|
| `num_tokens` | `int` | Number of input tokens (rows in output). Must be > 0. |
| `topk` | `int` | Experts selected per token. Must be > 0. |
| `output` | `void*` | Row-major `[num_tokens, ldc_output]` output buffer. First `D` columns written (FP32 or BF16 matching expert `dst` dtype). |
| `ldc_output` | `int` | Leading dimension of `output` (>= `D`, where `D = N[0]`). |
| `topk_weights` | `const float*` | Tightly-packed `[num_tokens, topk]` routing weights (row-major). Entry `[t * topk + k]` scales token `t`'s `k`-th expert. Required unless `skip_weighted == true`. |
| `skip_weighted` | `bool` | When `true`, all routing weights are implicitly 1.0; `topk_weights` may be `nullptr`. |
| `row_ptrs` | `const void**` | Pre-gathered row pointers: flat array of size `num_tokens * topk`. Entry `row_ptrs[t * topk + k]` points to the start of a D-wide row (FP32 or BF16) in an expert `dst` buffer. Must be non-null. |

### How the caller builds `row_ptrs`

During the token-to-expert scatter (grouping tokens into expert batches), build `row_ptrs` alongside the scatter:

```python
row_ptrs = [None] * (num_tokens * topk)
for t in range(num_tokens):
    for k in range(topk):
        expert_id = topk_indices[t, k]
        row_j = current_count[expert_id]
        expert_src[expert_id][row_j] = hidden_states[t]         # scatter
        row_ptrs[t * topk + k] = dst[expert_id] + row_j * ldc   # gather (fused!)
        current_count[expert_id] += 1
```

### Validation rules

- Parallel mode only (`src.size() > 1`).
- `row_ptrs` and `output` must be non-null.
- All `N[i]` equal (hidden dim `D`); `ldc_output >= D`.
- Every expert's `dst` dtype must be uniform **FP32 or BF16**.

## Thread configuration

| Mode | Typical thread use |
|------|---------------------|
| Sequential | Each op uses the resolved thread count |
| Parallel (auto, ALGO=0) | Auto-selects: V2 (per-expert) when experts >= threads, V3 (multilevel) otherwise |
| Sequential (ALGO=1) | Experts run serially, each GEMM uses all threads |
| Per-expert (ALGO=2) | Parallel-for over experts, 1 thread per expert |
| Multilevel (ALGO=3) | CCD-aware nested OMP: multi-CCD for large M, round-based for small M. Best standalone perf. |
| Flat CCD M-slice (ALGO=4) | Single-level OMP, CCD-aware M-slicing. Proportional CCD allocation for few experts, round-based for many. No nested OMP — framework-safe. |

Set via `OMP_NUM_THREADS` or `params[i].num_threads` (0 = library default).

## Usage examples

### Parallel group matmul (no MoE)

```cpp
int lowoha_group_matmul_example() {
  using namespace zendnnl::lowoha::matmul;

  constexpr int NUM_OPS = 4;
  std::vector<int> Ms = {64, 128, 32, 256};
  std::vector<int> Ns = {128, 64, 256, 64};
  std::vector<int> Ks = {256, 256, 128, 128};

  std::vector<std::vector<float>> src_buf(NUM_OPS), wei_buf(NUM_OPS),
                                   bias_buf(NUM_OPS), dst_buf(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    src_buf[i].resize(Ms[i] * Ks[i], 1.0f);
    wei_buf[i].resize(Ks[i] * Ns[i], 1.0f);
    bias_buf[i].resize(Ns[i], 0.0f);
    dst_buf[i].resize(Ms[i] * Ns[i], 0.0f);
  }

  std::vector<const void *> src(NUM_OPS), wei(NUM_OPS), bias(NUM_OPS);
  std::vector<void *> dst(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    src[i]  = src_buf[i].data();
    wei[i]  = wei_buf[i].data();
    bias[i] = bias_buf[i].data();
    dst[i]  = dst_buf[i].data();
  }

  std::vector<char> layout(NUM_OPS, 'r');
  std::vector<bool> tA(NUM_OPS, false), tB(NUM_OPS, false);
  std::vector<float> alpha(NUM_OPS, 1.f), beta(NUM_OPS, 0.f);
  std::vector<bool> wconst(NUM_OPS, true);
  std::vector<int> lda(NUM_OPS), ldb(NUM_OPS), ldc(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    lda[i] = Ks[i]; ldb[i] = Ns[i]; ldc[i] = Ns[i];
  }

  std::vector<matmul_params> params(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    params[i].dtypes = {data_type_t::f32, data_type_t::f32,
                        data_type_t::f32, data_type_t::f32,
                        data_type_t::f32};
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
  }

  status_t st = group_matmul_direct(
      layout, tA, tB, Ms, Ns, Ks, alpha,
      src, lda, wei, ldb, bias, beta, dst, ldc,
      wconst, params,
      nullptr);   // no MoE post-op

  return (st == status_t::success) ? 0 : -1;
}
```

### Sequential (chained) matmul

```cpp
int lowoha_sequential_matmul_example() {
  using namespace zendnnl::lowoha::matmul;

  const int NUM_OPS = 3, M = 64;
  std::vector<int> Ms = {M, M, M};
  std::vector<int> Ks = {128, 256, 128};
  std::vector<int> Ns = {256, 128, 64};

  std::vector<float> input(M * Ks[0], 1.0f);
  std::vector<std::vector<float>> wei_buf(NUM_OPS), dst_buf(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    wei_buf[i].resize(Ks[i] * Ns[i], 0.5f);
    dst_buf[i].resize(Ms[i] * Ns[i], 0.0f);
  }

  std::vector<const void *> src = {input.data()};
  std::vector<const void *> wei(NUM_OPS), bias(NUM_OPS, nullptr);
  std::vector<void *> dst(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    wei[i] = wei_buf[i].data();
    dst[i] = dst_buf[i].data();
  }

  std::vector<char> layout(NUM_OPS, 'r');
  std::vector<bool> tA(NUM_OPS, false), tB(NUM_OPS, false);
  std::vector<float> alpha(NUM_OPS, 1.f), beta(NUM_OPS, 0.f);
  std::vector<int> lda = Ks, ldb = Ns, ldc = Ns;
  std::vector<bool> wconst(NUM_OPS, false);

  std::vector<matmul_params> params(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    params[i].dtypes = {data_type_t::f32, data_type_t::f32,
                        data_type_t::f32, data_type_t::f32,
                        data_type_t::f32};
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
  }

  status_t st = group_matmul_direct(
      layout, tA, tB, Ms, Ns, Ks, alpha,
      src, lda, wei, ldb, bias, beta, dst, ldc,
      wconst, params,
      nullptr);   // MoE not supported in sequential mode

  return (st == status_t::success) ? 0 : -1;
}
```

## Notes and best practices

1. **Vector lengths**: All per-op vectors must have length `num_ops`; `src` length selects the mode.
2. **Chaining**: In sequential mode `K[i+1] == N[i]`.
3. **Parallel independence**: Each op uses its own `src[i]`, `weight[i]`, `dst[i]`.
4. **MoE**: Pass `nullptr` when not needed. When wired, provide `row_ptrs` (built during scatter) and `topk_weights`.
5. **Weight caching**: `is_weights_const[i] == true` enables caching on repeated calls.
6. **Alignment**: Align buffers for best kernel performance.
7. **Errors**: On failure, check logs for dimension / dtype / MoE validation messages.
