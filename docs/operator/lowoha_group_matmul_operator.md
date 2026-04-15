
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
- N-tiling for large-N experts: up to 3.9× faster than sequential for Mixtral gate_proj
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
| `layout` | `vector<char>` | `'r'`/`'R'` row-major or `'c'` column-major per op; tiled algos (2/3) require row-major |
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

### Return value

- `status_t::success` — all operations (and optional MoE post-op) completed
- `status_t::failure` — validation or kernel failure

## Execution modes

| Condition | Mode | Description |
|-----------|------|-------------|
| `src.size() == 1` | Sequential | Chain: `dst[i-1]` feeds op `i` |
| `src.size() > 1` | Parallel | Independent GEMMs + optional MoE post-op |

## Parallel strategy selection (`ZENDNNL_GRP_MATMUL_ALGO`)

| ALGO | Name | Strategy | OMP nesting |
|------|------|----------|-------------|
| **0** | Auto | Selects 1, 2, or 3 based on expert count, M, and N | — |
| **1** | Sequential | Experts serial, each GEMM uses all threads. Default for ≤4 experts and large K×N. | None |
| **2** | Adaptive tile | Per-expert M-tile or N-tile decision. No nested OMP — framework-safe. | Flat |
| **3** | N-tile | Pure N-tiling with decode-optimized sequential regime. Up to 3.9× for Mixtral. | Flat |
| **4** | Multilevel | CCD-aware nested OMP. Multi-CCD for large M, round-based for small M. | Nested |
| **5** | Per-expert | Parallel-for over experts, 1 thread each. Best when experts ≥ threads. | Flat |

### Auto-select decision tree (ALGO=0)

```
num_ops ≤ 4                            → ALGO 1 (sequential)
num_ops ≥ 32, max_M ≥ 8               → ALGO 3 (N-tile)
num_ops ≥ 32, max_M < 8, max_N ≤ 2048 → ALGO 3 (N-tile, small B fits L3)
num_ops ≥ 32, max_M < 8, max_N > 2048 → ALGO 2 (adaptive tile)
num_ops ≥ 8, max_M ≥ 8                → ALGO 3 (N-tile)
num_ops ≥ 16, max_M < 8               → ALGO 2 (adaptive tile)
num_ops ≥ 8, max_M ≤ 1                → ALGO 2 (adaptive tile)
fallback                               → ALGO 1 (sequential)
```

Tiled algos (2/3) require row-major layout, uniform dtypes, no quantization, no post-ops, and standard unpacked A and B (`mem_format_a=='n'`, `mem_format_b=='n'`, `pack_format_b==0`). If these conditions are not met, auto-select falls back to ALGO 1.

Set the strategy via environment variable: `ZENDNNL_GRP_MATMUL_ALGO=0|1|2|3|4|5`.

## MoE post-op (parallel mode only)

When `moe_postop != nullptr`, a weighted-reduce runs after the parallel expert GEMMs.

### What it does

For each token `t` and hidden dim `d`:

```
output[t, d] = Σ_{k=0}^{topk-1}  topk_weights[t, k] × row_ptrs[t·topk + k][d]
```

When `skip_weighted == true`, every weight is implicitly 1.0 (plain gather-sum).

The library performs **only** the weighted-reduce — not the gather. The caller must pre-build the `row_ptrs` array during the token-to-expert scatter step.

### `group_matmul_moe_postop_params`

```cpp
struct group_matmul_moe_postop_params {
  int num_tokens = 0;           // rows in the output buffer
  int topk = 0;                 // experts per token
  void *output = nullptr;       // [num_tokens, ldc_output] FP32 or BF16
  int ldc_output = 0;           // leading dim of output (>= D)
  const float *topk_weights = nullptr;  // [num_tokens, topk] routing weights
  bool skip_weighted = false;   // true → all weights = 1.0
  const void **row_ptrs = nullptr;      // [num_tokens * topk] pre-gathered row pointers
};
```

| Field | Requirement |
|-------|-------------|
| `num_tokens` | > 0 |
| `topk` | > 0 |
| `output` | Non-null. dtype must match expert `dst` dtype (FP32 or BF16). |
| `ldc_output` | ≥ D (where D = N[0]) |
| `topk_weights` | Required unless `skip_weighted == true` |
| `row_ptrs` | Non-null. Each entry points to a D-wide row in an expert dst buffer. |

### How the caller builds `row_ptrs`

During token-to-expert scatter (grouping tokens into expert batches), build `row_ptrs` alongside the scatter. This fuses the gather pointer construction with the scatter in a single pass:

```text
# Python/Zentorch-side pseudocode
row_ptrs = [None] * (num_tokens * topk)
current_count = [0] * num_experts

for t in range(num_tokens):
    for k in range(topk):
        expert_id = topk_indices[t, k]
        row_j = current_count[expert_id]

        # Scatter: copy token's hidden state into expert's input batch
        expert_src[expert_id][row_j] = hidden_states[t]

        # Gather pointer: where this token's result will be in expert output
        row_ptrs[t * topk + k] = &expert_dst[expert_id][row_j * ldc]

        current_count[expert_id] += 1
```

### C++ MoE post-op example (BF16, 4 experts, topk=2)

```cpp
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
using namespace zendnnl::lowoha::matmul;

// After setting up expert GEMMs (src, weight, dst, params, etc.)...

const int NUM_TOKENS = 8, TOPK = 2, N = 64, NUM_EXPERTS = 4;

// Build row_ptrs: token t → experts (t%4) and ((t+1)%4)
std::vector<const void *> row_ptrs(NUM_TOKENS * TOPK);
std::vector<float> weights(NUM_TOKENS * TOPK, 0.5f);  // equal routing

for (int t = 0; t < NUM_TOKENS; ++t) {
  int e0 = t % NUM_EXPERTS, e1 = (t + 1) % NUM_EXPERTS;
  row_ptrs[t * TOPK + 0] =
      static_cast<const uint16_t *>(dst_ptrs[e0]) + t * N;
  row_ptrs[t * TOPK + 1] =
      static_cast<const uint16_t *>(dst_ptrs[e1]) + t * N;
}

// MoE output buffer
std::vector<uint16_t> moe_output(NUM_TOKENS * N, 0);

// Fill the post-op struct
group_matmul_moe_postop_params moe;
moe.num_tokens = NUM_TOKENS;
moe.topk = TOPK;
moe.output = moe_output.data();
moe.ldc_output = N;
moe.topk_weights = weights.data();
moe.skip_weighted = false;
moe.row_ptrs = row_ptrs.data();

// Execute: expert GEMMs + fused weighted-reduce
status_t st = group_matmul_direct(
    layouts, transAs, transBs, Ms, Ns, Ks, alphas,
    src_ptrs, ldas, weight_ptrs, ldbs, bias_ptrs, betas,
    dst_ptrs, ldcs, is_weights_const, params, &moe);
// moe_output now contains the reduced [NUM_TOKENS, N] tensor
```

See `examples/lowoha_group_matmul_example.cpp` for complete runnable FP32, BF16, and MoE post-op examples with verification.

## Notes and best practices

1. **Vector lengths**: All per-op vectors must have length `num_ops`; `src` length selects the mode.
2. **Chaining**: In sequential mode `K[i+1] == N[i]`.
3. **Parallel independence**: Each op uses its own `src[i]`, `weight[i]`, `dst[i]`.
4. **MoE**: Pass `nullptr` when not needed. When enabled, provide `row_ptrs` (built during scatter) and `topk_weights`.
5. **Weight caching**: `is_weights_const[i] == true` enables caching. N-tiling creates per-slice cache entries (O(experts × tiles) — manageable for MoE).
6. **Tiled algos (2/3)**: Require row-major layout, uniform dtypes across experts, no quantization, no post-ops, and standard unpacked B. Falls back to ALGO 1 automatically when these conditions are not met.
7. **Errors**: On failure, check logs for dimension / dtype / MoE validation messages.
8. **Strategy override**: Set `ZENDNNL_GRP_MATMUL_ALGO=N` (1-5) to force a specific strategy. `0` (default) auto-selects.
