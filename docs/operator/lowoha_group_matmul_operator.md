
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA Group MatMul Operator

## Overview

The **Group MatMul API** (`group_matmul_direct`) executes multiple independent matrix multiplications in a single call.  Each operation can have its own dimensions, data buffers, and parameters — the buffers do **not** need to be contiguous in memory.

It supports two execution modes:

- **Sequential (linear)**: Chained operations where the output of one layer feeds the next (multi-layer perceptrons, transformer FFN stacks).
- **Parallel**: Independent GEMMs run concurrently with CCD-aware thread scheduling.  Use cases include multi-head attention projections, parallel Q/K/V computation, and MoE (Mixture of Experts) expert layers.

An optional **MoE post-op** can be attached to the parallel path to fuse expert outputs into a single token-major result via weighted-reduce, avoiding a separate kernel launch.

An optional **gated activation post-op** can be attached to fuse gate+up projections: after the GEMM (which uses fused `[gate_W | up_W]` weights producing `N = 2*dim` output columns), the activation computes in-place `dst[:, 0:dim] = act(gate) * up`.  Supported activations: `silu_and_mul` (Mixtral, Llama, Qwen), `gelu_and_mul`, `swiglu_oai_mul` (interleaved layout).  Applied after GEMM and before the MoE weighted-reduce (if both are set).

An optional **fused MoE** parameter expresses the entire MoE block — Op1 (gate+up) → gated activation → Op2 (down_proj) — as a single API call.  In the current V1 implementation all `GRP_ALGO` values execute this path as two internal dispatches (Pass 1: Op1 + activation, Pass 2: Op2) behind the single-call interface.  Per-expert or per-M-tile deep fusion that keeps intermediate data hot in cache across all three operations is a future optimization.

Include `lowoha_operators/matmul/lowoha_matmul.hpp` (namespace `zendnnl::lowoha::matmul`).

### Key benefits

- Single API call for multiple independent GEMMs — lower function-call overhead
- Non-contiguous buffers: each operation has its own src, weight, bias, and dst pointers
- Sequential chaining **or** parallel execution in one API
- CCD-aware adaptive parallelization (auto-selects best strategy)
- N-tiling for large-N experts: up to 3.9× faster than sequential for Mixtral gate_proj
- Optional MoE weighted-reduce post-op (`moe_postop`, `nullptr` to disable)
- Optional gated activation post-op (`gated_act`) for fused gate+up projections (silu_and_mul, gelu_and_mul, swiglu_oai_mul)
- Optional single-call fused MoE interface (`fused_moe`) for Op1(gate+up) → activation → Op2(down_proj); currently two-pass internally in V1
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
  const group_matmul_moe_postop_params *moe_postop = nullptr,
  const grp_matmul_gated_act_params *gated_act = nullptr,
  const grp_matmul_fused_moe_params *fused_moe = nullptr);
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
| `params` | `vector<matmul_params>&` | dtypes, threads, post-ops, plus the optional `active_matmul / total_matmul` prepack-extras hint (see [Framework prepack-extras contract](#framework-prepack-extras-contract)) |
| `moe_postop` | `const group_matmul_moe_postop_params*` | Optional MoE post-op; **`nullptr`** disables (default) |
| `gated_act` | `const grp_matmul_gated_act_params*` | Optional gated activation; **`nullptr`** disables (default). Requires N even, dst dtype f32/bf16. Applied after GEMM, before `moe_postop`. |
| `fused_moe` | `const grp_matmul_fused_moe_params*` | Optional fused MoE: Op1(gate+up) → activation → Op2(down_proj) in one call; **`nullptr`** disables (default). See [Fused MoE](#fused-moe-op1--activation--op2). |

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

Tiled algos (2/3) require row-major layout, uniform per-expert dtypes, and standard unpacked A and B (`mem_format_a=='n'`, `mem_format_b=='n'`, `pack_format_b==0`). ALGO 2 (M-tile, env-only) additionally supports the full quantization stack (WOQ, static W8A8, and dynamic INT8 with row-local src granularity — `{M, 1}` per-token or `{M, G}` per-group) and most post-ops; it rejects packed B, softmax/pooling, and dynamic-quant with non-row-local src granularity (per-tensor / per-column / per-channel). ALGO 3 (N-tile) rejects any quantization scales/zero-points and any buffer-bearing post-op (column slicing cannot offset N-indexed metadata). Auto-select never picks ALGO 2 — reach it via the env override — and falls back to ALGO 1 when the safety gate fails.

Set the strategy via environment variable: `ZENDNNL_GRP_MATMUL_ALGO=0|1|2|3|4|5`.

## Framework prepack-extras contract

For MoE deployments where the framework holds **all** expert weights but only a routing-decided subset is firing on any given call, set two fields on `params[0]` (the dispatcher reads them from the first entry only):

| Field | Type | Meaning |
|---|---|---|
| `params[0].total_matmul` | `uint32_t` | Total expert slots whose weights are present in `weight[]`, `K[]`, `N[]`, `ldb[]`, `transB[]`, `is_weights_const[]`, `bias[]`.  `0` means "no prepack-extras tail" (the dispatcher resolves it to `active_matmul`). |
| `params[0].active_matmul` | `uint32_t` | Number of firing experts (`<= M.size()`).  These occupy the **first `active_matmul`** entries of every weight-side vector.  On the input side, only the **first `active_matmul`** entries are consumed from `M[]`, `src[]`, `lda[]`, `dst[]`, `ldc[]`, ... — see the two accepted sizing patterns below. |

When `active_matmul > 0` the dispatcher accepts weight-side vectors of size `>= active_matmul` (instead of requiring `== num_ops`).  Input-side vectors must satisfy `size >= active_matmul`; two layouts are both legitimate:

- **Compact (vLLM-style):** `M.size() == active_matmul`.  Input vectors carry only the firing experts.  Weight-side vectors carry `total_matmul` entries (firing experts in `[0, active_matmul)`, prepack-extras tail in `[active_matmul, total_matmul)`).
- **Padded (gtest-style):** `M.size() == total_matmul`.  Input vectors are sized to the full pool with `M[active_matmul..total_matmul) == 0` placeholders; the dispatcher skips the zero-M slots and computes only the first `active_matmul` GEMMs.

The library:

1. **Computes** only the first `active_matmul` GEMMs (zero-M placeholders in the padded form are no-ops).
2. **Pre-warms** the inner-kernel weight cache for **all** `total_matmul` slots ahead of time, so any expert that fires on a future call hits a warm cache (no on-the-fly reorder spike).

Leave both fields at their default `0` to keep the legacy contract: `num_ops = M.size()`, all weight-side vectors must be exactly `num_ops` long, all of them fire.  The library still pre-warms the firing experts when `ZENDNNL_GRP_MATMUL_PREPACK=1` (default) — see [Environment variables](#environment-variables).

### Example (firing 4 of 8 experts)

```cpp
const int FIRING = 4, TOTAL = 8;

// Weight-side vectors carry all 8 experts; the first 4 are the
// "firing" ones the runtime expert-selection routed to this call,
// the last 4 are pre-pack extras whose weights are warmed but
// not computed.
std::vector<const void *> weight(TOTAL, /*...*/);
std::vector<int> K(TOTAL, /*...*/), N(TOTAL, /*...*/), ldb(TOTAL, /*...*/);
std::vector<bool> transB(TOTAL, false), is_weights_const(TOTAL, true);

// Input-side vectors only carry the firing 4 (M, src, dst, ldas, etc.):
std::vector<int> M(FIRING, /*...*/);
std::vector<const void *> src(FIRING, /*...*/);
std::vector<void *> dst(FIRING, /*...*/);
// ... lda, ldc, alpha, beta, layout, transA also of size FIRING ...

std::vector<matmul_params> params(FIRING);
params[0].active_matmul = FIRING;
params[0].total_matmul  = TOTAL;
// (Other params[i].active_matmul / total_matmul are unused.)

status_t st = group_matmul_direct(
    layout, transA, transB, M, N, K, alpha,
    src, lda, weight, ldb, bias, beta,
    dst, ldc, is_weights_const, params);
```

## Environment variables

User-facing knobs that affect `group_matmul_direct` behaviour.  All are read once and cached on first reference (except `ZENDNNL_GRP_MATMUL_ALGO`, intentionally re-read per call so production runs can flip ALGO between phases without restart).  Set them before the first `group_matmul_direct` call.

| Variable | Default | Effect |
|---|---|---|
| `ZENDNNL_GRP_MATMUL_ALGO` | `0` (auto) | Force a parallel strategy. `0` = auto, `1`-`5` = ALGO 1-5. See the table above. |
| `ZENDNNL_GRP_MATMUL_PREPACK` | `1` (ON) | Master switch for ahead-of-time weight prepack. When ON the dispatcher eagerly populates the inner-kernel weight cache for `max(M.size(), total_matmul)` experts on first reference, eliminating per-expert reorder spikes during inference. Set `0` to fall back to lazy on-first-touch reorder (lower first-call latency, but cache-miss spikes are visible during steady-state when a fresh expert routes). |
| `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL` | `0` (OFF) | Enables an in-house BF16-only AVX-512 microkernel for ALGO 3 (N-tile). Wins on single- and few-thread workloads; can lose to AOCL DLP at high thread counts on large MoE shapes. Falls back to the standard backend when the contract fails (non-BF16, transA, alpha != 1, ...). |
| `ZENDNNL_GRP_MATMUL_AOCL_STABLE_NTILE` | `1` (ON) | Pin ALGO 3's per-expert thread count to a `num_threads`-only formula so the AOCL DLP weight-reorder cache key is stable across MoE routing variation (active-expert filtering, batch-size shifts, ...). Disable only for A/B comparison; the fully-relaxed planner can degrade cache hit-rate under churn. |
| `ZENDNNL_MATMUL_WEIGHT_CACHE` | `1` (ON) | Standard weight-reorder cache for AOCL DLP / BRGEMM. Setting `0` disables both lazy and prepack populations — prepack short-circuits to a no-op rather than wasting CPU on entries that won't be cached. |

### Weight caching, prepack, and memory

When `ZENDNNL_GRP_MATMUL_PREPACK=1` (default), each firing call eagerly reorders weights for **all** `total_matmul` experts (or `M.size()` experts in legacy mode) on the **first** invocation that observes that configuration.  Subsequent calls short-circuit via a per-thread fingerprint cache (~100 ns overhead) and benefit from warm caches.

Trade-offs by deployment scenario:

| Scenario | Behaviour | Recommendation |
|---|---|---|
| Production MoE inference (32 experts, BF16, ALGO 3) | First call: ~325 ms one-time reorder cost. Steady state: warm caches, no per-token spikes. ~2 GB persistent weight cache (held for process lifetime). | Default `PREPACK=1` is the right choice. The first-call cost lands inside warm-up. |
| Latency-critical first-call (interactive serving without warm-up) | The ~325 ms first-call latency is unacceptable. | Set `ZENDNNL_GRP_MATMUL_PREPACK=0`. Falls back to lazy reorder — first call ~10 ms parallel, subsequent calls add a per-fresh-expert reorder spike. |
| Memory-bounded (multi-tenant, container quota < 4 GB) | Eager prepack populates ~2 GB of LRU per process for gpt-oss-20B-class shapes. | Either `ZENDNNL_GRP_MATMUL_PREPACK=0` (no eager warm), or `ZENDNNL_MATMUL_WEIGHT_CACHE=0` (kills both lazy and prepack — every call re-reorders, lowest memory but slowest steady-state). |
| Single-shot inference (eg. unit tests) | First-call cost dominates; no steady-state benefit. | `PREPACK=0` is fine. Default `1` is also fine — the cost is bounded and the test path itself is dominated by gtest fixture overhead. |

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

See `examples/lowoha_group_matmul_example.cpp` for complete runnable FP32, BF16, MoE post-op, gated activation, and fused MoE examples with verification.

## Gated activation post-op (parallel mode only)

When `gated_act != nullptr`, an in-place gated activation runs after the expert GEMMs and before the MoE weighted-reduce (if both are set).

### What it does

For each expert `e`, row `m`, and output column `d` (where `dim = N/2`):

```
dst[e][m, d] = act(gate[d]) * up[d]
```

The GEMM uses fused `[gate_W | up_W]` weights producing `N = 2*dim` output columns.  The first `dim` columns are the gate projection, the second `dim` columns are the up projection.  After activation, only `dst[:, 0:dim]` is meaningful.

For `swiglu_oai_mul`, the layout is interleaved: `[g0, u0, g1, u1, ...]`.

### `grp_matmul_gated_act_params`

```cpp
enum class grp_matmul_gated_act_t : int {
  none = 0,           // No activation (default).
  silu_and_mul = 1,   // SiLU(gate) * up — Mixtral, Llama, Qwen.
  gelu_and_mul = 2,   // GELU(gate) * up — some GPT variants.
  swiglu_oai_mul = 3  // SwigluOAI — interleaved gate/up layout.
};

struct grp_matmul_gated_act_params {
  grp_matmul_gated_act_t act = grp_matmul_gated_act_t::none;
};
```

### Constraints

| Constraint | Requirement |
|------------|-------------|
| `N` | Must be even (`N = 2 * dim`) for all experts |
| `dst dtype` | Must be FP32 or BF16 (uniform across experts) |
| `layout` | Must be row-major (`'r'` / `'R'`) for all experts |
| Mode | Parallel only (`src.size() > 1`) |

### C++ gated activation example

```cpp
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
using namespace zendnnl::lowoha::matmul;

// After setting up expert GEMMs with N = 2*dim...

grp_matmul_gated_act_params act;
act.act = grp_matmul_gated_act_t::silu_and_mul;

status_t st = group_matmul_direct(
    layouts, transAs, transBs, Ms, Ns, Ks, alphas,
    src_ptrs, ldas, weight_ptrs, ldbs, bias_ptrs, betas,
    dst_ptrs, ldcs, is_weights_const, params,
    nullptr,   // no MoE weighted-reduce
    &act);     // gated activation: silu(gate) * up
// dst[:, 0:dim] now contains the activated output
```

## Fused MoE (Op1 → activation → Op2)

When `fused_moe != nullptr`, the entire MoE block — gate+up projection (Op1), gated activation, and down projection (Op2) — executes as a single fused API call.

### How it works

| ALGO | Strategy | Cache benefit |
|------|----------|---------------|
| **All (V1)** | Two-pass — Op1+Act via parallel dispatch, then Op2 via parallel dispatch | Single API call; ALGO env honored for Op1 |

The weighted-reduce (`moe_postop`) always runs in a separate pass after Op2 since it requires all experts' outputs to be complete.

### `grp_matmul_fused_moe_params`

```cpp
struct grp_matmul_fused_moe_params {
  std::vector<const void *> down_weight;  // Per-expert down_proj weights [dim, N_down[i]].
  std::vector<int> N_down;                // Per-expert output columns of down_proj.
  std::vector<int> ldb_down;              // Leading dimension of down_weight per expert.
  std::vector<const void *> bias_down;    // Per-expert bias for down_proj (nullptr OK).
  data_type_t bias_dt_down = data_type_t::none;  // Bias dtype for Op2 (none = no bias).
  std::vector<void *> dst_down;           // Per-expert down_proj output [M, N_down[i]].
  std::vector<int> ldc_down;              // Leading dimension of dst_down per expert.
};
```

All vectors must have size `num_ops` (matching the main Op1 vectors).

| Field | Description |
|-------|-------------|
| `down_weight` | Per-expert weight matrices for the down projection. |
| `N_down` | Per-expert output column count (typically all the same: `hidden_size`). |
| `ldb_down` | Leading dimension of each `down_weight[i]` (>= `N_down[i]`). |
| `bias_down` | Per-expert bias (each entry can be `nullptr` for no bias). |
| `bias_dt_down` | Bias data type for Op2 (`none` = no bias; must match actual bias buffer dtype). |
| `dst_down` | Per-expert output buffers for down_proj results. |
| `ldc_down` | Leading dimension of each `dst_down[i]` (>= `N_down[i]`). |

### Constraints

| Constraint | Requirement |
|------------|-------------|
| `down_weight.size()` | `== num_ops` (all fused vectors must match) |
| `N` | Must be even (`N = 2 * dim`) for all experts |
| `K_down` | `= N_gate_up / 2` (the `dim` after gated activation) |
| `dst_down[i]` | Must hold at least `[M[i], N_down[i]]` elements |
| `layout` | Must be row-major (`'r'` / `'R'`) for all experts |
| Mode | Parallel only (`src.size() > 1`) |
| `fused_moe` | Must be `nullptr` to disable; non-null requires all vectors populated |

### C++ fused MoE example

```cpp
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
using namespace zendnnl::lowoha::matmul;

// Op1 params: gate+up projection with N = 2*dim
// (layouts, transAs, transBs, Ms, Ns, Ks, alphas, srcs, ldas,
//  gate_up_weights, ldbs, biases, betas, gate_up_dsts, ldcs,
//  is_weights_const, params already set up)

// Set up fused MoE params for Op2 (down_proj)
grp_matmul_fused_moe_params fused;
fused.down_weight = down_proj_weights;   // vector<const void*>
fused.N_down = std::vector<int>(num_ops, hidden_size);
fused.ldb_down = std::vector<int>(num_ops, hidden_size);
fused.bias_down = std::vector<const void*>(num_ops, nullptr);
fused.dst_down = down_proj_outputs;      // vector<void*>
fused.ldc_down = std::vector<int>(num_ops, hidden_size);

// Gated activation
grp_matmul_gated_act_params act;
act.act = grp_matmul_gated_act_t::silu_and_mul;

// Single call: Op1 → silu(gate)*up → Op2
status_t st = group_matmul_direct(
    layouts, transAs, transBs, Ms, Ns, Ks, alphas,
    srcs, ldas, gate_up_weights, ldbs, biases, betas,
    gate_up_dsts, ldcs, is_weights_const, params,
    nullptr,   // no MoE weighted-reduce (or pass &moe for final reduce)
    &act,      // gated activation
    &fused);   // fused down_proj
// fused.dst_down now contains the down_proj output per expert
```

## Notes and best practices

1. **Vector lengths**: By default all per-op vectors must have length `num_ops = M.size()`; `src` length selects the mode.  When the [framework prepack-extras contract](#framework-prepack-extras-contract) is engaged (`params[0].active_matmul > 0`), the dispatcher accepts weight-side vectors of size `>= active_matmul` while input-side vectors stay at `M.size()`.
2. **Chaining**: In sequential mode `K[i+1] == N[i]`.
3. **Parallel independence**: Each op uses its own `src[i]`, `weight[i]`, `dst[i]`.
4. **MoE**: Pass `nullptr` when not needed. When enabled, provide `row_ptrs` (built during scatter) and `topk_weights`.
5. **Gated activation**: Pass `nullptr` when not needed. When enabled, N must be even and dst dtype must be FP32 or BF16.  Applied after GEMM, before MoE weighted-reduce.
6. **Weight caching and prepack**: `is_weights_const[i] == true` enables caching for op `i`.  By default (`ZENDNNL_GRP_MATMUL_PREPACK=1`) the library *eagerly* warms the cache on the first observation of a configuration; subsequent calls hit warm caches with negligible overhead.  See [Weight caching, prepack, and memory](#weight-caching-prepack-and-memory) for the trade-offs (~325 ms first-call cost vs no per-token spikes; ~2 GB persistent cache for 32-expert MoE).
7. **Tiled algos (2/3)**: Both require row-major layout, uniform per-expert dtypes, and standard unpacked A/B. ALGO 2 (M-tile, `m_tile_safe`) additionally supports the full quantization stack — weight-only (S4 sym, U4 asym), static W8A8 (per-tensor / per-channel / per-group / per-token), and **dynamic INT8** (BF16/F32 source quantised at runtime; per-token `{M, 1}` and per-group `{M, G}` activation scales only) — and most post-ops. It blocks: packed B (GGML Q8_0), softmax/pooling, and dynamic-quant with non-row-local src granularity (per-tensor `{}` / `{1}` / `{1, 1}`, per-column `{1, K}`, per-channel-on-src `{1, N}`) because the per-thread reorder would race on the shared scale/zp buffer and use slice-local statistics. M-indexed source-quant metadata (`src_scale` / `src_zp`) is row-offset and dim-sliced per thread inside `group_matmul_m_tile.cpp::offset_quant_by_row` so the dynamic-quant per-group reorder dispatch sees a slice-shaped `src_shape × dims`. ALGO 3 (N-tile, `n_tile_safe`) is stricter: only buffer-free element-wise post-ops (relu, gelu, swish, etc.) are safe under column slicing, and any non-null quant scale/zero-point buffer disables it. Falls back to ALGO 1 automatically when the required safety check fails.
8. **Errors**: On failure, check logs for dimension / dtype / MoE / gated-act validation messages.
9. **Environment variables**: Strategy override (`ZENDNNL_GRP_MATMUL_ALGO`), prepack toggle (`ZENDNNL_GRP_MATMUL_PREPACK`), and other knobs are listed in the [Environment variables](#environment-variables) section above.  See `docs/runtime_env.md` for the master env-var reference.
