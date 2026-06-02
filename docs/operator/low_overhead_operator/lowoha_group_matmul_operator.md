
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
| `src.size() == num_ops` (`> 1`) | Parallel | Independent GEMMs + optional MoE post-op |

`src.size()` must be exactly `1` (sequential) or `num_ops` (parallel); any other size is rejected up front with `status_t::failure`.

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

Auto-select classifies each call by its largest per-expert M into a **decode** or **prompt** phase, then routes to a per-phase default, with a couple of structural guards on top:

```
num_threads ≤ 1  OR  num_ops == 0      → ALGO 1 (sequential)
num_ops > 256                          → ALGO 5 (per-expert; N-tile planner capacity ceiling)
max_M ≤ 32   (decode phase)            → ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO  (default 3, N-tile)
max_M > 32   (prompt phase)            → ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO  (default 2, M-tile)
```

Each phase knob accepts `0` or `1..5`. A value of `1..5` pins that ALGO for the phase (subject to the safety clamps below); a value of `0` defers to the legacy 3-rule cascade:

```
num_ops ≥ num_threads                  → ALGO 3 (N-tile)
num_ops ≤ 8                            → ALGO 1 (sequential)
prompt (max_M > 32)                    → ALGO 1 (sequential)
decode (max_M ≤ 32)                    → ALGO 3 (N-tile)
```

**Safety clamps:** when the chosen tiled algo is not legal for the call, auto-select falls back to ALGO 1 — ALGO 2 requires `m_tile_safe`, ALGO 3 requires `n_tile_safe`. Because the default prompt policy is ALGO 2, **auto-select picks ALGO 2 by default on prompt-phase shapes** (whenever `m_tile_safe` holds). The per-phase knobs (`ZENDNNL_GRP_MATMUL_AUTO_PROMPT_ALGO`, `ZENDNNL_GRP_MATMUL_AUTO_DECODE_ALGO`) are internal tuning overrides; production deployments normally leave them unset.

Tiled algos (2/3) require row-major layout, uniform per-expert dtypes, and standard unpacked A and B (`mem_format_a=='n'`, `mem_format_b=='n'`, `pack_format_b==0`). ALGO 2 (M-tile) additionally supports the full quantization stack (WOQ, static W8A8, and dynamic INT8 with row-local src granularity — `{M, 1}` per-token or `{M, G}` per-group) and most post-ops; it rejects packed B, softmax/pooling, and dynamic-quant with non-row-local src granularity (per-tensor / per-column / per-channel). ALGO 3 (N-tile) accepts **one** quant configuration: `params[i].dynamic_quant=true` with `{M, 1}` per-token source scale and per-channel `{1, N}` (or `{N}`) weight scale (statically quantised weight buffer supplied by the caller). The source-side reorder runs once per expert in `flat_n_tile`'s pre-OMP hoist (`HoistedSrcQuant` in `n_tile/group_matmul_n_tile.cpp`); per-tile threads then read the shared S8 src + column-sliced wei scale. Everything else — static src quant, per-tensor src/wei, per-group `{G, N}` wei, per-group `{M, G}` src, pure WOQ S4/U4/S8, and buffer-bearing post-ops — stays on ALGO 1. See `check_n_tile_extra`'s SCOPE NOTE for the boundaries. Auto-select routes prompt-phase shapes to ALGO 2 by default (when `m_tile_safe` holds) and falls back to ALGO 1 when the safety gate fails.

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

User-facing knobs that affect `group_matmul_direct` behaviour.  All are read once and cached on first reference (including `ZENDNNL_GRP_MATMUL_ALGO`).  Set them before the first `group_matmul_direct` call; mid-process changes have no effect.

| Variable | Default | Effect |
|---|---|---|
| `ZENDNNL_GRP_MATMUL_ALGO` | `0` (auto) | Force a parallel strategy. `0` = auto, `1`-`5` = ALGO 1-5. See the table above. |
| `ZENDNNL_MATMUL_WEIGHT_CACHE` | `1` (ON) | Standard weight-reorder cache for AOCL DLP / BRGEMM. Setting `0` disables both lazy and prepack populations — prepack short-circuits to a no-op rather than wasting CPU on entries that won't be cached. |

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

  // ── Optional Op2 (down_proj) weight quantization ─────────────────────
  // Nested helper, same shape as `matmul_quantization_params_t::matmul_quant_t` —
  // kept separate to keep the fused-MoE public API self-contained.
  struct down_weight_quant_t {
    const void *buff;
    data_type_t dt;
    std::vector<int64_t> dims;
    down_weight_quant_t() : buff(nullptr), dt(data_type_t::none), dims() {}
  };
  std::vector<down_weight_quant_t> down_scale;  // Per-expert weight scale for down_weight[i].
  std::vector<down_weight_quant_t> down_zp;     // Per-expert weight zero-point (asymmetric only).

  std::vector<void *> dst_down;           // Per-expert down_proj output [M, N_down[i]].
  std::vector<int> ldc_down;              // Leading dimension of dst_down per expert.
};
```

> Field order above matches the actual declaration in `zendnnl/src/lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp` — relevant only if you ever do brace / aggregate initialisation against this struct.  The recommended caller pattern is `grp_matmul_fused_moe_params fused;` followed by named-field assignment, which is order-independent.

All required vectors must have size `num_ops` (matching the main Op1 vectors).  `down_scale` / `down_zp` (the quant pair sandwiched between the weight-side metadata and `dst_down`) are **optional**: leave them empty for an un-quantized down_proj (legacy behaviour, fully backward compatible).

| Field | Description |
|-------|-------------|
| `down_weight` | Per-expert weight matrices for the down projection. |
| `N_down` | Per-expert output column count (typically all the same: `hidden_size`). |
| `ldb_down` | Leading dimension of each `down_weight[i]` (>= `N_down[i]`). |
| `bias_down` | Per-expert bias (each entry can be `nullptr` for no bias). |
| `bias_dt_down` | Bias data type for Op2 (`none` = no bias; must match actual bias buffer dtype). |
| `down_scale` | **Optional** per-expert weight scale tensor for `down_weight[i]` (the only Op2-specific quant artefact — every other quant knob is inherited from `params[i]`).  Each entry holds `{buff, dt, dims}`.  Empty vector ⇒ Op2 weight un-quantized.  When populated, must hold ≥ `num_ops` entries; size-validated against partial-vector hazards.  See [Op2 quantization](#op2-quantization-optional) below. |
| `down_zp`    | **Optional** per-expert weight zero-point for `down_weight[i]` (asymmetric quant only).  Same shape as `down_scale`.  Leave empty (or each entry's `dims` empty) for symmetric quant.  Size-validated independently of `down_scale`. |
| `dst_down` | Per-expert output buffers for down_proj results. |
| `ldc_down` | Leading dimension of each `dst_down[i]` (>= `N_down[i]`). |

### Op2 quantization (optional)

The fused-MoE dispatcher enforces **one quant scheme for both passes**: every quantization *knob* on Op1 (`params[i].dynamic_quant`, `params[i].dtypes.compute`, `params[i].quant_params.src_scale.dims`, `params[i].dtypes.wei`, etc.) is inherited verbatim by Op2's internal `params_down[i]` inside `group_matmul_fused_moe.cpp`.  The ONE thing that must be carried separately is the weight scale (and optional zero-point) of `down_weight[i]`, because `down_weight[i]` is a different tensor from Op1's `weight[i]` and therefore has its own per-channel / per-group / per-tensor scale buffer.

> **Note**: dynamic *source* quant in the fused path supports **per-token (`{M, 1}`) only** — per-group `{M, ngroups}` with `ngroups > 1` is rejected with `status_t::failure` because Op1 and Op2 reduce over different K and Op1's `ngroups` cannot transfer to Op2 verbatim.  See "Per-token-only constraint on dynamic source quant" below.

That's the entire purpose of `fused.down_scale` and `fused.down_zp` — nothing more.  Behaviour matrix:

| Scheme on Op2 (inherited from Op1) | `params[i].dynamic_quant` | `params[i].dtypes.wei` | `fused.down_scale` | `fused.down_zp` | Notes |
|---|---|---|---|---|---|
| Un-quantized (default) | `false` | bf16 / f32 | empty | empty | Legacy behaviour — no source changes for existing callers. |
| WOQ-S4 | `false` | `s4` | populated, per-channel `{1, N_down}` or per-group `{G, N_down}` | usually empty (symmetric) | AOCL DLP's pure WOQ fast path (`is_woq` gate at `aocl_postop.cpp:178`). |
| WOQ-U4 (asymmetric) | `false` | `u4` | populated | populated | Same as S4 but asymmetric — `down_zp[i]` provides the zero-points. |
| **Dynamic INT8 (recommended for Op2 INT8)** | `true` | `s8` | populated, per-channel `{1, N_down}` | usually empty (symmetric s8) | Op2 inherits `dtypes.compute = s8` + `src_scale.dims` (typically per-token `{M[i], 1}`) from `params[i]`; kernel allocates the runtime BF16→S8 reorder scratch internally for the Op2 source (= activated Op1 intermediate). |

Note on pure WOQ-S8: AOCL DLP's WOQ fast path is gated to `s4 || u4` only — a `bf16 src + s8 wei` combination without a caller-provided `src_scale` falls into the BF16-INT8 pre-quant path and is rejected.  Prefer S4 for WOQ, or pair S8 weights with `dynamic_quant = true` on `params[i]` (the dynamic INT8 row above).

Cross-cutting constraints:

* The dispatcher resets `params_down[i].quant_params` per call from the new fields — a stale scale-buffer pointer from a previous call cannot leak into the persistent thread-local Op2 scratch.
* For quantized Op2, ALGO 3 (N-tile) is **not** selected for per-tensor `down_scale[i]` / `down_zp[i]` (`{}` / `{1}`); those cases still route the down_proj through ALGO 1 / 2 / 4 / 5.  The ALGO 3 quantized path is limited to the documented Dynamic INT8 configuration: `params[i].dynamic_quant=true` (carried over to `params_down`) with row-local source granularity (`{M, 1}` per-token) and per-channel weight scale (`down_scale.dims = {N}` / `down_zp.dims = {1, N}` when asymmetric).  Per-group weight quant (`{G, N}` with `G > 1`) likewise disables Op2 ALGO 3 via `check_n_tile_extra` and routes the down_proj through ALGO 1 / 2 / 4 / 5 instead.  In that supported Dynamic INT8 case, `flat_n_tile`'s pre-OMP `HoistedSrcQuant` hoist runs the BF16→S8 source reorder once and the per-tile threads share the resulting S8 src; the explicit `dynamic_quant` rejection in `check_n_tile_extra` was dropped alongside that hoist.
* When `params[i].dynamic_quant = true`, Op2's `src_scale.dims` is inherited from `params[i].quant_params.src_scale.dims`, but **only per-token (`{M, 1}`) granularity is allowed** — see below.

#### Per-token-only constraint on dynamic source quant

Op1 reduces over `K[i]` (= `K_in`); Op2 reduces over `K_down = op2_k_for_act(N[i], act)` (= `N[i]/2` for gated activations, `N[i]` for `act=none`).  The two K values are typically different, so a per-group `src_scale.dims = {M, ngroups_op1}` on `params[i]` does **not** transfer to Op2 verbatim — the documented contract (`docs/operator/low_overhead_operator/lowoha_matmul_operator.md:227`: *"the number of groups (G) must match between source and weight"*) is K-dependent and applies independently per pass.

The fused-MoE dispatcher therefore **rejects** any per-group dynamic source quant up front with a clear `log_error` and `status_t::failure`, before any kernel work is done.  The same check is applied to `src_zp.dims` when asymmetric source quant is enabled.

Allowed source-side `params[i].quant_params.src_scale.dims` when `dynamic_quant = true`:

| Dims | Granularity | Result |
|---|---|---|
| `{1}` or `{1, 1}` | per-tensor | Accepted; copied verbatim to Op2. |
| `{M, 1}` | **per-token** (recommended) | Accepted; copied verbatim to Op2 (K-independent). |
| `{M, ngroups}` with `ngroups > 1` | per-group | **Rejected** with `log_error` directing the caller to use per-token. |

If you need per-group activation quantization, do the matmul at the single-matmul layer (`group_matmul_direct` per pass) where `K` is one value and per-group dims are well-defined — the fused-MoE wrapper deliberately stays narrower than that to keep the same-scheme-on-both-passes contract clean.

The weight side is unaffected by this restriction.  `fused.down_scale[i]` / `fused.down_zp[i]` are caller-provided per-pass for Op2's `down_weight[i]` directly, so the WOQ-S4 / U4 row above can still use per-group weight scales (`{G_down, N_down}` where `G_down = K_down / group_size`) — only the *dynamic source* path is restricted.

### Constraints

| Constraint | Requirement |
|------------|-------------|
| `down_weight.size()` | `== num_ops` (all required fused vectors must match) |
| `N` | Must be even (`N = 2 * dim`) for all experts |
| `K_down` | `= N_gate_up / 2` (the `dim` after gated activation) |
| `dst_down[i]` | Must hold at least `[M[i], N_down[i]]` elements |
| `layout` | Must be row-major (`'r'` / `'R'`) for all experts |
| Mode | Parallel only (`src.size() > 1`) |
| `fused_moe` | Must be `nullptr` to disable; non-null requires all required vectors populated |
| `down_scale` | **Optional**.  Either empty (un-quantized Op2 weight) or size `>= num_ops`.  A non-empty undersized vector is rejected up front — a partial vector would silently leave the tail experts un-quantized and produce wrong numerics. |
| `down_zp`    | **Optional**.  Same size rule as `down_scale`; empty for symmetric quant (sym S4 / s8). |

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

### C++ fused MoE example — Op2 weight-only INT4 (WOQ-S4)

Builds on the example above by quantising `down_weight` to S4 with a per-channel F32 scale and plumbing the scale through `fused.down_scale`.  Op1 carries its own wei_scale on `params[i].quant_params.wei_scale` (same as a stand-alone WOQ-S4 matmul); only the down_weight scale needs the new field because that tensor is per-pass.

```cpp
// Per-expert s4 down_weights with attached per-channel f32 scale.
// (See test_quant.cpp::WOQ_BF16_S4 for the same pattern at the
// single-matmul level.)
std::vector<tensor_t> down_scale(num_ops), w2_s4(num_ops);
for (int i = 0; i < num_ops; ++i) {
  down_scale[i] = factory.uniform_dist_tensor({1, hidden_size},
                                            data_type_t::f32, 2.0);
  w2_s4[i]    = factory.uniform_dist_tensor({dim, hidden_size},
                                            data_type_t::s4, 7.0,
                                            /*transposed=*/false,
                                            down_scale[i]);
}

std::vector<const void *> down_weight_raw(num_ops);
for (int i = 0; i < num_ops; ++i)
  down_weight_raw[i] = w2_s4[i].get_raw_handle_unsafe();

fused.down_weight = down_weight_raw;       // s4-packed buffers
fused.N_down      = std::vector<int>(num_ops, hidden_size);
fused.ldb_down    = std::vector<int>(num_ops, hidden_size);
fused.bias_down   = std::vector<const void *>(num_ops, nullptr);

// Plumb only the Op2 weight scale through the new field — every
// other quant knob (dtypes.wei = s4, dynamic_quant = false, etc.)
// is inherited from `params[i]` by the fused-MoE setup loop.
fused.down_scale.resize(num_ops);
for (int i = 0; i < num_ops; ++i) {
  auto &q = fused.down_scale[i];
  q.buff = w2_s4[i].get_quant_scale_raw_handle_const();
  q.dt   = w2_s4[i].get_quant_scale_data_type();
  auto sz = w2_s4[i].get_quant_scale_size();
  q.dims.assign(sz.begin(), sz.end());
}
// (Symmetric S4 → leave fused.down_zp empty.  For asymmetric WOQ-U4
//  populate fused.down_zp[i] the same way using
//  `w2_u4[i].get_quant_zero_*()`.)

// `params[i].dtypes.wei = s4` and `params[i].quant_params.wei_scale`
// set the same way on Op1.  The dispatcher routes both passes
// through AOCL DLP's WOQ fast path (`is_woq = true`).
```

### C++ fused MoE example — full dynamic INT8 (Op1 + Op2)

Self-contained example showing **dynamic INT8 on BOTH Op1 and Op2** through a single `group_matmul_direct` call.

| Layer | Dtypes | Quant scheme | Carries scales via |
|---|---|---|---|
| Op1 (gate+up) | BF16 src + S8 wei → BF16 dst | dynamic INT8 (runtime BF16→S8 reorder of the source) | `params[i].dynamic_quant = true` + `params[i].quant_params.{src_scale, wei_scale}` |
| swiglu_oai_mul | BF16 in / BF16 out (gate+up halved to `dim`) | — | `gated_act` param |
| Op2 (down_proj) | BF16 src (activated Op1 dst) + S8 wei → BF16 dst | dynamic INT8 on Op2 (runtime BF16→S8 reorder of the intermediate) | Op2 inherits `dynamic_quant`, `dtypes.compute`, `src_scale.dims` from `params[i]`.  Only the down_weight scale is per-pass — carried via `fused.down_scale[i]` (and optional `fused.down_zp[i]`). |

Pre-conditions:
- `M[i] >= 16` per expert.  Per-token `{M, 1}` src_scale dims are rejected for very small M by the BF16-INT8 reorder kernel — see [test_quant.cpp::INT8_DYNAMIC_GEMM_BF16](../../../zendnnl/gtests/group_matmul/test_quant.cpp) for the same constraint at the single-matmul level.
- `K_in` (= `H`) and `N_gate_up` (= `2 * dim`) multiples of 4 for clean S8 K-blocking.
- All experts share the same dtype tuple (BF16 src, S8 wei, BF16 dst).

```cpp
#include "lowoha_operators/matmul/lowoha_matmul.hpp"
#include "examples/example_utils.hpp"   // tensor_factory_t, quant_params_compute

using namespace zendnnl::lowoha::matmul;
using zendnnl::common::data_type_t;

const int num_ops    = 4;          // experts
const int dim        = 32;         // Op1 cols halved (N_gate_up = 2*dim)
const int N_gate_up  = 2 * dim;    // Op1 cols
const int H          = 32;         // hidden_size (= K_in = N_down)
const int K_in       = H;
const int K_down     = dim;        // Op2 K (gated activation halves Op1 output)
const int M          = 16;         // tokens per expert (>= 16 for dynamic quant)

tensor_factory_t factory;

// ─────────────────────────────────────────────────────────────────────
// (1) Op1 source — BF16 with a per-token F32 src_scale attached.
//     The scale tensor is zero-allocated; the dispatcher fills it at
//     runtime when it reorders the BF16 source to S8.
// ─────────────────────────────────────────────────────────────────────
std::vector<tensor_t> src_t(num_ops), src_scale_t(num_ops);
for (int i = 0; i < num_ops; ++i) {
  src_scale_t[i] = factory.zero_tensor(
      {static_cast<uint64_t>(M), static_cast<uint64_t>(1)},  // per-token {M, 1}
      data_type_t::f32);
  src_t[i] = factory.uniform_dist_tensor(
      {static_cast<uint64_t>(M), static_cast<uint64_t>(K_in)},
      data_type_t::bf16, 2.0, /*transposed=*/false,
      src_scale_t[i], tensor_t{});
}

// ─────────────────────────────────────────────────────────────────────
// (2) Op1 + Op2 weights — pre-quantize a BF16 reference to S8 with a
//     per-channel F32 scale via `quant_params_compute`.
// ─────────────────────────────────────────────────────────────────────
std::vector<tensor_t> w1_s8(num_ops), w1_scale(num_ops), w1_zp(num_ops);
std::vector<tensor_t> w2_s8(num_ops), down_scale(num_ops), down_zp(num_ops);
for (int i = 0; i < num_ops; ++i) {
  // Op1 weight: [K_in, N_gate_up], per-channel scale {1, N_gate_up}.
  auto w1_ref = factory.uniform_dist_tensor(
      {static_cast<uint64_t>(K_in), static_cast<uint64_t>(N_gate_up)},
      data_type_t::bf16, 2.0);
  quant_params_compute(factory, w1_ref,
                       data_type_t::bf16, data_type_t::s8,
                       /*scale_dims=*/{1, N_gate_up},
                       data_type_t::f32,
                       w1_scale[i], w1_zp[i], &w1_s8[i]);

  // Op2 weight: [K_down, H], per-channel scale {1, H}.
  auto w2_ref = factory.uniform_dist_tensor(
      {static_cast<uint64_t>(K_down), static_cast<uint64_t>(H)},
      data_type_t::bf16, 2.0);
  quant_params_compute(factory, w2_ref,
                       data_type_t::bf16, data_type_t::s8,
                       /*scale_dims=*/{1, H},
                       data_type_t::f32,
                       down_scale[i], down_zp[i], &w2_s8[i]);
}

// ─────────────────────────────────────────────────────────────────────
// (3) Pull raw pointers + per-call vectors for the dispatcher.
// ─────────────────────────────────────────────────────────────────────
std::vector<const void *> srcs(num_ops), wei1_p(num_ops), wei2_p(num_ops);
for (int i = 0; i < num_ops; ++i) {
  srcs[i]   = src_t[i] .get_raw_handle_unsafe();   // bf16 source
  wei1_p[i] = w1_s8[i] .get_raw_handle_unsafe();   // s8 gate+up
  wei2_p[i] = w2_s8[i] .get_raw_handle_unsafe();   // s8 down_proj
}

std::vector<bfloat16_t> d1_dst_buf(num_ops * M * N_gate_up);
std::vector<bfloat16_t> d2_dst_buf(num_ops * M * H);
std::vector<void *> dst1_p(num_ops), dst2_p(num_ops);
for (int i = 0; i < num_ops; ++i) {
  dst1_p[i] = d1_dst_buf.data() + i * M * N_gate_up;
  dst2_p[i] = d2_dst_buf.data() + i * M * H;
}

const std::vector<char>  layout(num_ops, 'r');
const std::vector<bool>  transA(num_ops, false),  transB(num_ops, false);
const std::vector<int>   Ms(num_ops, M), Ns(num_ops, N_gate_up), Ks(num_ops, K_in);
const std::vector<float> alpha(num_ops, 1.0f),    beta(num_ops, 0.0f);
const std::vector<int>   lda(num_ops, K_in), ldb(num_ops, N_gate_up),
                         ldc(num_ops, N_gate_up);
const std::vector<bool>  is_wc(num_ops, true);    // WOQ requires const weights
const std::vector<const void *> no_bias(num_ops, nullptr);

// ─────────────────────────────────────────────────────────────────────
// (4) Op1 matmul_params — dynamic INT8 contract.
//     `dtypes.compute = s8` arms `reorder_quantization_wrapper`'s
//     BF16→S8 reorder; the kernel writes per-token scales into
//     `src_scale.buff` (which we attached in step (1)) at runtime.
// ─────────────────────────────────────────────────────────────────────
std::vector<matmul_params> params(num_ops);
for (int i = 0; i < num_ops; ++i) {
  auto &p = params[i];
  p.dtypes.src     = data_type_t::bf16;
  p.dtypes.wei     = data_type_t::s8;
  p.dtypes.dst     = data_type_t::bf16;
  p.dtypes.compute = data_type_t::s8;
  p.dynamic_quant  = true;
  // Pull src_scale.{buff, dt, dims} from the bf16 source tensor's
  // attached quant metadata (set by `uniform_dist_tensor(..., scale, …)`).
  p.quant_params.src_scale.buff = src_t[i].get_quant_scale_raw_handle_const();
  p.quant_params.src_scale.dt   = src_t[i].get_quant_scale_data_type();
  auto src_sz = src_t[i].get_quant_scale_size();
  p.quant_params.src_scale.dims.assign(src_sz.begin(), src_sz.end());
  // Pull wei_scale from the s8 weight tensor.
  p.quant_params.wei_scale.buff = w1_s8[i].get_quant_scale_raw_handle_const();
  p.quant_params.wei_scale.dt   = w1_s8[i].get_quant_scale_data_type();
  auto wei_sz = w1_s8[i].get_quant_scale_size();
  p.quant_params.wei_scale.dims.assign(wei_sz.begin(), wei_sz.end());
}

// ─────────────────────────────────────────────────────────────────────
// (5) Gated activation between Op1 and Op2 (swiglu_oai_mul halves the
//     N_gate_up columns down to `dim`, so K_down = dim for Op2).
// ─────────────────────────────────────────────────────────────────────
grp_matmul_gated_act_params act{};
act.act = grp_matmul_gated_act_t::swiglu_oai_mul;

// ─────────────────────────────────────────────────────────────────────
// (6) Op2 quant — only the down_weight scale is per-pass and needs
//     a dedicated carrier on `fused`.  Op2's `dynamic_quant`,
//     `dtypes.compute = s8`, and per-token `src_scale.dims = {M, 1}`
//     are inherited from `params[i]` by the fused-MoE setup loop
//     in `group_matmul_fused_moe.cpp` — same scheme on both passes
//     by construction.  Op2's source is the activated Op1 output
//     (BF16 intermediate), so the dispatcher leaves the inherited
//     `src_scale.buff = nullptr` and the kernel allocates the
//     per-token F32 reorder scratch internally.
// ─────────────────────────────────────────────────────────────────────
grp_matmul_fused_moe_params fused;
fused.down_weight = wei2_p;
fused.N_down      = std::vector<int>(num_ops, H);
fused.ldb_down    = std::vector<int>(num_ops, H);
fused.bias_down   = no_bias;
fused.dst_down    = dst2_p;
fused.ldc_down    = std::vector<int>(num_ops, H);

fused.down_scale.resize(num_ops);
for (int i = 0; i < num_ops; ++i) {
  auto &q = fused.down_scale[i];
  // Per-channel F32 wei_scale attached to the S8 down_weight tensor.
  q.buff = w2_s8[i].get_quant_scale_raw_handle_const();
  q.dt   = w2_s8[i].get_quant_scale_data_type();
  auto sz = w2_s8[i].get_quant_scale_size();
  q.dims.assign(sz.begin(), sz.end());
}
// (Symmetric S8 → leave fused.down_zp empty.)

// ─────────────────────────────────────────────────────────────────────
// (7) Single fused-MoE call.
// ─────────────────────────────────────────────────────────────────────
status_t st = group_matmul_direct(
    layout, transA, transB, Ms, Ns, Ks, alpha,
    srcs, lda, wei1_p, ldb, no_bias, beta,
    dst1_p, ldc, is_wc, params,
    /*moe_postop=*/nullptr,
    &act,        // swiglu_oai_mul between Op1 and Op2
    &fused);     // dynamic INT8 on Op2 via the new fields

// After the call:
//   * `dst1_p[i]` holds the activated [M, dim] intermediate
//     (first `dim` cols are the gated output; cols [dim, 2*dim) are
//     left-over raw matmul state — caller should not read those).
//   * `dst2_p[i]` (= fused.dst_down[i]) holds the final
//     [M, H] down_proj output per expert.
//   * `src_scale_t[i].get_raw_handle_unsafe()` holds the Op1 runtime
//     per-token scales the kernel computed during the BF16→S8 reorder
//     (useful for diagnostics / activation-aware calibration).
```

What the dispatcher actually does under the hood for this call:
1. Validates all per-vector sizes + the new `down_scale` / `down_zp` size guards in `group_matmul_fused_moe.cpp`.
2. **Pass 1** — runs `group_matmul_run_parallel_dispatch` for Op1.  Whether auto-select picks ALGO 3 (N-tile) or ALGO 1 (sequential_experts) depends on `num_ops` / `num_threads` / `max_M` (see [Auto-select decision](#strategy-selection)); both are correct because dynamic INT8 with row-local source granularity (`{M, 1}` per-token) is supported on both algos.  On ALGO 3, `flat_n_tile`'s pre-OMP hoist runs `reorder_quantization_wrapper` once per expert to produce an S8 src + per-token scale buffer; the per-tile threads then run the S8×S8 GEMM in parallel.  On ALGO 1, `execute_expert_slice` → `reorder_quantization_wrapper` does the same reorder inline per expert.  Activation is fused inline (`swiglu_oai_mul` on the BF16 epilogue) in both paths.
3. **Pass 2** — Op2 setup builds `scratch.params_down[i]` by **inheriting** `dynamic_quant`, `dtypes.compute`, and `quant_params.src_scale.{dt, dims}` from `params[i]`, then copying `fused.down_scale[i]` into `params_down[i].quant_params.wei_scale` (and `fused.down_zp[i]` into `wei_zp` if present).  `src_scale.buff` is left `nullptr` so `reorder_quantization_wrapper` allocates the per-call F32 scratch sized from the `{M, 1}` dims, reorders the activated BF16 intermediate to S8, then runs the S8×S8 down_proj.  Same dispatch path as Op1.
4. Returns `status_t::success`; `dst2_p` holds the final BF16 output.

Set `ZENDNNL_API_LOG_LEVEL=3` to see the dispatch trail; the per-call summary will read:

```
[GRP_MATMUL Level1] num_ops=4 mode=fused_moe_2pass(op1=sequential_experts,op2=sequential_experts)
                    threads=… dtype=bf16>s8>bf16 layout=r transA=N transB=N alpha[0]=1 beta[0]=0
                    wconst[0]=1 lda[0]=32 ldb[0]=64 ldc[0]=64 N[0]=64 K[0]=32 M=[16,16,16,16](sum=64)
                    fused=[act=swiglu_oai_mul,down_proj=N_down[0]=32] sequential_chain=0 …
```

## Notes and best practices

1. **Vector lengths**: By default all per-op vectors must have length `num_ops = M.size()`; `src` length selects the mode.  When the [framework prepack-extras contract](#framework-prepack-extras-contract) is engaged (`params[0].active_matmul > 0`), the dispatcher accepts weight-side vectors of size `>= active_matmul` while input-side vectors stay at `M.size()`.
2. **Chaining**: In sequential mode `K[i+1] == N[i]`.
3. **Parallel independence**: Each op uses its own `src[i]`, `weight[i]`, `dst[i]`.
4. **MoE**: Pass `nullptr` when not needed. When enabled, provide `row_ptrs` (built during scatter) and `topk_weights`.
5. **Gated activation**: Pass `nullptr` when not needed. When enabled, N must be even and dst dtype must be FP32 or BF16.  Applied after GEMM, before MoE weighted-reduce.
7. **Fused-MoE Op2 quantization**: Op2 (down_proj) uses the **same** quant scheme as Op1 by construction — the dispatcher inherits `dynamic_quant`, `dtypes.compute`, and `quant_params.src_scale.{dt, dims}` from `params[i]` into Op2's internal `params_down[i]`.  The only Op2-specific quant artefact is the down_weight scale (because `down_weight[i]` is a different tensor from Op1's `weight[i]`), carried via the new optional `fused.down_scale` and `fused.down_zp` vectors (see [Op2 quantization](#op2-quantization-optional)).  Both default to empty for backward compatibility.  Use WOQ-S4 or dynamic INT8 on Op2 — pure WOQ-S8 (BF16 src + S8 wei + only `wei_scale`) is rejected by AOCL DLP's `is_woq` gate (s4 / u4 only).
8. **Tiled algos (2/3)**: Both require row-major layout, uniform per-expert dtypes, and standard unpacked A/B. ALGO 2 (M-tile, `m_tile_safe`) additionally supports the full quantization stack — weight-only (S4 sym, U4 asym), static W8A8 (per-tensor / per-channel / per-group / per-token), and **dynamic INT8** (BF16/F32 source quantised at runtime; per-token `{M, 1}` and per-group `{M, G}` activation scales only) — and most post-ops. It blocks: packed B (GGML Q8_0), softmax/pooling, and dynamic-quant with non-row-local src granularity (per-tensor `{}` / `{1}` / `{1, 1}`, per-column `{1, K}`, per-channel-on-src `{1, N}`) because the per-thread reorder would race on the shared scale/zp buffer and use slice-local statistics. M-indexed source-quant metadata (`src_scale` / `src_zp`) is row-offset and dim-sliced per thread inside `m_tile/group_matmul_m_tile.cpp::offset_quant_by_row` so the dynamic-quant per-group reorder dispatch sees a slice-shaped `src_shape × dims`. ALGO 3 (N-tile, `n_tile_safe`) is stricter: only buffer-free element-wise post-ops (relu, gelu, swish, etc.) are safe under column slicing, and any non-null quant scale/zero-point buffer disables it. Falls back to ALGO 1 automatically when the required safety check fails.
