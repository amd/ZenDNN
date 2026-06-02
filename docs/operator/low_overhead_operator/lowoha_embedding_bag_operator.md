
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA EmbeddingBag Operator

## Overview

The **LowOHA EmbeddingBag Operator** is a high-performance, low-overhead embedding-lookup and aggregation operator designed for **latency-sensitive inference workloads** such as recommendation systems (DLRM), wide-and-deep models, and LLM token embeddings. It provides direct C API entry points to the AVX-512 / AVX-512-FP16 vectorized kernels with minimal per-call overhead.

Unlike the standard EmbeddingBag operator which uses the operator factory pattern (`embag_operator_t` + `embag_context_t`), LowOHA EmbeddingBag provides a **function-based interface** optimized for:
- Minimal execution overhead — no operator / context object lifecycle on the hot path
- FP32, BF16, FP16, INT8, and INT4 (S4 / U4 packed) embedding tables
- Sum, mean, and max aggregation, plus a pure-lookup mode
- Native AVX-512 and AVX-512-FP16 kernels with optional FBGEMM dispatch
- Quantized INT8 / INT4 embeddings with per-row scale + zero-point (FP16 or FP32)
- Group batching of multiple embedding-bag operations in a single call
- Direct control over execution parameters (data types, threads, padding index)


## EmbeddingBag Operation

Let:

- *T* ∈ ℝ<sup>R×D</sup> : Embedding table (R rows of D-dimensional embeddings)
- *I* ∈ ℤ<sup>N</sup>   : Indices array of length N into the table
- *O* ∈ ℤ<sup>B+1</sup> : Offsets array defining bag boundaries for B bags
- *W* ∈ ℝ<sup>N</sup>   : Optional per-index weights (sum-only)
- *P* ∈ ℤ              : Optional padding index ignored during aggregation
- *Algo*               : Aggregation algorithm (`sum`, `mean`, `max`, or `none`)
- *E* ∈ ℝ<sup>B×D</sup> : Output embeddings (one row per bag)
- *Scale*, *ZeroPoint* : Per-row quantization parameters (INT8 / INT4 tables)

For each bag `b` covering indices `[O[b], O[b+1])`:

$$
E[b, :] =
\begin{cases}
\sum_{i} W[i] \cdot T[I[i], :] & \text{Algo = sum} \\
\frac{1}{|valid_i|} \sum_{i} T[I[i], :] & \text{Algo = mean} \\
\max_i \, T[I[i], :] & \text{Algo = max}
\end{cases}
\quad \text{where } I[i] \neq P
$$

For quantized tables, the lookup row is first dequantized:

$$
T[I[i], :] = \mathrm{Scale\_row} \times (\mathrm{raw\_row} - \mathrm{ZeroPoint\_row})
$$

When `Algo = none`, the operator performs a plain **embedding lookup** — one output row per index, no offsets, no reduction.

### Operation Flow

```text
Embedding Table T [R x D]    Indices I [N]   Offsets O [B+1]   Weights W [N]
        |                       |                  |           (optional)
        |                       |                  |                |
        +-----------------------+------------------+                |
                    |           |             |                     |
                    |       Index Lookup      |                     |
                    |           |             |                     |
                    +-----------v-------------+                     |
                    |      Embedding[i]       |                     |
                    +-----------v-------------+                     |
                                |                                   |
                    +-----------v-------------+                     |
                    |    Apply Weight         |<--------------------+
                    +-----------v-------------+
                                |
                    +-----------v-------------+
                    |    Filter Padding       |
                    +-----------v-------------+
                                |
                    +-----------v-------------+
                    |     Aggregation         |  <- Sum / Mean / Max per bag
                    +-----------v-------------+
                                |
                        Output E [B x D]
```


## Core APIs

The LowOHA EmbeddingBag operator exposes three direct entry points under the `zendnnl::lowoha::embag` namespace.

### `embedding_bag_direct`

Primary API for embedding bag with reduction:

```cpp
status_t embedding_bag_direct(
  const void   *table,    // Embedding table [num_embeddings x embedding_dim]
  const void   *indices,  // Indices array (s32 or s64)
  const void   *offsets,  // Offsets array (s32 or s64); required when algo != none
  const float  *weights,  // Optional per-index weights (sum only; may be nullptr)
  void         *dst,      // Output buffer [num_bags x embedding_dim]
  embag_params_t params   // Embedding-bag parameters
);
```

### `embedding_direct`

Simplified API for pure embedding lookup (no offsets, no reduction). Internally sets `algo = embag_algo_t::none` and calls `embedding_bag_direct`:

```cpp
status_t embedding_direct(
  const void   *table,    // Embedding table [num_embeddings x embedding_dim]
  const void   *indices,  // Indices array (s32 or s64)
  const float  *weights,  // Optional per-index weights (may be nullptr)
  void         *dst,      // Output buffer [num_indices x embedding_dim]
  embag_params_t params   // Parameters (algo is forced to none internally)
);
```

### `group_embedding_bag_direct`

Batches multiple independent embedding-bag operations into a single call. Useful in DLRM-style sparse-feature lookups where dozens of tables are queried per inference:

```cpp
status_t group_embedding_bag_direct(
  const std::vector<const void*>     &tables,
  const std::vector<const void*>     &indices,
  const std::vector<const void*>     &offsets,
  const std::vector<const float*>    &weights,
  const std::vector<void*>           &dsts,
  const std::vector<embag_params_t>  &params
);
```

All vectors must have the same length; element `i` describes the `i`-th embedding-bag operation.

### Return Value

| Value                          | Description |
|--------------------------------|-------------|
| `status_t::success`            | Operation completed successfully |
| `status_t::failure`            | Validation or runtime error (check logs for details) |
| `status_t::isa_unsupported`    | An F16 buffer was requested but the host lacks AVX-512-FP16 |


## Parameters Structure

### `embag_params_t`

The main configuration structure for LowOHA EmbeddingBag:

```cpp
struct embag_params_t {
  embag_data_types_t dtypes;      // Data types for each operand
  embag_algo_t       algo;        // Reduction algorithm (sum / mean / max / none)

  uint64_t num_embeddings;        // R: number of rows in the embedding table
  uint64_t embedding_dim;         // D: dimension of each embedding vector
  uint64_t num_indices;           // N: total number of indices across all bags
  uint64_t num_bags;              // B: number of output rows (bags)

  bool     is_weights;            // True if per-index weights are provided (sum only)
  bool     include_last_offset;   // True if offsets has length B+1 (PyTorch style)
  int64_t  padding_idx;           // Index to ignore (-1 = no padding)
  bool     fp16_scale_bias;       // True: scale/ZP in FP16; False: FP32 (INT8/INT4)
  uint64_t dst_stride;            // Output row stride (0 = embedding_dim)

  int32_t  num_threads;           // 0 = auto (OMP_NUM_THREADS / system default)
  embag_kernel_t kernel;          // Backend kernel selection (none = auto)
};
```

### `embag_data_types_t`

```cpp
struct embag_data_types_t {
  data_type_t table;      // Table dtype:    f32 / bf16 / f16 / s8 / s4 / u4
  data_type_t output;     // Output dtype:   f32 / bf16 / f16
  data_type_t indices;    // s32 or s64    (default s64)
  data_type_t offsets;    // s32 or s64    (default s64)
  data_type_t scale;      // f32 or f16    (INT8 / INT4 only; default f32)
  data_type_t bias;       // f32 or f16    (INT8 / INT4 only; default f32)
};
```

### Aggregation Algorithms

```cpp
enum class embag_algo_t {
  none = -1,   // Pure embedding lookup (no reduction)
  sum  =  0,   // Σ embeddings (only mode that supports weights)
  mean =  1,   // (1 / valid_indices) × Σ embeddings
  max  =  2    // Elementwise max across embeddings in each bag
};
```

### Supported Data Type Combinations

| Table dtype  | Indices / Offsets dtype | Weights dtype     | Output dtype      | Aggregation     |
|--------------|-------------------------|-------------------|-------------------|-----------------|
| FP32         | s32 / s64               | FP32 (sum only)   | FP32, BF16, FP16  | Sum, Mean, Max  |
| BF16         | s32 / s64               | FP32 (sum only)   | FP32, BF16        | Sum, Mean, Max  |
| FP16         | s32 / s64               | FP32 (sum only)   | FP32, FP16        | Sum, Mean, Max  |
| INT8 (s8)    | s32 / s64               | FP32 (sum only)   | FP32, BF16, FP16  | Sum, Mean, Max  |
| INT4 (s4/u4) | s32 / s64               | FP32 (sum only)   | FP32, BF16, FP16  | Sum, Mean, Max  |

**Notes:**
- **Quantized tables (INT8 / INT4):** Per-row `scale` and `zero_point` are stored appended after the row bytes. For INT4 (s4 / u4), two 4-bit values are packed per byte. The scale/ZP element type is controlled by `fp16_scale_bias`.
- **FP16 tables/output:** Require AVX-512-FP16. On unsupported hardware the API returns `status_t::isa_unsupported`.

### Indices and Offsets

- **Indices** (`s32` or `s64`): A flat array of length `num_indices`. Each value must be in `[0, num_embeddings)` (or equal to `padding_idx` to be skipped).
- **Offsets** (`s32` or `s64`): Bag boundaries. By default the array has length `num_bags`, with bag `b` covering indices `[offsets[b], offsets[b+1])` and the last bag implicitly ending at `num_indices`. Set `include_last_offset = true` for the PyTorch-style layout where `offsets` has length `num_bags + 1` and the final offset equals `num_indices` explicitly.

### Padding Index

Set `params.padding_idx` to a non-negative value to skip any index equal to that value — it is excluded from the aggregation **and** from the mean denominator. This is the standard way to handle variable-length sequences (a common convention is to pad with index `0`).

### Kernel Selection

```cpp
enum class embag_kernel_t {
  none             = -1,   // Auto-select (recommended)
  dynamic_dispatch =  0,   // Reserved (falls back to fbgemm)
  auto_tuner       =  1,   // Reserved (falls back to fbgemm)
  native           =  2,   // Native AVX-512 / AVX-512-FP16 kernels
  fbgemm           =  3,   // FBGEMM kernel (default for FP32 / BF16)
  reference        =  4    // Scalar reference (debugging / portability)
};
```

Leave `params.kernel = embag_kernel_t::none` for auto-selection (`fbgemm` for FP32/BF16, native AVX-512 for FP16/INT4/INT8). Override at runtime with the environment variable `ZENDNNL_EMBAG_ALGO`.


## Usage Examples

### Example 1: FP32 Sum Aggregation

The simplest LowOHA path — float32 table, integer indices, sum reduction.

```cpp
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

int embedding_bag_fp32_sum_example() {
  using namespace zendnnl::lowoha::embag;

  constexpr uint64_t R = 1000;   // num_embeddings
  constexpr uint64_t D = 128;    // embedding_dim
  constexpr uint64_t N = 50;     // total indices
  constexpr uint64_t B = 10;     // num_bags

  std::vector<float>   table(R * D, 1.0f);
  std::vector<int64_t> indices(N);
  std::vector<int64_t> offsets(B);
  std::vector<float>   output(B * D, 0.0f);

  for (uint64_t i = 0; i < N; ++i) indices[i] = i % R;
  for (uint64_t b = 0; b < B; ++b) offsets[b] = b * (N / B);

  // Configure parameters
  embag_params_t params;
  params.dtypes.table   = data_type_t::f32;
  params.dtypes.output  = data_type_t::f32;
  params.dtypes.indices = data_type_t::s64;
  params.dtypes.offsets = data_type_t::s64;
  params.algo           = embag_algo_t::sum;
  params.num_embeddings = R;
  params.embedding_dim  = D;
  params.num_indices    = N;
  params.num_bags       = B;
  params.padding_idx    = -1;  // no padding

  // Execute
  status_t status = embedding_bag_direct(
      table.data(), indices.data(), offsets.data(),
      /*weights=*/nullptr, output.data(), params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 2: BF16 Mean Aggregation

```cpp
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

int embedding_bag_bf16_mean_example() {
  using namespace zendnnl::lowoha::embag;

  constexpr uint64_t R = 2048, D = 256, N = 100, B = 16;

  // BF16 buffers stored as uint16_t
  std::vector<uint16_t> table(R * D);
  std::vector<int32_t>  indices(N);
  std::vector<int32_t>  offsets(B);
  std::vector<uint16_t> output(B * D, 0);

  // Initialize input...

  embag_params_t params;
  params.dtypes.table   = data_type_t::bf16;
  params.dtypes.output  = data_type_t::bf16;
  params.dtypes.indices = data_type_t::s32;
  params.dtypes.offsets = data_type_t::s32;
  params.algo           = embag_algo_t::mean;
  params.num_embeddings = R;
  params.embedding_dim  = D;
  params.num_indices    = N;
  params.num_bags       = B;

  status_t status = embedding_bag_direct(
      table.data(), indices.data(), offsets.data(),
      /*weights=*/nullptr, output.data(), params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 3: FP16 Sum Aggregation (Native AVX-512-FP16)

FP16 embeddings keep memory bandwidth low and, on AVX-512-FP16-capable hosts, accumulate natively in `__m512h` registers for ~2× throughput vs the FP32 fallback.

```cpp
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

int embedding_bag_fp16_sum_example() {
  using namespace zendnnl::lowoha::embag;

  constexpr uint64_t R = 1024, D = 128, N = 64, B = 8;

  // FP16 stored as uint16_t
  std::vector<uint16_t> table(R * D);
  std::vector<int64_t>  indices(N);
  std::vector<int64_t>  offsets(B);
  std::vector<uint16_t> output(B * D, 0);

  // Initialize input...

  embag_params_t params;
  params.dtypes.table   = data_type_t::f16;
  params.dtypes.output  = data_type_t::f16;
  params.dtypes.indices = data_type_t::s64;
  params.dtypes.offsets = data_type_t::s64;
  params.algo           = embag_algo_t::sum;
  params.num_embeddings = R;
  params.embedding_dim  = D;
  params.num_indices    = N;
  params.num_bags       = B;

  status_t status = embedding_bag_direct(
      table.data(), indices.data(), offsets.data(),
      /*weights=*/nullptr, output.data(), params);

  if (status == status_t::isa_unsupported) {
    // Host lacks AVX-512-FP16 -- skip or fall back to FP32
    return 0;
  }
  return (status == status_t::success) ? 0 : -1;
}
```

### Example 4: Quantized U4 Sum Aggregation (FP16 Scale/ZP)

Packed 4-bit embeddings with per-row FP16 scale and zero-point — common in LLM token-embedding tables to reduce memory by ~8× vs FP32.

```cpp
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

int embedding_bag_u4_sum_example() {
  using namespace zendnnl::lowoha::embag;

  constexpr uint64_t R = 100, D = 64, N = 10, B = 5;

  // Each row layout: D/2 packed nibble bytes + FP16 scale + FP16 zero-point
  constexpr size_t row_bytes = (D / 2) + 2 * sizeof(uint16_t);
  std::vector<uint8_t> table(R * row_bytes);

  std::vector<int64_t> indices(N);
  std::vector<int64_t> offsets(B);
  std::vector<float>   output(B * D, 0.0f);

  // Initialize table (packed values + per-row FP16 scale and zero-point)...

  embag_params_t params;
  params.dtypes.table    = data_type_t::u4;
  params.dtypes.output   = data_type_t::f32;
  params.dtypes.indices  = data_type_t::s64;
  params.dtypes.offsets  = data_type_t::s64;
  params.dtypes.scale    = data_type_t::f16;
  params.dtypes.bias     = data_type_t::f16;
  params.algo            = embag_algo_t::sum;
  params.num_embeddings  = R;
  params.embedding_dim   = D;
  params.num_indices     = N;
  params.num_bags        = B;
  params.fp16_scale_bias = true;

  status_t status = embedding_bag_direct(
      table.data(), indices.data(), offsets.data(),
      /*weights=*/nullptr, output.data(), params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 5: Quantized INT8 with FP16 Output

```cpp
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

int embedding_bag_s8_f16_example() {
  using namespace zendnnl::lowoha::embag;

  constexpr uint64_t R = 1024, D = 128, N = 64, B = 8;

  // Each row layout: D INT8 bytes + FP16 scale + FP16 zero-point
  constexpr size_t row_bytes = D + 2 * sizeof(uint16_t);
  std::vector<uint8_t> table(R * row_bytes);

  std::vector<int64_t>  indices(N);
  std::vector<int64_t>  offsets(B);
  std::vector<uint16_t> output(B * D, 0);   // FP16 output

  embag_params_t params;
  params.dtypes.table    = data_type_t::s8;
  params.dtypes.output   = data_type_t::f16;
  params.dtypes.indices  = data_type_t::s64;
  params.dtypes.offsets  = data_type_t::s64;
  params.dtypes.scale    = data_type_t::f16;
  params.dtypes.bias     = data_type_t::f16;
  params.algo            = embag_algo_t::sum;
  params.num_embeddings  = R;
  params.embedding_dim   = D;
  params.num_indices     = N;
  params.num_bags        = B;
  params.fp16_scale_bias = true;

  status_t status = embedding_bag_direct(
      table.data(), indices.data(), offsets.data(),
      /*weights=*/nullptr, output.data(), params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 6: Sum with Per-Index Weights and Padding

```cpp
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

int embedding_bag_weighted_with_padding_example() {
  using namespace zendnnl::lowoha::embag;

  constexpr uint64_t R = 500, D = 64, N = 40, B = 8;

  std::vector<float>   table(R * D, 1.0f);
  std::vector<int64_t> indices(N);
  std::vector<int64_t> offsets(B);
  std::vector<float>   weights(N, 0.5f);   // per-index weights (sum only)
  std::vector<float>   output(B * D, 0.0f);

  // Mark some indices as padding (will be skipped)
  for (uint64_t i = 0; i < N; ++i) indices[i] = i % R;
  indices[3] = 0;   // padding index

  embag_params_t params;
  params.dtypes.table   = data_type_t::f32;
  params.dtypes.output  = data_type_t::f32;
  params.dtypes.indices = data_type_t::s64;
  params.dtypes.offsets = data_type_t::s64;
  params.algo           = embag_algo_t::sum;
  params.num_embeddings = R;
  params.embedding_dim  = D;
  params.num_indices    = N;
  params.num_bags       = B;
  params.is_weights     = true;
  params.padding_idx    = 0;   // skip indices equal to 0

  status_t status = embedding_bag_direct(
      table.data(), indices.data(), offsets.data(),
      weights.data(), output.data(), params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 7: Pure Embedding Lookup (no reduction)

Use `embedding_direct` when you need one output row per index (equivalent to `nn.Embedding` rather than `nn.EmbeddingBag`):

```cpp
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

int embedding_lookup_fp32_example() {
  using namespace zendnnl::lowoha::embag;

  constexpr uint64_t R = 1000, D = 128, N = 32;

  std::vector<float>   table(R * D, 1.0f);
  std::vector<int64_t> indices(N);
  std::vector<float>   output(N * D, 0.0f);

  embag_params_t params;
  params.dtypes.table   = data_type_t::f32;
  params.dtypes.output  = data_type_t::f32;
  params.dtypes.indices = data_type_t::s64;
  params.num_embeddings = R;
  params.embedding_dim  = D;
  params.num_indices    = N;
  // params.algo is forced to embag_algo_t::none inside embedding_direct

  status_t status = embedding_direct(
      table.data(), indices.data(),
      /*weights=*/nullptr, output.data(), params);

  return (status == status_t::success) ? 0 : -1;
}
```

### Example 8: Group Embedding Bag (Batched DLRM-Style Lookups)

Recommendation models query many embedding tables per sample. `group_embedding_bag_direct` amortizes dispatch and threading overhead across all tables in a single call:

```cpp
#include "lowoha_operators/embedding_bag/lowoha_embedding_bag.hpp"

int group_embedding_bag_example() {
  using namespace zendnnl::lowoha::embag;

  constexpr size_t NUM_TABLES = 3;
  const std::vector<size_t> R = {100, 150, 80};
  const std::vector<size_t> D = {16, 32, 24};
  const std::vector<size_t> N = {10, 15, 8};
  const std::vector<size_t> B = {5, 5, 4};

  // Per-table buffers
  std::vector<std::vector<float>>   tables_data(NUM_TABLES);
  std::vector<std::vector<int64_t>> indices_data(NUM_TABLES);
  std::vector<std::vector<int64_t>> offsets_data(NUM_TABLES);
  std::vector<std::vector<float>>   outputs_data(NUM_TABLES);

  // Pointer vectors handed to the API
  std::vector<const void*>     tables(NUM_TABLES);
  std::vector<const void*>     indices(NUM_TABLES);
  std::vector<const void*>     offsets(NUM_TABLES);
  std::vector<const float*>    weights(NUM_TABLES, nullptr);
  std::vector<void*>           outputs(NUM_TABLES);
  std::vector<embag_params_t>  params(NUM_TABLES);

  for (size_t t = 0; t < NUM_TABLES; ++t) {
    tables_data[t].assign(R[t] * D[t], 0.1f * (t + 1));
    indices_data[t].resize(N[t]);
    offsets_data[t].resize(B[t]);
    outputs_data[t].assign(B[t] * D[t], 0.0f);

    tables[t]  = tables_data[t].data();
    indices[t] = indices_data[t].data();
    offsets[t] = offsets_data[t].data();
    outputs[t] = outputs_data[t].data();

    params[t].dtypes.table   = data_type_t::f32;
    params[t].dtypes.output  = data_type_t::f32;
    params[t].dtypes.indices = data_type_t::s64;
    params[t].dtypes.offsets = data_type_t::s64;
    params[t].algo           = embag_algo_t::sum;
    params[t].num_embeddings = R[t];
    params[t].embedding_dim  = D[t];
    params[t].num_indices    = N[t];
    params[t].num_bags       = B[t];
  }

  // Execute all embedding-bag operations in a single call
  status_t status = group_embedding_bag_direct(
      tables, indices, offsets, weights, outputs, params);

  return (status == status_t::success) ? 0 : -1;
}
```


## FP16 Accumulation Modes

FP16 embedding bag requires **AVX-512-FP16** and **GCC >= 12**. On hardware without AVX-512-FP16, `embedding_bag_direct` returns `status_t::isa_unsupported`. By default the library uses the native F16 FMA kernel; a build-time toggle is provided to force FP32 accumulation for numerical reproducibility.

| Mode                                          | Compiler  | ISA          | Accumulation       | Kernel                          | Throughput        |
|-----------------------------------------------|-----------|--------------|--------------------|---------------------------------|-------------------|
| **F16 FMA** (default)                         | GCC >= 12 | AVX-512-FP16 | FP16 (`__m512h`)   | `embag_avx512_f16_fma_kernel`   | 32 elements / ZMM |
| **F32 FMA** (`-DZENDNNL_NATIVE_F32_ACCUM=ON`) | GCC >= 12 | AVX-512F     | FP32 (`__m512`)    | `embag_avx512_kernel`           | 16 elements / ZMM |

- **F16 FMA mode** keeps the entire reduction loop in `__m512h`, performing FMA / max / division natively in FP16. Conversions to/from FP32 happen only at the load/store boundary.
- **F32 FMA mode** widens FP16 inputs to FP32 on load, accumulates in `__m512`, and narrows back to FP16 on store. Slower but bit-reproducible against the FP32 reference path. The CMake flag is library-wide and also covers the LowOHA normalization operator's F16 FMA fast path.

### Quantized INT8 / INT4 with FP16 Output

The same two-path mechanism applies when quantized tables (s8, s4, u4) produce FP16 output:

| Path             | Kernel                                  | Required ISA                                    |
|------------------|-----------------------------------------|-------------------------------------------------|
| F16 FMA          | `embag_avx512_int8_int4_f16_fma_kernel` | AVX-512F + AVX-512BW + AVX-512-FP16             |
| F32 FMA fallback | `embag_avx512_int8_int4_kernel`         | AVX-512F + AVX-512_BF16 + F16C                  |

On AMD hardware, every Zen 4+ CPU supports AVX-512F + AVX-512_BF16, so the FP32 fallback runs anywhere AVX-512F is available.

> **Precision note:** F16 FMA mode accumulates in half-precision; intermediate results may differ slightly from the F32 fallback due to FP16 rounding at every accumulation step. For reproducibility studies, build with `-DZENDNNL_NATIVE_F32_ACCUM=ON`.


## Performance Considerations

- **Algorithm Selection:** Leave `params.kernel = embag_kernel_t::none` for auto-selection (`fbgemm` for FP32/BF16; native AVX-512 for FP16/INT4/INT8). Override at runtime with `ZENDNNL_EMBAG_ALGO`.
- **Threading:** `params.num_threads` controls intra-op parallelism (`0` = auto, honors `OMP_NUM_THREADS`). The kernel splits work across **table rows**, **bags**, or **CCDs** depending on the selected thread algorithm (overridable via `ZENDNNL_EMBAG_THREAD_ALGO`).
- **Memory Layout:** Embedding tables are **contiguous row-major** `[R, D]`. For quantized INT8 / INT4 tables, each row layout is: `D` data values, then `scale`, then `zero_point` (FP16 or FP32 each, controlled by `fp16_scale_bias`).
- **Output Stride:** `params.dst_stride` defaults to `embedding_dim`; override only when writing into a wider, padded buffer.
- **Batching Strategy:**

  | Pattern                              | Recommended API               |
  |--------------------------------------|-------------------------------|
  | Single table, multiple bags          | `embedding_bag_direct`        |
  | Single table, one row per index      | `embedding_direct`            |
  | Many tables (DLRM, wide-and-deep)    | `group_embedding_bag_direct`  |

  The group API parallelizes across tables, so dispatch and thread-pool setup costs are paid **once** rather than per table.


## Integration Workflow

1. **Allocate** the embedding table (and per-row scale / zero-point for INT8 / INT4).
2. **Populate** `embag_params_t` with dimensions, dtypes, algorithm, and padding index.
3. **Build** the indices and offsets arrays (`s32` or `s64`). For PyTorch-style offsets that include the final endpoint, set `include_last_offset = true`.
4. **Call** `embedding_bag_direct` (or `embedding_direct` / `group_embedding_bag_direct`).
5. **Check** the returned `status_t`. Treat `status_t::isa_unsupported` separately from `failure` so FP16 paths can fall back gracefully on hosts without AVX-512-FP16.

```cpp
using namespace zendnnl::lowoha::embag;

embag_params_t params;                                   // 1. configure
params.dtypes.table   = data_type_t::f32;
params.dtypes.output  = data_type_t::f32;
params.algo           = embag_algo_t::sum;
params.num_embeddings = R;
params.embedding_dim  = D;
params.num_indices    = N;
params.num_bags       = B;

status_t s = embedding_bag_direct(                       // 2. invoke
    table, indices, offsets, weights, output, params);

if (s == status_t::isa_unsupported) { /* skip / fall back */ }
else if (s != status_t::success)    { /* handle error   */ }
```


## Validation

The operator performs the following validations (always-on, regardless of `ZENDNNL_DIAGNOSTICS_ENABLE`):

1. **Null pointer checks:** `table`, `indices`, and `dst` must not be null.
2. **Dimension checks:** `num_embeddings > 0`, `embedding_dim > 0`, `num_indices > 0`.
3. **Bag count:** `num_bags > 0` when `algo != embag_algo_t::none`.
4. **Data types:** `dtypes.table` and `dtypes.output` must be specified (not `data_type_t::none`).
5. **Offsets:** `offsets != nullptr` when `algo != embag_algo_t::none`.
6. **F16 ISA:** When either `dtypes.table` or `dtypes.output` is `f16`, the host must expose AVX-512-FP16 (returns `status_t::isa_unsupported` otherwise).

## Diagnostics and Profiling

- **Input validation** runs by default. Toggle with `ZENDNNL_DIAGNOSTICS_ENABLE` (defaults to `1`); set to `0` to skip optional validation on production hot paths.
- **Profiling** is controlled by `ZENDNNL_ENABLE_PROFILER=1` and `ZENDNNL_PROFILE_LOG_LEVEL=4`. When active, `embedding_bag_direct` logs execution time and parameters via `apilog_info` / `profilelog_verbose`.
- **Kernel / threading overrides:**
  - `ZENDNNL_EMBAG_ALGO` — overrides `params.kernel` (kernel selection).
  - `ZENDNNL_EMBAG_THREAD_ALGO` — overrides the internal thread-splitting strategy (`table_threaded`, `batch_threaded`, `ccd_threaded`, `hybrid_threaded`).
- **F16-FMA build-time toggle:** Build with `-DZENDNNL_NATIVE_F32_ACCUM=ON` to force the FP32-accumulating AVX-512 kernel for FP16 / quantized-FP16-output paths. Useful for A/B precision studies and reproducibility. The flag is library-wide and also covers the LowOHA normalization operator.
