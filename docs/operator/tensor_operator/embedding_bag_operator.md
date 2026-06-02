
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# EmbeddingBag Operator

## Overview

This section provides a high-level overview of embedding bag (`embag`) operations using the operator-factory pattern (`embag_context_t` + `embag_operator_t`). The operator supports FP32, BF16, FP16, INT8, and INT4 (S4/U4) embedding tables and three aggregation algorithms (sum, mean, max). The support matrix summarizes valid combinations of table, indices, offsets, weights, and output data types. Practical examples from `embedding_bag_example.cpp` demonstrate these configurations, such as `embedding_bag_f32_kernel_example`, `embedding_bag_f16_kernel_example`, and `embedding_bag_u4_kernel_example`.

> For the **function-based low-overhead direct API** (`embedding_bag_direct`, `embedding_direct`, `group_embedding_bag_direct`), see the [LOWOHA EmbeddingBag Operator](../low_overhead_operator/lowoha_embedding_bag_operator.md) documentation.

## General EmbeddingBag Operation

Let:

- *T* ∈ ℝᴿˣᴰ : Embedding table with `R` rows and embedding dimension `D`
- *I* ∈ ℤᴺ : Indices array of length `N` pointing to embedding table rows
- *O* ∈ ℤᴮ⁺¹ : Offsets array defining bag boundaries for `B` bags
- *W* ∈ ℝᴺ : Optional weights array (same length as indices, sum-only)
- *P* ∈ ℤ : Optional padding index to ignore during aggregation
- *Algo* : Aggregation algorithm (`sum`, `mean`, `max`)
- *E* ∈ ℝᴮˣᴰ : Output embeddings for `B` bags after aggregation
- *Scale* : Per-row scaling factor for quantized tables (INT8/INT4)
- *ZeroPoint* : Per-row zero-point offset for quantized tables (INT8/INT4)

## Steps to Perform EmbeddingBag Operation

1. **Index Lookup**:
   For each bag `b` and each index `i` in range `[O[b], O[b+1])`:
   ```
   embedding_i = T[I[i], :]
   ```
   For quantized embedding tables:
   ```
   embedding_i = Scale_row × (T[I[i], :] - ZeroPoint_row)
   ```

2. **Weight Application (optional)**:
   ```
   weighted_embedding_i = W[i] × embedding_i
   ```

3. **Padding Filtering**:
   Skip indices where `I[i] == P` (padding index).

4. **Aggregation**:
   Apply the specified algorithm:
   - **Sum**: `E[b, :] = Σ embedding_i`
   - **Mean**: `E[b, :] = (1 / |valid_indices|) × Σ embedding_i`
   - **Max**: `E[b, :] = max(embedding_i)`

5. **Store Result**:
   ```
   Output[b, :] = E[b, :]
   ```

### Example with Sum Aggregation and Weights

```
E[b, :] = Σ(i = O[b] to O[b+1] - 1) W[i] × T[I[i], :]    where I[i] ≠ P
```

### EmbeddingBag Operation Flow Diagram

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

## EmbeddingBag Support Table

This table outlines the supported configurations across data types, indices/offsets types, weights, and aggregation algorithms.

| Table<br>Data Type | Indices / Offsets<br>Data Type | Weights<br>Data Type | Output<br>Data Type | Aggregation<br>Algorithm |
|--------------------|--------------------------------|----------------------|---------------------|--------------------------|
| FP32               | INT32 / INT64                  | FP32 (Only Sum)      | FP32                | Sum, Mean, Max           |
| FP32               | INT32 / INT64                  | FP32 (Only Sum)      | BF16                | Sum, Mean, Max           |
| FP32               | INT32 / INT64                  | FP32 (Only Sum)      | FP16                | Sum, Mean, Max           |
| BF16               | INT32 / INT64                  | FP32 (Only Sum)      | FP32                | Sum, Mean, Max           |
| BF16               | INT32 / INT64                  | FP32 (Only Sum)      | BF16                | Sum, Mean, Max           |
| FP16               | INT32 / INT64                  | FP32 (Only Sum)      | FP32                | Sum, Mean, Max           |
| FP16               | INT32 / INT64                  | FP32 (Only Sum)      | FP16                | Sum, Mean, Max           |
| INT8 (s8)          | INT32 / INT64                  | FP32 (Only Sum)      | FP32, BF16, FP16    | Sum, Mean, Max           |
| INT4 (s4/u4)       | INT32 / INT64                  | FP32 (Only Sum)      | FP32, BF16, FP16    | Sum, Mean, Max           |

**Notes:**
- **Quantized tables (INT8/INT4):** Per-row `scale` and `zero_point` are appended to each table row. `INT4` packs two 4-bit values per byte. The scale/ZP element type is selected by `set_fp16_scale_bias(true|false)`.
- **FP16 tables/output:** Require AVX512-FP16. On unsupported hardware the operator returns `status_t::isa_unsupported`.

## Examples

### 1. embedding_bag_f32_kernel_example

This example performs embedding bag with `float32 (f32)` table and output, using `sum` aggregation.

**Key Components**

- **Embedding Table**
  - Uniform tensor with dimensions `{R, D}` in FP32
- **Indices and Offsets**
  - Indices: integer array (s64) over `[0, R)`
  - Offsets: integer array (s64) defining bag boundaries
- **Aggregation**
  - Applies `sum` aggregation across embeddings in each bag

```cpp
int embedding_bag_f32_kernel_example() {
  try {
    // Status variable to track success or failure of operations
    status_t status;

    // Factory object to create tensors with specified properties
    tensor_factory_t tensor_factory;

    // Create an embedding table with dimensions [R, D]
    // initialized with uniform value 1.0
    auto table = tensor_factory.uniform_tensor({EMB_ROW, EMB_DIM},
                                               data_type_t::f32,
                                               1.0, "table");

    // Create an embedding bag context with sum aggregation
    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table)
                                 .set_algo(embag_algo_t::sum)
                                 .create();

    // Check if the context was created successfully
    if (!embedding_bag_context.check()) {
      testlog_error("embedding bag context creation failed");
      return NOT_OK;
    }

    // Create an embedding bag operator using the defined context
    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("embedding_bag_f32")
                                  .set_context(embedding_bag_context)
                                  .create();

    // Check if the operator was created successfully
    if (embedding_bag_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    // Create indices tensor with random values within vocabulary range
    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s64, indices, "indices");

    // Create offsets tensor defining bag boundaries
    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s64, offsets, "offsets");

    // Create output tensor with dimensions [NUM_BAGS, EMBEDDING_DIM]
    auto output_tensor = tensor_factory.zero_tensor({EMB_BATCH_SIZE, EMB_DIM},
                         data_type_t::f32, "output");

    // Set the input and output tensors and execute the embag operator
    status = embedding_bag_operator
             .set_input("indices", indices_tensor)
             .set_input("offsets", offsets_tensor)
             .set_output("output", output_tensor)
             .execute();

    // Log the result of the execution
    if (status == status_t::success) {
      testlog_info("<", embedding_bag_operator.get_name(), ">",
                   " operator execution successful.");
    } else {
      testlog_error("<", embedding_bag_operator.get_name(), ">",
                    " operator execution failed.");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    // Catch and print any exceptions that occur during execution
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  // Return success status
  return OK;
}
```

### 2. embedding_bag_f16_kernel_example

This example performs embedding bag with `float16 (f16)` table and output, using `sum` aggregation. FP16 minimizes memory bandwidth and enables the native AVX512-FP16 `__m512h` FMA fast path on capable hosts.

**Key Components**

- **Embedding Table**
  - Uniform tensor with dimensions `{R, D}` in FP16
- **Indices and Offsets**
  - Indices: integer array (s64) over `[0, R)`
  - Offsets: integer array (s64) defining bag boundaries
- **Output**
  - FP16 output tensor — accumulation precision depends on the kernel selected at compile and runtime (see [FP16 Accumulation Modes](#fp16-accumulation-modes))
- **Aggregation**
  - Applies `sum` aggregation across embeddings in each bag

```cpp
int embedding_bag_f16_kernel_example() {
  try {
    status_t status;
    tensor_factory_t tensor_factory;

    // Create an FP16 embedding table with dimensions [R, D]
    auto table = tensor_factory.uniform_tensor({EMB_ROW, EMB_DIM},
                                               data_type_t::f16,
                                               1.0, "table");

    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table)
                                 .set_algo(embag_algo_t::sum)
                                 .create();

    if (!embedding_bag_context.check()) {
      testlog_error("embedding bag context creation failed");
      return NOT_OK;
    }

    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("embedding_bag_f16")
                                  .set_context(embedding_bag_context)
                                  .create();

    if (embedding_bag_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s64, indices, "indices");

    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s64, offsets, "offsets");

    // Create FP16 output tensor with dimensions [NUM_BAGS, EMBEDDING_DIM]
    auto output_tensor = tensor_factory.zero_tensor({EMB_BATCH_SIZE, EMB_DIM},
                         data_type_t::f16, "output");

    status = embedding_bag_operator
             .set_input("indices", indices_tensor)
             .set_input("offsets", offsets_tensor)
             .set_output("output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("<", embedding_bag_operator.get_name(), ">",
                   " operator execution successful.");
    } else {
      testlog_error("<", embedding_bag_operator.get_name(), ">",
                    " operator execution failed.");
      return NOT_OK;
    }

  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}
```

### 3. embedding_bag_u4_kernel_example

This example performs embedding bag with a `U4` quantized table and FP16 scale/zero-point, using `sum` aggregation. U4 packs two 4-bit unsigned values per byte and appends `scale` and `zero_point` to each table row.

**Key Components**

- **Embedding Table**
  - Quantized random tensor with dimensions `{R, D}`, values stored as packed U4
  - Per-row `scale` and `zero_point` appended to the table
- **Indices and Offsets**
  - Indices: integer array (s64) over `[0, R)`
  - Offsets: integer array (s64) defining bag boundaries
- **FP16 Scale / Zero-Point**
  - `set_fp16_scale_bias(true)` selects FP16 scale/ZP storage
- **Aggregation**
  - Applies `sum` aggregation across dequantized embeddings in each bag

```cpp
int embedding_bag_u4_kernel_example() {
  try {
    status_t status;
    tensor_factory_t tensor_factory;

    // Create a quantized U4 table; scale and zero_point are appended per row
    auto table = tensor_factory.quantized_embedding_tensor_random(
                   {EMB_ROW, EMB_DIM}, data_type_t::u4, "table", true);

    // Create context with sum aggregation and FP16 scale/zero-point
    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table)
                                 .set_algo(embag_algo_t::sum)
                                 .set_fp16_scale_bias(true)
                                 .create();

    if (!embedding_bag_context.check()) {
      testlog_error("embedding bag context creation failed");
      return NOT_OK;
    }

    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("embedding_bag_u4_operator")
                                  .set_context(embedding_bag_context)
                                  .create();

    if (embedding_bag_operator.is_bad_object()) {
      testlog_error(" operator ", embedding_bag_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                          data_type_t::s64, indices, "indices");

    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s64, offsets, "offsets");

    // FP32 output (BF16 / FP16 also supported)
    auto output_tensor = tensor_factory.zero_tensor({EMB_BATCH_SIZE, EMB_DIM},
                         data_type_t::f32, "output");

    status = embedding_bag_operator
             .set_input("indices", indices_tensor)
             .set_input("offsets", offsets_tensor)
             .set_output("output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("<", embedding_bag_operator.get_name(), ">",
                   " operator execution successful.");
    } else {
      testlog_error("<", embedding_bag_operator.get_name(), ">",
                    " operator execution failed.");
      return NOT_OK;
    }

    // Free the table buffer after use (quantized_embedding_tensor_random
    // allocates a raw buffer that the caller owns)
    free(table.get_raw_handle_unsafe());
    table.reset();
  } catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}
```

**Key Points for Quantization**

- **Packed weights:** `s4` (signed) / `u4` (unsigned) values are packed two-per-byte; `s8` is one-per-byte.
- **Per-row scale and zero-point:** Appended to each table row; element type is selected by `set_fp16_scale_bias(true|false)`.
- **Dequantization formula:** `dequant = scale × (raw_value − zero_point)`.

## FP16 Accumulation Modes

FP16 embedding bag requires **AVX512-FP16** and **GCC >= 12**. On hardware without AVX512-FP16, the operator returns `status_t::isa_unsupported`. By default the library uses the native F16 FMA kernel; a build-time toggle is provided to force FP32 accumulation for numerical reproducibility.

| Mode                                         | Compiler  | ISA          | Accumulation       | Kernel                          | Throughput        |
|----------------------------------------------|-----------|--------------|--------------------|---------------------------------|-------------------|
| **F16 FMA** (default)                        | GCC >= 12 | AVX512-FP16  | FP16 (`__m512h`)   | `embag_avx512_f16_fma_kernel`   | 32 elements / ZMM |
| **F32 FMA** (`-DZENDNNL_NATIVE_F32_ACCUM=ON`)| GCC >= 12 | AVX-512F     | FP32 (`__m512`)    | `embag_avx512_kernel`           | 16 elements / ZMM |

- **F16 FMA mode** keeps the entire reduction loop in `__m512h`. Conversions to/from FP32 happen only at load/store boundaries.
- **F32 FMA mode** widens FP16 inputs to FP32 on load, accumulates in `__m512`, and narrows back to FP16 on store. Slower but bit-reproducible against the FP32 reference path. The flag is library-wide and also covers the LOWOHA normalization operator's F16 FMA fast path.

The same two-path mechanism applies to quantized INT8/INT4 tables that produce FP16 output:

| Path             | Kernel                                  | Required ISA                                    |
|------------------|-----------------------------------------|-------------------------------------------------|
| F16 FMA          | `embag_avx512_int8_int4_f16_fma_kernel` | AVX-512F + AVX-512BW + AVX-512-FP16             |
| F32 FMA fallback | `embag_avx512_int8_int4_kernel`         | AVX-512F + AVX-512_BF16 + F16C                  |

## Parameter Naming Convention

**Important:** The string identifiers used in `.set_param()`, `.set_input()`, and `.set_output()` are fixed and must not be changed. These names are internally mapped and executed by the operator implementation.

Required identifiers:

- `.set_param("table", ...)` → must use `"table"`
- `.set_input("indices", ...)` → must use `"indices"`
- `.set_input("offsets", ...)` → must use `"offsets"`
- `.set_output("output", ...)` → must use `"output"`

Changing these names will result in incorrect behavior or operator failure.
