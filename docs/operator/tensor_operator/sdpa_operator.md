
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# SDPA Operator

## Overview

This section provides a high-level overview of the **Scaled Dot-Product Attention (SDPA) encoder operator**, the object-oriented (Tensor Operator API) counterpart to the LowOHA `sdpa_direct` flow. It computes multi-head attention over 4D `[B, H, S, D]` query, key, and value tensors using a portable **reference kernel** that supports FP32, BF16, and F16 data types.

Unlike the LowOHA flash backend, the SDPA encoder operator dispatches to a single **reference kernel** (`sdpa_encoder_ref_kernel_t`) that performs all arithmetic in FP32 internally, widening typed inputs to float at load time and narrowing the result at store time. Because it never uses a reduced-precision ISA, the reference kernel runs on **any CPU** — including F16 inputs on hardware without AVX512-FP16, which the flash backend would reject with `isa_unsupported`.

The reference kernel prioritizes **correctness and portability**: it is the baseline against which the optimized flash backend is validated, and the fallback for layouts or platforms the optimized path does not cover.

A practical example from `sdpa_example.cpp` (`sdpa_example`) demonstrates the operator end-to-end.

## SDPA Computation

The operator implements the standard attention computation:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{d_k}} + M\right) \cdot V
$$

Let:

- *Q* ∈ ℝ<sup>B×H×S<sub>q</sub>×D</sup> : Query tensor
- *K* ∈ ℝ<sup>B×H×S<sub>kv</sub>×D</sup> : Key tensor
- *V* ∈ ℝ<sup>B×H×S<sub>kv</sub>×D</sup> : Value tensor
- *M* : Optional additive attention mask (broadcastable 2-D or 4-D)
- *scale* : Scaling factor applied to *Q · K<sup>T</sup>* (typically 1/√*d<sub>k</sub>*)

For **self-attention** *S<sub>q</sub>* == *S<sub>kv</sub>*. For **cross-attention** (e.g. encoder-decoder models) they may differ; *K* and *V* must share the same *S<sub>kv</sub>*.

## Steps to Perform the SDPA Operation

For each `(batch, head)` slice, the reference kernel computes:

1. **Scaled scores (Q × Kᵀ)**:
   ```
   scores[i, j] = (Σ_k Q[i, k] * K[j, k]) * scale
   ```
   The score buffer is always FP32 for numerical stability.

2. **Causal masking (optional)**:
   ```
   scores[i, j] = -inf  for j > i
   ```
   Applied when `is_causal = true` (query position *i* attends only to key positions `[0..i]`).

3. **Additive mask (optional)**:
   ```
   scores[i, j] = scores[i, j] + mask[i, j]
   ```
   Each mask element is converted to FP32 before the add.

4. **Row-wise softmax** (numerically stable):
   ```
   scores[i, :] = exp(scores[i, :] - row_max) / row_sum
   ```

5. **Weighted values (scores × V)**:
   ```
   output[i, j] = Σ_k scores[i, k] * V[k, j]
   ```
   The FP32 result is narrowed back to the Q/K/V data type on store.

The `(batch, head)` slices are processed in parallel with `#pragma omp parallel for collapse(2)`.

### SDPA Operation Flow Diagram

```text
        Query [B,H,S_q,D]   Key [B,H,S_kv,D]   Value [B,H,S_kv,D]
              |                   |                    |
              +---------x---------+                    |
                        |                              |
                   Q x K^T * scale                     |
                        |                              |
                 +------v------+                       |
                 | Causal mask |  (optional)           |
                 +------v------+                       |
                 | Add mask    |  (optional)           |
                 +------v------+                       |
                 |  Softmax    |  (row-wise, FP32)     |
                 +------v------+                       |
                        +--------------x---------------+
                                       |
                                  scores x V
                                       |
                                       v
                              Output [B,H,S_q,D]
```

## Supported Configurations

| Q/K/V Data Type | Mask Data Type | Output Data Type | ISA Requirement |
|-----------------|----------------|------------------|-----------------|
| FP32 | FP32 / none | FP32 | Any |
| BF16 | FP32 / BF16 / none | BF16 | Any |
| F16  | FP32 / F16 / none  | F16  | Any (FP32 internal compute; no AVX512-FP16 required) |

**Notes:**
- **Internal precision is always FP32.** Typed inputs are widened to float at load and narrowed back at store, so softmax is numerically stable for every input dtype.
- **Q, K, V, and output must share the same data type.** The kernel is dispatched on Q's dtype and casts the other buffers to it; a mismatch is rejected by `validate()`.
- **Mask dtype rules:** FP32 QKV pairs only with an FP32 mask; BF16 QKV accepts an FP32 or BF16 mask; F16 QKV accepts an FP32 or F16 mask.
- Unsupported QKV dtypes are rejected with `status_t::unimplemented` from the kernel factory.

## Supported Layouts

The Q, K, V, and output tensors are **logically** 4D `[B, H, S, D]`; the **physical** layout is encoded entirely in the strides. Each tensor may independently use either of two layouts:

| Layout | Stride pattern `(s_B, s_H, s_S, s_D)` | Origin |
|--------|----------------------------------------|--------|
| **BHSD** (canonical contiguous) | `(H·S·D, S·D, D, 1)` | Standard `[B, H, S, D]` memory |
| **BSHD** (logical BHSD via transpose) | `(S·H·D, D, H·D, 1)` | PyTorch `[B, S, H, D]` after `.transpose(1, 2)` |

The head-dimension axis must be contiguous (`s_D == 1`) whenever `D > 1`. A stride of `0` is permitted on any size-1 dimension (broadcast convention).

## Attention Mask

The optional additive mask is applied before the softmax and supports two layouts:

| Mask rank | Shape | Broadcasting |
|-----------|-------|--------------|
| `2` | `[S_q, S_kv]` | Broadcast across batch and heads |
| `4` | `[B\|1, H\|1, S_q, S_kv]` | Size-1 leading dims broadcast |

The inner `[S_q, S_kv]` slab must be canonical row-major contiguous (kv stride = 1, q stride = `S_kv`); broadcast leading dims may carry stride `0`. The `set_has_mask(...)` flag and the presence of a `"mask"` parameter must agree, or `validate()` fails.

## Parameters, Inputs, and Outputs

| Kind | Name | Required | Description |
|------|------|----------|-------------|
| Parameter | `query` | Yes | Query tensor `[B, H, S_q, D]` |
| Parameter | `key` | Yes | Key tensor `[B, H, S_kv, D]` |
| Parameter | `value` | Yes | Value tensor `[B, H, S_kv, D]` |
| Parameter | `mask` | No | Additive attention mask (2-D or 4-D) |
| Output | `sdpa_output` | Yes | Output tensor `[B, H, S_q, D]`, same dtype as Q/K/V |

### Context configuration

| Setter | Description |
|--------|-------------|
| `.set_scale(float)` | Scaling factor applied to `Q · Kᵀ` (e.g. `1 / sqrt(head_dim)`) |
| `.set_is_causal(bool)` | Enable causal (upper-triangular) masking |
| `.set_has_mask(bool)` | Declare that a `"mask"` parameter is present |
| `.set_is_dropout(bool)` | Dropout flag (kept for interface parity) |

## Example

### sdpa_example

This example performs FP32 Scaled Dot-Product Attention using the `sdpa_encoder_operator_t`.

**Key Components**

- **Q / K / V Initialization**
  - Uniform tensors with dimensions `{BS, NUM_HEADS, SEQ_LEN, HEAD_DIM}`

- **Context**
  - Sets the `query`, `key`, `value` parameters, the attention `scale`, and the `is_causal` / `has_mask` flags.

- **Execution**
  - Creates the operator from the context, sets the output tensor, and executes.

```cpp
int sdpa_example() {
  try {
    tensor_factory_t tensor_factory;

    // Create Q, K, V tensors with dimensions [B, H, S, D]
    auto query_tensor = tensor_factory.uniform_tensor({BS, NUM_HEADS, SEQ_LEN, HEAD_DIM},
                        data_type_t::f32, 0.1f, "query");
    auto key_tensor = tensor_factory.uniform_tensor({BS, NUM_HEADS, SEQ_LEN, HEAD_DIM},
                      data_type_t::f32, 0.1f, "key");
    auto value_tensor = tensor_factory.uniform_tensor({BS, NUM_HEADS, SEQ_LEN, HEAD_DIM},
                        data_type_t::f32, 0.1f, "value");

    // Create output tensor
    auto output_tensor = tensor_factory.zero_tensor({BS, NUM_HEADS, SEQ_LEN, HEAD_DIM},
                         data_type_t::f32, "output");

    // Create SDPA encoder context with default parameters
    float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
    auto sdpa_context = sdpa_encoder_context_t()
                        .set_param("query", query_tensor)
                        .set_param("key", key_tensor)
                        .set_param("value", value_tensor)
                        .set_scale(scale)
                        .set_is_dropout(false)
                        .set_is_causal(false)
                        .set_has_mask(false)
                        .create();

    if (!sdpa_context.check()) {
      testlog_error("SDPA encoder context creation failed");
      return NOT_OK;
    }

    // Create SDPA encoder operator
    auto sdpa_operator = sdpa_encoder_operator_t()
                         .set_name("SDPA Encoder FP32")
                         .set_context(sdpa_context)
                         .create();

    if (sdpa_operator.is_bad_object()) {
      log_error("SDPA encoder operator creation failed");
      return NOT_OK;
    }

    // Set output tensor and execute
    sdpa_operator
      .set_output("sdpa_output", output_tensor)
      .execute();

  }
  catch (const exception_t &ex) {
    std::cout << "Exception: " << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}
```

## Parameter Naming Convention

**Important:** The string identifiers used in `.set_param()` and `.set_output()` are fixed and must not be changed. These names are internally mapped and executed by the operator implementation.

Required identifiers:

- `.set_param("query", ...)` → must use `"query"`
- `.set_param("key", ...)` → must use `"key"`
- `.set_param("value", ...)` → must use `"value"`
- `.set_param("mask", ...)` → must use `"mask"` (only when `has_mask` is true)
- `.set_output("sdpa_output", ...)` → must use `"sdpa_output"`

Changing these names will result in incorrect behavior or operator failure.

## Common Variables

- **tensor_factory_t**: Utility for creating tensors with specific shapes, types, and initial values.
- **sdpa_encoder_context_t**: Context configuration for the SDPA encoder operator.
- **sdpa_encoder_operator_t**: Operator class for executing the SDPA computation.
- **status_t**, **exception_t**: Status and exception handling types.
- **Logging utilities**: `testlog_info`, `testlog_error`.

## Error Handling

The operator's `validate()` performs comprehensive up-front checks and fails fast before kernel dispatch:

1. **Mandatory output**: `sdpa_output` must be set.
2. **Mandatory parameters**: `query`, `key`, `value` must be present.
3. **Mask agreement**: `has_mask` and the presence of a `"mask"` parameter must agree in both directions.
4. **Rank**: Q, K, V, and output must all be 4D.
5. **Dimension compatibility**: B, H, and head_dim must match across Q/K/V; K and V must share `S_kv`; the output must match Q's `[B, H, S_q, D]`.
6. **Dtype consistency**: Q, K, V, and output must share the same dtype.
7. **Layout**: Each tensor's strides must describe a supported BHSD or BSHD layout.
8. **Mask validation**: mask dtype, rank (2 or 4), inner `[S_q, S_kv]` dims, broadcast leading dims, and canonical row-major strides.

Unsupported QKV dtypes return `status_t::unimplemented`; other invalid configurations return `status_t::failure`.

## Logger

Utility functions such as `testlog_info` and `testlog_error` are used for logging information and errors, respectively, in the operation flow.

This operator demonstrates the object-oriented Tensor Operator API for attention, complementing the high-performance LowOHA `sdpa_direct` path documented in [LowOHA SDPA Operator](../low_overhead_operator/lowoha_sdpa_operator.md).
