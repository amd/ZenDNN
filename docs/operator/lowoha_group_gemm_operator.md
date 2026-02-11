
(Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.)

# LowOHA Group GEMM Operator

## Overview

The **Group GEMM API** (`group_gemm_direct`) enables executing multiple matrix multiplication operations in a single function call. It supports two execution modes:

- **Sequential (Linear)**: Chained operations where the output of one feeds as input to the next, commonly used in MLP layers of transformer models
- **Parallel**: Independent operations executed concurrently, used for multi-head attention, MoE expert routing, and parallel Q/K/V projections

### Key Benefits

- **Reduced Function Call Overhead**: Single API call for multiple operations
- **Dual Execution Modes**: Sequential chaining or parallel execution based on workload
- **Flexible Configurations**: Each operation can have independent dimensions and parameters
- **Unified Error Handling**: Returns success only if all operations complete successfully

## API Signature

```cpp
status_t group_gemm_direct(
  const std::vector<char> &layout,           // Layout for each operation ('r' for row-major)
  const std::vector<bool> &transA,           // Transpose input A for each operation
  const std::vector<bool> &transB,           // Transpose weight B for each operation
  const std::vector<int> &M,                 // Number of rows in A (and C) for each operation
  const std::vector<int> &N,                 // Number of columns in B (and C) for each operation
  const std::vector<int> &K,                 // Number of columns in A / rows in B for each operation
  const std::vector<float> alpha,            // Scaling factor for A*B for each operation
  const std::vector<const void *> src,       // Input matrices A for each operation
  const std::vector<int> lda,                // Leading dimension of A for each operation
  const std::vector<const void *> weight,    // Weight matrices B for each operation
  const std::vector<int> ldb,                // Leading dimension of B for each operation
  const std::vector<const void *> bias,      // Optional bias vectors for each operation
  const std::vector<float> beta,             // Scaling factor for output C for each operation
  const std::vector<void *> dst,             // Output matrices C for each operation
  const std::vector<int> ldc,                // Leading dimension of C for each operation
  const std::vector<bool> is_weights_const,  // Weight caching flag for each operation
  std::vector<matmul_params> params          // LowOHA parameters for each operation
);
```

### `matmul_params`

The main configuration structure for LowOHA MatMul:

```cpp
struct matmul_params {
  matmul_data_types dtypes;                  // Data types for tensors
  std::vector<matmul_post_op> postop_;       // Post-operations
  matmul_quantization_params_t quant_params; // Quantization parameters
  char mem_format_a;                         // Memory format for A ('n'=non-reordered, 'r'=reordered)
  char mem_format_b;                         // Memory format for B ('n'=non-reordered, 'r'=reordered)
  matmul_algo_t lowoha_algo;                 // Preferred backend algorithm
  uint64_t num_threads;                      // Number of threads (0 = auto)
  std::string plugin_op;                     // Plugin op name
};
```


### `matmul_data_types`

Specifies the data types for each tensor:

```cpp
struct matmul_data_types {
  data_type_t src;      // Input data type
  data_type_t wei;      // Weight data type
  data_type_t dst;      // Output data type
  data_type_t bias;     // Bias data type
  data_type_t compute;  // Computation type
};
```

## Parameters

All parameters are vectors where each element corresponds to one matrix multiplication operation. All vectors must have the same size (number of operations).

| Parameter | Type | Description |
|-----------|------|-------------|
| `layout` | `std::vector<char>` | Memory layout for each operation: `'r'` for row-major, `'c'` for column-major |
| `transA` | `std::vector<bool>` | Whether to transpose input matrix A for each operation |
| `transB` | `std::vector<bool>` | Whether to transpose weight matrix B for each operation |
| `M` | `std::vector<int>` | Number of rows in A (and C) for each operation |
| `N` | `std::vector<int>` | Number of columns in B (and C) for each operation |
| `K` | `std::vector<int>` | Shared dimension (columns of A / rows of B) for each operation |
| `alpha` | `std::vector<float>` | Scaling factor applied to A×B product for each operation |
| `src` | `std::vector<const void*>` | Pointers to input matrices A for each operation |
| `lda` | `std::vector<int>` | Leading dimension of input matrix A for each operation |
| `weight` | `std::vector<const void*>` | Pointers to weight matrices B for each operation |
| `ldb` | `std::vector<int>` | Leading dimension of weight matrix B for each operation |
| `bias` | `std::vector<const void*>` | Pointers to bias vectors for each operation (can contain `nullptr`) |
| `beta` | `std::vector<float>` | Scaling factor for accumulation into output C for each operation |
| `dst` | `std::vector<void*>` | Pointers to output matrices C for each operation |
| `ldc` | `std::vector<int>` | Leading dimension of output matrix C for each operation |
| `is_weights_const` | `std::vector<bool>` | Whether weights are constant (enables caching) for each operation |
| `params` | `std::vector<matmul_params>` | LowOHA configuration (data types, post-ops, etc.) for each operation |

## Return Value

- `status_t::success` - All operations completed successfully
- `status_t::failure` - One or more operations failed

## Execution Modes

### Mode Selection

The execution mode is determined by `src.size()`:

| Condition | Mode | Description |
|-----------|------|-------------|
| `src.size() == 1` | Sequential | Chained execution: `dst[i-1]` → `src[i]` |
| `src.size() > 1` | Parallel | Independent operations in parallel |

### Sequential (Linear)

When `src.size() == 1`, operations are executed **sequentially** in a chain. The output of each operation becomes the input for the next:

- **Op 0**: Uses `src[0]` as input
- **Op 1**: Uses `dst[0]` (output of Op 0) as input
- **Op 2**: Uses `dst[1]` (output of Op 1) as input
- ...and so on

Each operation uses **all available threads** for maximum throughput.

```
┌──────────────────────────────────────────────────────────┐
│              group_gemm_direct (Sequential GEMM)         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  src[0] ──► ┌──────────┐                                 │
│             │  Op 0    │ (all T threads)                 │
│             │ M₀×K₀×N₀ │                                 │
│             └────┬─────┘                                 │
│                  │ dst[0]                                │
│                  ▼                                       │
│             ┌──────────┐                                 │
│             │  Op 1    │ (all T threads)                 │
│             │ M₁×K₁×N₁ │                                 │
│             └────┬─────┘                                 │
│                  │ dst[1]                                │
│                  ▼                                       │
│             ┌──────────┐                                 │
│             │  Op 2    │ (all T threads)                 │
│             │ M₂×K₂×N₂ │                                 │
│             └────┬─────┘                                 │
│                  │ dst[2]                                │
│                  ▼                                       │
│              (output)                                    │
└──────────────────────────────────────────────────────────┘
```

### Parallel

When `src.size() > 1`, operations are executed **in parallel** using OpenMP, with each operation assigned to its own thread:

```
┌────────────────────────────────────────────────────────────┐
│              group_gemm_direct (Parallel GEMM)             │
├────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │           OpenMP Parallel Execution                 │   │
│  │    #pragma omp parallel for num_threads(T)          │   │
│  └─────────────────────────────────────────────────────┘   │
│       │           │           │                 │          │
│       ▼           ▼           ▼                 ▼          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐     │
│  │ Thread 0│ │ Thread 1│ │ Thread 2│  ...  │Thread T-1│    │
│  └────┬────┘ └────┬────┘ └────┬────┘       └────┬────┘     │
│       │           │           │                 │          │
│       ▼           ▼           ▼                 ▼          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐     │
│  │  Op 0   │ │  Op 1   │ │  Op 2   │  ...  │  Op N-1 │     │
│  │ M₀×K₀×N₀│ │ M₁×K₁×N₁│ │ M₂×K₂×N₂│       │ Mₙ×Kₙ×Nₙ │     │
│  └─────────┘ └─────────┘ └─────────┘       └─────────┘     │
└────────────────────────────────────────────────────────────┘
```

## Thread Configuration

The number of threads used for execution can be controlled via:

1. **Environment Variable:**
   ```bash
   export OMP_NUM_THREADS=8
   ```

2. **Execution Mode Thread Behavior:**

| Mode | Thread Usage |
|------|-------------|
| Sequential | Each operation uses all `T` threads |
| Parallel | Operations distributed across `T` threads, each op uses 1 thread |

## Usage Example

### Example: Group GEMM with Multiple Operations

```cpp
int lowoha_group_gemm_example() {
  using namespace zendnnl::lowoha::matmul;
  
  // Number of independent matmul operations
  constexpr int NUM_OPS = 4;
  
  // Define dimensions for each operation (can be different)
  std::vector<int> Ms = {64, 128, 32, 256};
  std::vector<int> Ns = {128, 64, 256, 64};
  std::vector<int> Ks = {256, 256, 128, 128};
  
  // Allocate buffers for each operation
  std::vector<std::vector<float>> src_buffers(NUM_OPS);
  std::vector<std::vector<float>> weight_buffers(NUM_OPS);
  std::vector<std::vector<float>> bias_buffers(NUM_OPS);
  std::vector<std::vector<float>> dst_buffers(NUM_OPS);
  
  for (int i = 0; i < NUM_OPS; ++i) {
    src_buffers[i].resize(Ms[i] * Ks[i], 1.0f);
    weight_buffers[i].resize(Ks[i] * Ns[i], 1.0f);
    bias_buffers[i].resize(Ns[i], 0.0f);
    dst_buffers[i].resize(Ms[i] * Ns[i], 0.0f);
  }
  
  // Create pointer vectors
  std::vector<const void*> src_ptrs(NUM_OPS);
  std::vector<const void*> weight_ptrs(NUM_OPS);
  std::vector<const void*> bias_ptrs(NUM_OPS);
  std::vector<void*> dst_ptrs(NUM_OPS);
  
  for (int i = 0; i < NUM_OPS; ++i) {
    src_ptrs[i] = src_buffers[i].data();
    weight_ptrs[i] = weight_buffers[i].data();
    bias_ptrs[i] = bias_buffers[i].data();
    dst_ptrs[i] = dst_buffers[i].data();
  }
  
  // Configure common parameters
  std::vector<char> layouts(NUM_OPS, 'r');
  std::vector<bool> transA(NUM_OPS, false);
  std::vector<bool> transB(NUM_OPS, false);
  std::vector<float> alphas(NUM_OPS, 1.0f);
  std::vector<float> betas(NUM_OPS, 0.0f);
  std::vector<bool> is_weights_const(NUM_OPS, true);
  
  // Calculate leading dimensions
  std::vector<int> ldas(NUM_OPS);
  std::vector<int> ldbs(NUM_OPS);
  std::vector<int> ldcs(NUM_OPS);
  
  for (int i = 0; i < NUM_OPS; ++i) {
    ldas[i] = Ks[i];  // Row-major, no transpose
    ldbs[i] = Ns[i];
    ldcs[i] = Ns[i];
  }
  
  // Configure matmul params for each operation
  std::vector<matmul_params> params(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    params[i].dtypes.src = data_type_t::f32;
    params[i].dtypes.wei = data_type_t::f32;
    params[i].dtypes.dst = data_type_t::f32;
    params[i].dtypes.bias = data_type_t::f32;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
  }
  
  // Execute group GEMM
  status_t status = group_gemm_direct(
                      layouts, transA, transB,
                      Ms, Ns, Ks, alphas,
                      src_ptrs, ldas,
                      weight_ptrs, ldbs,
                      bias_ptrs, betas,
                      dst_ptrs, ldcs,
                      is_weights_const,
                      params);
  
  if (status == status_t::success) {
    std::cout << "Group GEMM executed successfully for " 
              << NUM_OPS << " operations." << std::endl;
  }
  
  return (status == status_t::success) ? 0 : -1;
}
```

### Example: Sequential GEMM

```cpp
int lowoha_sequential_gemm_example() {
  using namespace zendnnl::lowoha::matmul;
  
  // 3-layer MLP: Input(64,128) -> Hidden1(64,256) -> Hidden2(64,128) -> Output(64,64)
  const int NUM_OPS = 3;
  const int M = 64;  // Batch size (constant across layers)
  
  // Dimensions: K[i+1] must equal N[i] for chaining
  std::vector<int> Ms = {M, M, M};
  std::vector<int> Ks = {128, 256, 128};   // Input dims per layer
  std::vector<int> Ns = {256, 128, 64};    // Output dims per layer
  
  // Allocate input (single src triggers sequential mode)
  std::vector<float> input_buffer(M * Ks[0], 1.0f);
  
  // Allocate weights and outputs for each layer
  std::vector<std::vector<float>> weight_buffers(NUM_OPS);
  std::vector<std::vector<float>> dst_buffers(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    weight_buffers[i].resize(Ks[i] * Ns[i], 0.5f);
    dst_buffers[i].resize(Ms[i] * Ns[i], 0.0f);
  }
  
  // src has only 1 entry → triggers sequential mode
  std::vector<const void*> src_ptrs = {input_buffer.data()};
  
  std::vector<const void*> weight_ptrs(NUM_OPS);
  std::vector<const void*> bias_ptrs(NUM_OPS, nullptr);
  std::vector<void*> dst_ptrs(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    weight_ptrs[i] = weight_buffers[i].data();
    dst_ptrs[i] = dst_buffers[i].data();
  }
  
  // Common parameters
  std::vector<char> layouts(NUM_OPS, 'r');
  std::vector<bool> transA(NUM_OPS, false);
  std::vector<bool> transB(NUM_OPS, false);
  std::vector<float> alphas(NUM_OPS, 1.0f);
  std::vector<float> betas(NUM_OPS, 0.0f);
  std::vector<int> ldas = Ks;
  std::vector<int> ldbs = Ns;
  std::vector<int> ldcs = Ns;
  std::vector<bool> is_weights_const(NUM_OPS, false);
  
  // Configure matmul params
  std::vector<matmul_params> params(NUM_OPS);
  for (int i = 0; i < NUM_OPS; ++i) {
    params[i].dtypes.src = data_type_t::f32;
    params[i].dtypes.wei = data_type_t::f32;
    params[i].dtypes.dst = data_type_t::f32;
    params[i].dtypes.bias = data_type_t::f32;
    params[i].mem_format_a = 'n';
    params[i].mem_format_b = 'n';
  }
  
  // Execute sequential GEMM: Input -> Layer1 -> Layer2 -> Layer3
  status_t status = group_gemm_direct(
                      layouts, transA, transB,
                      Ms, Ns, Ks, alphas,
                      src_ptrs, ldas,
                      weight_ptrs, ldbs,
                      bias_ptrs, betas,
                      dst_ptrs, ldcs,
                      is_weights_const,
                      params);
  
  return (status == status_t::success) ? 0 : -1;
}
```

## Notes and Best Practices

1. **Vector Size Consistency**: All input vectors must have the same size (number of operations), except `src` which determines the execution mode.

2. **Sequential Chaining**: In sequential mode, ensure that dimensions are compatible across chained operations (output dimensions of op `i` must match input dimensions of op `i+1`).

3. **Parallel Independence**: In parallel mode, each operation is fully independent and can have different dimensions, data types, and configurations.

4. **Weight Caching**: Set `is_weights_const[i] = true` for operations with constant weights to enable caching and improve performance on repeated executions.

5. **Memory Alignment**: Ensure input/output buffers are properly aligned for optimal performance.

6. **Error Handling**: The function returns `status_t::failure` if any operation fails. Check individual operation configurations if failure occurs.

