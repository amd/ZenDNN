
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# EmbeddingBag Operator

## Overview

This section provides a high-level overview of embedding bag (`embag`) operations with support for FP32 data type and multiple aggregation algorithms (sum, mean, max). The support matrix summarizes valid combinations of table, indices, offsets, weights, and output data types along with supported operations. Practical examples from `embedding_bag_example.cpp` demonstrate these configurations, such as `embedding_bag_f32_kernel_example`, which performs FP32 embedding bag operations with sum aggregation methods.

# General EmbeddingBag Operation

Let:

- *T* ∈ ℝᴿˣᴰ : Embedding table with number of embedding rows R and embedding dimension D
- *I* ∈ ℤᴺ : Indices array of length N pointing to embedding table rows
- *O* ∈ ℤᴮ⁺¹ : Offsets array defining bag boundaries for B bags
- *W* ∈ ℝᴺ : Optional weights array (same length as indices)
- *P* ∈ ℤ : Optional padding index to ignore during aggregation
- *Algo* : Aggregation algorithm (Example: sum, mean, max)
- *E* ∈ ℝᴮˣᴰ : Output embeddings for B bags after aggregation

The computation can be expressed as:

## Steps to Perform EmbeddingBag Operation

1. **Index Lookup**:
   For each bag `b` and each index `i` in range `[O[b], O[b+1])`:
   ```
   embedding_i = T[I[i], :]
   ```

2. **Weight Application (optional)**:
   ```
   weighted_embedding_i = W[i] × embedding_i
   ```

3. **Padding Filtering**:
   Skip indices where I[i] = P (padding index)

4. **Aggregation**:
   Apply the specified algorithm:
   - **Sum**: E[b, :] = E[b, :] = Σ embedding_i
   - **Mean**: E[b, :] = (1/|valid_indices|) × Σ embedding_i
   - **Max**: E[b, :] = max(embedding_i)

5. **Store Result**:
   ```
   Output[b, :] = E[b, :]
   ```

## Example with Sum Aggregation

For sum aggregation with weights:
```
E[b, :] = Σ(i=O[b] to O[b+1]-1) W[i] × T[I[i], :] where I[i] ≠ P
```

# EmbeddingBag Operation Support Overview

- **Aggregation Algorithms**: Three different methods for combining embeddings within each bag:
  - **Sum**: Simple summation of embeddings
  - **Mean**: Average of embeddings (normalized by count of valid indices)
  - **Max**: Element-wise maximum across embeddings

- **Weight Handling**: Optional per-index weights for weighted aggregation:
  - **Unweighted**: All embeddings contribute equally (weight = 1.0)
  - **Weighted**: Each embedding is scaled by its corresponding weight

- **Padding Index**: Optional index value to ignore during aggregation:
  - Commonly used for variable-length sequences
  - Padding indices are excluded from aggregation and count calculations

### EmbeddingBag Operation Flow Diagram

```text
Embedding Table T [R x D]    Indices I [N]   Offsets O [B+1]   Weights W [N]
        |                       |                  |           (optional)
        |                       |                  |                |
        |                       |                  |                |
        +-----------------------+------------------+                |
                    |           |             |                     |
                    |       Index Lookup      |                     |
                    |           |             |                     |
                    +-----------v--------------+                    |
                    |      Embedding[i]       |                     |
                    +-----------v-------------+                     |
                               |                                    |
                    +-----------v-------------+                     |
                    |    Apply Weight         |<--------------------+
                    +-----------v-------------+
                               |
                    +-----------v-------------+
                    |    Filter Padding       |
                    +-----------v-------------+
                               |
                    +-----------v-------------+
                    |     Aggregation         |  ← Sum/Mean/Max per bag
                    +-----------v-------------+
                               |
                        Output E [B x D]
```

## Supported Configurations
This table provides a detailed overview of supported configurations for embedding bag operations across various data types and aggregation methods.

| Table<br>Data Type | Indices<br>Data Type	| Weights<br>Data Type |	Output<br>Data Type	| Aggregation<br>Algorithm	|
|--------------------|----------------------|----------------------|----------------------|---------------------------|
|FP32	               |INT32	                |FP32	(Only Sum)       |FP32	                |Sum, Mean, Max	            |
|FP32	               |INT32	                |FP32	(Only Sum)       |BF16	                |Sum, Mean, Max	            |
|BF16	               |INT32	                |FP32	(Only Sum)       |FP32	                |Sum, Mean, Max	            |
|BF16	               |INT32	                |FP32	(Only Sum)       |FP32	                |Sum, Mean, Max	            |

## Tensor
A **tensor** is a multi-dimensional array that serves as the primary data structure in deep learning models. It generalizes vectors (1D), matrices (2D) to higher dimensions (3D, etc.), Tensors are a fundamental building block for neural network computations, facilitating efficient data manipulation and mathematical operations.

In the context of the EmbeddingBag operator, tensors are used to represent:
- Embedding Table: Learnable lookup table containing dense vector representations
- Indices: Integer array specifying which embeddings to retrieve
- Offsets: Integer array defining bag boundaries for grouping indices
- Weights: Optional scaling factors for each embedding
- Output: Aggregated embeddings for each bag

### Key Properties of EmbeddingBag Tensors

Each tensor is defined by several important attributes that determine how it behaves in computations:

#### Shape Requirements
Specifies the required dimensions for each tensor:

- **Embedding Table**: `[R, D]` where `R` is num of embeddings, `D` is embedding dimension
- **Indices Tensor**: `[N]` where `N` is total number of indices across all bags
- **Offsets Tensor**: `[B+1]` where `B` is number of bags (includes end offset)
- **Weights Tensor**: `[N]` (optional, same length as indices)
- **Output Tensor**: `[B, D]` where `B` is number of bags
```cpp
table.set_size({R, D});
indices.set_size({N});
offsets.set_size({B + 1});
weights.set_size({N});  // optional
output.set_size({B, D});
```

#### Data Type Constraints
Indices and Offsets: INT32
Table, Weights, Output: FP32
```cpp
indices.set_data_type(data_type_t::s32);
offsets.set_data_type(data_type_t::s32);
table.set_data_type(data_type_t::f32);
output.set_data_type(data_type_t::f32);
```

#### Storage
The storage of a tensor defines how and where its data is allocated or managed in memory.

- **Default Storage Allocation**
- **Aligned Storage Allocation**
- **Borrowing Memory from a Raw Pointer**
- **Sharing Storage with Another Tensor**
```cpp
tensor.set_storage();
```

#### Layout
Describes how the tensor is stored in memory, which affects performance and access patterns.

- **Contiguous**:*(default)* Linear, row-major format
- **Blocked**: Data is stored in blocks for optimized access patterns.
```cpp
tensor.set_layout(tensor_layout_t::contiguous);
```

#### Example:
```cpp
auto table = tensor_t()
             .set_name("embedding_table")
             .set_size({R, D})
             .set_data_type(data_type_t::f32)
             .set_layout(tensor_layout_t::contiguous)
             .set_storage()
             .create();
```

#### Examples

### 1. embag_sum_f32_kernel_example

This example performs embedding bag operation with `float32 (f32)` data types, using `sum` aggregation algorithm.

**Key Components**

- **Embedding Table Initialization**
  - Table: Uniform tensor with dimensions `{R, D}`
- **Indices and Offsets**
  - Indices: Random integers within vocabulary range
  - Offsets: Define bag boundaries for grouping embeddings
- **Aggregation**
  - Applies `sum` aggregation across embeddings in each bag

```cpp
int embag_sum_f32_kernel_example() {
  try {
    // Status variable to track success or failure of operations
    status_t status;

    // Factory object to create tensors with specified properties
    tensor_factory_t tensor_factory;

    std::vector<uint32_t> indices = generate_random_indices(INDICES_SIZE);
    std::vector<uint32_t> offsets = generate_offsets(BATCH_SIZE);

    // Create an embedding table with dimensions [VOCAB_SIZE, EMBEDDING_DIM]
    // initialized with uniform value 1.0
    auto embedding_table = tensor_factory.uniform_tensor({R, D},
                                                         data_type_t::f32,
                                                         1.0, "table");

    // Create indices tensor with random values within vocabulary range
    auto indices_tensor = tensor_factory.non_uniform_tensor({indices.size()},
                                                            data_type_t::s32,
                                                            indices, "indices");

    // Create offsets tensor defining bag boundaries
    // offsets = [0, bag1_size, bag1_size + bag2_size, ...]
    auto offsets_tensor = tensor_factory.non_uniform_tensor({BATCH_SIZE},
                                                            data_type_t::s32,
                                                            offsets, "offsets");

    // Create embedding bag context with sum aggregation
    auto embag_context = embag_context_t()
      .set_param("table", embedding_table)
      .set_algo(embag_algo_t::sum)
      .set_padding_index(-1)
      .create();

    // Create embedding bag operator using the defined context
    auto embag_operator = embag_operator_t()
      .set_name("embag_sum_f32")
      .set_context(embag_context)
      .create();

    // Check if the operator was created successfully
    if (!embag_operator.check()) {
      testlog_error(" operator ", embag_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    // Create output tensor with dimensions [NUM_BAGS, EMBEDDING_DIM]
    auto output_tensor = tensor_factory.zero_tensor({B, D},
                                                    data_type_t::f32,
                                                    "output");

    // Set the input and output tensors and execute the embag operator
    status = embag_operator
      .set_input("indices", indices_tensor)
      .set_input("offsets", offsets_tensor)
      .set_output("output", output_tensor)
      .execute();

    // Log the result of the execution
    if (status == status_t::success) {
      testlog_info("<", embag_operator.get_name(), ">", " operator execution successful.");
    } else {
      testlog_error("<", embag_operator.get_name(), ">", " operator execution failed.");
      return NOT_OK;
    }

  } catch (const exception_t& ex) {
    // Catch and print any exceptions that occur during execution
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  // Return success status
  return OK;
}

```

## Parameter Naming Convention
**Important:** The string identifiers used in .set_param(), .set_input(), and .set_output() are fixed and must not be changed. These names are internally mapped and executed by the operator implementation.

Required Identifers:

- .set_param("table", ...) → must use "table"
- .set_input("indices", ...) → must use "indices"
- .set_input("offsets", ...) → must use "offsets"
- .set_output("output", ...) → must use "output"

Changing these names will result in incorrect behavior or operator failure.

## Common Variables

- **tensor_factory_t**: Utility for creating tensors with specific shapes, types, and initial values.
- **embag_context_t**: Context configuration for the Embedding Bag operator.
- **embag_operator_t**: Operator class for executing the Embedding Bag operation.
- **status_t**, **exception_t**: Status and exception handling types.
- **Logging utilities**: `testlog_info`, `testlog_error`.

## Error Handling

Each example includes error checking where the operator creation and execution status is checked, and relevant logging is provided.

## Logger

Utility functions such as `testlog_info` and `testlog_error` are used for logging information and errors, respectively, in the operation flow.

These examples demonstrate the versatility and composability of embedding bag operations within the `ZenDNN*` library.

>ZenDNN* : ZenDNN is currently undergoing a strategic re-architecture and refactoring to enhance performance, maintainability, and scalability.