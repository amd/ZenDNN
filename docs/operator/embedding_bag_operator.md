
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# EmbeddingBag Operator

## Overview

This section provides a high-level overview of embedding bag (`embag`) operations with support for FP32, BF16, INT8, INT4 data types and multiple aggregation algorithms (sum, mean, max). The support matrix summarizes valid combinations of table, indices, offsets, weights, and output data types along with supported operations. Practical examples from `embedding_bag_example.cpp` demonstrate these configurations, such as `embedding_bag_f32_kernel_example`, `embedding_bag_u4_kernel_example`, which performs embedding bag operations with sum aggregation methods.

# General EmbeddingBag Operation

Let:

- *T* ∈ ℝᴿˣᴰ : Embedding table with number of embedding rows R and embedding dimension D
- *I* ∈ ℤᴺ : Indices array of length N pointing to embedding table rows
- *O* ∈ ℤᴮ⁺¹ : Offsets array defining bag boundaries for B bags
- *W* ∈ ℝᴺ : Optional weights array (same length as indices)
- *P* ∈ ℤ : Optional padding index to ignore during aggregation
- *Algo* : Aggregation algorithm (Example: sum, mean, max)
- *E* ∈ ℝᴮˣᴰ : Output embeddings for B bags after aggregation
- *Scale* : Scaling factor for quantized data (For INT8/INT4 data type appended to table tensor)
- *ZeroPoint* : Zero-point offset for quantized data (For INT8/INT4 data type appended to table tensor)

The computation can be expressed as:

## Steps to Perform EmbeddingBag Operation

1. **Index Lookup**:
   For each bag `b` and each index `i` in range `[O[b], O[b+1])`:
   ```
   embedding_i = T[I[i], :]
   ```
   For quantized EmbeddingBag
   ```
   embedding_i = T[I[Scale]] * (T[I[i], :] - T[I[ZeroPoint]]) 
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

- **FP16 Scale Bias**: Boolean value used to specify the data type of scale and bias:
  - **True**: Indicates scale and zeropoint are of type FP16.
  - **False**: Indicates scale and zeropoint are of type FP32.

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

## Quantization

Quantization is a technique used to reduce the precision of numerical computations, enabling faster execution and reduced memory usage. In the context of the EmbeddingBag operator, quantization is primarily applied to INT8/INT4 data types, where floating-point values are mapped to 8-bit integers using a scale and zero-point.

### Key Components of Quantization

1. **Scale**:
   - A multiplier used to scale the quantized values back to their original floating-point range.
   - Defined per channel.

2. **Zero-Point**:
   - An offset added to the quantized values to represent zero in the integer domain.
   - Helps in handling signed and unsigned integer representations.
   - Defined per channel.

3. **Quantization Formula**:
   - The relationship between a floating-point value \( x \) and its quantized representation \( q \) is given by:
     $$
     q = \text{round}\left(\frac{x}{\text{Scale}}\right) + \text{ZeroPoint}
     $$
   - The dequantization process to recover the floating-point value is:
     $$
     x = \text{Scale} \cdot (q - \text{ZeroPoint})
     $$

### Quantized EmbeddingBag Workflow

1. **Input Quantization**:
   - User passes input tensors of data type INT8/INT4 along with the scale and zero-point.
   - For INT4 2 values of 4-bit are packed per byte  

2. **Dequantization**:
   - Convert the quantized values into FP32 using the scale and zero-point for further processing.

3. **EmbeddingBag Operation**:
   - Library performs the EmbeddingBag operation in the FP32 domain.

## Supported Configurations
This table provides a detailed overview of supported configurations for embedding bag operations across various data types and aggregation methods.

| Table<br>Data Type | Indices<br>Data Type	| Weights<br>Data Type |	Output<br>Data Type	| Aggregation<br>Algorithm	|
|--------------------|----------------------|----------------------|----------------------|---------------------------|
|FP32	               |INT32/INT64	          |FP32	(Only Sum)       |FP32	                |Sum, Mean, Max	            |
|FP32	               |INT32/INT64           |FP32	(Only Sum)       |BF16	                |Sum, Mean, Max	            |
|BF16	               |INT32/INT64           |FP32	(Only Sum)       |FP32	                |Sum, Mean, Max	            |
|BF16	               |INT32/INT64           |FP32	(Only Sum)       |BF16	                |Sum, Mean, Max	            |
|INT8	               |INT32/INT64           |FP32	(Only Sum)       |FP32	                |Sum, Mean, Max	            |
|INT8	               |INT32/INT64           |FP32	(Only Sum)       |BF16	                |Sum, Mean, Max	            |
|INT4	               |INT32/INT64           |FP32	(Only Sum)       |FP32	                |Sum, Mean, Max	            |
|INT4	               |INT32/INT64           |FP32	(Only Sum)       |BF16	                |Sum, Mean, Max	            | 


## Tensor
A **tensor** is a multi-dimensional array that serves as the primary data structure in deep learning models. It generalizes vectors (1D), matrices (2D) to higher dimensions (3D, etc.), Tensors are a fundamental building block for neural network computations, facilitating efficient data manipulation and mathematical operations.

In the context of the **EmbeddingBag operator**, tensors are used to represent:
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
Table: FP32, BF16, INT8, INT4
Output: FP32, BF16
Indices and Offsets: INT32, INT64
Weights: FP32

```cpp
indices.set_data_type(data_type_t::s32);
offsets.set_data_type(data_type_t::s64);
table.set_data_type(data_type_t::f32);
table.set_data_type(data_type_t::s4); // For signed int4
table.set_data_type(data_type_t::u4); // For unsigned int4
output.set_data_type(data_type_t::bf16);
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

#### Stride
The **stride** of a tensor defines the number of elements to skip in memory to move to the next element along each dimension.

**Example:**
For output tensor with shape `[B, D]` where `B=4` and `D=128`:
- Default contiguous stride: `[128, 1]`
- Custom padded stride (e.g., aligned to 256): `[256, 1]`

```cpp
// Set stride for output tensor
// For shape [B, D], stride specifies memory layout
output.set_stride({D, 1});  // Row-major contiguous layout

// Example with custom stride for memory alignment
output.set_stride({256, 1});  // Padded row stride for alignment
```
#### Example:
```cpp
auto table = tensor_t()
             .set_name("embedding_table")
             .set_size({R, D})
             .set_data_type(data_type_t::f32)
             .set_layout(tensor_layout_t::contiguous)
             .set_stride({D, 1})  // Row-major contiguous stride
             .set_storage()
             .create();
```

Tensor can be created in two ways:
1. Direct Tensor creation (Fine grained control over attributes)
```cpp
auto table = tensor_t()
             .set_name("embedding_table")
             .set_size({R, D})
             .set_data_type(data_type_t::f32)
             .set_layout(tensor_layout_t::contiguous)
             .set_stride({D, 1})  // Row-major contiguous stride
             .set_storage()
             .create();
```

2. Using Tensor Factory
The *tensor_factory_t* class provides utility functions to create tensors with predefined configurations.
Available APIs:
- **zero_tensor**: Creates a tensor initialized with zeros.
- **uniform_tensor**: Creates a tensor with a uniform value.
- **uniform_dist_tensor**: Creates a tensor with uniform random values.
- **quantized_embedding_tensor_random**: Creates a tensor with quantized random values and appeneds scale and zeropoint for each embedding row.

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

    // Set stride for output tensor - defines memory layout
    // For shape [B, D], stride {D, 1} specifies row-major contiguous layout
    output_tensor.set_stride({D, 1});

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

### 2. embedding_bag_u4_kernel_example

This example performs embedding bag operation with U4 table data type, using `sum` aggregation algorithm.

**Key Components**

- **Embedding Table Initialization**
  - Table: Quantized random tensor with dimensions `{R, D}`
  - Values are stored as 4-bit usigned integers (U4), packed 2 values per byte
  - With appended Scale and ZeroPoint to the table tensor
- **Indices and Offsets**
  - Indices: Random integers within vocabulary range
  - Offsets: Define bag boundaries for grouping embeddings
- **Aggregation**
  - Applies `sum` aggregation across embeddings in each bag
- **FP16 Scale ZeroPoint**
  - Applies `FP16` scale and zeropoint

```cpp
int embedding_bag_u4_kernel_example() {
  try {
    // Status variable to track success or failure of operations
    status_t status;

    // Factory object to create tensors with specified properties
    tensor_factory_t tensor_factory;

    std::vector<uint32_t> indices = generate_random_indices(INDICES_SIZE);
    std::vector<uint32_t> offsets = generate_offsets(BATCH_SIZE);

    // Create an embedding table with dimensions [VOCAB_SIZE, EMBEDDING_DIM]
    // initialized with random values
    auto table = tensor_factory.quantized_embedding_tensor_random({R, D},
                 data_type_t::u4, "table", true);

    // Create embedding bag context with sum aggregation, scale and zp as fp16
    auto embedding_bag_context = embag_context_t()
                                 .set_param("table", table)
                                 .set_algo(embag_algo_t::sum)
                                 .set_fp16_scale_bias(true)
                                 .create();

    // Check if the context was created successfully
    if (! embedding_bag_context.check()) {
      testlog_error("embedding bag context creation failed");
      return NOT_OK;
    }

    // Create embedding bag operator using the defined context
    auto embedding_bag_operator = embag_operator_t()
                                  .set_name("embedding_bag_int4_operator")
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
    // offsets = [0, bag1_size, bag1_size + bag2_size, ...]
    auto offsets_tensor = tensor_factory.non_uniform_tensor({EMB_BATCH_SIZE},
                          data_type_t::s64, offsets, "offsets");

    // Create output tensor with dimensions [NUM_BAGS, EMBEDDING_DIM]
    auto output_tensor = tensor_factory.zero_tensor({EMB_BATCH_SIZE, EMB_DIM},
                         data_type_t::f32, "output");

    // Set stride for output tensor - defines memory layout
    // For shape [EMB_BATCH_SIZE, EMB_DIM], stride {EMB_DIM, 1} specifies
    // row-major contiguous layout
    output_tensor.set_stride({EMB_DIM, 1});

    // Set the input and output tensors and execute the embag operator
    status = embedding_bag_operator
             .set_input("indices", indices_tensor)
             .set_input("offsets", offsets_tensor)
             .set_output("output", output_tensor)
             .execute();

    // Log the result of the execution
    if (status == status_t::success) {
      testlog_info("<",embedding_bag_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",embedding_bag_operator.get_name(),">",
                    " operator execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    // Catch and print any exceptions that occur during execution
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  // Return success status
  return OK;
}
```

**Key Points for Quantization:**
- **Quantized Table Tensor**: Created with `data_type_t::s8/s4/u4`, for `s4` and `u4` the values are stored as 4-bit signed/unsigned integers, packed 2 values per byte.
- **Per-channel Scale and ZeroPoint**: Scale and ZeroPoint are appended to the table for each embedding row `R`.
- **Dequantization**: During computation, S8/S4/U4 weights are dequantized using the formula: `dequant = scale * (embedding_i - zeropoint)`.

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