
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# Reorder Operator

## Overview

This section provides a high-level overview of reorder operations with support for FP32 and BF16 data types, using Out-of-Place and In-Place memory. To achieve best performance, certain compute-intensive operations such as matrix multiplication (Matmul) require data in a specalized memory layout known as blocked memory format. The reorder operator efficiently transfers data from contigious memory layout to optimized blocked format based on backend.

Practical examples from `reorder_example.cpp` demonstrate these configurations, such as `reorder_outofplace_f32_kernel_example` and `reorder_inplace_bf16_example`.

## General Reorder Operation

Let:

- \( B \in \mathbb{R}^{K \times N} \)
- \( \text{Backend}(x) \): Backend used for reorder computation (e.g., AOCL, OneDNN)
- \( \text{Data type} \): Supported data types for reorder (e.g., FP32, BF16)

## Steps to Perform Reorder Operation

1. **Backend Algo selection**:
   - Specify the backend algorithm for reordering using `set_algo_format()`.

2. **Reorder size computation**:
   - Compute the reorderd size using `get_reorder_size()`.

3. **Output Tensor creation**:
    - Out-of-Place : Tensor creation with reordered size and blocked layout.
    - In-Place : Tensor creation with same view (memory) as Input tensor and blocked layout.

4. **Reorder Execution**:
    - Converts memory from contigious to blocked format.

# Reorder Support Table

This table outlines the support for Reorder operations with various data types, backend, and memory storage type.

| Input Data Type | Output Data Type | Backend | Memory Storage type    |
|-----------------|------------------|---------|------------------------|
| FP32            | FP32             | AOCL    | Out-of-Place, In-Place |
| BF16            | BF16             | AOCL    | Out-of-Place, In-Place |

## Examples

### 1. reorder_outofplace_f32_kernel_example

This example performs reorder with `float32 (f32)` data type using `AOCL` backend.

**Key Components**

- **Weights and Initialization**
  - Weights: Uniform tensor with dimensions `{ROWS, COLS}`

- **Execution**
  - Performs the reorder, sets the input tensor, backend, and executes the operator with Out-of-Place memory.

```cpp

int reorder_outofplace_f32_kernel_example() {
  try {
    // Status variable to track success or failure of operations
    status_t status;

    // Factory object to create tensors with specified properties
    tensor_factory_t tensor_factory;

    // Create a tensor with dimensions [K, N], data type float32, layout
    // contigious initialized with uniform values of 1.0, and named
    // "reorder_input"
    auto input_tensor = tensor_factory.uniform_tensor({ROWS, COLS},
                                                      data_type_t::f32,
                                                      1.0, "reorder_input");

    // Create a reorder context by backend algo
    // This context encapsulates all parameters required for the reorder operation
    auto reorder_context = reorder_context_t()
      .set_algo_format("aocl")
      .create();

    // Check if the context was created successfully
    if (!(reorder_context.check())) {
      testlog_error("context creation failed");
      return NOT_OK;
    }

    // Create a reorder operator using the defined context and set input
    // tensor, This operator will execute the reorder operation
    auto reorder_operator = reorder_operator_t()
      .set_name("reorder_f32_operator")  // Assign a name to the operator
      .set_context(reorder_context)      // Attach the context
      .create()
      .set_input("reorder_input", input_tensor);

    // Check if the operator was created successfully
    if (!reorder_operator.check()) {
      testlog_error(" operator ", reorder_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    // Compute the reorder size based on selected backend algo in defined
    // context and returns the size.
    size_t reorder_size = reorder_operator.get_reorder_size();

    // Create new memory using aligned alloc with reorderd size
    void *reorder_weights = aligned_alloc(64, reorder_size);

    // Create a Pair of storage params [reorder size and reorder weights] and
    // use it in tensor creation
    StorageParam buffer_params = std::make_pair(reorder_size, reorder_weights);

    // Create output tensor with dimensions [K, N], data type float32,
    // layout blocked, buffer parama [reorder size, reorder weights]
    // and named "reorder_output".
    auto output_tensor = tensor_factory.blocked_tensor({ROWS, COLS},
                                                       data_type_t::f32,
                                                       buffer_params,
                                                       "reorder_output");

    // Set output tensors and execute the reorder operator
    status = reorder_operator
      .set_output("reorder_output", output_tensor)
      .execute();

    // Log the result of the execution
    if (status == status_t::success) {
      testlog_info("<", reorder_operator.get_name(), ">", " operator execution successful.");
    } else {
      testlog_error("<", reorder_operator.get_name(), ">", " operator execution failed.");
      return NOT_OK;
    }

    // Free the buffer.
    free(reorder_weights);

  } catch (const exception_t& ex) {
    // Catch and print any exceptions that occur during execution
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  // Return success status
  return OK;
}

```
### 2. reorder_inplace_bf16_example

This example performs reorder with `bfloat16 (bf16)` data type using `AOCL` backend.

**Key Components**

- **Weights and Initialization**
  - Weights: Uniform tensor with dimensions `{ROWS, COLS}`

- **Execution**
  - Performs the reorder, sets the input tensor, backend, and executes the operator with In-Place memory.

```cpp

int reorder_inplace_bf16_example() {
  try {
    // Status variable to track success or failure of operations
    status_t status;

    // Factory object to create tensors with specified properties
    tensor_factory_t tensor_factory;

    // Create a tensor with dimensions [K, N], data type bfloat16, layout
    // contigious initialized with uniform values of 1.0, and named
    // "reorder_input"
    auto input_tensor = tensor_factory.uniform_tensor({ROWS, COLS},
                                                      data_type_t::bf16,
                                                      1.0, "reorder_input");

    // Create a reorder context by backend algo
    // This context encapsulates all parameters required for the reorder operation
    auto reorder_context = reorder_context_t()
      .set_algo_format("aocl")
      .create();

    // Check if the context was created successfully
    if (!(reorder_context.check())) {
      testlog_error("context creation failed");
      return NOT_OK;
    }

    // Create a reorder operator using the defined context and set input
    // tensor, This operator will execute the reorder operation
    auto reorder_operator = reorder_operator_t()
      .set_name("inplace_reorder_bf16_operator")  // Assign a name to the operator
      .set_context(reorder_context)      // Attach the context
      .create()
      .set_input("reorder_input", input_tensor);

    // Check if the operator was created successfully
    if (!reorder_operator.check()) {
      testlog_error(" operator ", reorder_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    // Compute the reorder size based on selected backend algo in defined
    // context and returns the size.
    size_t reorder_size = reorder_operator.get_reorder_size();

    // Get the input Buffer size.
    size_t input_buffer_size    = input_tensor.get_buffer_sz_bytes();

    // Inplace reorder takes place when reorder buffer size is same as
    // input buffer size
    if (reorder_size == input_buffer_size) {
      // Assign input_tensor to buffer_params as a tensor_t variant
      StorageParam buffer_params = input_tensor;

      // Create output tensor with dimensions [K, N], data type bfloat16,
      // layout blocked, buffer params [reorder size, reorder weights]
      // and named "reorder_output".
      auto output_tensor = tensor_factory.blocked_tensor({ROWS, COLS},
                                                         data_type_t::bf16,
                                                         buffer_params,
                                                         "reorder_output");

      // Set output tensors and execute the reorder operator
      status = reorder_operator
        .set_output("reorder_output", output_tensor)
        .execute();

      // Log the result of the execution
      if (status == status_t::success) {
        testlog_info("<", reorder_operator.get_name(), ">", " operator execution successful.");
      } else {
        testlog_error("<", reorder_operator.get_name(), ">", " operator execution failed.");
        return NOT_OK;
      }
    }
    else {
      // Inplace reorder is not possible as there is mismatch in the size.
      testlog_error("Inplace reorder is not possible for given input");
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

## Common Variables

- **tensor_factory_t**: Utility for creating tensors with specific shapes, types, and initial values.
- **reorder_context_t**: Context configuration for the Reorder operator.
- **reorder_operator_t**: Operator class for executing the Reorder operation.
- **status_t**, **exception_t**: Status and exception handling types.
- **Logging utilities**: `testlog_info`, `testlog_error`.

## Error Handling

Each example includes error checking where the operator creation and execution status is checked, and relevant logging is provided.

## Logger

Utility functions such as `testlog_info` and `testlog_error` are used for logging information and errors, respectively, in the operation flow.

These examples demonstrate the versatility and composability of reorder operations, showcasing the use of different data types and memory formats in the `ZenDNN*` library.

>ZenDNN* : ZenDNN is currently undergoing a strategic re-architecture and refactoring to enhance performance, maintainability, and scalability.
