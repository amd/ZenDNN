
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# MatMul Operator

## Overview

This section provides a high-level overview of matrix multiplication (`matmul`) operations with support for FP32 and BF16 data types, optional bias addition, and a flexible sequence of post-processing operations such as activation functions (ReLU, GELU, etc.) and binary operations (Add, Mul). The support matrix summarizes valid combinations of source, weight, and output data types along with supported operations. Practical examples from `matmul_example.cpp` demonstrate these configurations, such as `matmul_relu_f32_kernel_example` and `matmul_mul_silu_mul_f32_kernel_example`, which performs FP32 matmul with activation and binary post-op.


# General MatMul Operation

Let:

- *A* ∈ ℝᴹˣᴷ : Input matrix 
- *B* ∈ ℝᴷˣᴺ : Weight matrix
- *C* ∈ ℝᴹˣᴺ : Output matrix
- *Bias* ∈ ℝ¹ˣᴺ : Optional Bias vector
- *Activation(x)* : Optional activation function (Example: ReLU, GELU, etc.)
- *BinaryOp(x, y)* : Optional binary post-operation (Example: element-wise add/mul with another matrix)
- *D* ∈ ℝᴹˣᴺ : Optional second operand for binary operations
- *Transpose(A, B)* : Optional transpose operation on matrices A or B (Example: Aᵀ, Bᵀ)
- *Strides(A, B)* : Optional strides for matrix A or B, defining the step size for accessing elements

The computation can be expressed as:

$$
C = \text{BinaryOp}(\text{Activation}(A \cdot B + \text{Bias}), D)
$$

## Steps to Perform MatMul Operation

1. **Matrix Multiplication**:  
   ```
   Z = A × B
   ```

2. **Bias Addition (optional)**:  
   ```
   Z = Z + Bias
   ```

3. **Activation Function (optional)**:  
   ```
   Z = Activation(Z)
   ```

4. **Binary Post-Op (optional)**:  
   ```
   Z = BinaryOp(Z, D)
   ```

5. **Store Result**:  
   ```
   C = Z
   ```

## Example with ReLU and Add Post-Op

If using ReLU as activation and element-wise addition with matrix \( D \):

$$
C = \text{ReLU}(A \times B + \text{Bias}) + D
$$

# MatMul Operation Support Overview

- **Bias Handling**: Bias is typically applied **per channel**, meaning a unique bias value is added to each output feature or channel. This is common in neural network layers to allow each output neuron to learn an independent offset.

- **Binary Post-Operations**: Binary post-operations are element-wise operations applied after the MatMul and optional activation.

  They are supported in two forms:
  - **Per-channel**: A vector is added or multiplied with each column of the output matrix.
  - **Full matrix (element-wise)**: A matrix of the same shape as the output is used for element-wise operations.

- **Activation Functions**: Activation functions introduce non-linearity into the model, which is essential for learning complex patterns. These are applied after the MatMul and bias steps.

  Supported activations include:

  - ReLU: Rectified Linear Unit
  - Sigmoid
  - Tanh
  - GELU (both erf and tanh variants)
  - SiLU (also known as Swish)

- **Flexible Composition**: The MatMul operator supports chaining multiple post-operations, allowing for expressive and customizable computation pipelines.
  - Multiple BinaryOps
  - Multiple Activations
  - Mix of Binary and Activation

### Matmul Operation Flow Diagram
```text
  Input A [M x K]                    Weights B [K x N]
    (Optional:                         (Optional: 
Transpose/Strides)                 Transpose/Strides)
       |                                    |
       +------------------x-----------------+
                          |
                       MatMul
                          |
                   +------v------+ 
                   |   Add Bias  |
                   +------v------+
                   |   Post-Op   |  ← Activation / Binary Op D
                   +------v------+
                       Output C

```

## Supported Configurations
This table provides a detailed overview of supported configurations for matrix multiplication (MatMul) operations across various data types, including bias application, activation functions, and binary post-processing options.

| Src<br>Data Type | Weight<br>Data Type | Bias<br> Data Type | Output<br>Data Type | Activation     | Binary<br>Post-Op |
|------------------|---------------------|--------------------|---------------------|----------------|-------------------|
| FP32             | FP32                | FP32               | FP32                | ReLU           | Add               |
| BF16             | BF16                | FP32, BF16         | FP32, BF16          | Sigmoid        | Mul               |
|                  |                     |                    |                     | Tanh           |                   |
|                  |                     |                    |                     | GELU (erf)     |                   |
|                  |                     |                    |                     | GELU (tanh)    |                   |
|                  |                     |                    |                     | SiLU           |                   |

## Tensor
A **tensor** is a multi-dimensional array that serves as the primary data structure in deep learning models. It generalizes vectors (1D), matrices (2D) to higher dimensions (3D, etc.), Tensors are a fundamental building block for neural network computations, facilitating efficient data manipulation and mathematical operations.

In the context of the **MatMul operator** tensors are used to represent:
- Input data: Activations or feature maps from previous layers in the neural network.
- Weights: Learnable parameters that are optimized during training.
- Biases: Optional offsets that can be added to the output.
- Output results: Results of computations.

### Key Properties of a Tensor

Each tensor is defined by several important attributes that determine how it behaves in computations:

---

#### Shape
Specifies the dimensions of the tensor. It defines how data is organized and processed.

- **Input Tensor**: \([M, K]\)
- **Weight Tensor**: \([K, N]\)
- **Output Tensor**: \([M, N]\)
```cpp
tensor.set_size({M, K}); 
```
---

#### Data Type
Indicates the precision of the values stored in the tensor.

- **FP32**:*(Default)* 32-bit floating point
- **BF16**: 16-bit Brain Floating Point
```cpp
tensor.set_data_type(data_type_t::f32);
```
---
#### Storage
The storage of a tensor defines how and where its data is allocated or managed in memory.

- **Default Storage Allocation**
- **Aligned Storage Allocation**
- **Borrowing Memory from a Raw Pointer**
- **Sharing Storage with Another Tensor**
```cpp
tensor.set_storage();
```
---
#### Layout
Describes how the tensor is stored in memory, which affects performance and access patterns.

- **Contiguous**:*(default)* Linear, row-major format
- **Blocked**: Data is stored in blocks for optimized access patterns.
```cpp
tensor.set_layout(tensor_layout_t::blocked);
```
---

#### Transpose (Optional)
Specifies whether the tensor is transposed, which swaps its rows and columns.

- **Original**:*(default)* *A* ∈ ℝᴹˣᴷ
```cpp
tensor.set_order("ab");
```
- **Transposed**: *Aᵀ* ∈ ℝᴷˣᴹ
```cpp
tensor.set_order("ba");
```
---

#### Strides (Optional)
Defines how many memory elements to skip to move between elements along each dimension.

- Enables efficient access to non-contiguous data
- Useful for custom memory layouts and batched operations
```cpp
tensor.set_stride_size({stride_m, stride_k});
```

#### Example:
A tensor of shape **[M, K]** with strides **[stride_M, stride_K]** defines how memory is accessed along each dimension.

#### Offset Calculation
The memory offset for accessing an element at position \([i, j]\) is calculated as:

```
Offset = i * stride_M + j * stride_K
```

#### Where:
- *i* : Row index *i* ∈ [ 0 , M )
- *j* : Column index *j* ∈ [ 0 , K )
```cpp
auto tensor = tensor_t()
              .set_name("strided_example")       // Set tensor name
              .set_size({M, K})                // Define tensor dimensions
              .set_stride_size({stride_M, stride_K}) // Define strides
              .set_storage()                  // Allocate storage
              .create();
```
---
Tensor can be created in two ways:
1. Direct Tensor creation (Fine grained control over attributes)
```cpp
auto tensor = tensor_t()
              .set_name("example_tensor")       // Set tensor name
              .set_size({M, K})                // Define tensor dimensions
              .set_data_type(data_type::f32)   // Set data type
              .set_stride_size({stride_1, stride_2}) // Define strides
              .set_storage()                  // Allocate storage
              .set_layout()                   // Set layout
              .create();
```

2. Using Tensor Factory
The *tensor_factory_t* class provides utility functions to create tensors with predefined configurations.
Available APIs:
- **uniform_dist_strided_tensor**: Creates a tensor with uniform random values and custom strides.
- **zero_tensor**: Creates a tensor initialized with zeros.
- **uniform_tensor**: Creates a tensor with a uniform value.
- **uniform_dist_tensor**: Creates a tensor with uniform random values.
- **blocked_tensor**: Creates a tensor with a blocked memory layout.

## Examples

### 1. matmul_relu_f32_kernel_example

This example performs matrix multiplication with `float32 (f32)` data types, applying a ReLU activation as a post-operation.

**Key Components**

- **Weights and Bias Initialization** 
  - Weights: Uniform tensor with dimensions `{MATMUL_K, MATMUL_N}`
  - Bias: Uniform tensor with dimensions `{MATMUL_N}`

- **Post-Operation**
  - Applies the ReLU operation on the result.

- **Execution**
  - Performs the matrix multiplication, sets the input and output tensors, and executes the operator.


```cpp
int matmul_relu_f32_kernel_example() {
  try {
    // Status variable to track success or failure of operations
    status_t status;

    // Factory object to create tensors with specified properties
    tensor_factory_t tensor_factory;

    // Create a weights tensor with dimensions [K, N], data type float32,
    // initialized with uniform values of 1.0, and named "weights"
    auto weights = tensor_factory.uniform_tensor({MATMUL_K, MATMUL_N},
                                                 data_type_t::f32,
                                                 1.0, "weights");

    // Create a bias tensor with dimensions [N], data type float32,
    // initialized with uniform values of -10.0, and named "bias"
    auto bias = tensor_factory.uniform_tensor({MATMUL_N},
                                              data_type_t::f32,
                                              -10.0, "bias");

    // Define a ReLU post-operation to be applied after matrix multiplication
    auto relu_post_op = post_op_t{post_op_type_t::relu};

    // Create a matmul context by setting weights, bias, and post-operation
    // This context encapsulates all parameters required for the matmul operation
    auto matmul_context = matmul_context_t()
      .set_param("weights", weights)
      .set_param("bias", bias)
      .set_post_op(relu_post_op)
      .create();

    // Create a matmul operator using the defined context
    // This operator will execute the matmul operation
    auto matmul_operator = matmul_operator_t()
      .set_name("matmul_f32")  // Assign a name to the operator
      .set_context(matmul_context)  // Attach the context
      .create();

    // Check if the operator was created successfully
    if (!matmul_operator.check()) {
      testlog_error(" operator ", matmul_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    // Create an input tensor with dimensions [M, K], data type float32,
    // initialized with uniform values of 1.0, and named "matmul_input"
    auto input_tensor = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_K},
                                                      data_type_t::f32,
                                                      1.0, "matmul_input");

    // Create an output tensor with dimensions [M, N], data type float32,
    // initialized to zero, and named "matmul_output"
    auto output_tensor = tensor_factory.zero_tensor({MATMUL_M, MATMUL_N},
                                                    data_type_t::f32,
                                                    "matmul_output");

    // Set the input and output tensors and execute the matmul operator
    status = matmul_operator
      .set_input("matmul_input", input_tensor)
      .set_output("matmul_output", output_tensor)
      .execute();

    // Log the result of the execution
    if (status == status_t::success) {
      testlog_info("<", matmul_operator.get_name(), ">", " operator execution successful.");
    } else {
      testlog_error("<", matmul_operator.get_name(), ">", " operator execution failed.");
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


### 2. matmul_mul_silu_mul_f32_kernel_example

This example showcases an advanced operation combining matrix multiplication with a series of binary multiplication and SiLU operations.

**Key Components**

- **Post-Operations**  
  - Applies a binary multiplication, SiLU, followed by another binary multiplication.

- **Execution**  
  - Involves setting multiple input tensors corresponding to the operations defined in the context.

```cpp
// Function to demonstrate a fused MatMul operator with binary_mul, SiLU, and another binary_mul
int matmul_mul_silu_mul_f32_kernel_example() {
  testlog_info("**matmul binary_mul + silu + binary_mul operator f32 kernel example.");

  try {
    status_t status;
    tensor_factory_t tensor_factory;

    // Create a weights tensor with shape {K, N}, data type f32, filled with uniform values
    auto weights = tensor_factory.uniform_tensor({MATMUL_K, MATMUL_N},
                                                 data_type_t::f32,
                                                 1.0, "weights");

    // Create a bias tensor with shape {N}, data type f32
    auto bias = tensor_factory.uniform_tensor({MATMUL_N},
                                              data_type_t::f32,
                                              3.0, "bias");

    // Define the first binary multiplication post-op with scale 1.0
    binary_mul_params_t binary_mul;
    binary_mul.scale = 1.0;

    auto binary_mul_po   = post_op_t{binary_mul};                    // First binary_mul
    auto silu_post_op    = post_op_t{post_op_type_t::swish};         // SiLU activation
    auto binary_mul_po_2 = post_op_t{post_op_type_t::binary_mul};    // Second binary_mul

    // Define the MatMul context with weights, bias, and post-ops
    auto matmul_context = matmul_context_t()
      .set_param("weights", weights)
      .set_param("bias", bias)
      .set_post_op(binary_mul_po)
      .set_post_op(silu_post_op)
      .set_post_op(binary_mul_po_2)
      .create();

    // Create the MatMul operator with the defined context
    auto matmul_operator = matmul_operator_t()
      .set_name("matmul_f32_operator")
      .set_context(matmul_context)
      .create();

    // Check if the operator was created successfully
    if (!matmul_operator.check()) {
      testlog_error(" operator ", matmul_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    // Create input tensor for MatMul
    auto input_tensor = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_K},
                                                      data_type_t::f32,
                                                      1.0, "matmul_input");

    // Create tensor for first binary multiplication
    auto mul_tensor = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_N},
                                                    data_type_t::f32,
                                                    2.0, "binary_mul_0");

    // Create tensor for second binary multiplication
    auto mul_tensor_2 = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_N},
                                                      data_type_t::bf16,
                                                      3.0, "binary_mul_1");

    // Create output tensor initialized to zero
    auto output_tensor = tensor_factory.zero_tensor({MATMUL_M, MATMUL_N},
                                                    data_type_t::f32,
                                                    "matmul_output");

    // Set inputs and outputs for the operator and execute
    status = matmul_operator
      .set_input("matmul_input", input_tensor)
      .set_input(matmul_context.get_post_op(0).binary_mul_params.tensor_name, mul_tensor)
      .set_input(matmul_context.get_post_op(2).binary_mul_params.tensor_name, mul_tensor_2)
      .set_output("matmul_output", output_tensor)
      .execute();

    // Log the result of execution
    if (status == status_t::success) {
      testlog_info("operator ", matmul_operator.get_name(), " execution successful.");
    } else {
      testlog_info("operator ", matmul_operator.get_name(), " execution failed.");
      return NOT_OK;
    }

  } catch(const exception_t& ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}
```

## Parameter Naming Convention
**Important:** The string identifiers used in .set_param(), .set_input(), and .set_output() are fixed and must not be changed. These names are internally mapped and executed by the operator implementation.

Required Identifers:

- .set_param("weights", ...) → must use "weights"
- .set_param("bias", ...) → must use "bias"
- .set_input("matmul_input", ...) → must use "matmul_input"
- .set_output("matmul_output", ...) → must use "matmul_output"

Changing these names will result in incorrect behavior or operator failure.

## Common Variables

- **tensor_factory_t**: Utility for creating tensors with specific shapes, types, and initial values.
- **matmul_context_t**: Context configuration for the MatMul operator.
- **matmul_operator_t**: Operator class for executing the MatMul operation.
- **post_op_t**: Represents post-processing operations applied after MatMul.
- **status_t**, **exception_t**: Status and exception handling types.
- **Logging utilities**: `testlog_info`, `testlog_error`.



## Error Handling

Each example includes error checking where the operator creation and execution status is checked, and relevant logging is provided.



## Logger

Utility functions such as `testlog_info` and `testlog_error` are used for logging information and errors, respectively, in the operation flow.



These examples demonstrate the versatility and composability of matrix multiplication operations, showcasing the use of different data types and post-processing operations in the computational graphs defined within the `ZenDNN*` library.

>ZenDNN* : ZenDNN is currently undergoing a strategic re-architecture and refactoring to enhance performance, maintainability, and scalability.
