
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# MatMul Operator

## Overview

This section provides a high-level overview of matrix multiplication (`matmul`) operations with support for FP32, BF16, INT8 data types, and Weight-Only Quantization (WOQ) with S4 weights. The operator supports optional bias addition and a flexible sequence of post-processing operations such as activation functions (ReLU, GELU, etc.) and binary operations (Add, Mul). The support matrix summarizes valid combinations of source, weight, and output data types along with supported operations. Practical examples from `matmul_example.cpp` and `batchmatmul_example.cpp` demonstrate these configurations, such as `matmul_relu_f32_kernel_example`, `matmul_mul_silu_mul_f32_kernel_example` (FP32 2D-matmul with activation and binary post-op), `matmul_woq_bf16_kernel_example` (Weight-Only Quantization with BF16 input and S4 weights), and `batch_matmul_relu_bf16_kernel_example` and `batch_matmul_inp2d_relu_f32_kernel_example` which perform batched matmul with activation.


# General MatMul Operation

Let:

- *A* ∈ ℝ<sup>mxk</sup> or ℝ<sup>bsxmxk</sup>: Input Matrix  or Batched Input Matrix
- *B* ∈ ℝ<sup>kxn</sup> or ℝ<sup>bsxkxn</sup>: Weight Matrix or Batched Weight Matrix
- *C* ∈ ℝ<sup>mxn</sup> or ℝ<sup>bsxmxn</sup>: Output Matrix or Batched Output Matrix
- *Bias* ∈ ℝ¹ˣᴺ : Optional Bias vector
- *Scale* : Scaling factor for quantized data (INT8)
- *ZeroPoint* : Zero-point offset for quantized data (INT8)
- *Activation(x)* : Optional activation function (Example: ReLU, GELU, etc.)
- *BinaryOp(x, y)* : Optional binary post-operation (Example: element-wise add/mul with another matrix)
- *D* ∈ ℝᴹˣᴺ : Optional second operand for binary operations
- *Transpose(A, B)* : Optional transpose operation on matrices A or B (Example: Aᵀ, Bᵀ)
- *Strides(A, B)* : Optional strides for matrix A or B, defining the step size for accessing elements

- Note: If Input and Weight matrices both are not batched, then Output matrix can not be batched.

The computation can be expressed as:

$$
C = \text{BinaryOp}(\text{Activation}(A \cdot B + \text{Bias}), D)
$$

The computation for quantized MatMul can be expressed as:

$$
C = \text{Scale} \cdot (\text{BinaryOp}(\text{Activation}(A \cdot B + \text{Bias}), D) + \text{ZeroPoint})
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

3. **Scaling and Zero-Point Adjustment (INT8 only)**:  
   ```
   Z = Scale * (Z + ZeroPoint)
   ```

4. **Activation Function (optional)**:  
   ```
   Z = Activation(Z)
   ```

5. **Binary Post-Op (optional)**:  
   ```
   Z = BinaryOp(Z, D)
   ```

6. **Store Result**:  
   ```
   C = Z
   ```

- Above steps(#1 to 5) are repated for number of batches in batchedmatmul operation with proper offset.

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
       Input A                            Weights B 
  ([M x K] or [BS x M x K])         ([K x N] or [BS x K x N])
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
## Quantization

Quantization is a technique used to reduce the precision of numerical computations, enabling faster execution and reduced memory usage. In the context of the MatMul operator, quantization is primarily applied to INT8 data types, where floating-point values are mapped to 8-bit integers using a scale and zero-point.

### Key Components of Quantization

1. **Scale**:
   - A multiplier used to scale the quantized values back to their original floating-point range.
   - Defined per tensor or per channel, depending on the use case.

2. **Zero-Point**:
   - An offset added to the quantized values to represent zero in the integer domain.
   - Helps in handling signed and unsigned integer representations.

3. **Quantization Formula**:
   - The relationship between a floating-point value \( x \) and its quantized representation \( q \) is given by:
     $$
     q = \text{round}\left(\frac{x}{\text{Scale}}\right) + \text{ZeroPoint}
     $$
   - The dequantization process to recover the floating-point value is:
     $$
     x = \text{Scale} \cdot (q - \text{ZeroPoint})
     $$

### Quantized MatMul Workflow

1. **Input Quantization**:
   - User passes input tensors of data type INT8 along with the scale and zero-point.

2. **Matrix Multiplication**:
   - Library performs the MatMul operation in the INT8 domain for improved performance.

3. **Dequantization**:
   - Convert the INT8 results back to FP32/BF16 using the scale and zero-point for further processing.

### Benefits of Quantization

- **Performance**: Reduced precision allows for faster computation on hardware optimized for INT8 operations.
- **Memory Efficiency**: INT8 tensors consume less memory compared to FP32 or BF16 tensors.
- **Energy Efficiency**: Lower precision computations require less energy, making them suitable for edge devices.

### Example: Quantized tensor creation

```cpp
// Define scale and zero-point for input and output tensors
auto scale_tensor = tensor_t()
                    .set_name("scale_tensor")        // Set tensor name
                    .set_size({1, N})                // Define tensor dimensions (per channel)
                    .set_data_type(data_type_t::f32) // data type of buffer
                    .set_storage()                   // Allocate storage
                    .create();

auto zp_tensor = tensor_t()
                  .set_name("scale_tensor")        // Set tensor name
                  .set_size({1, 1})                // Define tensor dimensions (per tensor)
                  .set_data_type(data_type_t::s8)  // data type of buffer
                  .set_storage()                   // Allocate storage
                  .create();

// Quantize input tensor
auto quantized_input = tensor_t()
                        .set_name("scale_tensor")        // Set tensor name
                        .set_size({1, 1})                // Define tensor dimensions (per tensor)
                        .set_data_type(data_type_t::s8)  // data type of buffer
                        .set_storage()                   // Allocate storage
                        .set_quant_scale(scale_tensor)   // Set scale tensor
                        .set_quant_zero_point(zp_tensor) // Set zero_point tensor
                        .create();
```

### Supported Quantization Configurations

| Tensor         | Scale (mandatory)                          | Zero-Point (per-tensor)|
|----------------|--------------------------------------------|------------------------|
| Input          | Yes (FP32/BF16)  Per-tensor                | Yes (INT8/UINT8/INT32) |
| Weights        | Yes (FP32/BF16)  Per-tensor or per-channel | Yes (INT8/UINT8/INT32) |
| Output         | Yes (FP32/BF16)  Per-tensor                | Yes (INT8/UINT8/INT32) |

Quantization in the MatMul operator enables efficient computation while maintaining acceptable accuracy for many deep learning workloads.

## Weight-Only Quantization (WOQ)

Weight-Only Quantization (WOQ) is a specialized quantization technique where **only the weights are quantized** to low precision (e.g., 4-bit integers), while the input activations remain in higher precision (BF16). This approach is particularly effective for **Large Language Model (LLM) inference** where:

- Memory bandwidth is the primary bottleneck
- Weight matrices are large and reused across tokens
- Activation precision needs to be preserved for accuracy

### WOQ Configuration

| Parameter | Supported Values | Description |
|-----------|------------------|-------------|
| Input (Src) | BF16 | Source activations in BFloat16 |
| Weights | S4 (signed 4-bit) | Packed weights (2 values per byte) |
| Output | FP32, BF16 | Destination data type |
| Bias | FP32 | Optional bias vector |

### S4 Weight Format

S4 weights use **signed 4-bit integers** with a range of [-8, 7]. Two S4 values are packed into a single byte:
- **Low nibble** (bits 0-3): First S4 value
- **High nibble** (bits 4-7): Second S4 value

### WOQ Quantization Granularity

WOQ supports flexible quantization granularity for weight scales and zero-points:

| Granularity | Scale Dimensions | Description |
|-------------|------------------|-------------|
| Per-tensor | `{1, 1}` | Single scale for entire weight matrix |
| Per-channel | `{1, N}` | One scale per output channel |
| Per-group | `{G, N}` | G groups along K dimension (G = K / group_size) |

**Per-group quantization** is commonly used in LLM inference with typical group sizes of 32 or 128.

### WOQ Dequantization Formula

The dequantization process to recover floating-point values from S4 weights:

$$
W_{dequant} = \text{Scale} \cdot (W_{s4} - \text{ZeroPoint})
$$

Where:
- \( W_{s4} \): Quantized S4 weight value
- \( \text{Scale} \): Per-tensor, per-channel, or per-group scale factor
- \( \text{ZeroPoint} \): Optional zero-point offset (for asymmetric quantization)

### WOQ Tensor Creation Example

```cpp
// Create weight scale tensor (per-channel: {1, N})
auto wei_scale = tensor_factory.uniform_tensor({1, MATMUL_N},
                 data_type_t::f32,
                 0.25, "scale_tensor");

// Create S4 quantized weights with scale
auto weights = tensor_factory.uniform_tensor({MATMUL_K, MATMUL_N},
               data_type_t::s4,
               1.0, "weights", wei_scale);

// Optionally with zero-point for asymmetric quantization:
auto wei_zp = tensor_factory.uniform_tensor({1, MATMUL_N},
              data_type_t::s8,
              0, "zp_tensor");

auto weights_asym = tensor_factory.uniform_tensor({MATMUL_K, MATMUL_N},
                    data_type_t::s4,
                    1.0, "weights", wei_scale, wei_zp);
```

### WOQ Supported Quantization Configurations

| Parameter | Scale (mandatory) | Zero-Point | Granularity |
|-----------|-------------------|------------|-------------|
| Weights (S4) | Yes (FP32/BF16) | Optional (S8) | Per-tensor, per-channel, or per-group |

## Supported Configurations
This table provides a detailed overview of supported configurations for matrix multiplication (MatMul) operations across various data types, including bias application, activation functions, and binary post-processing options.

| Src<br>Data Type | Weight<br>Data Type | Bias<br> Data Type | Output<br>Data Type            | Scale | ZeroPoint |
|------------------|---------------------|--------------------|--------------------------------|-------|-----------|
| FP32             | FP32                | FP32               | FP32                           | N/A   | N/A       |
| BF16             | BF16                | FP32, BF16         | FP32, BF16                     | N/A   | N/A       |
| UINT8/INT8       | INT8                | FP32, BF16, INT8   | FP32, BF16, INT32, UINT8, INT8 | Yes   | Yes       |
| BF16             | S4 (WOQ)            | FP32, BF16         | FP32, BF16                     | Yes   | Optional  |
---
| Activation     | Description                     |
|----------------|---------------------------------|
| ReLU           | Rectified Linear Unit          |
| Sigmoid        | Sigmoid Activation Function    |
| Tanh           | Hyperbolic Tangent Function    |
| GELU (erf)     | Gaussian Error Linear Unit (erf variant) |
| GELU (tanh)    | Gaussian Error Linear Unit (tanh variant) |
| SiLU           | Sigmoid Linear Unit (Swish)    |
---
| Binary Post-Op | Description                     |
|----------------|---------------------------------|
| Add            | Element-wise addition           |
| Mul            | Element-wise multiplication     |

**Note** Binary post-ops require extra buffer from user.

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

- **Input Tensor**: \([M, K]\)  or \([BS, M, K]\)
- **Weight Tensor**: \([K, N]\) or \([BS, K, N]\)
- **Output Tensor**: \([M, N]\) or \([BS, M, N]\)
```cpp
tensor.set_size({M, K});
tensor.set_size({BS, M, K}); 
```
---

#### Data Type
Indicates the precision of the values stored in the tensor.

- **FP32**:*(Default)* 32-bit floating point
- **BF16**: 16-bit Brain Floating Point
- **INT8/S8/U8**: 8-bit signed/unsigned integer (for quantization)
- **S4**: 4-bit signed integer (for WOQ, packed 2 values per byte)
```cpp
tensor.set_data_type(data_type_t::f32);
tensor.set_data_type(data_type_t::s4);  // For WOQ weights
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
tensor.set_order("ab");   //for 2D case
tensor.set_order("abc");  //for 3D case
```
- **Transposed**: *Aᵀ* ∈ ℝᴷˣᴹ
```cpp
tensor.set_order("ba");   //for 2D case
tensor.set_order("acb");  //for 3D case
```
---

#### Strides (Optional)
Defines how many memory elements to skip to move between elements along each dimension.

- Enables efficient access to non-contiguous data
- Useful for custom memory layouts and batched operations
- default stride = {K,1} if not passed for 2D tensor of size {M,K}
- default stride = {MxK,K,1} if not passed for 3D tensor of size {BS,M,K}
```cpp
tensor.set_stride({stride_m, stride_k});
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
              .set_stride({stride_M, stride_K}) // Define strides
              .set_storage()                  // Allocate storage
              .create();
```
---
Tensor can be created in two ways:
1. Direct Tensor creation (Fine grained control over attributes)
```cpp
auto tensor = tensor_t()
              .set_name("example_tensor")       // Set tensor name
              .set_size({M, K})                // Define tensor dimensions, can be {BS, M, K} for batched tensor
              .set_data_type(data_type::f32)   // Set data type
              .set_stride({stride_1, stride_2}) // Define strides
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
  - Bias: Uniform tensor with dimensions `{1, MATMUL_N}`

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

    // Create a bias tensor with dimensions [1, N], data type float32,
    // initialized with uniform values of -10.0, and named "bias"
    auto bias = tensor_factory.uniform_tensor({1, MATMUL_N},
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

    // Create a bias tensor with shape {1, N}, data type f32
    auto bias = tensor_factory.uniform_tensor({1, MATMUL_N},
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


### 3. batch_matmul_relu_bf16_kernel_example

This example showcases an advanced operation combining batched matrix multiplication with ReLU operation for BFloat16 datatype.

**Key Components**

- **Post-Operations**  
  - Applies the ReLU operation on the result.

- **Execution**  
  - Involves setting multiple input tensors corresponding to the operations defined in the context.

```cpp
int batch_matmul_relu_bf16_kernel_example() {
  try {
    status_t status;
    tensor_factory_t tensor_factory;
    // Create a weights tensor with shape {BS, K, N}, data type bf16, filled with uniform values
    auto weights = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_K, BATCH_MATMUL_N},
                   data_type_t::bf16,
                   1.0, "weights");
    // Create a bias tensor with shape {1, 1, N}, data type f32
    auto bias    = tensor_factory.uniform_tensor({1, 1, BATCH_MATMUL_N},
                   data_type_t::f32,
                   3.0, "bias");

    // Define a ReLU post-operation to be applied after matrix multiplication
    auto relu_post_op = post_op_t{post_op_type_t::relu};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(relu_post_op)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch_matmul context creation failed");
      return NOT_OK;
    }

    //define batch_matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_bf16_operator")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error(" operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    // Create an input tensor with dimensions [BS, M, K], data type BFloat16
    auto input_tensor = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::bf16,
                        1.0, "matmul_input");

    // Create an output tensor with dimensions [BS, M, N], data type BFloat16
    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::bf16,
                         "matmul_output");

    // Set inputs and outputs for the operator, force aocl-blis kernel and execute
    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .set_forced_kernel("aocl_blis")
             .execute();

    // Log the result of execution
    if (status == status_t::success) {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution successful.");
    }
    else {
      testlog_info("operator ", batch_matmul_operator.get_name(),
                   " execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return NOT_OK;
}
```


### 4. batch_matmul_inp2d_relu_f32_kernel_example

This example showcases an advanced operation combining batched matrix multiplication with 2D input_tensor and ReLU postop for float32 datatype.

**Key Components**

- **Post-Operations**  
  - Applies the ReLU operation on the result.

- **Execution**  
  - Involves setting multiple input tensors corresponding to the operations defined in the context and performing batched Matrix Mulitplication with 2D Input and 3D Weights.

```cpp
int batch_matmul_inp2d_relu_f32_kernel_example() {
  try {
    status_t status;
    tensor_factory_t tensor_factory;
    // Create a weights 3D tensor with shape {BS, K, N}, data type float32, filled with uniform values
    auto weights = tensor_factory.uniform_tensor({BATCH_SIZE, BATCH_MATMUL_K, BATCH_MATMUL_N},
                   data_type_t::f32,
                   1.0, "weights");

    // Create a bias tensor with shape {1, 1, N}, data type float32
    auto bias    = tensor_factory.uniform_tensor({1, 1, BATCH_MATMUL_N},
                   data_type_t::f32,
                   -10.0, "bias");

    // Define a ReLU post-operation to be applied after matrix multiplication
    auto relu_post_op = post_op_t{post_op_type_t::relu};

    //define batch_matmul context
    auto batch_matmul_context = matmul_context_t()
                                .set_param("weights", weights)
                                .set_param("bias", bias)
                                .set_post_op(relu_post_op)
                                .create();

    if (! batch_matmul_context.check()) {
      testlog_error("batch_matmul context creation failed");
      return NOT_OK;
    }

    //define batch_matmul operator
    auto batch_matmul_operator = matmul_operator_t()
                                 .set_name("batch_matmul_f32")
                                 .set_context(batch_matmul_context)
                                 .create();

    if (! batch_matmul_operator.check()) {
      testlog_error(" operator ", batch_matmul_operator.get_name(),
                    " creation failed.");
      return NOT_OK;
    }

    // Create an output 2D tensor with dimensions [M, K], data type float32
    auto input_tensor = tensor_factory.uniform_tensor({BATCH_MATMUL_M, BATCH_MATMUL_K},
                        data_type_t::f32,
                        1.0, "matmul_input");
    
    // Create an output 3D tensor with dimensions [BS, M, N], data type float32
    auto output_tensor = tensor_factory.zero_tensor({BATCH_SIZE, BATCH_MATMUL_M, BATCH_MATMUL_N},
                         data_type_t::f32, "matmul_output");
    
    // Set inputs and outputs for the operator, force aocl-blis kernel and execute
    status = batch_matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .set_forced_kernel("aocl_blis")
             .execute();

    // Log the result of execution
    if (status == status_t::success) {
      testlog_info("<",batch_matmul_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",batch_matmul_operator.get_name(),">",
                    " operator execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}
```


### 5. matmul_woq_bf16_kernel_example

This example demonstrates Weight-Only Quantization (WOQ) with BF16 input and S4 (4-bit signed integer) weights, commonly used for efficient LLM inference.

**Key Components**

- **S4 Quantized Weights**
  - Weights are stored as 4-bit signed integers (S4), packed 2 values per byte
  - Per-channel scale tensor is used for dequantization

- **Post-Operation**
  - Applies GELU (erf variant) activation on the result.

- **Execution**
  - Creates S4 weights with associated scale tensor, performs matrix multiplication with BF16 input.

```cpp
int matmul_woq_bf16_kernel_example() {
  testlog_info("**WOQ matmul operator bf16s4 kernel example.");
  try {
    status_t status;
    tensor_factory_t tensor_factory;

    // Create weight scale tensor (per-channel: {1, N})
    // Scale is used to dequantize S4 weights during computation
    auto wei_scales = tensor_factory.uniform_tensor({1, MATMUL_N},
                      data_type_t::f32,
                      0.25, "scale tensor");

    // Create S4 quantized weight tensor with associated scale
    // S4 weights are packed: 2 x 4-bit values per byte, range [-8, 7]
    auto weights = tensor_factory.uniform_tensor({MATMUL_K, MATMUL_N},
                   data_type_t::s4,
                   1.0, "weights", wei_scales);

    // Create bias tensor
    auto bias = tensor_factory.uniform_tensor({1, MATMUL_N},
                data_type_t::f32,
                5.0, "bias");

    // Define GELU post-operation
    auto gelu_post_op = post_op_t{post_op_type_t::gelu_erf};

    // Define matmul context with WOQ weights
    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights)
                          .set_param("bias", bias)
                          .set_post_op(gelu_post_op)
                          .create();

    if (!matmul_context.check()) {
      testlog_error("matmul context creation failed");
      return NOT_OK;
    }

    // Define matmul operator
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul_bf16s4")
                           .set_context(matmul_context)
                           .create();

    if (matmul_operator.is_bad_object()) {
      testlog_error(" operator ", matmul_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    // Create BF16 input tensor
    auto input_tensor = tensor_factory.uniform_tensor({MATMUL_M, MATMUL_K},
                        data_type_t::bf16,
                        1.0, "matmul_input");

    // Create FP32 output tensor
    auto output_tensor = tensor_factory.zero_tensor({MATMUL_M, MATMUL_N},
                         data_type_t::f32, "matmul_output");

    // Execute WOQ matmul
    status = matmul_operator
             .set_input("matmul_input", input_tensor)
             .set_output("matmul_output", output_tensor)
             .execute();

    if (status == status_t::success) {
      testlog_info("<", matmul_operator.get_name(), ">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<", matmul_operator.get_name(), ">",
                    " operator execution failed.");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}
```

**Key Points for WOQ:**
- **S4 Weight Tensor**: Created with `data_type_t::s4` and an associated scale tensor.
- **Per-channel Scale**: Scale tensor with dimensions `{1, N}` provides one scale value per output channel.
- **Dequantization**: During computation, S4 weights are dequantized using the formula: `W_dequant = scale * W_s4`.
- **BF16 Input**: Source activations remain in BFloat16 for accuracy.
- **Post-ops Support**: WOQ MatMul supports the same post-operations as regular MatMul (ReLU, GELU, SiLU, binary ops, etc.).

## Parameter Naming Convention
**Important:** The string identifiers used in .set_param(), .set_input(), and .set_output() are fixed and must not be changed. These names are internally mapped and executed by the operator implementation.

Required Identifers:

- .set_param("weights", ...) → must use "weights"
- .set_param("bias", ...) → must use "bias"
- .set_input("matmul_input", ...) → must use "matmul_input"
- .set_output("matmul_output", ...) → must use "matmul_output"

Changing these names will result in incorrect behavior or operator failure.

## Runtime Variables
To achieve optimal performance for the MatMul operator, you can configure runtime variables using either a JSON configuration file or environment variables.

### Supported Matmul Kernels
| Algo |       Kernel       |
|------|--------------------|
| 1    | aocl_blis_blocked  |
| 2    | onednn_blocked     |
| 3    | -NA-               |
| 4    | aocl_blis          |
| 5    | onednn             |
| 6    | -NA-               |
| 7    | batched_sgemm      |
| 8    | -NA-               |
| 9    | reference          |

### Configuration methods

#### 1. JSON Configuration
Specify the desired kernel in your JSON config file:
```json
{
  "matmul" : {
      "kernel" : "<kernel name>"
  }
}
```
Example: `"kernel": "aocl_blis_blocked"`

#### 2. Environment Variable
Set the environment variable before running your application:
```
export ZENDNNL_MATMUL_ALGO = <algo>
```
Example: `export ZENDNNL_MATMUL_ALGO = 1` (for aocl_blis_blocked)

- Note: Runtime variable can be set via either method, precedence is given to JSON configuration if both are provided.


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