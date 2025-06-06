
# MatMul Operator

## Overview

This section provides a high-level overview of matrix multiplication (`matmul`) operations with support for FP32 and BF16 data types, bias addition, and a range of post-operations including activations (ReLU, Sigmoid, Tanh, GELU, SiLU) and binary ops (Add, Mul, Mul+Add). The support matrix summarizes valid combinations of source, weight, and output data types along with supported operations. Practical examples from `matmul_example.cpp` demonstrate these configurations, such as `matmul_relu_f32_kernel_example` and `matmul_mul_silu_mul_f32_kernel_example`, which performs FP32 matmul with activation and binary post-op.


## General MatMul Operation

Let:

- \f$A \in \mathbb{R}^{M \times K}\f$  
- \f$B \in \mathbb{R}^{K \times N}\f$  
- \f$\text{Bias} \in \mathbb{R}^{1 \times N}\f$ 
- \f$\text{Activation}(x)\f$: optional activation function (e.g., ReLU, GELU)  
- \f$\text{BinaryOp}(x, y)\f$: optional binary post-operation (e.g., element-wise add/mul with another matrix)  
- \f$D \in \mathbb{R}^{M \times N}\f$: optional second operand for binary op  
- \f$C \in \mathbb{R}^{M \times N}\f$: the result



The computation can be expressed as:

$$
C = \text{BinaryOp}(\text{Activation}(A \cdot B + \text{Bias}), D)
$$


## Step-by-Step Operation

1. **Matrix Multiplication**:
   $$
   Z = A \times B
   $$

2. **Bias Addition (if present)**:
   $$
   Z = Z + \text{Bias}
   $$

3. **Activation Function (if present)**:
   $$
   Z = \text{Activation}(Z)
   $$

4. **Binary Post-Op (if present)**:
   $$
   Z = \text{BinaryOp}(Z, D)
   $$

5. **Store Result**:
   $$
   C = Z
   $$



## Example with ReLU and Add Post-Op

If using ReLU as activation and element-wise addition with matrix \( D \):

$$
C = \text{ReLU}(A \times B + \text{Bias}) + D
$$




# MatMul Support Table

This table outlines the support for MatMul operations with various data types, bias, and post-operations.

| Src Data Type | Weight Data Type | Output Data Type | Bias | Activation | Binary Post-Op |
|---------------|------------------|------------------|------|------------|----------------|
| FP32          | FP32             | FP32             | Yes  | ReLU       | Add            |
| BF16          | BF16             | FP32, BF16       | Yes  | Sigmoid    | Mul            |
|               |                  |                  |      | Tanh       | Mul+Add        |
|               |                  |                  |      | GELU (erf) |                |
|               |                  |                  |      | GELU (tanh)|                |
|               |                  |                  |      | SiLU       |                |
|               |                  |                  |      | SiLU+Mul   |                |



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

    auto binary_mul_po   = post_op_t{binary_mul};                     // First binary_mul
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
