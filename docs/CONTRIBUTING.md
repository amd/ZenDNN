
# Adding New Operator

This document serves as a comprehensive guide for developers looking to integrate a new operator into **ZenDNN***, AMD’s CPU-optimized inference library. We use the `Sample Operator` as a reference template, demonstrating how to build and plug in a custom operator within the modular architecture of ZenDNN*.

> **Audience**: This guide is intended for C++ developers familiar with deep learning libraries and AMD CPU architectures.


## ZenDNN* Main Components

### Tensor
A **Tensor** is a multi-dimensional data structure and a foundational element of computation in ZenDNN*.

**Components**:
- **Metadata**: Dimensions, sizes, strides, data type.
- **Quantization**: Scale and zero-point for INT8 inference.
- **Memory**: Allocated buffer to hold data.

**Roles**:
- Carry input data, intermediate results, and final outputs.

### Context
A **Context** defines an operator's runtime state, encapsulating everything needed for computation.

**Contains**:
- Weight and bias tensors.
- Element-wise post-ops (e.g., ReLU).
- Algorithm configuration (e.g., reduction mode in embedding bag).

### Operator
An **Operator** performs tensor computations using a selected kernel. Operators are building blocks for neural network layers.

**Examples**:
- Fused MatMul: Combines MatMul with activation.
- Reorder: Optimizes memory access pattern.
- Embedding Bag: Lookup + reduction (sum, mean, max).

### Kernel
A **Kernel** is a low-level function that executes the core logic of an operator, optimized for specific ISAs (e.g., AVX512).

**Factors**:
- Instruction Set Architecture (ISA).
- Input tensor data type.
- Problem size.


## 1. Overview

The Sample Operator in ZenDNN* is a template for creating new operators. It includes the following components:

- Operator interface (`sample_operator.hpp` and `sample_operator.cpp`)
- Operator context (`sample_context.hpp` and `sample_context.cpp`)
- Kernel list (`sample_kernel_list.hpp`)
- Operator kernel (`sample_f32_avx512_kernel.hpp` and `sample_f32_avx512_kernel.cpp`)

This guide will help you replicate the structure and functionality of the Sample Operator to create your own custom operator.

## 2. Steps to Add a New Operator

### 2.1 Step 1: Create Folder
Navigate to the `zendnnl/src/operators` directory, and create a folder for your new operator:

```bash
mkdir src/operators/new_operator
```

### 2.2 Step 2: Add Required Files

Within `new_operator`, add:

| File Name              | Description |
|------------------------|-------------|
| `NewOp_operator.hpp`   | Declares the main operator class |
| `NewOp_operator.cpp`   | Implements the operator class |
| `NewOp_context.hpp`    | Declares context for the operator |
| `NewOp_context.cpp`    | Defines context implementation |
| `NewOp_kernel_list.hpp`| Lists supported kernels |
| `NewOp_kernel.hpp`     | Declares kernel interface |
| `NewOp_kernel.cpp`     | Implements the kernel logic |

### 2.3 Step 3: File Structure and Responsibilities

#### 2.3.1 NewOp_operator.hpp
Declares `new_operator_t`, defines validation and kernel selection.

```cpp
#ifndef _NEW_OPERATOR_HPP_
#define _NEW_OPERATOR_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator.hpp"
#include "NewOp_context.hpp"

namespace zendnnl {
namespace ops {

/** @class new_operator_t
 *  @brief A new operator class for demonstration and implementation.
 *
 * @par Synopsys
 *
 * Invokes a kernel based on the input data type. The kernel performs the
 * operation defined by this operator.
 *
 * @par Parameters, Inputs, Outputs
 *
 * The operator has the following parameters and input/outputs:
 * - Parameter(s)
 *   1. (mandatory) new_param  : An arbitrary tensor.
 * - Inputs
 *   1. (mandatory) new_input  : An arbitrary tensor.
 * - Output(s)
 *   1. (mandatory) new_output : An arbitrary tensor.
 */
class new_operator_t final : public operator_t<new_operator_t, new_context_t> {
public:
  using parent_type = operator_t<new_operator_t, new_context_t>;
protected:
  status_t validate() override;
  status_t kernel_factory() override;
};

} // namespace ops
namespace interface {
using new_operator_t = zendnnl::ops::new_operator_t;
} // namespace interface
} // namespace zendnnl

#endif
```

#### 2.3.2 NewOp_context.hpp

- Defines the context for the operator, including parameters and configurations.
- Example:

```cpp
#ifndef _NEW_OPERATOR_CONTEXT_HPP_
#define _NEW_OPERATOR_CONTEXT_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_context.hpp"

namespace zendnnl {
namespace ops {

/** @class new_context_t
 *  @brief Context for the new operator.
 */
class new_context_t : public operator_context_t {
public:
  new_context_t() = default;
  ~new_context_t() = default;

  void set_param(const std::string &key, const tensor_t &value) {
    params[key] = value;
  }
};

} // namespace ops
} // namespace zendnnl

#endif
```

#### 2.3.3 New_op_kernel_list.hpp

- Lists the kernels supported by the operator.

#### 2.3.4 New_op_operator.hpp

- Defines the operator interface and integrates it with the framework.

## 3. Integrate the New Operator

### 3.1 Update Build System

Add the new files to the build system (e.g., `CMakeLists.txt` or `Makefile`):

```cmake
add_library(NewOp
    src/operators/new_operator/New_op.cpp
    src/operators/new_operator/New_op_context.cpp
    src/operators/new_operator/New_op_operator.cpp
)
```

## 4. Example Usage

### 4.1 Example Code

```cpp
int new_op_example() {
    testlog_info("New operator example");
    try {
        tensor_factory_t tensor_factory;

        auto input_tensor = tensor_factory.uniform_tensor({ROWS, COLS},
                              data_type_t::f32,
                              1.0, "new_op_input");

        // Call to new operator
        new_operator_execute(input_tensor);
    }
    catch (const exception_t &ex) {
        std::cout << ex.what() << std::endl;
        return NOT_OK;
    }
    return OK;
}
```

### 4.2 Execution

- Compile and run the example:

```bash
./new_op_example
```

## 5. Testing

- Add unit tests for the new operator in the `tests` directory.
- Example:

```cpp
TEST(NewOpTest, Execute) {
    zendnnl::ops::NewOp op;
    op.execute();
    ASSERT_TRUE(true); // Replace with actual validation
}
```

Following these steps and using the Sample Operator as a reference, you can successfully add and integrate a new operator into ZenDNN*.


## System Flow Recap

```text
[Tensor] --> [Context] --> [Operator.validate()] --> [Kernel Selection] --> [Kernel Execution] --> [Output Tensor]
```


## Design Best Practices

| Principle       | Description |
|----------------|-------------|
| **Modularity** | Each component (Tensor, Context, Operator, Kernel) is pluggable |
| **ISA Awareness** | Kernels are ISA-specific (AVX2, AVX512) for optimal performance |
| **Extensibility** | Add new operators with minimal changes elsewhere |
| **Separation of Concerns** | Operator logic separate from kernel implementation |



## Summary

Following this guide and leveraging the modular architecture of ZenDNN*, you can rapidly prototype and deploy new operators optimized for AMD CPUs. Whether your target is a new quantized convolution or a custom transformer layer, ZenDNN*’s flexible operator/kernel structure makes it possible.

> Be sure to benchmark and validate new kernels for correctness and performance on AMD hardware.

>ZenDNN* : ZenDNN is currently undergoing a strategic re-architecture and refactoring to enhance performance, maintainability, and scalability.

