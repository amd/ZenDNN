
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNN Architecture

## Overview
ZenDNN is a high-performance CPU inference library designed to accelerate deep learning workloads. It provides optimized implementations for several key operations. The architecture is modular, extensible, and integrates seamlessly with popular deep learning frameworks. Its layered architecture ensures flexibility, extensibility, and high performance across a variety of hardware platforms and use cases.


<img src="images/zendnnl_architecture.png" alt="ZenDNN Architecture" width="500" height="600"/>


## 1. Frontend Layer
The **Frontend** is where model developers define their neural networks using high-level APIs provided by frameworks such as **PyTorch**, **TensorFlow**, **Llama.cpp**. These frameworks handle model construction, training, and inference logic. ML operators (e.g. MatMul, EmbeddingBag, Reorder) are defined at this layer. When a model is executed, the frontend translates these operations into a format that ZenDNN can understand and dispatches them to the ZenDNN Library below.

## 2. Regular and Low Overhead API Layer
The **Regular and Low Overhead API Layer** is the critical interface that connects the frontend (PyTorch, TensorFlow, Llama.cpp, etc.) with the optimized computational backend of ZenDNN. It exposes two paths:

- **Regular API path:** A **modularized, object-oriented approach** that is extensible and framework-agnostic. It ensures that ZenDNN can be easily integrated into various ecosystems without requiring deep changes in the framework internals, and enables rapid experimentation and deployment of new operators and optimizations.
- **Low Overhead API (LowOHA) path:** **Lightweight** and minimal; it minimizes API overhead and is **critical for small GEMM-heavy workloads** such as Scaled Dot-Product Attention (SDPA) and Batched Matrix Multiply (BMM), where per-call overhead can dominate runtime.

### Regular API: Features

#### 2.1. Tensor Creation and Management
- Accepts input and output buffers from the framework.
- Wraps them into ZenDNN-compatible tensor objects.
- Handles metadata such as shape, data type (FP32, BF16, INT8), and memory layout (blocked and non blocked).
- Ensures zero-copy or minimal-copy data handling to reduce overhead.

#### 2.2. Operator Registration and Dispatch
- Maintains a registry of supported operators (Example: MatMul, Fused MatMul, Reorder, etc.).
- Maps framework-level operations to ZenDNN core implementations.
- Combines multiple operations into a single kernel to reduce memory bandwidth and improve cache locality.

#### 2.3. Precision Control
- Allows frameworks to specify the desired precision for inference:
  - **FP32**: Full precision for accuracy-sensitive tasks.
  - **BF16**: Balanced precision for performance and accuracy.
  - **INT8**: Quantized precision for reduced memory and faster inference.

#### 2.4. Execution Context Management
- Initializes and manages execution contexts (Example: constant buffers, post ops, etc. )
- Provides hooks for profiling and logging.

### Low Overhead API (LowOHA)
The Low Overhead API path provides direct, function-based APIs that bypass the regular operator factory and context setup. It is optimized for latency-sensitive inference and small GEMM-heavy workloads (e.g. SDPA, BMM), where minimizing per-call overhead is essential. Supported operations and their documentation are listed in [Low Overhead API (LowOHA): Supported Operations](#low-overhead-api-lowoha-supported-operations).


## 3. ZenDNN Library
The ZenDNN Library is the core layer that implements the APIs and orchestrates execution. It is built around the following:

### Multi-Backend Approach
ZenDNN uses a **multi-backend approach** that **dynamically selects the best backend based on problem size**. The Decision Tree, Auto Tuner, and runtime heuristics work together to choose among native kernels and third-party libraries (AOCL, OneDNN, LibXSMM, FBGEMM) for each operation.

### Core Modules
- **Decision Tree (DT):** Drives backend and kernel selection based on problem characteristics and heuristics.
- **AutoTuner:** Automatically selects the best algorithm and backend for the current workload and hardware, based on TBP (Time Based Profiling).
- **Caching (LRU + Reorder):** LRU cache and cached reorder operations reduce redundant work and improve data locality.
- **Threading Strategies:**
  - Batch, Table and Hybrid for EmbeddingBag Operation.
- **Parallel Primitive:** Enables scalable parallel execution.

## 4. Third Party Libraries
ZenDNN leverages several low-level libraries to provide foundational building blocks for performance-critical operations:

- **AOCL DLP:** Optimized GEMM for AMD CPUs.
- **FBGEMM:** A low-precision, high-performance matrix multiplication and embedding bag library.
- **OneDNN:** Intel's deep learning primitives for x86 CPUs.
- **LibXSMM:** Specialized in small matrix multiplications (Example: 64x64 or smaller).

---

The following sections describe the design and execution model in more detail: the core building blocks of the Regular API, the supported Low Overhead API operations, the hardware layer, execution flows for both API paths, and design principles.


## Core Design Principles: Regular API
The **Regular API** is built on four core building blocks. These are the design elements of the ZenDNN core used by the regular path:

- **Tensor:** Manages data layout, memory, and shape transformations.
- **Context:** Maintains constant model parameters, post op operation, runtime settings like threading and logging and profiling information.
- **Operator:** Defines high-level operations like MatMul, Reorder, etc.
- **Kernel:** Implements low-level, hardware-optimized routines for each operator.


<img src="images/basic_concepts.png" alt="ZenDNN Basic Modules" width="800"/>

### Tensor
A **Tensor** is a fundamental building block in deep learning libraries. It represents multi-dimensional arrays of data. In the context of CPU inference, tensors are used to store and manipulate data efficiently.

### Key Components of a Tensor:
1. **Tensor Metadata**:
   - **Dimensions**: Specifies the shape of the tensor (Example: 2x3 matrix).
   - **Sizes**: Indicates the size of each dimension.
   - **Stride**: Defines the step size to move from one element to the next in memory.
   - **Data Type**: Specifies the type of data stored (Example: float32, int8).

2. **Tensor Quantization Data**:
   - **Scale**: A factor used to map the tensor values to a quantized range.
   - **Zero Point**: An offset used in quantization to map zero in the original data to a quantized value.

3. **Memory Buffer**:
   - A contiguous block of memory allocated to store the tensor data.

### Use Cases:
- **Input Data**: Tensors are used to represent input data fed into the neural network.
- **Intermediate Results**: During computation, tensors store intermediate results.
- **Output Data**: The final output of the neural network is stored in tensors.

### Context
The **Context** of an operator encompasses all the information required to perform its computations. Given an operator, its parameter tensors (for example, weight and bias) and any other parameters (for example, element-wise post-ops or an embedding bag algorithm such as add, mean, or max) together form the operator context.

### Key Components of Context:
1. **Parameter Tensors**:
   - **Weight**: Represents the learned weights of the neural network.
   - **Bias**: Represents the bias values added to the weighted sum.

2. **Additional Parameters**:
   - **Element-wise Post-ops**: Operations applied element-wise after the main computation (Example: activation functions).
   - **Embedding Bag Algorithm**: Algorithms like add, mean, or max used in embedding layers.

### Use Cases:
- **Operator Computation**: Context provides all the necessary data for an operator to perform its computation.
- **Optimization**: Context helps in optimizing the computation by providing relevant parameters.

### Operator
An **Operator** is a function or a set of functions that perform computations on tensors. An operator in a computational graph is a node that takes input tensors, performs some computations on them, and produces output tensors. An operator can implement a simple computation on input tensors (add, concat...), computations involving other tensors acting as operator parameters (matrix multiplication with weight and bias as operator parameters), or a complex subgraph (attention layer in LLMs).

### Key Components of an Operator:
1. **Simple Computations**:
   - **Matrix Multiplication**: Performs multiplication of input tensors with weight and bias as parameters.
   - **Element-wise Operations**: Applies operations like activations, addition, subtraction, etc., on tensors.

2. **Complex Subgraphs**:
   - **Attention Layer**: Implements attention mechanisms used in large language models (LLMs).

### Operator Type

#### Fused MatMul
Fused Matrix Multiplication (Fused MatMul) is an optimized operation that combines multiple matrix multiplications and element-wise operations into a single kernel. This reduces memory bandwidth requirements and improves computational efficiency.

##### Key Components:
1. **Matrix Multiplication**: Performs the multiplication of input tensors.
2. **Element-wise Operations**: Applies operations like activations, addition, subtraction, etc., in a fused manner.

##### Use Cases:
- **Neural Network Layers**: Commonly used in fully connected layers and transformer models.
- **Performance Optimization**: Reduces the number of memory accesses and improves cache utilization.

#### Reorder
Reorder operation changes the memory layout of a tensor to improve data locality and access patterns. This is crucial for optimizing performance on different hardware architectures.

##### Key Components:
1. **Memory Layout Transformation**: Changes the order of data storage in memory.
2. **Data Locality**: Improves access patterns for subsequent operations.

##### Use Cases:
- **Data Preprocessing**: Reorders data to match the expected input format of specific kernels.
- **Performance Optimization**: Enhances cache utilization and reduces memory access latency.

#### Embedding Bag
Embedding Bag operation is used to look up embeddings for a set of indices and combine them using a specified reduction algorithm (Example: sum, mean, max).

##### Key Components:
1. **Embedding Lookup**: Retrieves embeddings for given indices.
2. **Reduction Algorithm**: Combines the retrieved embeddings using a specified method.

##### Use Cases:
- **Natural Language Processing**: Commonly used in models for handling categorical data, such as word embeddings.
- **Recommendation Systems**: Utilized for representing user and item features.

### Kernel
A **Kernel** is an implementation of an operator. Kernels ensure efficient execution of operators on different architectures. An operator can have multiple kernels depending on machine ISA, problem size, backend and quantization level.

### Key Components of a Kernel:
1. **Machine ISA**:
   - **Instruction Set Architecture**: Kernels are optimized for specific ISAs (Example: AVX, SSE).

2. **Problem Size**:
   - **Small vs. Large**: Kernels are tailored to handle different problem sizes efficiently.

3. **Backend and Quantization Level**:
   - **Backend**: Specifies the computational backend (Example: CPU, GPU).
   - **Quantization Level**: Indicates the level of quantization applied to the data.

### Use Cases
- **Performance Optimization**: Kernels ensure that operators run efficiently on the target hardware.
- **Scalability**: Kernels allow operators to scale across different problem sizes and hardware configurations.

### Interaction Between Main Classes

Understanding how **Tensor**, **Context**, **Operator**, and **Kernel** interact is essential for designing efficient and modular deep learning systems.

### Data Flow Overview

1. **Input Tensor Creation**:
   - The inference process begins with the creation of input tensors that hold the raw data (Example: images, text embeddings).

2. **Context Initialization**:
   - A context is initialized with parameters such as weights, biases, and configuration settings (Example: activation functions, quantization details).

3. **Operator Execution**:
   - The operator receives the input tensor and context. It validates the inputs and selects the appropriate kernel based on the data type, hardware, and problem size.

4. **Kernel Invocation**:
   - The selected kernel performs the actual computation, leveraging hardware-specific optimizations (Example: AVX512 instructions on Intel CPUs).

5. **Output Tensor Generation**:
   - The result of the kernel execution is stored in an output tensor, which can be passed to the next operator in the computation graph.



## Low Overhead API (LowOHA): Supported Operations
The Low Overhead API path exposes direct, Low Overhead APIs for specific operations. Each operation has dedicated documentation:

| Operation | Description | Documentation |
|-----------|-------------|---------------|
| **Group MatMul** | Multiple independent GEMMs in one call via `group_matmul_direct`; sequential chaining or parallel execution; optional MoE weighted-reduce post-op; optional gated activation post-op (silu_and_mul, gelu_and_mul, swiglu_oai_mul) for fused gate+up projections. | [lowoha_group_matmul_operator.md](operator/lowoha_group_matmul_operator.md) |
| **MatMul / Batched MatMul** | Direct matrix multiplication and batched matmul with weight caching, fused post-ops; latency-sensitive inference. | [lowoha_matmul_operator.md](operator/lowoha_matmul_operator.md) |
| **Reorder** | Data type conversion and reorder (e.g. BF16/FP32/INT8); quantization and dequantization. | [lowoha_reorder_operator.md](operator/lowoha_reorder_operator.md) |
| **Embedding Bag** | Low overhead embedding bag and group embedding bag. | [embedding_bag_operator.md](operator/embedding_bag_operator.md) |
| **Conv2D** | Low overhead 2D convolution. | [lowoha_conv2d_operator.md](operator/lowoha_conv2d_operator.md) |
| **Pooling** | Low overhead pooling operations. | [lowoha_pooling_operator.md](operator/lowoha_pooling_operator.md) |
| **Softmax** | Low overhead softmax. | [lowoha_softmax_operator.md](operator/lowoha_softmax_operator.md) |

## 5. Hardware Layer
ZenDNN is engineered to extract maximum performance from **general-purpose CPUs**, making it ideal for server-side inference, edge computing, and CPU-only environments. This layer is where all computations are ultimately executed, and its efficiency directly impacts the overall throughput and latency of deep learning models.

### Key Optimization Strategies

#### 5.1 Instruction Set Utilization
ZenDNN is optimized to leverage modern CPU instruction sets that enable vectorized and parallel computation:
- **AVX2 (Advanced Vector Extensions 2):** Widely supported on modern x86 CPUs, enables 256-bit SIMD operations.
- **AVX-512:** Offers 512-bit SIMD operations, allowing more data to be processed per instruction cycle. Ideal for FP32 and BF16 workloads.

These instruction sets are detected at runtime, and ZenDNN dynamically selects the best kernel path based on the available hardware.

#### 5.2 Multi-threading
ZenDNN supports parallel execution using:
- **OpenMP:** A widely-used API for multi-threaded programming in C/C++.

This allows ZenDNN to scale across multiple CPU cores, improving throughput for batch inference and large models.

#### 5.3 Cache Optimization
ZenDNN kernels are designed to:
- Maximize data reuse within L1/L2/L3 caches.
- Use blocking and tiling strategies to reduce cache misses.
- Align memory access patterns with CPU prefetching behavior.


## Execution Flow
ZenDNN supports two execution paths. The flow differs depending on whether the frontend uses the **Regular API** or the **Low Overhead API (LowOHA)**.

### Regular API path
For the regular path, a tensor moves through the ZenDNN stack as follows:

1. **Input Tensor** is passed from the frontend (Example: PyTorch).
2. The **Regular API** receives the tensor and determines the appropriate operator (creates context and operator).
3. **ZenDNN Core** processes the tensor using the selected operator and kernel.
4. The **Kernel** invokes routines from **Third Party Libraries** (Example: AOCL, OneDNN).
5. Computation is executed on the **CPU**.
6. The **Output Tensor** is returned back to the frontend.

```
Input Tensor (from Frontend)
          ↓
   Regular API
(Receives tensor, creates context and operator)
          ↓
      ZenDNN Core
(Processes tensor using operator & kernel)
          ↓
   Third Party Libraries
(Invokes optimized routines: AOCL, OneDNN, etc.)
          ↓
         CPU
(Performs computation)
          ↓
     Output Tensor
(Returned to Frontend)
```

### Low Overhead API (LowOHA) path
For the Low Overhead API path, execution is more direct to minimize per-call overhead:

1. **Input** is passed from the frontend to a **direct Low Overhead API** (e.g. `matmul_direct`, `group_matmul_direct`).
2. **Backend selection** (e.g. via Auto Tuner or Decision Tree) chooses the best backend and native kernel for the problem size.
3. **Native kernel** or **Third Party Libraries** (AOCL, LibXSMM, OneDNN, FBGEMM) perform the computation.
4. Computation is executed on the **CPU**.
5. **Output** is returned to the frontend.

```
Input (from Frontend)
          ↓
   Low Overhead API (direct call)
          ↓
   Backend selection (Auto Tuner / Decision Tree)
          ↓
   Native Kernel or Third Party Libraries
          ↓
         CPU
          ↓
     Output (to Frontend)
```


## Design Principles

#### 1. Modularity
- Each class is designed to be independent and reusable.
- Operators can be swapped or extended without modifying the core tensor or context logic.

#### 2. Hardware Abstraction
- Kernels abstract away hardware-specific details, allowing operators to remain platform-agnostic.

#### 3. Extensibility
- New operators and kernels can be added with minimal changes to the existing framework, supporting rapid prototyping and deployment.

#### 4. Performance Optimization
- Leverages the best available kernel for each scenario.

#### 5. User Transparency
- The complexity of backend selection and tensor conversion is hidden from the user, ensuring a clean and consistent interface.