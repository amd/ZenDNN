
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNN Google Test Infrastructure

## **Overview of GTest**
Google Test (GTest) is a unit testing framework for C++ that provides a robust and flexible way to write and execute tests. In this project, GTest is used to validate the functionality and performance of ZenDNN operations. It supports parameterized tests, filtering, and detailed reporting, making it ideal for testing complex numerical computations.

## **Purpose and Audience**
The ZenDNN GTest infrastructure is designed for:
- **Developers**: To verify the correctness of ZenDNN operations during development.
- **Test Engineers**: To run automated tests and validate functionality across different configurations.
- **Researchers**: To experiment with numerical precision and post-operations in deep learning workloads.

This infrastructure ensures that ZenDNN operations meet expected accuracy standards under various conditions.

## **Flexible Configuration**
ZenDNN GTest provides flexibility in configuring tests through command-line arguments and code-level parameters. Users can:
- **Filter Tests**: Run specific test suites or cases using `--gtest_filter`.
- **Set Random Seed**: Provide a seed value for reproducible random test data generation.
- **Specify Post-Operations**: Apply post-operations like `relu`, `gelu_tanh`, etc., during matrix multiplication tests.
- **Backend Selection**: Choose specific computational backends using `--backend` parameter to control algorithm selection (e.g., `aocl_dlp`, `onednn`, `libxsmm`).
- **LOWOHA**: Enable or disable Low Overhead API using `--lowoha` parameter.
- **Thread Control**: Specify the number of threads for parallel execution using `--num_threads` parameter.
- **Input File Support**: Use `--input_file` with `--op` and optionally `--ndims` parameters to read test configurations from a file instead of random generation.

## **Configurable Parameters**
You can modify the following parameters in the source code (`gtest_main.cpp`):
- `MATMUL_F32_TOL`: Tolerance for floating-point precision in tests (default: `0.001`).
- `MATMUL_BF16_TOL`: Tolerance for BF16 precision in tests (default: `0.01`).
- F16 tests use the same tolerance as BF16 (`MATMUL_BF16_TOL`).
- `NORM_F32_TOL`: Tolerance for normalization F32 tests (default: `0.001`).
- `NORM_BF16_TOL`: Tolerance for normalization BF16 tests (default: `0.01`).
- `TEST_NUM`: Number of test cases to generate (default: `100`).

## **Matrix Dimension Ranges**

The following dimension ranges are used for randomly generated test cases (defined in `gtest_utils.hpp`):

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| **M** | 1 | 3000 | Number of rows in the output matrix |
| **K** | 1 | 3000 | Inner dimension (columns of A / rows of B) |
| **N** | 1 | 3000 | Number of columns in the output matrix |
| **Batch Size** | 1 | 256 | Batch size for batch matrix multiplication |

**Dimension Formula:**
```
dimension = MATMUL_SIZE_START + rand() % MATMUL_SIZE_END
```
Where `MATMUL_SIZE_START = 1` and `MATMUL_SIZE_END = 3000`.

## **Buffer Value Distribution**

Test tensors are initialized with uniformly distributed random values. The distribution ranges vary by data type and tensor role:

### **Input and Weight Tensors**

| Data Type | Distribution Range | Description |
|-----------|-------------------|-------------|
| **F32** | `[-2.0, 2.0]` | Standard floating-point tests |
| **BF16** | `[-2.0, 2.0]` | BFloat16 tests |
| **F16** | `[-2.0, 2.0]` | Half-precision (IEEE 754) tests |
| **S8** (INT8 weights) | `[-25.0, 25.0]` | Signed 8-bit integer quantized weights |
| **U8** (INT8 source) | `[0, 25.0]` | Unsigned 8-bit integer quantized input |
| **S4** (WOQ weights) | `[-8, 7]` | 4-bit signed integer for weight-only quantization |

### **Other Tensors**

| Tensor Type | Distribution Range | Description |
|-------------|-------------------|-------------|
| **Bias** | `[-2.0, 2.0]` | Bias tensor (F32 or BF16) |
| **Binary Post-op** | `[-2.0, 2.0]` | Binary add/mul tensors |
| **Output** | `[-2.0, 2.0]` | Initial output buffer values |
| **Quantization Scales** | `[-0.2, 0.2]` to `[-2.0, 2.0]` | Scale factors for quantized operations |
| **Zero Points** | Integer values | Zero points for asymmetric quantization |

### **Uniform Constant Value Tensors**

Some tensors are initialized with a single constant value (all elements set to the same value) using `uniform_tensor()` instead of random distribution:

| Tensor Type | Constant Value | Data Type | Description |
|-------------|---------------|-----------|-------------|
| **Weight Zero Point (WOQ)** | `0` | S8 | Zero point for S4 weight-only quantization |
| **Source Zero Point (INT8)** | `16` | S32 | Zero point for U8 asymmetric quantization |
| **Weight Zero Point (INT8)** | `16` | S32 | Zero point for per-tensor weight quantization |
| **Destination Zero Point** | `53` | S32 | Zero point for U8 output quantization |

### **Alpha and Beta Parameters**
- **Alpha**: Uniformly distributed in `[0.0, 10.0]`
- **Beta**: Uniformly distributed in `[0.0, 10.0]`
- **Note**: For LIBXSMM backends, alpha is fixed to `1.0` and beta is `0` or `1`.

## **Accuracy Validation**

ZenDNN GTest validates numerical accuracy by comparing optimized implementations against reference results using dynamic tolerance algorithms that adapt to matrix dimensions and data types.

### **Tolerance Calculation**

**BF16/F16 Operations:**
```
abs_bound = k * epsilon_bf16
allowed_error = abs_bound + rtol_bf16 * |reference_value|
```
> **Note:** F16 uses the same tolerance bounds as BF16 since both are low-precision 16-bit formats.

**F32 Operations:**
```
abs_bound = ((20 + log2(k)/4) * k + 15) * epsilon_f32
allowed_error = abs_bound + rtol_f32 * |reference_value|
```

Where `k` is the inner matrix dimension that determines accumulation error scaling.

**Relaxation Features:**
- **Zero-reference tolerance**: `8e-4` for near-zero reference values (`< 1e-6`)
- **Additional slack**: `2e-4` buffer for F32 operations

**Note**: Relaxation can be enabled via `ENABLE_F32_RELAXATION` macro in `gtest_utils.hpp` and is enabled by default for LIBXSMM backends.

### **Configuration**
Key tolerance parameters in `gtest_main.cpp`:
```cpp
const float epsilon_f32     = 1.19e-7;  // Base F32 tolerance
const float epsilon_bf16    = 9.76e-4;    // Base BF16 tolerance
const float rtol_f32        = 1e-5;    // F32 relative tolerance
const float rtol_bf16       = 1e-2;;    // BF16 relative tolerance
```

### **Adaptive Scaling**
- **Small matrices**: Tolerance based on base epsilon values
- **Large matrices**: Logarithmic scaling prevents excessive tolerance growth
- **Post-operations**: Additional margin (P=15) for accumulation errors

This ensures accuracy validation scales with computational complexity while maintaining precision standards across different backends and data types.


## **Supported Post-Operations**
The following post-operations are supported for matmul and its variations:
- `relu`
- `gelu_tanh`
- `gelu_erf`
- `sigmoid`
- `swish`
- `tanh`
- `binary_add`
- `binary_mul`

**Note:** If specified post-op is not from above mention list, then no post-op will be applied to operator.

## **Bias Support**
- **Default Behavior**: Bias is enabled by default for all matrix multiplication tests
- **Data Types**: Bias tensors support both F32 and BF16 data types, randomly selected during test execution

**Note**: GTest currently excludes bias operations for LIBXSMM BF16 backends

## **Directory Structure**

The `zendnnl/gtests/` directory contains all test-related files and subdirectories. Below is the structure:

```plaintext
gtests/
├── gtest_main.cpp           # Entry point for all tests.
├── test_matmul.cpp          # Matmul testsuite with different test cases.
├── test_batchmatmul.cpp     # Batch matmul testsuite with different test cases.
├── test_reorder.cpp         # Reorder testsuite (regular + LOWOHA quantization/dequantization).
├── test_embag.cpp           # Embedding bag testsuite with different test cases.
├── test_embedding.cpp       # Embedding testsuite with different test cases.
├── test_normalization.cpp   # Normalization testsuite with different test cases.
├── test_omp_api.cpp         # OpenMP thread-control utility testsuite (omp_thread_control.hpp).
└── gtest_utils.cpp/hpp      # Utility functions for tests.
```

## **Configure and build GTest with ZenDNN**
- Configure:
```bash
cmake -DZENDNNL_BUILD_GTEST=ON ..
```
- Build:
```bash
cmake --build .
```
## **Running Tests**

### **General Command Structure**
```bash
./install/gtests/gtests --gtest_filter=<TestSuite>/<TestCase>[/<Index>] --seed <Seed>  --postop <PostOp> --test <num_of_tests> --backend <Backend> --lowoha <true/false> --num_threads <num_threads>
```

### **Command Structure with Input File**
```bash
./install/gtests/gtests --gtest_filter=<TestSuite>/<TestCase>[/<Index>] --input_file <InputFile> --op <Operator> --ndims <Dimensions> --lowoha <true/false> --num_threads <num_threads>
```

## **Parameters**
1. **`--gtest_filter=<TestSuite>/<TestCase>`** (Optional):
   - Specifies the test suite and test case to run.
   - *Example*:
     - `Reorder/TestReorder.BF16_F32/*`: Runs all tests for BF16 reorder followed by Matrix (BF16 Input and FP32 Output).
     - `Matmul/TestMatmul.BF16_BF16/23`: Runs the 23rd test case for BF16 Input and BF16 Output with matrix multiplication.

2. **`--seed <Seed>`** (Optional):
   - Sets the seed based on the seed value for generating test data.
   - *Example*:
     - `1245`: Uses 1245 as the seed for randomization.

3. **`--postop <PostOp>`** (Optional):
   - Specifies the post-operation to apply during tests.

4. **`--test <num_of_tests>`** (Optional):
   - Specifies the number of tests to be run for test-suite.

5. **`--backend <Backend>`** (Optional):
   - Specifies the computational backend/algorithm to use.
   - *Supported backends*:
     - `aocl_dlp`
     - `aocl_dlp_blocked`
     - `onednn`
     - `onednn_blocked`
     - `libxsmm`
     - `libxsmm_blocked`
   - *Example*:
     - `aocl_dlp`: Uses AOCL DLP backend for computations

6. **`--lowoha <true/false>`** (Optional):
   - Specifies whether to use LOWOHA (Low Overhead API) mode.
   - *Example*:
     - `true`: Use Low Overhead API
     - `false`: Use Regular API
   - **Default behavior varies by test suite**:
     - **Reorder tests**: Random 50/50 selection between LOWOHA and regular reorder tests per instance
       - `--lowoha true`: Only LOWOHA reorder tests run (regular tests are skipped)
       - `--lowoha false`: Only regular reorder tests run (LOWOHA tests are skipped)
     - **Matmul/BatchMatmul tests**: Tests are automatically partitioned into thirds:
       - First third: LOWOHA off
       - Second third: LOWOHA on
       - Last third: LOWOHA on with LIBXSMM backend
     - **Embedding/EmbeddingBag tests**: Random 50/50 selection between LOWOHA on/off
     - **Normalization tests**: Always use LOWOHA API (normalization is a LOWOHA-only operator)
   - **Note**: For LIBXSMM backends, LOWOHA is always enabled by default

7. **`--input_file <InputFile>`** (Optional):
   - Specifies an input file containing test configurations for operators.
   - When provided, test inputs will be read from the file instead of random generation.
   - Must be used in conjunction with `--op` parameter.
   - *Example*:
     - `input.txt`: Reads test configurations from input.txt file

8. **`--op <Operator>`** (Optional):
   - Specifies the operator type when using input file mode.
   - Required when `--input_file` is provided to determine how to parse the file.
   - *Supported operators*:
     - `matmul`: Matrix multiplication operator
     - `reorder`: Reorder operator
   - *Example*:
     - `matmul`: Process matmul test configurations from input file

9. **`--ndims <Dimensions>`** (Optional):
   - Specifies the number of dimensions for the operator.
   - Required when `--input_file` is provided to distinguish between operation types.
   - **Required for batch matmul** (3D matmul) operations.
   - *Supported values*:
     - `2`: 2D matmul (standard matrix multiplication)
     - `3`: 3D matmul (batch matrix multiplication) - **required for batch matmul**
   - *Example*:
     - `2`: Process 2D matmul test configurations
     - `3`: Process batch matmul test configurations

10. **`--num_threads <num_threads>`** (Optional):
   - Specifies the number of threads to use for parallel execution.
   - *Example*:
     - `8`: Uses 8 threads for execution
   - *Default*: Random value is selected from available thread count

**Note:**
 - If **`<PostOp>`** parameter is not provided, gtest will pick postop randomly from supported post-ops.
 - If **`<Seed>`** parameter is not provided, gtest sets the seed based on timestamp for generating test data.
 - If **`<num_of_tests>`** parameter is not provided, gtest sets the number of tests to a default value i.e. 1000.
 - If **`<Backend>`** parameter is not provided, gtest will randomly select from available backends based on compilation flags.
 - If **`<lowoha>`** parameter is not provided:
   - **Reorder tests**: Random 50/50 selection between LOWOHA and regular reorder tests per instance
   - **Matmul/BatchMatmul tests**: Tests are partitioned to cover both LOWOHA and non-LOWOHA scenarios
   - **Embedding/EmbeddingBag tests**: Random 50/50 selection between LOWOHA on/off
 - If **`<num_threads>`** parameter is not provided, a random value is selected from available thread count.
 - If **`<InputFile>`**, **`<Operator>`**, or **`<Dimensions>`** are not provided, tests will use randomly generated parameters instead of reading from a file.
 - If no parameters are provided, It will run all available tests with seed sets based on timestamp and randomly selected postops and backends.

## **Input File Format**

When using `--input_file` parameter, the input file must follow specific formats based on the operator type:

### **Matmul (2D) Input File Format**
Each line should contain comma-separated values in the following order:
```
M,K,N,postOp,kernel,transA,transB,alpha,beta
```
**Example:**
```
128,256,512,relu,aocl_dlp,false,false,1.0,0.0
64,128,256,gelu_tanh,onednn,false,true,1.0,0.0
```

### **Batch Matmul (3D) Input File Format**
Each line should contain comma-separated values in the following order:
```
BS,M,K,N,postOp,kernel,transA,transB,alpha,beta
```
**Where BS is the batch size.**

**Example:**
```
32,128,256,512,relu,aocl_dlp,false,false,1.0,0.0
16,64,128,256,,libxsmm,false,true,1.0,0.0
```

### **Reorder Input File Format**
Each line should contain comma-separated values in the following order:
```
M,K,N,postOp,kernel,transA,transB,inplace_reorder
```
**Example:**
```
128,256,512,relu,aocl_dlp,false,false,false
64,128,256,gelu_tanh,onednn,false,true,true
```

**Field Descriptions:**
- **M, K, N**: Matrix dimensions
- **BS**: Batch size (for batch matmul only)
- **postOp**: Post-operation (relu, gelu_tanh, gelu_erf, sigmoid, swish, tanh, binary_add, binary_mul, or none)
- **kernel**: Backend algorithm (aocl_dlp, aocl_dlp_blocked, onednn, onednn_blocked, libxsmm, libxsmm_blocked)
- **transA, transB**: Transpose flags (0 or 1)
- **alpha, beta**: Scaling factors for matmul operations
- **inplace_reorder**: Reorder mode flag (0 or 1)

## **LOWOHA Reorder Tests**

LOWOHA reorder tests validate quantization, dequantization, and type conversion kernels using a **round-trip methodology**. Each test performs a forward operation followed by its inverse, and then compares the result against the original input to verify correctness.

### **Test Categories**

| Category | Test Cases | Description |
|----------|-----------|-------------|
| **Static Quantization** | `BF16_QUANT_DEQUANT`, `FP32_QUANT_DEQUANT` | Round-trip: Source → INT8 (S8/U8) → Source using user-provided scale/zp |
| **Strided Static Quantization** | `BF16_QUANT_DEQUANT_STRIDED`, `FP32_QUANT_DEQUANT_STRIDED` | Same as above but with strided (non-contiguous) source memory |
| **Type Conversion** | `FP32_BF16_CVT` | Round-trip: FP32 ↔ BF16 without scale/zp |
| **Scaled Type Conversion** | `FP32_BF16_CVT_SCALED` | Round-trip: FP32 ↔ BF16 with scale/zp |
| **Strided Scaled Conversion** | `FP32_BF16_CVT_STRIDED` | Same as scaled conversion but with strided source memory |
| **Dynamic Quantization** | `FP32_DYN_QUANT`, `BF16_DYN_QUANT` | Round-trip: Source → INT8 (S8/U8) → Source where scale/zp are computed by the kernel |

### **Round-Trip Methodology**

All LOWOHA reorder tests follow a three-step pattern:

1. **Forward pass**: Apply the operation (quantize, dequantize, or convert) on the source tensor to produce an intermediate result.
2. **Backward pass**: Apply the inverse operation on the intermediate result to reconstruct the original data type.
3. **Compare**: Compare the reconstructed output against the original input using a tolerance-based comparison.

### **Comparison Methods**

- **Quantization tests** (`*QUANT_DEQUANT*`, `*DYN_QUANT*`, `*CVT_SCALED*`, `*CVT_STRIDED*`): Use `compare_lowoha_quant_output`, which computes a scale-aware tolerance:
  ```
  tolerance = max_scale / 2  +  epsilon
  ```
  Where `epsilon = 0.03` for BF16 sources and `0.001` for FP32 sources. The `max_scale / 2` term accounts for the maximum rounding error introduced by quantization.

- **Type conversion tests** (`FP32_BF16_CVT`): Use `compare_lowoha_reorder_output` with fixed tolerances based on the output data type (BF16 tolerance for BF16, FP32 tolerance for FP32).

### **Randomized Test Parameters**

Each LOWOHA reorder test instance is parameterized with randomly generated values:

| Parameter | Range / Distribution | Description |
|-----------|---------------------|-------------|
| **Dimensionality** | 1D (20%), 2D (50%), 3D (30%) | Tensor dimensionality |
| **M** | [1, 3000] | Row dimension (fixed to 1 for 1D) |
| **N** | [1, 3000] | Column dimension |
| **Batch** | [2, 256] for 3D | Batch dimension |
| **Symmetric/Asymmetric** | 50/50 random | S8 (symmetric, no zp) or U8 (asymmetric, with zp) |
| **Granularity** | per-tensor (60-70%), per-channel (30%), per-group (10%) | Quantization granularity |
| **Strided padding** | [0, 16] elements | Row padding for strided tests (50% chance of zero padding) |
| **Threads** | [1, max_threads] | Number of OMP threads |

### **Quantization Granularities**

LOWOHA reorder tests support five quantization granularities:

| Granularity | Scale/ZP Shape (2D) | Scale/ZP Shape (3D) | Description |
|------------|--------------------|--------------------|-------------|
| **Per-tensor** | `{1, 1}` | `{1, 1, 1}` | Single scale/zp for all elements |
| **Per-channel-row** | `{M, 1}` | `{1, M, 1}` | One scale/zp per row (per-token) |
| **Per-channel-col** | `{1, N}` | `{1, 1, N}` | One scale/zp per column |
| **Per-group-row** | `{G, N}` | `{1, G, N}` | Groups along M dimension (M/G rows per group) |
| **Per-group-col** | `{M, G}` | `{1, M, G}` | Groups along N dimension (N/G cols per group) |

For 1D tensors, only per-tensor and per-channel granularities are supported.

## **Examples**

### Reorder Tests
 - The Reorder TestSuite contains both LOWOHA reorder tests and Regular reorder tests (Reorder + Matmul).
 - By default (no `--lowoha` flag), each test instance is randomly assigned as LOWOHA or regular (50/50).
 - Use `--lowoha true` to run only LOWOHA reorder tests, or `--lowoha false` to run only regular reorder tests.
 - **LOWOHA test cases**: `BF16_QUANT_DEQUANT`, `FP32_QUANT_DEQUANT`, `FP32_BF16_CVT`, `FP32_BF16_CVT_SCALED`, `BF16_QUANT_DEQUANT_STRIDED`, `FP32_QUANT_DEQUANT_STRIDED`, `FP32_BF16_CVT_STRIDED`, `FP32_DYN_QUANT`, `BF16_DYN_QUANT`
 - **Regular test cases**: `F32_F32`, `BF16_F32`, `BF16_BF16`, `F32`, `BF16`, `S8`, `F32_F32_Stride`, `BF16_F32_Stride`, `BF16_BF16_Stride`

#### LOWOHA Reorder Tests

1. Run all LOWOHA reorder tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.* --lowoha true
```
2. Run LOWOHA static quantization round-trip tests (BF16 and FP32):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_QUANT_DEQUANT/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_QUANT_DEQUANT/* --lowoha true
```
3. Run LOWOHA strided quantization round-trip tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_QUANT_DEQUANT_STRIDED/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_QUANT_DEQUANT_STRIDED/* --lowoha true
```
4. Run LOWOHA type conversion tests (FP32 ↔ BF16):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_BF16_CVT/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_BF16_CVT_SCALED/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_BF16_CVT_STRIDED/* --lowoha true
```
5. Run LOWOHA dynamic quantization tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.FP32_DYN_QUANT/* --lowoha true
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_DYN_QUANT/* --lowoha true
```
6. Run a specific LOWOHA test with fixed seed and thread count:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_QUANT_DEQUANT/0 --seed 42 --num_threads 8 --lowoha true
```

#### Regular Reorder Tests (Reorder + Matmul)

7. Run all regular reorder tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.* --lowoha false
```
8. Run FP32 reorder + matmul tests:
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.F32_F32/* --lowoha false
```
9. Run BF16 reorder + matmul tests (F32 output):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_F32/* --lowoha false
```
10. Run BF16 reorder + matmul tests (BF16 output):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_BF16/* --lowoha false
```

### Matmul Tests
 - Matmul TestSuite has nine testcases (F32_F32, BF16_F32, BF16_BF16, F16_F16, F16_F32, F32_F32_Stride, BF16_F32_Stride, BF16_BF16_Stride, F16_F16_Stride)

> **Note:** F16 tests (F16_F16, F16_F32, F16_F16_Stride) require **AVX512-FP16** or **AVX-NE-CONVERT** ISA support. On unsupported platforms, these tests are automatically skipped via `GTEST_SKIP()` with an informative message.

1. Run all BF16 Input, F32 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_F32/*
```
2. Run all BF16 Input, BF16 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/*
```
3. Run all F32 Input, F32 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32/*
```
4. Run F32_F32_Stride matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32_Stride/*
```
5. Run all F16 Input, F16 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F16_F16/*
```
6. Run all F16 Input, F32 Output matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F16_F32/*
```
7. Run F16_F16_Stride matmul tests:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F16_F16_Stride/*
```

### Embedding Bag Tests
 - Embedding Bag TestSuite has four testcases(F32_F32, F32_BF16, BF16_F32, BF16_BF16, INT8_F32, INT8_BF16, S4_F32, S4_BF16, U4_F32, U4_BF16)
1. Run all F32 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.F32_F32/*
```
2. Run all F32 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.F32_BF16/*
```
3. Run all BF16 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.BF16_F32/*
```
4. Run all BF16 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.BF16_BF16/*
```
5. Run all INT8 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.INT8_F32/*
```
6. Run all INT8 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.INT8_BF16/*
```
7. Run all S4 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.S4_F32/*
```
8. Run all S4 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.S4_BF16/*
```
9. Run all U4 Input, F32 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.U4_F32/*
```
10. Run all U4 Input, BF16 Output embedding bag tests:
``` bash
./install/gtests/gtests --gtest_filter=EmbeddingBag/TestEmbag.U4_BF16/*
```

### Embedding Tests
 - Embedding TestSuite has four testcases(F32_F32, F32_BF16, BF16_F32, BF16_BF16, INT8_F32, INT8_BF16, S4_F32, S4_BF16, U4_F32, U_BF16)
1. Run all F32 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.F32_F32/*
```
2. Run all F32 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.F32_BF16/*
```
3. Run all BF16 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.BF16_F32/*
```
4. Run all BF16 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.BF16_BF16/*
```
5. Run all INT8 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.INT8_F32/*
```
6. Run all INT8 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.INT8_BF16/*
```
7. Run all S4 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.S4_F32/*
```
8. Run all S4 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.S4_BF16/*
```
9. Run all U4 Input, F32 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.U4_F32/*
```
10. Run all U4 Input, BF16 Output embedding tests:
``` bash
./install/gtests/gtests --gtest_filter=Embedding/TestEmbedding.U4_BF16/*
```

### Normalization Tests
 - Normalization TestSuite has four testcases (F32_F32, BF16_BF16, BF16_F32, F32_BF16)
 - Supports four normalization types: LayerNorm, RMSNorm, FusedAddRMSNorm, BatchNorm
 - Test parameters (shape, norm_ndims, use_scale, use_shift, gamma/beta data types) are randomly generated
 - Validates native kernel output against the reference (scalar) kernel output

1. Run all F32 Input, F32 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.F32_F32/*
```
2. Run all BF16 Input, BF16 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.BF16_BF16/*
```
3. Run all BF16 Input, F32 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.BF16_F32/*
```
4. Run all F32 Input, BF16 Output normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/TestNormalization.F32_BF16/*
```
5. Run all normalization tests:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/*
```
6. Run normalization tests with 8 threads:
``` bash
./install/gtests/gtests --gtest_filter=Normalization/* --num_threads 8
```

### OMP API Tests
 - OMP API TestSuite validates the OpenMP thread-control utilities in `omp_thread_control.hpp`
 - Uses fixed `TEST_F` tests (not parameterized) since each test targets a specific code path or contract
 - Self-contained: does not depend on `gtest_utils.hpp` or randomized data from `gtest_main.cpp`
 - The `OmpApiTest` fixture manages OpenMP state (disables `omp_set_dynamic`, restores ICVs between tests)

#### Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| **resolve_num_threads** | 6 tests | Pure logic tests for the thread-count resolution function (auto, explicit, single-thread, boundary, single-core, negative) |
| **thread_guard::max_threads** | 3 tests | Singleton caching: stability across repeated calls, cache immunity to external ICV mutation, thread-safe static init |
| **thread_guard RAII** | 8 tests | Set/restore correctness: no-op elision, two-arg top-level, single-arg capture, per-task ICV inside parallel regions, loop pattern, nested guards, sequential guards, over-request restore |
| **Combined / Production** | 2 tests | Full operator entry-point pattern (`resolve_num_threads` + `thread_guard`), combined `thread_guard` + `scoped_active_levels` (group_gemm pattern) |
| **scoped_active_levels** | 2 tests | RAII correctness: no-op elision when desired equals current, set/restore of max-active-levels ICV |

#### Environment Setup

Set these environment variables before running OMP API tests to ensure deterministic behavior:
```bash
export OMP_NUM_THREADS=8
export OMP_MAX_ACTIVE_LEVELS=2
```

#### Running OMP API Tests

1. Run all OMP API tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.*'
```
2. Run only resolve_num_threads tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.ResolveNumThreads*'
```
3. Run only thread_guard RAII tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.*ThreadGuard*'
```
4. Run only scoped_active_levels tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.ScopedActiveLevels*'
```
5. Run only the production pattern and combined tests:
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.ProductionPattern*:OmpApiTest.Combined*'
```
6. Run only the parallel-region tests (per-task ICV, loop pattern, concurrent max_threads):
```bash
./install/gtests/gtests --gtest_filter='OmpApiTest.*ParallelRegion*:OmpApiTest.*PerTaskIcv*:OmpApiTest.*ParallelThreads*'
```

### Example with more Arguments Support
1. Run the 23rd BF16 Input, BF16 Output matmul test with seed 1245:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/23 --seed 1245
```
2. Run the 23rd BF16 Input, BF16 Output matmul test with seed 1245 and relu post-operation:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/23 --seed 1245 --postop relu
```
3. Run the BF16 Input, BF16 Output matmul test for 10 randomly generated valid tests with random seed and random postop selection:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/* --test 10
```
4. Run F32 matmul tests with AOCL DLP backend:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32/* --backend aocl_dlp
```
5. Run matmul tests with oneDNN backend and LOWOHA enabled:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32/* --backend onednn --lowoha true
```
6. Run matmul tests using input file configurations for 2D matmul:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/* --input_file test.txt --op matmul --ndims 2
```
7. Run batch matmul tests using input file configurations for 3D matmul:
``` bash
./install/gtests/gtests --gtest_filter=BatchMatmul/* --input_file batch_tests.txt --op matmul --ndims 3
```
8. Run matmul tests with 16 threads:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.F32_F32/* --num_threads 16
```

### Run All testcases of a TestSuite
```bash
./install/gtests/gtests --gtest_filter=<TestSuite>/*
```

### Run All available unit tests
```bash
./install/gtests/gtests
```

## **Future Enhancements**
- Add dynamic tolerance values based on tensor dimensions, data type, and value range.
- Implement a command-line parser to avoid hardcoding parameters.
