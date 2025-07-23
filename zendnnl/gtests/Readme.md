
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNN* Google Test Infrastructure

## **Overview of GTest**
Google Test (GTest) is a unit testing framework for C++ that provides a robust and flexible way to write and execute tests. In this project, GTest is used to validate the functionality and performance of ZenDNN* operations. It supports parameterized tests, filtering, and detailed reporting, making it ideal for testing complex numerical computations.

## **Purpose and Audience**
The ZenDNN* GTest infrastructure is designed for:
- **Developers**: To verify the correctness of ZenDNN* operations during development.
- **Test Engineers**: To run automated tests and validate functionality across different configurations.
- **Researchers**: To experiment with numerical precision and post-operations in deep learning workloads.

This infrastructure ensures that ZenDNN* operations meet expected accuracy standards under various conditions.

## **Flexible Configuration**
ZenDNN* GTest provides flexibility in configuring tests through command-line arguments and code-level parameters. Users can:
- **Filter Tests**: Run specific test suites or cases using `--gtest_filter`.
- **Set Random Seed**: Provide a seed value for reproducible random test data generation.
- **Specify Post-Operations**: Apply post-operations like `relu`, `gelu_tanh`, etc., during matrix multiplication tests.

## **Configurable Parameters**
You can modify the following parameters in the source code (`gtest_main.cpp`):
- `MATMUL_F32_TOL`: Tolerance for floating-point precision in tests (default: `0.001`).
- `MATMUL_BF16_TOL`: Tolerance for BF16 precision in tests (default: `0.01`).
- `TEST_NUM`: Number of test cases to generate (default: `100`).

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

## **Directory Structure**

The `zendnnl/gtests/` directory contains all test-related files and subdirectories. Below is the structure:

```plaintext
gtests/
├── gtest_main.cpp           # Entry point for all tests.
├── test_matmul.cpp          # Matmul testsuite with different test cases.
├── test_reorder.cpp         # Reorder testsuite with different test cases.
└── gtest_utils.cpp/hpp      # Utility functions for tests.
```

## **Configure and build GTest with ZenDNN***
- Configure:
```bash
cmake -ZENDNNL_BUILD_GTEST=ON ..
```
- Build:
```bash
cmake --build .
```
## **Running Tests**

### **General Command Structure**
```bash
./install/gtests/gtests --gtest_filter=<TestSuite>/<TestCase>[/<Index>] [<Seed>] [<PostOp>]
```

## **Parameters**
1. **`--gtest_filter=<TestSuite>/<TestCase>`** (Optional):
   - Specifies the test suite and test case to run.
   - *Example*:
     - `Reorder/TestReorder.BF16_F32/*`: Runs all tests for BF16 reorder followed by Matrix (BF16 Input and FP32 Output).
     - `Matmul/TestMatmul.BF16_BF16/23`: Runs the 23rd test case for BF16 Input and BF16 Output with matrix multiplication.

2. **`<Seed>`** (Optional):
   - Sets the seed based on the seed value for generating test data.
   - *Example*:
     - `1245`: Uses 1245 as the seed for randomization.

3. **`<PostOp>`** (Optional):
   - Specifies the post-operation to apply during tests.

**Note:**
 - If **`<PostOp>`** parameter is not provided, gtest will pick postop randomly from supported post-ops.
 - If **`<Seed>`** parameter is also not provided, gtest sets the seed based on timestamp for generating test data.
 - If no parameters are provided, It will run all available tests with seed sets based on timestamp and randomly selected postops.

## **Examples**
### Reorder Tests (Reorder + Matmul)
 - Reorder TestSuite has six testcases(F32_F32, BF16_F32, BF16_BF16, F32_F32_Stride, BF16_F32_Stride, BF16_BF16_Stride)
1. Run all BF16 reorder tests followed by Matmul(BF16 Input, F32 Output):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_F32/*
```
2. Run all BF16 reorder tests followed by Matmul(BF16 Input, BF16 Output):
```bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.BF16_BF16/*
```
3. Run all FP32 reorder tests followed by Matmul(F32 Input, F32 Output):
``` bash
./install/gtests/gtests --gtest_filter=Reorder/TestReorder.F32_F32/*
```

### Matmul Tests
 - Matmul TestSuite has six testcases(F32_F32, BF16_F32, BF16_BF16, F32_F32_Stride, BF16_F32_Stride, BF16_BF16_Stride)
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

### Example with more Arguments Support
1. Run the 23rd BF16 Input, BF16 Output matmul test with seed 1245:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/23 1245
```
2. Run the 23rd BF16 Input, BF16 Output matmul test with seed 1245 and relu post-operation:
``` bash
./install/gtests/gtests --gtest_filter=Matmul/TestMatmul.BF16_BF16/23 1245 relu
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

>ZenDNN* : ZenDNN is currently undergoing a strategic re-architecture and refactoring to enhance performance, maintainability, and scalability.
