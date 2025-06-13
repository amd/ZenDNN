# Overview
`benchdnn` is a high-performance benchmarking utility purpose-built to rigorously assess the `efficiency` and `correctness` of matrix multiplication and related computational operations within the `ZenDNN (Zen Deep Neural Network)` library. It plays a pivotal role in the ZenDNN ecosystem by enabling detailed performance analysis and validation of deep learning primitives.

# Purpose and Audience
This tool is indispensable for a wide range of users, including `developers`, `researchers`, and `performance engineers`. Whether you're optimizing kernel implementations, experimenting with new data types, or evaluating the impact of various optimization strategies, `benchdnn` provides the precision and flexibility needed to make informed decisions.

# Flexible Configuration
Designed with adaptability in mind, `benchdnn` supports a broad spectrum of configurable parameters. Users can specify matrix dimensions, data types, post-operations (such as activation functions or element-wise operations), and kernel backends. These configurations can be supplied either through intuitive command-line arguments or structured input files, making the tool suitable for both quick tests and large-scale automated benchmarking.

# Extensibility for Innovation
One of the core strengths of `benchdnn` is its extensible architecture. New operators, data types, and post-processing functions can be seamlessly integrated into the framework. This makes it an ideal platform for ongoing development, experimentation, and validation of emerging features within ZenDNN.

# Precision in Performance Measurement
Accuracy is at the heart of `benchdnn`'s design. To ensure that performance metrics reflect real-world behavior, the tool incorporates several advanced features:
- Warmup iterations to eliminate cold-start anomalies.
- Cache flushing to simulate realistic memory access patterns.
- Detailed timing breakdowns, including:
  - Context creation
  - Operator setup
  - Execution time
These capabilities help isolate performance bottlenecks and provide a reliable foundation for performance tuning and regression analysis

## Features
- **Operator Selection via Command-Line**: Use `--op=<operator>` to specify the operation to benchmark (e.g., `--op=matmul`, `--op=reorder`).
- **Flexible Input File**: Use `--input_file=<filename>` to provide a configuration file tailored to the selected operator.
- **Matmul Benchmarking**: For `--op=matmul`, the input file should contain one configuration per line with the following fields:
  - `m`: Rows in matrix A
  - `k`: Columns in matrix A / rows in matrix B
  - `n`: Columns in matrix B
  - `iters`: Number of benchmark iterations
  - `dt`: Data types (`src:weights:dst`, e.g., `f32:f32:f32`)
  - `bias`: `true` or `false` (whether to add bias)
  - `bias_dt`: Data type for bias (e.g., `f32`, `bf16`). If `bias` is `false`, this can be omitted or left empty.
  - `post_ops`: Post-operations (colon-separated, e.g., `relu`, `gelu_erf`, `binary_mul`, or combinations like `relu:binary_mul`). Binary post-ops are supported and will use an additional tensor as input.
  - `kernel`: Kernel backend (e.g., `aocl_blis_blocked`)
  - `warmup_iters` (optional): Number of warmup iterations (if not provided, defaults to 20% of `iters`)
- **Reorder Benchmarking**: For `--op=reorder`, the input file should contain one configuration per line with the following fields:
  - `rows`: Number of rows in the tensor to reorder
  - `cols`: Number of columns in the tensor to reorder
  - `iters`: Number of benchmark iterations
  - `dt`: Data type (e.g., `f32`, `bf16`)
  - `kernel`: Kernel backend (e.g., `aocl`)
  - `isInplace`: `true` or `false` (whether to perform in-place reorder)
  - `warmup_iters` (optional): Number of warmup iterations (defaults to 20% of `iters` if not provided)
- **Multiple Data Types**: Supports a range of data types (e.g., `f32`, `bf16`, `s8`, `u8`, etc.)
- **Detailed Timing**: Reports total time, GFLOPS, and detailed timing statistics (context creation, operator creation, execution)
- **Warmup Iterations**: Optional warmup runs to stabilize measurements
- **Cache Control**: Optional cache flush between runs for accurate timing (if enabled at compile time)
- **Comprehensive Output**: Results are printed to the console and saved to a timestamped CSV file for easy analysis

## Directory Structure
- `benchdnn.cpp` / `benchdnn.hpp`: Main benchmarking logic and configuration structures
- `matmul/matmul_benchdnn.cpp` / `matmul/matmul_benchdnn.hpp`: Matrix multiplication benchmarking implementation
- `matmul/matmul_parser.cpp` / `matmul/matmul_parser.hpp`: Input parsing for matmul benchmark
- `reorder/reorder_benchdnn.cpp` / `reorder/reorder_benchdnn.hpp`: Reorder benchmarking implementation
- `reorder/reorder_parser.cpp` / `reorder/reorder_parser.hpp`: Input parsing for reorder benchmark
- `utils/benchdnn_utils.hpp` / `utils/benchdnn_utils.cpp`: Utility functions for benchmarking (string conversion, cache flush, etc.)
- `build/input.txt`: Example input file for benchmarking (located in the build directory)
- `build/timings_<timestamp>.csv`: Output CSV files with timing results (created in the build directory)

## Build Instructions

From the ZenDNN root directory:

```sh
mkdir build && cd build
cmake .. -DZENDNNL_BUILD_BENCHDNN=ON
cmake --build .
```

## Running the Benchmark

Run the benchmark by passing the operator and input file as command-line arguments (from the build directory):

```sh
./bench/benchdnn --op=<operator> --input_file=input.txt
```

- `<operator>`: The operator to benchmark (e.g., `matmul`, `reorder`).
- `input.txt`: Path to the input file containing benchmark configurations (see below).

Both the input file (`input.txt`) and the output CSV files (`timings_<timestamp>.csv`) are located in the `build` directory by default.

## Input File Format

Each line in the input file specifies a benchmark configuration for the selected operator.

### Matmul Operator
For `--op=matmul`, each line should be:

```
m, k, n, iters, dt, bias, bias_dt, post_ops, kernel[, warmup_iters]
```

### Example (matmul)
```
100, 200, 500, 10, f32:f32:f32, true, f32, relu:gelu_erf, aocl_blis_blocked, 1
```

### Reorder Operator
For `--op=reorder`, each line should be:

```
rows, cols, iters, dt, kernel, isInplace[, warmup_iters]
```

### Example (reorder)
```
256, 5, 1000, f32, aocl, true
```
## Output

The benchmark prints the following for each input:
- Total execution time
- Achieved GFLOPS for matmul operator
- Detailed timing statistics for all iterations of each input, including:
  - Context creation time
  - Operator creation time
  - Operator execution time

Output is printed to the console and also saved to a CSV file named `timings_<current timestamp>.csv` in the `build` directory.

### Matmul Output
The CSV file contains one row per input configuration, with columns:

```
M, K, N, Iterations, Data type, Post Operation, Kernel name, Warmup iterations, Total time (ms), GFLOPS, Context Creation Time (ms), Operator Creation Time (ms), Operator Execution Time (ms)
```

Example row:
```
100, 200, 500, 10, f32:f32:f32, relu:gelu_erf, aocl_blis_blocked, 1, 108.39, 0.184519, 0.07195, 0.01625, 108.302
```

### Reorder Output
For reorder benchmarks, the CSV columns are:

```
Rows, Cols, Iterations, Data type, Kernel name, In-place, Warmup iterations, Total time (ms), Context Creation Time (ms), Operator Creation Time (ms), Operator Execution Time (ms), Others (ms)
```

Example row:
```
256, 5, 1000, f32, aocl, 0, 200, 8.81514, 0.576412, 0.457751, 6.80787, 0.973113
```

This CSV file contains detailed timing results for all benchmark runs and can be used for further analysis.

## Extending
- Add new post-operations or data types by updating enums and parsing logic in the utility and parser modules.
- For 3D tensor (BMM) support, see the TODO in the `MatmulConfig` struct.
- Follow Doxygen-style documentation and project code style for contributions.

## Documentation & Diagrams
- All major source/header files and functions are documented with Doxygen-style comments.
- See the `benchdnn` directory for high-level and data flow diagrams (if provided).

## License

Licensed under the Apache License, Version 2.0. See the ZenDNN root directory for details.
