# Overview
`benchdnn` is a high-performance benchmarking utility purpose-built to rigorously assess the `efficiency` of Matmul and Reorder operators within the `ZenDNN (Zen Deep Neural Network)` library. It plays a pivotal role in the ZenDNN ecosystem by enabling detailed performance analysis of deep learning primitives.

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
  - `n`: Columns in matrix B (for multi-layer matmul, specify as colon-separated values, e.g., `512:256:128`)
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
- **Detailed Timing**: Reports total time, GFLOPS, and detailed timing statistics (context creation, operator creation, execution), including percentage breakdowns for each stage (% of total time)
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

Each line in the input file specifies a benchmark configuration for the selected operator. Below are example input lines for each mode:

### Matmul Operator
For `--op=matmul`, each line should be:

- **Normal (single-layer) matmul:**
  ```
  128, 9216, 4096, 1, f32:f32:f32, true, f32, relu, aocl_blis_blocked, 30
  128, 9216, 4096, 100, f32:f32:f32, true, f32, relu, aocl_blis_blocked, 30
  ```

- **Multi-layer (pipeline) matmul:**
  ```
  768, 3072, 512:256, 100, f32:f32:f32, true, f32, gelu_erf, aocl_blis_blocked, 30
  4096, 768, 256:3072:512, 100, f32:f32:f32, true, f32, gelu_erf, aocl_blis_blocked, 30
  ```

### Reorder Operator
For `--op=reorder`, each line should be:

- **Reorder:**
  ```
  256, 5, 1000, f32, aocl, true
  3072, 768, 100, f32, aocl, true, 30
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

## Output Format (Matmul)

The benchmark prints results to both the console and a CSV file named `timings_<timestamp>.csv` in the `build` directory. The output format supports both multi-layer (pipeline) and normal (single-layer) matmul configurations, with detailed timing breakdowns for each layer or configuration.

### Matmul Output: Multi-Layer Pipeline (Console & CSV)

For pipeline (multi-layer) matmul, the output includes a `Summary` row for each configuration and a `Layer_N` row for each layer.
#### Example (multi-layer matmul, console/CSV)
```
Layer    M     K     N             Iters  Data_type    Bias_Enabled  Bias_dt  PostOp                              Kernel_Name        Warmup_iters  Total_time(ms, all iters)  GFLOPS  %_of_Total  Ctx_Creation(ms_%)  Op_Creation(ms_%)  Op_Execution(ms_%)  
Summary  768   3072  512:256       100    f32:f32:f32  0                      gelu_erf:binary_add                 aocl_blis_blocked  30            2989.89                                                                                                   
Layer_0  768   3072  512           100    f32:f32:f32  0                      gelu_erf:binary_add                 aocl_blis_blocked  30            1559.42                    154.92  52.16 %     0.61 (0.04 %)       0.15 (0.01 %)      1558.66 (99.95 %)   
Layer_1  768   512   256           100    f32:f32:f32  0                      gelu_erf:binary_add                 aocl_blis_blocked  30            1430.47                    14.07   47.84 %     0.66 (0.05 %)       0.17 (0.01 %)      1429.64 (99.94 %)
```

### Matmul Output: Normal (Single-Layer) (Console & CSV)

For normal (single-layer) matmul, each configuration produces a single row.
#### Example (normal matmul, console/CSV)
```
M     K     N     Iters  Data_type       Bias_Enabled  Bias_dt  PostOp      Kernel_Name        Warmup_iters  Total_time(ms, all iters)  GFLOPS   Ctx_Creation(ms_%)  Op_Creation(ms_%)  Op_Execution(ms_%)
128   9216  4096  1      f32:f32:f32     1             f32      relu        aocl_blis_blocked  30            12.73                      758.84   0.01 (0.06 %)       0.00 (0.01 %)      12.73 (99.92 %)
128   9216  4096  100    f32:f32:f32     1             f32      relu        aocl_blis_blocked  30            1364.92                    708.00   0.84 (0.06 %)       0.20 (0.01 %)      1363.89 (99.92 %)
```

### Notes
- All timing columns are in milliseconds.
- GFLOPS is calculated per layer for matmul.
- For multi-layer matmul, each layer's results are shown separately, with N as colon-separated values in the summary.
- The CSV format is suitable for further analysis and plotting.
- Context/Operator/Execution times and percentages help identify bottlenecks and optimize performance.

## Output Format (Reorder)

The benchmark prints results to both the console and a CSV file named `timings_<timestamp>.csv` in the `build` directory.

### Reorder Output (Console & CSV)

Each row contains timing and configuration details for a single reorder benchmark run.
#### Example (reorder, console/CSV)
```
Rows  Cols  Iterations  Data_type  Kernel_Name  In-place  Warmup_iters  Total_time(ms)  Ctx_Creation(ms_%)  Op_Creation(ms_%)  Op_Execution(ms_%)  Others(ms_%)
3072  768   100         f32        aocl         1         30            1318.81         0.16 (0.01 %)       0.21 (0.02 %)      1318.12 (99.95 %)   0.31 (0.02 %)
768   3072  100         f32        aocl         1         30            1288.75         0.16 (0.01 %)       0.20 (0.02 %)      1288.11 (99.95 %)   0.29 (0.02 %)
```

### Notes
- All timing columns are in milliseconds.
- Percentages show the proportion of total time spent in each stage.
- The CSV format is suitable for further analysis and plotting.

## Extending
- Add new post-operations or data types by updating enums and parsing logic in the utility and parser modules.
- For 3D tensor (BMM) support, see the TODO in the `MatmulConfig` struct.
- Follow Doxygen-style documentation and project code style for contributions.

## Documentation & Diagrams
- All major source/header files and functions are documented with Doxygen-style comments.
- See the `benchdnn` directory for high-level and data flow diagrams (if provided).

## License

Licensed under the Apache License, Version 2.0. See the ZenDNN root directory for details.
