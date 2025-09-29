(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

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

## Supported Features Overview

| Feature              | Supported Values                        |
|----------------------|----------------------------------------|
| Operators            | matmul, reorder                         |
| Multi-layer Matmul   | Supported                               |
| Data Types           | f32, bf16                               |
| Timing Modes         | end-to-end, detailed timing breakdowns  |
| Cache Modes          | hot cache, cold cache                   |
| Warmup Iterations    | Supported (configurable)                |
| Batched Matmul (BMM) | Supported (via 'bs' field and --ndims=3)         |

## Flow Diagram

Below is a high-level flow diagram illustrating the benchmarking process:

<img src="./images/BenchDNNFlow_diagram.png" alt="BenchDNN Flow Diagram" width="700"/>

### Benchmark Workflow Diagram

#### Cold Cache Control
The cold cache behavior is controlled by the macro `COLD_CACHE` defined in `utils/benchdnn_utils.hpp`:

- If `COLD_CACHE` is set to `1`, the CPU cache is flushed before each operation to simulate cold-start performance. This affects both individual and end-to-end timing modes.
- If `COLD_CACHE` is set to `0`, cache flushing is disabled and benchmarks run with warm cache.

Set this macro according to your benchmarking needs before building the project.
#### Timing Mode Selection
The timing mode is controlled by the macro `MEASURE_INDIVIDUAL_TIMINGS` defined in `benchdnn.hpp`:

- If `MEASURE_INDIVIDUAL_TIMINGS` is set to `1`, individual timings for each operation (context creation, operator creation, operator execution, other operations) are recorded.
- If `MEASURE_INDIVIDUAL_TIMINGS` is set to `0`, only end-to-end timings are recorded.

Set this macro according to your benchmarking needs before building the project.

Below is a workflow diagram showing the main steps in the benchmarking process:


<img src="./images/Benchmark_workflow.png" alt="Benchmark Workflow" width="450" height="450"/>

#### Benchmark Workflow Overview

The flow is split into two main sections:

**1. Individual Timings**
  - Each operation is timed separately using distinct timers:
    - T1: Context Creation
    - T2: Operator Creation
    - T3: Operator Execution
    - T4: Other Operations
  - If `cold_cache` is true, the CPU cache is flushed before each operation to simulate cold-start performance.

**2. End-to-End Timings**
  - All operations are timed together using a single timer:
    - T: Measures the total time for Context Creation + Operator Creation + Operator Execution + Other Operations.
  - Cache flushing also occurs here if `cold_cache` is enabled.



## Features
- **Operator Selection via Command-Line**: Use `--op=<operator>` to specify the operation to benchmark (e.g., `--op=matmul`, `--op=reorder`).
- **Flexible Input File**: Use `--input_file=<filename>` to provide a configuration file tailored to the selected operator.
- **Matmul Benchmarking**: Supports matrix multiplication benchmarks with options for single-layer, multi-layer, and batched matmul.
- **Reorder Benchmarking**: Supports tensor reorder benchmarks with configurable parameters.
- **Multiple Data Types**: Supports a range of data types (e.g., `f32`, `bf16`)
- **Detailed Timing**: Reports total time, GFLOPS, and detailed timing statistics (context creation, operator creation, execution), including percentage breakdowns for each stage (% of total time)
- **Warmup Iterations**: Optional warmup runs to stabilize measurements
- **Cache Control**: Optional cache flush between runs for accurate timing (if enabled at compile time)
- **Comprehensive Output**: Results are printed to the console and saved to a timestamped CSV file for easy analysis


## Build Instructions

From the ZenDNN root directory:

```sh
mkdir build && cd build
cmake .. -DZENDNNL_BUILD_BENCHDNN=ON
cmake --build .
```

## Running the Benchmark

To run a benchmark, specify the operator and input method as command-line arguments from the build directory. For example:

```sh
./benchdnn/benchdnn --op=<operator> [--input_file=<file>] [command-line options] [--input_model_file=<model_file>]
```

- `<operator>`: Operator can be one of the following :
  - [matmul](doc/matmul.md)
  - [reorder](doc/reorder.md)
- `--input_file=<file>`: Path to a configuration file with one or more test cases.
- `--input_model_file=<model_file>`: (Optional) Path to a model file for model-based benchmarking.
- `[command-line options]`: Command-line arguments to specify all required parameters directly. These can be used in combination with model files.

The output CSV files (`timings_<timestamp>.csv`) are located in the `build` directory.

## Extending
- Add new post-operations or data types by updating enums and parsing logic in the utility and parser modules.
- Follow Doxygen-style documentation and project code style for contributions.

## Documentation & Diagrams
- All major source/header files and functions are documented with Doxygen-style comments.
- See the `benchdnn` directory for high-level and data flow diagrams (if provided).

## License

Licensed under the Apache License, Version 2.0. See the ZenDNN root directory for details.
