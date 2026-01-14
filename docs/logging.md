
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# Logging Support and Control

## Overview

The logging system in ZenDNN provides a flexible mechanism to monitor and debug application execution. It categorizes logs by modules and log levels, and its behavior can be controlled via environment variables. This document explains the available log modules and levels, along with how to use environment variables to control log output.

## Log Modules

Log modules represent different areas or components within ZenDNN. Each module can have its own logging configuration. Some common log modules include:

- **common**: General logging for common operations.
- **api**: Logging related to API calls and interfaces.
- **test**: Logs for testing and validation processes.
- **profile**: Metrics and performance-related logs.
- **debug**: Detailed debugging information.

## Log Levels

Log levels determine the severity of the messages:

- **disabled**: No log messages are printed.
- **error**: Critical failures that require immediate attention.
- **warning**: Potential issues that are not immediately critical.
- **info**: General information regarding execution flow.
- **verbose**: Detailed debugging information for in-depth analysis.

The levels are typically numeric, where a higher number corresponds to increased verbosity. For example, if a module's log level is set to 4 (verbose), then all messages with a level of error, warning, info, and verbose will be printed.

## Environment Variables for Log Control

ZenDNN supports several environment variables to control logging behavior. These variables override default settings, allowing you to adjust verbosity at runtime without recompiling the code.

- **ZENDNNL_<log_module>_LOG_LEVEL**:
  Sets the default log level for a specific module. For example, the following command sets the log level of a module (Example: API, PROFILE) to verbose (4):
  Example:
  ```
  export ZENDNNL_<log_module>_LOG_LEVEL=4
  ```

**Implication:**
  When the log level is set to 4, all messages at the verbose level and below will be printed. This means:
  - **disabled**(level 0)
  - **error**   (level 1)
  - **warning** (level 2)
  - **info**    (level 3)
  - **verbose** (level 4)

  will all be output. Adjusting this variable helps in getting more detailed logs for debugging or reducing output during normal execution.

These settings enable runtime control of logging behavior, which is essential in different environments such as development, testing, and production.

### API Logs

ZenDNN provides API-level logs to monitor interactions with the library's interfaces. These logs are enabled by setting the `ZENDNNL_API_LOG_LEVEL` environment variable.

#### Enabling API Logs
To enable API logs, set the log level for the `API` module to verbose (4):
```bash
export ZENDNNL_API_LOG_LEVEL=4
```

#### API Log Details
When API logs are enabled, ZenDNN outputs detailed information about API calls, including:
- Function names and parameters.
- Execution status (success or failure).
- Any warnings or errors encountered during API calls.

**Sample API Log Entry**
```
[API    ][info   ][0.021067]:Tensor create - weights[5,4]:bf16:contiguous
[API    ][info   ][0.021111]:Tensor create - bias[1,4]:f32:contiguous
[API    ][info   ][0.021125]:Context create - weights[5,4]:bf16:contiguous,bias[1,4]:f32:contiguous,post-op:relu
[API    ][info   ][0.021139]:Operator create - matmul_bf16_operator
```

### Test Logs

ZenDNN provides test-level logs to monitor the execution of test cases and validation processes. These logs are enabled by setting the `ZENDNNL_TEST_LOG_LEVEL` environment variable.

#### Enabling Test Logs
To enable test logs, set the log level for the `TEST` module to verbose (4):
```bash
export ZENDNNL_TEST_LOG_LEVEL=4
```

#### Test Log Details
When test logs are enabled, ZenDNN outputs detailed information about test execution, including:
- Test case names and descriptions.
- Validation results (Example: successful or failed).
- Any warnings or errors encountered during testing.

**Sample Test Log Entry**
```
[TEST   ][info   ][0.016712]:operator matmul_f32_operator execution successful.
```

## Profile Logs

ZenDNN provides detailed profiling logs to monitor context creation time, kernel execution time, and operator performance. These logs are enabled by setting the `ZENDNNL_PROFILE_LOG_LEVEL` and `ZENDNNL_ENABLE_PROFILER` environment variable.

### Enabling Profile Logs
To enable profiling logs, set the log level for the `PROFILE` module to verbose (4):
```bash
export ZENDNNL_ENABLE_PROFILER=1
export ZENDNNL_PROFILE_LOG_LEVEL=4
```

### Profiling Log Details
When profiling logs are enabled, ZenDNN outputs detailed information about context creation and operator execution, including:
- Operator name and type.
- Operator context details.
- Input and output tensor details (dimensions, data types, and layouts).
- Post-operation details (Example: activation functions).
- Execution time in milliseconds.

**Sample Profiling Log Entry**
```
[PROF   ][info   ][0.017474]:Operator context - weights[5,4]:f32:contiguous,bias[1,4]:f32:contiguous,post-op:binary_mul:swish:binary_mul,time:0.0009ms
[PROF   ][info   ][0.017503]:Operator execute - matmul_f32,matmul_input[10,5]:f32:contiguous,weights[5,4]:f32:contiguous,bias[1,4]:f32:contiguous,matmul_output[10,4]:f32:contiguous,post-op:binary_mul:swish:binary_mul,time:0.00321ms
```

## Debug Logs

ZenDNN provides detailed debug logs to output comprehensive information about internal operations, including intermediate states and execution flow. These logs are enabled by setting the `ZENDNNL_DEBUG_LOG_LEVEL` environment variable **and are only available when the library is built in debug mode**.

### Debug Mode Requirement
Debug logs are only available when the library is built in debug mode. To build the library in debug mode, complete the following steps:

**Set Debug Build Type**:
In the `CMakeLists.txt` file, ensure the `CMAKE_BUILD_TYPE` is set to `Debug`:
```cmake
set(CMAKE_BUILD_TYPE Debug)
```

### Enabling Debug Logs
To enable debug logs, set the log level for the `DEBUG` module to verbose (4):
```bash
export ZENDNNL_DEBUG_LOG_LEVEL=4
```

### Debug Log Details
- Logs the file path and function name where the debug message is generated.
- A brief description of the operation being performed.

**Sample Debug Log Entry**
```
[DEBUG  ][verbose][0.000405]:[ZenDNN/zendnnl/src/memory/tensor.cpp], [zendnnl::memory::tensor_t& zendnnl::memory::tensor_t::create()]: Creating tensor object
[DEBUG  ][verbose][0.000681]:[ZenDNN/zendnnl/src/memory/tensor_options.cpp], [virtual std::size_t zendnnl::memory::tensor_option_t::hash()]: Generating tensor option hash
```

## Explanation of Log Fields
- **[MODULE]**: Indicates the log module (Example: `COMMON`, `API`, `TEST`, `PROF`, `DEBUG`).
- **[LEVEL]**: Log level (Example: `error`, `warning`, `info`, `verbose`).
- **[TIMESTAMP]**: Timestamp of the log entry, representing the time elapsed since the application started.
- **Description**: A brief description of the operation being logged.
- **Details**: Additional information specific to the log module.

## Example Usage

The following code snippet example illustrates logging messages across various log modules and levels:

```cpp
#include "common/zendnnl_global.hpp"

int main() {

    // Log an error message.
    zendnnl::error_handling::testlog_info("Executing matmul FP32 example.");

    return 0;
}
```

By using the environment variable settings outlined in this document, you can achieve fine-grained control over the logging behavior in ZenDNN, making it easier to troubleshoot and optimize your application.
