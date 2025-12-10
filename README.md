
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNN* : Accelerated Deep Learning Inference on AMD Zen Architecture

<!-- toc -->
- [About ZenDNN*](#1-about-zendnn)
  - [Overview](#11-overview)
  - [Structure](#12-code-structure)
  - [Third Party Libraries](#13-third-party-libraries)
  - [Supported OS](#14-supported-os)
- [Build and Install](#2-build-and-install)
- [Examples, Tests and Benchmarks](#3-examples-tests-and-benchmarks)
  - [Examples](#31-examples)
  - [GoogleTest](#32-gtests)
  - [BenchDNN](#33-benchdnn)
- [Inference Frameworks Integration](#4-inference-frameworks-integration)  
<!-- tocstop -->

# 1. About ZenDNN*

## 1.1. Overview

ZenDNN* is being designed as a re-architected and refactored ZenDNN. ZenDNN*, like its predecessor ZenDNN, is a performance primitive library for deep learning inference on AMD Zen architecture. In addition to features offered by ZenDNN, ZenDNN* is intended to support the following additional features:

1. Support to plug-and-play other primitive libraries like OneDNN, or other low level backends like BLAS libraries. For example, if a primitive is unimpemented in ZenDNN*, or more performant in OneDNN, the OneDNN primitive can be executed using OneDNN APIs, without getting into OneDNN code.
2. Provide integrated profiling and performance analysis tools, which will help in both development and deployment.For example, these tools could be used for instrumented profiling of a primitive, analysing performance issues, and to optimize the primitive. These tools could also be used in deployment; for example, to analyse scaling behaviour of primitives in a multi-instance deployment.

## 1.2. Code Structure

ZenDNN* has the following top level directory structure:

```
ZenDNN
|- build        : to build and install the library.
|- benchdnn     : contains benchmarking utilities for performance analysis.
|- cmake        : contains CMake modules.
|- dependencies : to download all dependencies.
|- examples     : contains tutorial examples illustrating how to use ZenDNN* APIs.
|- scripts      : contains supporting shell scripts.
|- docs         : contains documentation files.
|   |- doxygen  : doxygen config file and additional pages.
|
|- zendnnl      : contains library code.
|   |- gtests   : GoogleTest files.
|   |- src      : contains library code.
|   |   |- common : contains high level utilities needed by the library.
|   |   |- memory : implements tensor_t class.
|   |   |- operator : implements all operator classes.
|   |   |   |- common : implements base classes needed for the operators.
|   |   |   |- sample_operator : demonstrates how to create an operator.
|   |   |   |- matmul_operator : implements matrix multiplication with optional post-op.
|   |   |   |- reorder_operator : copies data between different memory formats.
|   |   |   |- compare_operator : perform element-wise comparision of tensors.
|   |   |   |- embag_operator : implements embedding bag and embedding operators.
```

## 1.3. Third Party Libraries

ZenDNN* depends on the following libraries.
 - AOCL UTILS (https://github.com/amd/aocl-utils)
 - GoogleTest (https://github.com/google/googletest)
 - NLOHMANN JSON (https://github.com/nlohmann/json) 

Apart from this ZenDNN* uses BLAS backends for matrix computations. It depends on any one of the following BLAS backends
 - AOCL BLIS (https://github.com/amd/blis)
 - AOCL DLP (https://github.com/amd/aocl-dlp)

ZenDNN* can also use the following other optional backends
 - OneDNN (https://github.com/uxlfoundation/oneDNN)
 - LibXSMM (https://github.com/libxsmm/libxsmm)

ZenDNN* downloads and builds these dependencies as a part of its build process. These dependencies are also forwarded to any downstream package using ZenDNN* in its build. Thus a downstream package does not need to figure out ZenDNN* dependencies and build them.

## 1.4. Supported OS

Refer to the [support matrix](https://www.amd.com/en/developer/zendnn.html#getting-started) for the list of supported operating systems.

# 2. Build and Install

Please refer to the [build system documentation](docs/zendnnl_build.md) for building and installing the library and its components.

# 3. Examples, Tests and Benchmarks

## 3.1. Examples

ZenDNN* provides many examples demonstrating the usage of the library API. Executables of these examples can be found in the `install/examples/bin/` folder.

For detailed logging support and control, please see [logging support documentation](docs/logging.md).

## 3.2. GTests

GTests are found in `install/gtests`, and can be executed as follows:
```bash
./gtests/gtests
```

For detailed usage instructions, refer to the [zendnn* gtests documentation](zendnnl/gtests/Readme.md).

## 3.3. BenchDNN

ZenDNN* includes `benchdnn`, a high-performance benchmarking utility designed to rigorously assess the efficiency of deep learning operators within the ZenDNN library. BenchDNN provides detailed performance analysis capabilities for primitives like matrix multiplication (matmul) and reorder operations.

For detailed usage instructions and configuration options, refer to the [zendnn* benchdnn documentation](benchdnn/README.md).

# 4. Inference Frameworks Integration

Please refer to the [build system documentation](docs/zendnnl_build.md) to find details of how ZenDNN* can be integrated with a inference or serving framework.

Note that testing has not been performed on including ZenDNN* by other methods such as add_subdirectory() or Fetch_Content().

>ZenDNN* : ZenDNN is currently undergoing a strategic re-architecture and refactoring to enhance performance, maintainability, and scalability.