
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# Zen Deep Neural Network Library (ZenDNN): Accelerated Deep Learning Inference on AMD Zen Architecture

<!-- toc -->
- [About ZenDNN](#1-about-zendnn)
  - [Overview](#11-overview)
  - [Structure](#12-code-structure)
  - [Third Party Libraries](#13-third-party-libraries)
  - [Supported Frameworks](#14-supported-frameworks)
  - [Supported OS](#15-supported-os)
- [Build and Install](#2-build-and-install)
- [Examples, Tests and Benchmarks](#3-examples-tests-and-benchmarks)
  - [Examples](#31-examples)
  - [GoogleTest](#32-gtests)
  - [BenchDNN](#33-benchdnn)
- [Inference Frameworks Integration](#4-inference-frameworks-integration)
- [License](#5-license)
- [Technical Support](#6-technical-support)
<!-- tocstop -->

# 1. About ZenDNN

## 1.1. Overview

ZenDNN (Zen Deep Neural Network) Library accelerates deep learning inference applications on AMD CPUs. This library, which includes APIs for basic neural network building blocks optimized for AMD CPUs, targets deep learning application and framework developers with the goal of improving inference performance on AMD CPUs across a variety of workloads, including computer vision, natural language processing (NLP), and recommender systems.

ZenDNN is a redesigned, re-architected, and refactored deep learning library, evolving from the original ZenDNN_legacy. The legacy version is retained for reference and backward compatibility (https://github.com/amd/ZenDNN/tree/zendnn_legacy).
In addition to features offered by ZenDNN_legacy, ZenDNN is intended to support the following additional features:

1. Support to plug-and-play other primitive libraries like OneDNN, LibXSMM, or other low level backends like BLAS libraries. For example, if a primitive is unimpemented in ZenDNN, or more performant in OneDNN, the OneDNN primitive can be executed using OneDNN APIs, without getting into OneDNN code.
2. Provide integrated profiling and performance analysis tools, which will help in both development and deployment. For example, these tools could be used for instrumented profiling of a primitive, analysing performance issues, and to optimize the primitive. These tools could also be used in deployment; for example, to analyse scaling behaviour of primitives in a multi-instance deployment.

## 1.2. Code Structure

ZenDNN has the following top level directory structure:

```
ZenDNN
|- build        : to build and install the library.
|- benchdnn     : contains benchmarking utilities for performance analysis.
|- cmake        : contains CMake modules.
|- dependencies : to download all dependencies.
|- examples     : contains tutorial examples illustrating how to use ZenDNN APIs.
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
|   |   |   |- compare_operator : perform element-wise comparison of tensors.
|   |   |   |- embag_operator : implements embedding bag and embedding operators.
```

## 1.3. Third Party Libraries

ZenDNN depends on the following libraries.
 - AOCL UTILS (https://github.com/amd/aocl-utils)
 - GoogleTest (https://github.com/google/googletest)
 - NLOHMANN JSON (https://github.com/nlohmann/json) 

Apart from this ZenDNN uses BLAS backends for matrix computations. It depends on any one of the following BLAS backends
 - AOCL DLP (https://github.com/amd/aocl-dlp)
 - AOCL BLIS (https://github.com/amd/blis)

ZenDNN can also use the following other optional backends
 - OneDNN (https://github.com/uxlfoundation/oneDNN)
 - LibXSMM (https://github.com/libxsmm/libxsmm)
 - FBGEMM (https://github.com/pytorch/FBGEMM)

ZenDNN downloads and builds these dependencies as a part of its build process. These dependencies are also forwarded to any downstream package using ZenDNN in its build. Thus a downstream package does not need to figure out ZenDNN dependencies and build them.

## 1.4. Supported Frameworks

ZenDNN library is intended to be used in conjunction with the frameworks mentioned below and cannot be used independently.

ZenDNN library is integrated with TensorFlow v2.20.0 (Plugin), and PyTorch v2.10.0 (Plugin).
- Python v3.9-v3.13 are supported versions to generate the TensorFlow v2.20.0 (Plugin) wheel files (*.whl).
- Python v3.10-v3.13 are supported versions to generate the PyTorch v2.10.0 (Plugin) wheel files (*.whl).

## 1.5. Supported OS

Build from source will be supported on
- Ubuntu® 22.04, 24.04
- Red Hat® Enterprise Linux® (RHEL) 9.2, 9.5


# 2. Build and Install

Please refer to the [build system documentation](docs/zendnnl_build.md) for building and installing the library and its components.

# 3. Examples, Tests and Benchmarks

## 3.1. Examples

ZenDNN provides many examples demonstrating the usage of the library API. Executables of these examples can be found in the `install/examples/bin/` directory, and can be executed as follows:

```bash
./install/examples/bin/examples
```

## 3.2. GTests

GTests are found in `install/gtests`, and can be executed as follows:
```bash
./install/gtests/gtests
```

For detailed usage instructions, refer to the [zendnn* gtests documentation](zendnnl/gtests/Readme.md).

## 3.3. BenchDNN

ZenDNN includes `benchdnn`, a high-performance benchmarking utility designed to rigorously assess the efficiency of deep learning operators within the ZenDNN library. BenchDNN provides detailed performance analysis capabilities for primitives like matrix multiplication (matmul) and reorder operations.

For detailed usage instructions and configuration options, refer to the [zendnn* benchdnn documentation](benchdnn/README.md).

## 3.4. Logging

For detailed usage instructions and configuration options, refer to the [zendnn logs documentation](docs/logging.md).

# 4. Inference Frameworks Integration

Please refer to the [build system documentation](docs/zendnnl_build.md) to find details of how ZenDNN can be integrated with a inference or serving framework.

# 5. License
Refer to the "[LICENSE](LICENCE)" file for the full license text and copyright notice.

This distribution includes third party software governed by separate license terms.

This third party software, even if included with the distribution of the Advanced Micro Devices software, may be governed by separate license terms, including without limitation, third party license terms,  and open source software license terms. These separate license terms govern your use of the third party programs as set forth in the **THIRD-PARTY-PROGRAMS** file.

# 6. Technical Support
Please email Zendnn.Maintainers@amd.com for questions, issues, and feedback on ZenDNN.

Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/amd/ZenDNN/issues) page.
