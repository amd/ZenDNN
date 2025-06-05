
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# ZENDNNL : Accelerated Deep Learning Inference on AMD Zen Architecture

<!-- toc -->
- [About ZenDNNL](#1-about-zendnnl)
  - [Motivation](#11-motivation)
  - [Overview](#12-overview)
  - [Structure](#13-structure)
  - [Third Party Libraries](#14-third-party-libraries)
  - [Supported OS](#15-supported-os)
- [ZenDNNL : Build and Install](#2-zendnnl-build-and-install)
  - [Build Dependencies](#21-build-dependencies)
  - [Build and install](#22-build-and-install)
- [ZenDNNL : Examples and Tests](#3-zendnnl-examples-tests)
  - [Examples](#31-examples)
  - [GoogleTest](#32-googletest)
<!-- tocstop -->

# 1. About ZenDNNL

## 1.1. Motivation

ZenDNN is a performance primitive inference library for AMD server class CPUs. This library started with a forked version of OneDNN and new features, optmizations and enhancements were added over the years. In the process ZenDNN acquired lot of technical debt and its architecture poses problems in adding and enhancing new features. The following list elaborates further problems with existing ZenDNN.

1. OneDNN has also evolved over time. Since OneDNN and ZenDNN took different evolution paths, bringing new features from OneDNN to existing ZenDNN is difficult and error prone.
2. ZenDNN has limited quantization support. Adding quantization support and mixed-precision arithmetic requires elaborate changes.
3. ZenDNN has rudimentary support for concurrent primitive execution. This prevents it from creating useful parallel patterns like fork-join, pipeline or task-pool for its primitives.
4. ZenDNN does not provide support to limit a primitive execution on a certain number of cores, or bind a primitive to cores.

## 1.2. Overview

ZenDNNL as a new primitive library is envisaged to address these problems with current ZenDNN. The following are the design goals of ZenDNN PlugIn.

1. Simplified design and architecture.
2. Support for plug-and-play other primitive libraries like OneDNN. If a primitive is unimpemented in ZenDNNL, or more performant in OneDNN, the OneDNN primitive can be executed using OneDNN APIs, without getting into OneDNN code.
3. Runtime support for concurrent primitive execution.
4. Detailed profiling and performance analysis. This will help to take informed decisions about performance bottlenecks.

## 1.3. ZenDNNL : Code Structure

ZenDNNL has the following top level directory structure

```
ZenDNN
|- build        : used to build and install the library.
|- cmake        : contains cmake modules.
|- dependencies : to downaload all dependencies.
|- examples     : tutorial examples on how to use ZenDNNL APIs.
|- scripts      : suporting shell scripts.
|- zendnnl      : contains library code.
|   |   |- doxygen : doxygen config file and additional pages.
|   |   |- gtests : GoogleTest files.
|   |   |- src : contains library code.
|   |   |   |- common : contains some high level utilities needed by the library.
|   |   |   |- memory : implements tensor_t class.
|   |   |   |- operator : implements all operator classes.
|   |   |   |   |- common : implements base classes needed for the operators.
|   |   |   |   |- sample_operator : demonstrates how to create an operator.
|   |   |   |   |- matmul_operator : implements matrix multiplication with optional post-op.
|   |   |   |   |- reorder_operator : copies data between different memory formats.
|   |   |   |   |- compare_operator : perform element-wise comparision of tensors.
```
## 1.4. Third Party Libraries

ZenDNNL depends on the following libraries.
 - [AOCL BLIS](https://github.com/amd/blis)
 - [GoogleTest](https://github.com/google/googletest)



## 1.5. Supported OS

Refer to the [support matrix](https://www.amd.com/en/developer/zendnn.html#getting-started) for the list of supported operating systems.

# 2. ZENDNNL : Build and Install

## 2.1. Build Dependencies

ZENDNNL needs the following tools to build
1. **CMake** >= version 3.25.
2. **g++** >= 11.2.0 toolchain.
3. **conda** >= 24.1.0.

Gcc 13.x could be installed using `scripts/zendnnl_install_gcc.sh`. A conda virtual environment with CMake 3.25 can be created using `scripts/zendnnl_conda_env_create.sh`.

ZenDNNL additionally depends on the following packages
1. AMD AOCL BLIS
2. Googletest

These dependencies can either be pre-installed and their location could be passed to ZenDNNL, or downloaded and built by ZenDNNL. Configuration options are available in `cmake/ConfigOptions.cmake`.

## 2.2. Build and install

### Option 1: Manual Build Steps

**Configure AMD BLIS and GoogleTest**
  - Open `cmake/ConfigOptions.cmake`.
  - To use a pre-installed AMD BLIS, set `ZENDNNL_AMDBLIS_USE_LOCAL_REPO` to `ON` and provide the local path to `ZENDNNL_AMDBLIS_DIR`.
  - If ZenDNNL is required to download AMD BLIS then leave the settings as they are.
  - To build GoogleTest, set `ZENDNNL_DEPENDS_GTEST` to `ON`.

**Activate Conda Environment**
  ```bash
  source scripts/zendnnl_conda_env_create.sh
  conda activate zendnnltorch
  ```

**Create a Build Directory**
  ```bash
  mkdir build && cd build
  ```

**Configure CMake**:
  ```bash
  cmake ../
  ```

**Build and Install ZenDNNL**
  ```bash
  cmake --build .
  ```

### Option 2: Using Build Scripts

ZenDNNL also provides build script to automate the build process, including dependency management.

#### Usage

To see all available options for the build script, use the `--help` flag:

```bash
cd scripts
source zendnnl_build.sh --help
```

#### Output

```
usage   : zendnnl-build <options>

options :
 --all      : build and install all targets.
 --clean    : clean all targets.
 --zendnnl  : build and install zendnnl lib.
 --examples : build and install examples.
 --doxygen  : build and install doxygen docs.
 --no-deps  : don't rebuild (or clean) dependencies.

examples :
 build all targets including dependencies
 source zendnnl_build.sh --all

 build all targets if dependencies are already built
 (will fail if dependencies are not built by previous build)
 source zendnnl_build.sh --no-deps --all
```

ZenDNNL will be installed in the `build/install` directory.

### Installation Directory Structure

```
ZenDNN
 |- build
     |- install
         |- doxygen  : contains doxygen documentation.
         |- examples : contains tutorial example executables.
         |- gtestss  : contains gtests executables.
         |- zendnnl  : contains zendnnl executables
             |- include  : ZenDNNL include files.
             |- lib      : ZenDNNL lib files.
```
# 3. ZENDNNL : Examples and Tests

## 3.1. Examples

Examples could be run by executing
```bash
./examples/examples
```

## 3.2. GoogleTest

GoogleTest could be run by executing
```bash
./gtests/gtests
```
