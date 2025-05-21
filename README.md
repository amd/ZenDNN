
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
- [ZenDNNL : Tests](#3-zendnnl-tests)
  - [Unit Tests](#31-unit-tests)
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
ZenDNNL
|- build        : used to build and install the library.
|- cmake        : contains cmake modules.
|- dependencies : to downaload all dependencies.
|- doxygen      : doxygen config file and additional pages.
|- examples     : tutorial examples on how to use ZenDNNL APIs.
|- scripts      : suporting shell scripts.
|- src          : contains library code.
|   |- common : contains some high level utilities needed by the library.
|   |- memory : implements tensor_t class.
|   |- operator : implements all operator classes.
|   |   |- common : implements base classes needed for the operators.
|   |   |- sample_operator : demonstrates how to create an operator.
|   |   |- matmul_operator : implements matrix multiplication with optional post-op.
|   |   |- compare_operator : perform element-wise comparision of tensors.
```
## 1.4. Third Party Libraries

ZenDNNL depends on the following library.
 * [AOCL BLIS](https://github.com/amd/blis)

## 1.5. Supported OS

Refer to the [support matrix](https://www.amd.com/en/developer/zendnn.html#getting-started) for the list of supported operating systems.

# 2. ZENDNNL : Build and Install

## 2.1. Build Dependencies

ZENDNNL needs the following tools to build
1. CMake >= version 3.25.
2. g++ >= 13.x toolchain.
3. conda version >= 24.1.0.

Gcc 13.x could be installed using scripts/zendnnl_install_gcc.sh. A conda virtual environment with CMake 3.25 can be created using scripts/zendnnl_conda_env_create.sh.

ZenDNNL additionally depends on the following packages
1. AMD AOCL BLIS
2. Googletest

These packages could either be preinstalled and their location could be passed to ZenDNNL, or
if they are not preinstalled, ZenDNNL can download and build them. These options are provided
in cmake/ConfigOptions.cmake.

## 2.2. Build and install

1. Open cmake/ConfigOptions.cmake. If pre-installed AMD BLIS is to be used, then set
   ZENDNNL_AMDBLIS_USE_LOCAL_REPO to ON and provide local path to ZENDNNL_AMDBLIS_DIR. If
   ZenDNNL is required to download AMD BLIS then leave the settings as they are.
2. To build googletest, open cmake/ConfigOptions.cmake and set ZENDNNL_DEPENDS_GTEST to ON.
3. Activate conda environment. (scripts/zendnnl_conda_env_create.sh creates a conda
   environment named zendnnltorch).

### 2.2.1. Create a build directory 
```bash
mkdir build && cd build
```
### 2.2.2. To configure cmake
```bash
cmake ../
```
### 2.2.3. To build and install ZenDNNL
```bash
make install
```

ZenDNNL is installed in build/install directory.
```
ZenDNNL
 |- build
     |- install
         |- doxygen  : contains doxygen documentation.
         |- examples : contains tutorial example executables.
         |- include  : ZenDNNL include files.
         |- lib      : ZenDNNL lib files.
```
# 3. ZENDNNL : Tests

## 3.1. Unit Tests

Examples could be run by executing
```bash
./examples/examples
```

## 3.2. GoogleTest

GoogleTest could be run by executing
```bash
./gtests/gtests
```

### known issues

1. Running "make install" again rebuilds AMD BLIS. A workaround for this is to
   delete everything inside build directory, configure cmake again with "cmake ../",
   and then do "make install".
