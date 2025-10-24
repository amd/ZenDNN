
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNN* : Accelerated Deep Learning Inference on AMD Zen Architecture

<!-- toc -->
- [About ZenDNN*](#1-about-ZenDNN*)
  - [Overview](#12-overview)
  - [Structure](#13-structure)
  - [Third Party Libraries](#14-third-party-libraries)
  - [Supported OS](#15-supported-os)
- [Build and Install](#2-build-and-install)
  - [Build Dependencies](#21-build-dependencies)
  - [Build and install](#22-build-and-install)
- [Examples, Tests and Benchmarks](#3-ZenDNN*-examples-tests-and-benchmarks)
  - [Examples](#31-examples)
  - [GoogleTest](#32-gtests)
  - [BenchDNN](#33-benchdnn)
<!-- tocstop -->

# 1. About ZenDNN*

## 1.1. Overview

ZenDNN* is being designed as a re-architected and refactored ZenDNN. ZenDNN*, like its predecessor ZenDNN, is a performance primitive library for deep learning inference on AMD Zen architecture. In addition to features offered by ZenDNN, ZenDNN* is intended to support the following additional features:

1. Support to plug-and-play other primitive libraries like OneDNN, or other low level backends like BLAS libraries. For example, if a primitive is unimpemented in ZenDNN*, or more performant in OneDNN, the OneDNN primitive can be executed using OneDNN APIs, without getting into OneDNN code.
2. Provide integrated profiling and performance analysis tools, which will help in both development and deployment.For example, these tools could be used for instrumented profiling of a primitive, analysing performance issues, and to optimize the primitive. These tools could also be used in deployment; for example, to analyse scaling behaviour of primitives in a multi-instance deployment.

## 1.2. ZenDNN* : Code Structure

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
 - [AOCL BLIS](https://github.com/amd/blis)
 - [AOCL UTILS](https://github.com/amd/aocl-utils)
 - [GoogleTest](https://github.com/google/googletest)

ZenDNN* downloads and builds these dependencies as a part of its build process. These dependencies are also forwarded to any downstream package using ZenDNN* in its build. Thus a downstream package does not need to figure out ZenDNN* dependencies and build them.

## 1.5. Supported OS

Refer to the [support matrix](https://www.amd.com/en/developer/zendnn.html#getting-started) for the list of supported operating systems.

# 2. Build and Install

## 2.1. Build Dependencies

Install the following prerequisites before installing ZenDNN*:

1. **CMake** >= version 3.25.
2. **g++** >= 11.2.0 toolchain.

Optionally, create a Conda virtual environment (Conda >= 24.1.0) with CMake >= 3.25.

## 2.2. Build and install

The ZenDNN* build system is a CMake super-build designed to compile the following package components:

1. Third party libraries (TPL), hitherto referred as "dependencies".
2. ZenDNN* library including its gtests.
3. Library usage examples demonstrating the usage of library APIs.
4. Doxygen documentation.

CMake build exposes the following component targets:

| Component         | CMake Target     | Depends On   |
|-------------------|------------------|--------------|
| Dependencies(TPL) | zendnnl-deps     | None         |
| ZenDNN* Library   | zendnnl          | zendnnl-deps |
| Examples          | zendnnl-examples | zendnnl      |
| BenchDNN          | zendnnl-benchdnn | zendnnl      |
| Doxygen Docs      | zendnnl-doxygen  | None         |


In addition, the CMake build also supports the standard "all" and "clean" CMake targets.

CMake build supports the following command line options.

| Option               | Values           | Default                  | Remarks                                |
|----------------------|------------------|--------------------------|----------------------------------------|
| CMAKE_BUILD_TYPE     | Debug, Release   | Release                  |                                        |
| CMAKE_INSTALL_PREFIX |                  | CMAKE_BINARY_DIR/install |                                        |
| ZENDNNL_BUILD_DEPS   | ON, OFF          | ON                       | OFF is used to prevent dependencies<br> to rebuild (or cleaned) for<br> incremental library builds.            |

The ZENDNNL_BUILD_DEPS option needs further elaboration. Downloading and building Dependencies is time-consuming.
During library development developers may want to download and build dependencies only once during
the initial build, and may want to turn off Dependencies build for incremental library development and build. This
option helps developers avoid repeatedly downloading and building Dependencies. As built and installed
dependencies must be prevented from "clean", setting ZENDNNL_BUILD_DEPS=OFF and reconfiguring CMake before
"clean" does not clean dependencies. A downstream package may be interested only in the library, and may want to enable examples, doxygen docs, and gtests. Examples are built by default, while doxygen docs and gtests are disabled by default. These can be
controlled by setting the appropriate options in `ZenDNN/cmake/ZenDnnlComponentsOptions.cmake`:
- `ZENDNNL_BUILD_EXAMPLES=ON/OFF` (default: ON)
- `ZENDNNL_BUILD_DOXYGEN=ON/OFF` (default: OFF)
- `ZENDNNL_BUILD_GTEST=ON/OFF` (default: OFF)
- `ZENDNNL_BUILD_BENCHDNN=ON/OFF` (default: OFF)

In addition, a shell script `ZenDNN/scripts/zendnnl_build.sh` is also provided to execute the build process easily.

### Command Line Build

Command Line Build refers to the usage of CMake command to configure and build the library. It consists of the
following steps:

1. Create and switch to a build directory. Generally, this build folder is inside the top level ZenDNN folder
   but it could be anywhere else. Follow these steps:

   2.1 Go to the ZenDNN folder.
   ```
   mkdir build && cd build
   ```

2. Configure CMake using the following command:

   ```
   cmake <options> <source folder>
   ```

   Example: If the build folder is created as described and no command line CMake option is to be given,
   configure CMake by:

   ```
   cmake ..
   ```

   Example: To disable rebuilding (or cleaning) dependencies:

   ```
   cmake -DZENDNNL_BUILD_DEPS=OFF ..
   ```

3. Build a target using:

   ```
   cmake --build . --target <target>
   ```

   Example: To build all targets

   ```
   cmake --build . --target all
   ```

   Example: To clean all targets:

   ```
   cmake --build . --target clean
   ```

   Example: To build only library and examples:

   ```
   cmake --build . --target zendnnl zendnnl-examples
   ```

   Example: To clean all targets except dependencies (may need reconfiguration):

   ```
   cmake -DZENDNNL_BUILD_DEPS=OFF ..
   cmake --build . --target clean
   ```

### Shell Script Build

A shell script `ZenDNN/scripts/zendnn_build.sh` is provided to assist the build process. To get
the usage instructions:

1. Go to `ZenDNN/scripts` folder.
2. bash zendnn_build.sh --help

This displays the various options and the usage.

ZenDNN* will be installed in the `build/install` directory.

### ZenDNN* Installation

The default installation folder is `CMAKE_BINARY_DIR/install`. If the build folder is `ZenDNN/build`, the 
install folder will be `ZenDNN/build/install`. All dependencies, libraries, examples, tests, and documentation
will be installed in this folder with the following folder structure:

```
ZenDNN
 |- build
     |- install
         |- deps     : contains installed dependencies
         |   |- amdblis   : amdblis installation
         |   |- aoclutils : aoclutils installation
         |   |- gtests    : gtests installation
         |- zendnnl  : zendnnl library installation
         |- doxygen  : doxygen documentation.
         |- examples : tutorial example executables.
         |- gtests   : gtests executables.
         |- benchdnn : benchdnn executables.
```

This installation also includes CMake packaging information in `install/zendnnl/lib/cmake`. Downstream
packages can use this to find zendnnl using CMake FindPackage().

# 3. ZenDNN* Examples, Tests and Benchmarks

## 3.1. Examples

ZenDNN* provides many examples demonstrating the usage of the library API. Executables of these examples can be found in the `install/examples/bin/` folder.

For detailed logging support and control, see [logging.md](docs/logging.md).

## 3.2. GTests

GTests are found in `install/gtests`, and can be executed as follows:
```bash
./gtests/gtests
```

For detailed usage instructions, refer to the [GTests's README](zendnnl/gtests/Readme.md).

## 3.3. BenchDNN

ZenDNN* includes `benchdnn`, a high-performance benchmarking utility designed to rigorously assess the efficiency of deep learning operators within the ZenDNN library. BenchDNN provides detailed performance analysis capabilities for primitives like matrix multiplication (matmul) and reorder operations.

For detailed usage instructions and configuration options, refer to the [BenchDNN's README](benchdnn/README.md).

# 4. Integrating ZenDNN* with Downstream Packages

As ZenDNN* builds, installs, and exports its dependencies, it is necessary for the downstream to build and install ZenDNN*, and then use its exported targets by using the CMake FindPackage() command.

```
set(zendnnl_ROOT "${ZENDNNL_INSTALL_DIR}")
set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
find_package(zendnnl REQUIRED)
if(zendnnl_FOUND)
  message(STATUS "zendnnl forund at ${zendnnl_ROOT}")
endif()
```
where, ZENDNNL_INSTALL_DIR is the ZenDNN* installation folder.

ZenDNN* CMake package exports `zendnnl::zendnnl_archive (archive lib)` and `zendnnl::zendnnl(shared lib)`. These can be linked to any downstream target using:

```
target_link_libraries(<target> INTERFACE zendnnl::zendnnl_archive)
```

Note that testing has not been performed on including ZenDNN* by other methods such as add_subdirectory() or Fetch_Content().

>ZenDNN* : ZenDNN is currently undergoing a strategic re-architecture and refactoring to enhance performance, maintainability, and scalability.