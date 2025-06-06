
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
- [Examples and Tests](#3-examples-tests)
  - [Examples](#31-examples)
  - [GoogleTest](#32-googletest)
<!-- tocstop -->

# 1. About ZenDNN*

## 1.1. Overview

ZenDNN* is being designed as a re-architected and refactored ZenDNN. ZenDNN*, like its predecessor ZenDNN is a performance primitive library for deep learning inference on AMD Zen architecture. In addition to features of ZenDNN, ZenDNN* is being designed to support the following additional features

1. Support for plug-and-play other primitive libraries like OneDNN, or other low level backends like BLAS libraries. For example, if a primitive is unimpemented in ZenDNN*, or more performant in OneDNN, the OneDNN primitive can be executed using OneDNN APIs, without getting into OneDNN code.
2. Integrated profiling and performance analysis tools. These tools will help in both development and deployment.For example, these tools could be used for instrumented profiling of a primitive, analysing performance issues and optimizing the primitive. These tools could also be used in deployment, for example,  to analyse scaling behaviour of primitives in multi-instance deployment.

## 1.2. ZenDNN* : Code Structure

ZenDNN* has the following top level directory structure

```
ZenDNN
|- build        : used to build and install the library.
|- cmake        : contains cmake modules.
|- dependencies : to downaload all dependencies.
|- examples     : tutorial examples on how to use ZenDNN* APIs.
|- scripts      : suporting shell scripts.
|- docs         : contains documentation files.
|   |- doxygen  : doxygen config file and additional pages.
|
|- zendnnl      : contains library code.
|   |- gtests   : GoogleTest files.
|   |- src      : contains library code.
|   |   |- common : contains some high level utilities needed by the library.
|   |   |- memory : implements tensor_t class.
|   |   |- operator : implements all operator classes.
|   |   |   |- common : implements base classes needed for the operators.
|   |   |   |- sample_operator : demonstrates how to create an operator.
|   |   |   |- matmul_operator : implements matrix multiplication with optional post-op.
|   |   |   |- reorder_operator : copies data between different memory formats.
|   |   |   |- compare_operator : perform element-wise comparision of tensors.
```
## 1.3. Third Party Libraries

ZenDNN* depends on the following libraries.
 - [AOCL BLIS](https://github.com/amd/blis)
 - [AOCL UTILS](https://github.com/amd/aocl-utils)
 - [GoogleTest](https://github.com/google/googletest)

ZenDNN* downloads and builds these dependencies as a part of its build process. These dependencies are also forwarded to any downstream package, that is using ZenDNN* in its build. Thus a downstream package need not figure out ZenDNN* dependencies and build them.

## 1.5. Supported OS

Please refer to the [support matrix](https://www.amd.com/en/developer/zendnn.html#getting-started) for the list of supported operating systems.

# 2. Build and Install

## 2.1. Build Dependencies

ZenDNN* needs the following tools to build

1. **CMake** >= version 3.25.
2. **g++** >= 11.2.0 toolchain.

Optionally a conda virtual environment (needs conda >= 24.1.0) with CMake 3.25 can be created using `ZenDNN/scripts/zendnnl_conda_env_create.sh`.

## 2.2. Build and install

ZenDNN* build system is a CMake super-build that builds the following package components

1. Third party libraries(TPL), hitherto referred as "dependencies".
2. ZenDNN* library. This includes its gtests also.
3. Library usage examples. These examples demonstrate how library APIs could be used.
4. Doxygen documentation.

CMake build exposes the following component targets

| Component         | CMake Target     | Depends On   |
|-------------------|------------------|--------------|
| Dependencies(TPL) | zendnnl-deps     | None         |
| ZenDNN* Library   | zendnnl          | zendnnl-deps |
| Examples          | zendnnl-examples | zendnnl      |
| Doxygen Docs      | zendnnl-doxygen  | None         |


In addition CMake build also supports standard "all" and "clean" CMake targets.

CMake build supports the following command line options.

| Option               | Values           | Default                  | Remarks                                |
|----------------------|------------------|--------------------------|----------------------------------------|
| CMAKE_BUILD_TYPE     | Debug, Release   | Release                  |                                        |
| CMAKE_INSTALL_PREFIX |                  | CMAKE_BINARY_DIR/install |                                        |
| ZENDNNL_BUILD_DEPS   | ON, OFF          | ON                       | OFF is used to prevent dependencies<br> to rebuilt (or cleaned) for<br> incremental library builds.            |

Option ZENDNNL_BUILD_DEPS needs further elaboration. Dependencies download and build is time-consuming.
During librrary development the developers may want to download and build dependencies only once during
initial build, and may want to turn off dependencies build for incremental library development and build. This
option helps developers to avoid repeated download and build of dependencies. Also since built and installed
dependencies need to be prevented from "clean", making ZENDNNL_BUILD_DEPS=OFF and reconfiguring CMake before
"clean" does not clean dependencies.

A downstream package may be interested only in the library, and may want to disable examples, doxygen docs
and gtests. Though no direct command line option is given to avoid building gtests, this can be done by
making ZENDNNL_BUILD_GTESTS=OFF in `ZenDNN/cmake/ZenDnnlComponentsOptions.cmake`.

In addition a shell script `ZenDNN/scripts/zendnnl_build.sh` is also provided to execute build process easily.

### Command Line Build

Command line build refers to use CMake command like to configure and build the library. It involves the
following steps

1. (Optional) Create and activate a conda environment having CMake version >= 3.25. A shell script in
   `scripts/zendnnl_conda_env_create.sh` can be used for this purpose. By default, this scripts creates
   a conda environment "zendnnl_build". This step is needed only if build machine does not support
   CMake >= 3.25. Exact steps are as follows

   ```
   1.1. go to ZenDNN/scripts folder.
   1.2. source zendnnl_conda_env_create.sh
   1.3  conda activate zendnnl_build
   ```

2. Create and switch to a build directory. Generally this build folder is inside top level ZenDNN folder
   but it could be anywhere. Exact steps are

   ```
   2.1 go to ZenDNN folder.
   2.2 mkdir build && cd build
   ```

3. Configure CMake using the following command

   ```
   CMake <options> <source folder>
   ```

   Example : If the build folder is created as above, and no command line CMake option is to be given
   then CMake can be configured by

   ```
   CMake ..
   ```

   Example : If we need to disable rebuilding (or cleaning) dependencies then

   ```
   CMake -DZENDNNL_BUILD_DEPS=OFF ..
   ```

4. Build a target using

   ```
   CMake --build . --target <target>
   ```

   Example : To build all targets

   ```
   CMake --build . --target all
   ```

   Example : To clean all targets

   ```
   CMake --build . --target clean
   ```

   Example : To build only library and examples

   ```
   CMake --build . --targets zendnnl zendnnl-examples
   ```

   Example : To clean all targets except dependencies (may need reconfiguration)

   ```
   1. CMake -DZENDNNL_BUILD_DEPS=OFF ..
   2. CMake --build . --target clean
   ```

### Shell Script Build

A shell script `ZenDNN/scripts/zendnn_build.sh` is provided to assist build process. In order to get
the usage

1. go to `ZenDNN/scripts` folder.
2. zendnn_build.sh --help

This will display the usage and various options. The options are self-explainatory.

ZenDNN* will be installed in the `build/install` directory.

### ZenDNN* Installation

By default installation folder is `CMAKE_BINARY_DIR/install`. If build folder is `ZenDNN/build` Then
install folder will be `ZenDNN/build/install`. All dependencies, library, examples, tests and documentation
is installed in this folder as follows

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
```

This installation also includes cmake packaging information in `install/zendnnl/lib/cmake`. Downstream
packages can use this to find zendnnl using CMake FindPackage().

# 3. ZenDNN* Examples and Tests

## 3.1. Examples

ZenDNN* provides many examples that demonstrate how the library API could be used. Executables of these examples could be found in `install/examples/bin/` folder.

## 3.2. GTests

GTests are found in `install/gtests`, and could be executed as follows
```bash
./gtests/gtests
```
# 4. Integrating ZenDNN* with Downstream Packages

Since ZenDNN* builds, installs and exports its dependencies also, it is necessary for the downstream to build and install ZenDNN*, and then use its exported targets by using CMake FindPackage() command.

```
set(zendnnl_ROOT "${ZENDNNL_INSTALL_DIR}")
set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
find_package(zendnnl REQUIRED)
if(zendnnl_FOUND)
  message(STATUS "zendnnl forund at ${zendnnl_ROOT}")
endif()
```
where ZENDNNL_INSTALL_DIR is the ZenDNN* installation folder.

ZenDNN(N) CMake package exports `zendnnl::zendnnl_archive (archive lib)` and `zendnnl::zendnnl(shared lib)`. These could be linked to any downstream target using

```
target_link_libraries(<target> INTERFACE zendnnl::zendnnl_archive)
```

Including ZenDNN* by other methods like add_subdirectory() or Fetch_Content() are not tested.

## Footnotes
ZenDNN* : ZenDNN is being designed as a re-architected and refactored ZenDNN.
