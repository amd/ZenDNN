
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# The ZenDNN Build System

## Table of Contents
- [Overview](#1-overview)
- [Required Toolchains](#2-required-toolchains)
- [Build Options](#3-build-options)
- [The Build Process](#4-the-build-process)

## 1. Overview

ZenDNN as an inference acceleration library, depends on multiple BLAS and other backends, builds multiple components including its dependencies, components like GTest, examples and benchmarking infrastructure. It can be built as a stand-alone library for development, test and benchmarking purposes, or it can be built as a part of a larger inference framework stack. It also needs to support both Debug and Release configurations. ZenDNN build system takes care of these needs.

### 1.1 Dependencies

ZenDNN depends on the following third party libraries (TPL), referred in this document as dependencies,

| Dependency | Description | Mandatory |
|------------|-------------|-----------|
| nlohmann-json | To read and write json files | Mandatory |
| aocl-utils | To detect runtime system information | Mandatory |
| amd-blis or aocl-dlp | As default BLAS backend. Any one of these two need to be present | Mandatory |
| onednn | In an unlikely situation of onednn kernels doing better than ZenDNN | Optional |
| libxsmm         | Optimized small matrix multiplication library | Optional |
| parlooper       | Parallel loop abstraction library             | Optional |
| fbgemm          | A low-precision, high-performance matrix multiplication and embedding bag library | Optional |

### 1.2 Components

ZenDNN builds the following components

| Component | Description | Mandatory |
|-----------|-------------|-----------|
| zendnnl-deps | All dependencies listed above. If these are not built at least once (or provided by embedding inference framework) the library build will fail. | Mandatory |
| zendnnl-lib-archive and zendnnl-lib-shared | Archive (static) or shared library. At least one of these two needs to be built. | Mandatory |
| zendnnl-gtests | GTests for the library. | Optional |
| zendnnl-examples | Examples illustrating the library API usage. | Optional |
| zendnnl-benchdnn | A benchmarking framework for ZenDNN. | Optional |
| zendnnl-doxygen-docs | ZenDNN doxygen documentation. | Optional |

### 1.3 Build Modes

In order to assist development and also integration with an inference framework ZenDNN build supports the following modes

#### 1.3.1. Standalone Build

Standalone build is primarily used for library development, and independent test and benchmarking purposes. In this build, the  build system discovers all its dependencies before building the library. The dependencies are discovered as follows
  - As a part of the build, they are downloaded and built before the library, or
  - The developer provides a local build of a dependency. This feature is useful if a developer need to experiment with unreleased/beta version of the dependency.

#### 1.3.2 Framework Integration Build

Framework integration build assumes that ZenDNN is part of an embedding inference framework (for example PyTorch or TensorFlow). In this kind of build, ZenDNN assumes that the framework may also be building few of its dependencies, and can provide these dependencies binary and include files to ZenDNN, and ZenDNN need not build them.

## 2. Required Toolchains

- **Build System** : CMake >= 3.25
- **Compilers**    : GNU g++ >= 11.2.0

## 3. Build Options

In order to configure the build according to dependencies, components, and stand-alone/ embedding framework, the build system provides the following options

### 3.1 Configuration Options

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_SOURCE_DIR | Path to ZenDNN source | PATH | ${CMAKE_SOURCE_DIR} |
| ZENDNNL_INSTALL_PREFIX | Library install prefix | PATH |${ZENDNNL_SOURCE_DIR}/build/install |
| ZENDNNL_BUILD_TYPE | Library build type (Release/Debug)| STRING | Release |
| ZENDNNL_MESSAGE_LOG_LEVEL | CMake log level | STRING | Debug |
| ZENDNNL_VERBOSE_MAKEFILE | CMake verbose makefile | BOOL | ON |

### 3.2 Components Options

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_BUILD_DEPS | Download and build the dependencies. This is generally used during development to avoid repeated download and build the dependencies. If dependencies are not built at least once before this option is turned off, the build will fail.| BOOL |ON |
| ZENDNNL_BUILD_EXAMPLES | Build zendnnl examples. These examples illustrate API and library usage in different contexts. | BOOL | ON |
| ZENDNNL_BUILD_GTESTS | Build zendnnl gtests. This is a comprehensive test suite to test all operators and features functionality of the library. | BOOL | ON |
| ZENDNNL_BUILD_DOXYGEN | Build doxygen documentation. | BOOL |OFF |
| ZENDNNL_BUILD_BENCHDNN | Build benchdnn benchmarking tool. This tool is used to benchmark individual operators for different workloads. | BOOL |ON |
| ZENDNNL_LIB_BUILD_ARCHIVE | Build zendnnl archive (static) library | BOOL |ON |
| ZENDNNL_LIB_BUILD_SHARED | Build zendnnl shared library. Build should be configure to build at least one of the archive or shared library.| BOOL |OFF |

### 3.3 Framework Integration Options

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_FWK_BUILD | Build ZenDNN as a part of framework build. This option need to be ON if the library is being built as a part of a framework. If this option is OFF the library will be built as a standalone package. | BOOL | OFF |
| ZENDNNL_AMDBLIS_FWK_DIR | If the framework is building a ZenDNN dependency (for example AMDBLIS in this case), it can inform the library where this dependency is installed, by populating this variable. If populated, ZenDNN assumes that the dependency is already built before library build starts, the binaries of the dependency are kept in ${ZENDNNL_AMDBLIS_FWK_DIR}/lib, and the include files are kept in ${ZENDNNL_AMDBLIS_FWK_DIR}/include. The framework build will have to ascertain by creating proper CMake dependency tree that this dependency is built before the library starts building. ZenDNN provides a CMake include file (fwk/ZenDNNLFwkIntergate.cmake) to enable it. If this variable is not populated, then ZenDNN assumes that the dependency is not being provided by the framework, and builds it the way it does for standalone build. | PATH | NO DEFAULT |
| ZENDNNL_AOCLDLP_FWK_DIR | Its usage is same as that of ZENDNNL_AMDBLIS_FWK_DIR except it is for AOCLDLP dependency | PATH | NO DEFAULT |
| ZENDNNL_ONEDNN_FWK_DIR | Its usage is same as that of ZENDNNL_AMDBLIS_FWK_DIR except it is for ONEDNN dependency | PATH | NO DEFAULT |
| ZENDNNL_LIBXSMM_FWK_DIR | Its usage is same as that of ZENDNNL_AMDBLIS_FWK_DIR except it is for LIBXSMM dependency | PATH | NO DEFAULT |
| ZENDNNL_PARLOOPER_FWK_DIR | Its usage is same as that of ZENDNNL_AMDBLIS_FWK_DIR except it is for PARLOOPER dependency | PATH | NO DEFAULT |
| ZENDNNL_FBGEMM_FWK_DIR | Its usage is same as that of ZENDNNL_AMDBLIS_FWK_DIR except it is for FBGEMM dependency | PATH | NO DEFAULT |

### 3.4 Dependencies Options

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_DEPENDS_AOCLDLP | aocl-dlp is default BLAS backend for ZenDNN. If this option is OFF, then amd-blis becomes default BLAS backend of ZenDNN | BOOL | ON |
| ZENDNNL_DEPENDS_ONEDNN | ONEDNN is a backend of ZenDNN | BOOL | ON |
| ZENDNNL_DEPENDS_LIBXSMM | LIBXSMM is a backend for optimized small matrix multiplication in ZenDNN | BOOL | ON |
| ZENDNNL_DEPENDS_PARLOOPER | PARLOOPER is a parallel loop abstraction library used by ZenDNN | BOOL | OFF |
| ZENDNNL_DEPENDS_FBGEMM | FBGEMM is a backend for embedding_bag in ZenDNN | BOOL | ON |
| ZENDNNL_LOCAL_AMDBLIS | Use a locally available code of amd-blis instead of downloading from a public repository. This option can be used by developers to test the library with the dependency version not yet publicly available, or a version still unstable. In such a case the developer still need to copy (or provide soft link) the dependency code to ${ZENDNNL_SOURCE_DIR}/dependencies directory. | BOOL | OFF |
| ZENDNNL_LOCAL_AOCLDLP | Its usage is same as that of ZENDNNL_LOCAL_AMDBLIS, except it is for AOCLDLP | BOOL | OFF |
| ZENDNNL_LOCAL_AOCLUTILS | Its usage is same as that of ZENDNNL_LOCAL_AMDBLIS, except it is for AOCLUTILS | BOOL | OFF |
| ZENDNNL_LOCAL_JSON | Its usage is same as that of ZENDNNL_LOCAL_AMDBLIS, except it is for JSON (nlohmann-json) | BOOL | OFF |
| ZENDNNL_LOCAL_ONEDNN | Its usage is same as that of ZENDNNL_LOCAL_AMDBLIS, except it is for ONEDNN | BOOL | OFF |
| ZENDNNL_LOCAL_LIBXSMM | Its usage is same as that of ZENDNNL_LOCAL_AMDBLIS, except it is for LIBXSMM | BOOL | OFF |
| ZENDNNL_LOCAL_PARLOOPER | Its usage is same as that of ZENDNNL_LOCAL_AMDBLIS, except it is for PARLOOPER | BOOL | OFF |
| ZENDNNL_LOCAL_FBGEMM | Its usage is same as that of ZENDNNL_LOCAL_AMDBLIS, except it is for FBGEMM | BOOL | OFF |

## 4. The Build Process

### 4.1 General Build Process

ZenDNN follows CMake super-build structure, where all the dependencies, the library, and various other components (examples, documentation and benchdnn benchmarking tool) are built as independent external projects, and the top level CMake orchestrates these independent external builds. In general build process is as follows

- A dependency (for example amdblis or aoclutils) is downloaded from its public repository to ${ZENDNNL_SOURCE_DIR}/dependencies/ (or provided by the developer in the same folder in case of a local/unreleased/beta dependency source code to be used). This dependency is then source built using cmake external project, and installed. All dependencies get installed in ${ZENDNNL_INSTALL_PREFIX}/deps/<dependency name> directory.

- The library itself is also built as an cmake external project. While building the library tries to find all the dependencies installed in ${ZENDNNL_INSTALL_PREFIX}/deps/ directory. If it finds any mandatory or enabled dependency missing, it reports it and exits the build.

- Library GTests, if enabled, are built as a part of the library.

- Post library build, other components (examples, benchdnn) are built as cmake external project. These components try to find the library in ${ZENDNNL_INSTALL_PREFIX}/zendnnl/ directory. A component is installed in ${ZENDNNL_INSTALL_PREFIX}/[component] (eg. ${ZENDNNL_INSTALL_PREFIX}/examples) etc directories.

- For each dependency (eg. amd-blis), the build adds a compiler option ZENDNNL_DEPENDS_[DEPENDENCY] (eg. ZENDNNL_DEPENDS_AMDBLIS). If the dependency is enabled, this is set to one, else set to zero. This compiler option can be used in c++  code to enable dependency specific code

```
#if ZENDNNL_DEPENDS_AMDBLIS
 <amd-blis dependent code>
#else
 <alternative amd-blis independent code>
#endif
```

The CMake build exposes the following component targets

| Component         | CMake Target     | Depends On   |
|-------------------|------------------|--------------|
| Dependencies(TPL) | zendnnl-deps     | None         |
| ZenDNN Library    | zendnnl          | zendnnl-deps |
| Examples          | zendnnl-examples | zendnnl      |
| GTests            | zendnnl-gtest    | zendnnl      |
| BenchDNN          | zendnnl-benchdnn | zendnnl      |
| Doxygen Docs      | zendnnl-doxygen  | None         |

### 4.2 Standalone Build

This mode is used primarily for library development, test and benchmarking its operators. In this mode, any optional dependencies and components can be enabled/disabled as given in [Build Options](#3-build-options). By default, the ZenDNN build will download and source build all the dependencies before trying to build the library.

#### 4.2.1 Providing Local Dependencies

In the general build process of ZenDNN, it downloads all dependencies source code from their public repositories, and source builds them. However there may be occasions, when the library need to be built with unreleased dependency source code. In such a case the following is needed to be done

- Enable build option ZENDNNL_LOCAL_[DEPENDENCY] (for example ZENDNNL_LOCAL_AMDBLIS).
- Copy (or create soft link of) the local dependency directory to ${ZENDNNL_SOURCE_DIR}/dependencies.

If a local dependency is provided, the build will not try to download dependency source code, and work with local source code.

#### 4.2.2 Command Line Build

The standalone build can be done using command line CMake configuration and build, giving command line CMake options as given in [Build Options](#3-build-options). However to assist the build process a bash script **ZenDNN/scripts/zendnnl_build.sh** is provided. In order to use this script

- Go to the `ZenDNN/scripts/` directory,
- Invoke `source zendnnl_build.sh --help` to list down all the build options.
- Invoke `source zendnnl_build.sh` with required build options to build the library (and its components).

##### Build Script Options

> **Important**: Use `--all` to build everything, OR use `--zendnnl` with other targets.
> Components like `--zendnnl-gtest`, `--examples`, and `--benchdnn` depend on the zendnnl library.
> Always include `--zendnnl` when building these components individually.

| Option | Description |
|--------|-------------|
| **Build Targets** | |
| `--all` | Build and install all targets |
| `--zendnnl` | Build and install zendnnl lib |
| `--zendnnl-gtest` | Build and install zendnnl gtest (requires --zendnnl) |
| `--examples` | Build and install examples (requires --zendnnl) |
| `--benchdnn` | Build and install benchdnn (requires --zendnnl) |
| `--doxygen` | Build and install doxygen docs |
| **Clean Options** | |
| `--clean` | Clean all targets |
| `--clean-all` | Clean dependencies and build folders |
| **Dependency Options** | |
| `--no-deps` | Don't rebuild (or clean) dependencies |
| `--enable-parlooper` | Enable parlooper |
| `--enable-amdblis` | Enable amdblis (disables aocldlp which is default) |
| **Local Dependency Options** | Requires source in `dependencies/<name>/` |
| `--local-amdblis` | Use local amdblis |
| `--local-aocldlp` | Use local aocldlp |
| `--local-aoclutils` | Use local aoclutils |
| `--local-json` | Use local json |
| `--local-onednn` | Use local onednn |
| `--local-libxsmm` | Use local libxsmm |
| `--local-parlooper` | Use local parlooper |
| `--local-fbgemm` | Use local fbgemm |
| **Build Options** | |
| `--nproc <N>` | Number of processes for parallel build (default: 1) |

##### Build Script Examples

```bash
# Build all targets including dependencies
source zendnnl_build.sh --all

# Build everything with parallel jobs
source zendnnl_build.sh --all --nproc 8

# Build library only
source zendnnl_build.sh --zendnnl

# Build library + gtests
source zendnnl_build.sh --zendnnl --zendnnl-gtest

# Build library + gtests + examples
source zendnnl_build.sh --zendnnl --zendnnl-gtest --examples

# Rebuild without re-downloading dependencies
source zendnnl_build.sh --no-deps --all

# Use local onednn source
source zendnnl_build.sh --zendnnl --local-onednn
```

### 4.3 Framework Integration Build

ZenDNN can be informed that it is being built as a part of an inference framework by enabling ZENDNNL_FWK_BUILD option.

#### 4.3.1 Providing Dependencies Pre-installed by the Framework

It is possible that few of the ZenDNN dependencies are being built by the framework, and rebuilding them again as a part of ZenDNN may result in bloated binaries, or symbol name clashes. ZenDNN provides a way to inject these dependencies being built by the framework into the build process of ZenDNN. The build system provides a uniform way to inject and dependency to ZenDNN build as follows
- It assumes that the dependency is preinstalled by the framework, with the following kind of directory configuration
```
dependency_install_directory
|
|- lib : contains all the dependency library files.
|  |- cmake/CMake (optional): provides cmake config files needed by find_package tool of CMake. 
|- include : contains all include files of the dependency.
|
```
- If the dependency provides cmake configuration files in lib/cmake, it uses config mode of CMake find_package tool to discover the dependency. If lib/cmake is not present, it has a Find\<dependency\>.cmake to find the dependency using module mode of CMake find_package tool.
- The cmake variable ZENDNNL_\<DEPENDENCY\>_FWK_DIR (eg. ZENDNNL_AMDBLIS_FWK_DIR) need to be pointed to the dependency_install_directory as shown above, and making ZenDNN build dependent on the dependency target.

#### 4.3.1 Integrating ZenDNN Build to Framework Build

In order to assist ZenDNN build integration to a framework, it provides CMake files for including it as an external project in a framework build. These files, kept in ${ZENDNL_SOURCE_DIR}/fwk/, are as follows

- ZenDnnlFwkMacros.cmake : This file defines few CMake macros needed by ZendnnlFwkIntegrate.cmake
- ZenDnnlFwkIntegrate.cmake : This file provides CMake code to include ZenDNN as an external project in framework CMake build.

ZenDNN build can be integrated to the framework build using these files as follows

- Put these files in CMAKE_MODULE_PATH of the framework build.
- ZenDnnlFwkIntegrate.cmake uses a macro called *zendnnl_add_option* to set ZenDNN build options. Edit this file to set the ZenDNN options as needed. In particular the following options are to be provided

  | Option | Description |
  |--------|-------------|
  | ZENDNN_SOURCE_DIR | ZenDNN source code. |
  | ZENDNNL_BINARY_DIR | Where ZenDNN will be built in the build tree. if unsure set ${CMAKE_CURRENT_BINARY_DIR}/zendnnl. |
  | ZENDNNL_INSTALL_DIR | Where ZenDNN will be built in the build tree. if unsure set ${CMAKE_INSTALL_PREFIX}/zendnnl. |
  | ZENDNNL_AMDBLIS_FWK_DIR | Install path of amd-blis if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_AOCLDLP_FWK_DIR | Install path of aocl-dlp if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_ONEDNN_FWK_DIR | Install path of onednn if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_LIBXSMM_FWK_DIR | Install path of libxsmm if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_PARLOOPER_FWK_DIR | Install path of parlooper if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_FBGEMM_FWK_DIR | Install path of fbgemm if framework is building it and wants to inject it to ZenDNN build. |

- Make *fwk_zendnnl* target dependent on injected dependencies by editing its "add_dependencies(fwk_zendnnl...)" command.
- Include the edited ZenDnnlFwkIntegrate.cmake in the framework build flow, where ZenDNN is to be built.
- Make any other targets that depend on ZenDNN dependent on *fwk_zendnnl* target.

**ManyLinux Docker Hack:**

ManyLinux Docker toolchain builds aocl-utils (a ZenDNN dependency) library binaries in ${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils/lib64, instead of ${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils/lib. This causes build to fail in ManyLinux Docker. ZenDnnlFwkIntegrate.cmake provides a hack, in the form of checking an environment variable ZENDNNL_MANYLINUX_BUILD. If this variable is set it uses aocl-utils library path suitable for ManyLinux Docker. ManyLinux Dockerfile need to set this environment variable.

### 4.4 ZenDNN Installation

ZenDNN install folder can be configured by providing CMAKE_INSTALL_PREFIX at the command line. The default installation folder is `CMAKE_BINARY_DIR/install`. If the folder is `ZenDNN/build`, the install folder will be `ZenDNN/build/install`. All dependencies, libraries, examples, tests, and documentation will be installed in this folder with the following folder structure:

```
ZenDNN
 |- build
     |- install
         |- deps     : contains installed dependencies
         |   |- aocldlp   : aocldlp installation
         |   |- aoclutils : aoclutils installation
         |   |- json      : json installation
         |   |- gtests    : gtests installation
         |   |- <any other optional dependencies>
         |- zendnnl  : zendnnl library installation
         |- doxygen  : doxygen documentation.
         |- examples : tutorial example executables.
         |- gtests   : gtests executables.
         |- benchdnn : benchdnn executables.
```

This installation also includes CMake packaging information in `install/zendnnl/lib/cmake`. Downstream
packages can use this to find zendnnl using CMake find_package tool.

Note that testing has not been performed on including ZenDNN by other methods such as add_subdirectory() or Fetch_Content().
