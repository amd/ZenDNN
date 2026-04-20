
(Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.)

# The ZenDNN Build System

## Table of Contents
- [Quick Start](#quick-start)
- [Overview](#1-overview)
- [Required Toolchains](#2-required-toolchains)
- [Build Options](#3-build-options)
- [The Build Process](#4-the-build-process)


## Quick Start

Build all targets with default settings:

```bash
cd ZenDNN/scripts/
source zendnnl_build.sh --all --nproc $(nproc)
```

Or use CMake commands directly:

```bash
cd ZenDNN
mkdir -p build && cd build

# Configure with default options
cmake ..

# Build with default dependencies
cmake --build . --target all -j$(nproc)
```

After the build completes, all artifacts are installed under `ZenDNN/build/install/`.

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
| zendnnl-lib-archive and zendnnl-lib-shared | Archive (static) or shared library. At least one of these two needs to be built; both can be built in the same tree when both options are ON. | Mandatory |
| zendnnl-gtests | GTests for the library. | Optional |
| zendnnl-examples | Examples illustrating the library API usage. | Optional |
| zendnnl-benchdnn | A benchmarking framework for ZenDNN. | Optional |
| zendnnl-doxygen-docs | ZenDNN doxygen documentation. | Optional |

### 1.3 Build Modes

In order to assist development and also integration with an inference framework ZenDNN build supports the following modes

#### 1.3.1. Standalone Build

Standalone build is primarily used for library development, and independent test and benchmarking purposes. In this build, the build system discovers all its dependencies before building the library. The dependencies are discovered in one of three ways:
  - **Default**: Downloaded from public repositories and built automatically.
  - **Local**: The developer provides a local source directory of a dependency. ZenDNN builds it from that source instead of cloning. Useful for testing unreleased/beta dependency versions.
  - **Injection**: The developer provides a pre-built install directory of a dependency. ZenDNN uses it as-is without any compilation. Useful when the dependency is already built and installed separately.

#### 1.3.2 Framework Integration Build

Framework integration build assumes that ZenDNN is part of an embedding inference framework (for example PyTorch or TensorFlow). In this kind of build, ZenDNN assumes that the framework may also be building few of its dependencies, and can provide these dependencies binary and include files to ZenDNN, and ZenDNN need not build them.

## 2. Required Toolchains

- **Build System** : CMake >= 3.26
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
| ZENDNNL_BUILD_EXAMPLES | Build ZenDNN examples. These examples illustrate API and library usage in different contexts. | BOOL | ON |
| ZENDNNL_BUILD_GTEST | Build ZenDNN gtests. This is a comprehensive test suite to test all operators and features functionality of the library. | BOOL | OFF |
| ZENDNNL_BUILD_DOXYGEN | Build doxygen documentation. | BOOL |OFF |
| ZENDNNL_BUILD_BENCHDNN | Build benchdnn benchmarking tool. This tool is used to benchmark individual operators for different workloads. | BOOL | OFF |
| ZENDNNL_LIB_BUILD_ARCHIVE | Build ZenDNN archive (static) library | BOOL |ON |
| ZENDNNL_LIB_BUILD_SHARED | Build ZenDNN shared library. Build should be configure to build at least one of the archive or shared library.| BOOL |OFF |

### 3.3 Dependency Injection Options

Dependency injection allows ZenDNN to use a pre-built installation of a dependency instead of downloading and building it from source. This works in both standalone and framework builds.

When an `INJECT_DIR` path is provided, ZenDNN creates a symlink to the specified install directory and uses it directly. The install directory is expected to have `lib/` and `include/` subdirectories.

In **standalone** builds, injected dependencies are linked with `WHOLE_ARCHIVE` so the resulting `libzendnnl_archive.a` is self-contained. In **framework** builds (`ZENDNNL_FWK_BUILD=ON`), injected dependencies use `COMPILE_ONLY` linking — the framework is responsible for linking the dependencies at the final binary stage.

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_FWK_BUILD | Build ZenDNN as a part of framework build. If ON, dependencies use COMPILE_ONLY linking (framework handles final link). If OFF (default), dependencies are linked with WHOLE_ARCHIVE for a self-contained archive. | BOOL | OFF |
| ZENDNNL_AMDBLIS_INJECT_DIR | Path to a pre-built AMD-BLIS install directory. If provided, ZenDNN skips building AMD-BLIS and uses this installation. The directory must contain `lib/` and `include/` subdirectories. | PATH | (empty) |
| ZENDNNL_AOCLDLP_INJECT_DIR | Same as ZENDNNL_AMDBLIS_INJECT_DIR but for AOCL-DLP. | PATH | (empty) |
| ZENDNNL_ONEDNN_INJECT_DIR | Same as ZENDNNL_AMDBLIS_INJECT_DIR but for oneDNN. | PATH | (empty) |
| ZENDNNL_LIBXSMM_INJECT_DIR | Same as ZENDNNL_AMDBLIS_INJECT_DIR but for LIBXSMM. | PATH | (empty) |
| ZENDNNL_PARLOOPER_INJECT_DIR | Same as ZENDNNL_AMDBLIS_INJECT_DIR but for PARLOOPER. | PATH | (empty) |
| ZENDNNL_FBGEMM_INJECT_DIR | Same as ZENDNNL_AMDBLIS_INJECT_DIR but for FBGEMM. | PATH | (empty) |

### 3.4 Dependencies Options

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_DEPENDS_AOCLDLP | aocl-dlp is default BLAS backend for ZenDNN. If this option is OFF, then amd-blis becomes default BLAS backend of ZenDNN | BOOL | ON |
| ZENDNNL_DEPENDS_ONEDNN | ONEDNN is a backend of ZenDNN | BOOL | ON |
| ZENDNNL_DEPENDS_LIBXSMM | LIBXSMM is a backend for optimized small matrix multiplication in ZenDNN | BOOL | ON |
| ZENDNNL_DEPENDS_PARLOOPER | PARLOOPER is a parallel loop abstraction library used by ZenDNN | BOOL | OFF |
| ZENDNNL_DEPENDS_FBGEMM | FBGEMM is a backend for embedding_bag in ZenDNN | BOOL | ON |
| ZENDNNL_LOCAL_AMDBLIS | Use a locally available source of amd-blis instead of downloading from a public repository. By default, the source is expected at `${ZENDNNL_SOURCE_DIR}/dependencies/amdblis`. Alternatively, pass `-DAMDBLIS_ROOT_DIR=<path>` to specify a custom source directory (a symlink is created automatically). | BOOL | OFF |
| ZENDNNL_LOCAL_AOCLDLP | Same as ZENDNNL_LOCAL_AMDBLIS but for AOCL-DLP. Custom path: `-DAOCLDLP_ROOT_DIR=<path>`. | BOOL | OFF |
| ZENDNNL_LOCAL_AOCLUTILS | Same as ZENDNNL_LOCAL_AMDBLIS but for AOCL-UTILS. | BOOL | OFF |
| ZENDNNL_LOCAL_JSON | Same as ZENDNNL_LOCAL_AMDBLIS but for JSON (nlohmann-json). | BOOL | OFF |
| ZENDNNL_LOCAL_ONEDNN | Same as ZENDNNL_LOCAL_AMDBLIS but for oneDNN. Custom path: `-DONEDNN_ROOT_DIR=<path>`. | BOOL | OFF |
| ZENDNNL_LOCAL_LIBXSMM | Same as ZENDNNL_LOCAL_AMDBLIS but for LIBXSMM. Custom path: `-DLIBXSMM_ROOT_DIR=<path>`. | BOOL | OFF |
| ZENDNNL_LOCAL_PARLOOPER | Same as ZENDNNL_LOCAL_AMDBLIS but for PARLOOPER. Custom path: `-DPARLOOPER_ROOT_DIR=<path>`. | BOOL | OFF |
| ZENDNNL_LOCAL_FBGEMM | Same as ZENDNNL_LOCAL_AMDBLIS but for FBGEMM. Custom path: `-DFBGEMM_ROOT_DIR=<path>`. | BOOL | OFF |

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

In the general build process of ZenDNN, it downloads all dependencies source code from their public repositories, and source builds them. However there may be occasions when the library needs to be built with unreleased dependency source code or a pre-built dependency. ZenDNN provides two mechanisms for this:

##### Local Build (from source)

Builds the dependency from a local source directory instead of downloading:

**Option A: Place source in the dependencies directory**
- Enable `ZENDNNL_LOCAL_<DEPENDENCY>` (e.g., `ZENDNNL_LOCAL_AOCLDLP=ON`).
- Copy or symlink the source to `${ZENDNNL_SOURCE_DIR}/dependencies/<name>`.

**Option B: Point to a source directory anywhere**
- Enable `ZENDNNL_LOCAL_<DEPENDENCY>=ON` and pass `-D<DEPENDENCY>_ROOT_DIR=<path>`.
- A symlink is created automatically in `${ZENDNNL_SOURCE_DIR}/dependencies/`.

```bash
# Option A: symlink in dependencies/
ln -s /path/to/aocl-dlp dependencies/aocldlp
cmake .. -DZENDNNL_LOCAL_AOCLDLP=ON

# Option B: custom path (no manual symlink needed)
cmake .. -DZENDNNL_LOCAL_AOCLDLP=ON -DAOCLDLP_ROOT_DIR=/path/to/aocl-dlp
```

##### Dependency Injection (pre-built install)

Uses a pre-built installation of a dependency without any compilation. Provide the install prefix path (containing `lib/` and `include/` subdirectories):

```bash
cmake .. -DZENDNNL_AOCLDLP_INJECT_DIR=<path-to-aocldlp-install>

# Multiple dependencies can be injected simultaneously
cmake .. \
  -DZENDNNL_AOCLDLP_INJECT_DIR=<path-to-aocldlp-install> \
  -DZENDNNL_ONEDNN_INJECT_DIR=<path-to-onednn-install> \
  -DZENDNNL_LIBXSMM_INJECT_DIR=<path-to-libxsmm-install> \
  -DZENDNNL_FBGEMM_INJECT_DIR=<path-to-fbgemm-install>
```

##### Comparison

| | Default | Local | Injection |
|---|---|---|---|
| Input | None (auto-download) | Source directory | Install directory (pre-built) |
| Git clone | Yes | No | No |
| Compilation | Yes | Yes | No |
| Flag | (none) | `-DZENDNNL_LOCAL_<DEP>=ON` | `-DZENDNNL_<DEP>_INJECT_DIR=<path>` |

#### 4.2.2 Command Line Build

The standalone build can be done either using CMake commands or using the provided build helper script.

##### CMake Commands

All CMake commands are run from the `ZenDNN/build/` directory. The general workflow is: configure once, then build the desired targets.

**Configure:**

```bash
cd ZenDNN
mkdir -p build && cd build

# Default build
cmake ..
```

**Configure with specific options:**

```bash
# Release build with gtests and benchdnn, but skip examples
cmake -DZENDNNL_BUILD_TYPE=Release \
      -DZENDNNL_BUILD_GTEST=ON \
      -DZENDNNL_BUILD_BENCHDNN=ON \
      -DZENDNNL_BUILD_EXAMPLES=OFF \
      ..

# Debug build with both static and shared libraries
cmake -DZENDNNL_BUILD_TYPE=Debug \
      -DZENDNNL_LIB_BUILD_ARCHIVE=ON \
      -DZENDNNL_LIB_BUILD_SHARED=ON \
      ..

# Build without optional dependencies (onednn, libxsmm, fbgemm)
cmake -DZENDNNL_BUILD_TYPE=Release \
      -DZENDNNL_DEPENDS_ONEDNN=OFF \
      -DZENDNNL_DEPENDS_LIBXSMM=OFF \
      -DZENDNNL_DEPENDS_FBGEMM=OFF \
      ..

```

**Build:**

```bash
# Build with default dependencies
cmake --build . --target all -j$(nproc)
```

**Clean:**

```bash
# Clean all build artifacts (does not remove dependencies)
cmake --build . --target clean

# For a full clean including dependencies installed, remove the build directory
cd ZenDNN
rm -rf build
```

##### Build Script

To assist the build process a bash script **ZenDNN/scripts/zendnnl_build.sh** is provided. The script wraps the CMake commands above and provides a convenient interface. In order to use this script

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
| `--zendnnl-gtest` | Build and install ZenDNN gtest (requires --zendnnl) |
| `--examples` | Build and install examples (requires --zendnnl) |
| `--benchdnn` | Build and install benchdnn (requires --zendnnl) |
| `--doxygen` | Build and install doxygen docs |
| **Clean Options** | |
| `--clean` | Clean all targets |
| `--clean-all` | Clean dependencies and build folders |
| **Dependency Options** | |
| `--no-deps` | Don't rebuild (or clean) dependencies |
| `--enable-<dep>` | Enable a dependency (e.g., `--enable-parlooper`, `--enable-amdblis`) |
| **Local Dependency Options** | Build from source in `dependencies/<name>/` |
| `--local-<dep>` | Use local source from `dependencies/<dep>/` (e.g., `--local-aocldlp`, `--local-onednn`) |
| **Local Dependency with Custom Path** | Build from source at specified directory |
| `--local-<dep>-dir <path>` | Use source from `<path>` (e.g., `--local-aocldlp-dir <path>`) |
| **Dependency Injection** | Use pre-built install (no compilation) |
| `--inject-<dep> <path>` | Inject pre-built dependency from `<path>` (e.g., `--inject-aocldlp <path>`) |
| **Build Options** | |
| `--nproc <N>` | Number of processes for parallel build (default: 1) |
| `--debug` | Build in debug mode (default: release) |
| `--shared` | Build shared library (.so) in addition to static |
| `--asan` | Enable AddressSanitizer |
| `--coverage` | Enable code coverage |
| `--install-prefix <path>` | Set custom install prefix |
| `--cc <path>` | Set C compiler |
| `--cxx <path>` | Set C++ compiler |

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

# Use local onednn source from default dependencies/ directory
source zendnnl_build.sh --zendnnl --local-onednn

# Use local aocl-dlp source from a custom path
source zendnnl_build.sh --zendnnl --local-aocldlp-dir <path-to-aocldlp-source>

# Inject pre-built dependencies (no compilation)
source zendnnl_build.sh --zendnnl \
  --inject-aocldlp <path-to-aocldlp-install> \
  --inject-libxsmm <path-to-libxsmm-install> \
  --nproc 16

# Debug build with ASAN
source zendnnl_build.sh --all --debug --asan --nproc 8
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
- The cmake variable ZENDNNL_\<DEPENDENCY\>_INJECT_DIR (eg. ZENDNNL_AMDBLIS_INJECT_DIR) need to be pointed to the dependency_install_directory as shown above, and making ZenDNN build dependent on the dependency target.

> **Note**: In framework builds (`ZENDNNL_FWK_BUILD=ON`), injected dependencies use `COMPILE_ONLY` linking. This means ZenDNN's archive library does not embed the dependency symbols — the framework is responsible for linking them at the final binary stage. This avoids duplicate symbols when multiple framework components share the same dependency.

#### 4.3.2 Integrating ZenDNN Build to Framework Build

In order to assist ZenDNN build integration to a framework, it provides CMake files for including it as an external project in a framework build. These files, kept in ${ZENDNNL_SOURCE_DIR}/fwk/, are as follows

- ZenDnnlFwkMacros.cmake : This file defines few CMake macros needed by ZenDnnlFwkIntegrate.cmake
- ZenDnnlFwkIntegrate.cmake : This file provides CMake code to include ZenDNN as an external project in framework CMake build.

ZenDNN build can be integrated to the framework build using these files as follows

- Put these files in CMAKE_MODULE_PATH of the framework build.
- ZenDnnlFwkIntegrate.cmake uses a macro called *zendnnl_add_option* to set ZenDNN build options. Edit this file to set the ZenDNN options as needed. In particular the following options are to be provided

  | Option | Description |
  |--------|-------------|
  | ZENDNNL_SOURCE_DIR | ZenDNN source code. |
  | ZENDNNL_BINARY_DIR | Where ZenDNN will be built in the build tree. if unsure set ${CMAKE_CURRENT_BINARY_DIR}/zendnnl. |
  | ZENDNNL_INSTALL_PREFIX | Where ZenDNN will be installed. If unsure set ${CMAKE_INSTALL_PREFIX}/zendnnl. |
  | ZENDNNL_AMDBLIS_INJECT_DIR | Install path of amd-blis if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_AOCLDLP_INJECT_DIR | Install path of aocl-dlp if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_ONEDNN_INJECT_DIR | Install path of onednn if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_LIBXSMM_INJECT_DIR | Install path of libxsmm if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_PARLOOPER_INJECT_DIR | Install path of parlooper if framework is building it and wants to inject it to ZenDNN build. |
  | ZENDNNL_FBGEMM_INJECT_DIR | Install path of fbgemm if framework is building it and wants to inject it to ZenDNN build. |

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
