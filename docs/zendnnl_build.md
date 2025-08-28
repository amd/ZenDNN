
(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNN* Build

## Overview

ZenDNN* as an inference acceleration library, depends on multiple BLAS and other backends, builds multiple components including its dependencies, components like GTest, examples and benchmarking infrastructure. It can be built as a stand-alone library for development, test and benchmarking puposes, or it can be built as a part of a larger inference framework stack. It also needs to support both Debug and Release configurations. ZenDNN* build system takes care of these needs.

ZenDNN* depends on the following dependencies

| Dependency | Description | Mandatory |
|------------|-------------|-----------|
| nlohmann-json | To read and write json files | Mandatory |
| aocl-utils | To detect runtime system information | Mandatory |
| amd-blis or aocl-dlp | As default BLAS backend. Any one of these two need to be present | Mandatory |
| onednn | In an unlikely situation of onednn kernels doing better than ZenDNN* | Optional |


ZenDNN* build the following components

| Component | Description | Mandatory |
|-----------|-------------|-----------|
| zendnnl-deps | All dependencies listed above. If these are not built at least once (or provided by embedding inference framework) the library build will fail. | Mandatory |
| zendnnl-lib-archive and zendnnl-lib-shared | Archive (static) or shared library. At least one of these two needs to be built. | Mandatory |
| zendnnl-gtests | GTests for the library. | Optional |
| zendnnl-examples | Examples illuminating the libray API usage. | Optional |
| zendnnl-benchdnn | A benchmarking framework for ZenDNN*. | Optional |
| zendnnl-doxygen-docs | ZenDNN* doxygen documentation. | Optional |


In order to assist development and also integration with an inference  framework ZenDNN* build supports the following modes

| Build Mode | Description |
|------------|-------------|
| Standalone Build | This build is primarily used by the developers of the library and its components. In this mode ZenDNN* discovers all its dependencies (by either downloading and building them, or explicitly being provided by the developers), builds the dependencies, library and other components. |
| Framework Build | Framework build assumes that ZenDNN* is part of an embedding inference framework (for example PyTorch or TensorFlow). In this kind of build, ZenDNN* assumes that the framework may also be buliding few of its dependencies, and can provide these dependencies binary and include files to ZenDNN*, and ZenDNN* need not build them.|


## Supported Toolchains

- **Build System** : CMake >= 3.25
- **Compilers**    : GNU g++ >= 13.3.0

## Build Options

In order to configure the build according to dependencies, components, and stand-alone/ embedding framework, ZenDNN* provides the following options

### Configuration Options

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_SOURCE_DIR | Path to ZenDNN* source | PATH | ${CMAKE_SOURCE_DIR} |
| ZENDNNL_INSTALL_PREFIX | Library install prefix | PATH |${ZENDNNL_SOURCE_DIR}/build/install |
| ZENDNNL_BUILD_TYPE | Library build type (Release/Debug)| STRING | Release |
| ZENDNNL_MESSAGE_LOG_LEVEL | CMake log level | STRING | Debug |
| ZENDNNL_VERBOSE_MAKEFILE | CMake verbose makefile | BOOL | ON |

### Components Options

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_BUILD_DEPS | Download and build the dependencies. This is generally used during development to avoid repeated download and build the dependencies. If dependencies are not built at least once before this option is turned off, the build will fail.| BOOL |ON |
| ZENDNNL_BUILD_EXAMPLES | Build zendnnl examples | BOOL | ON |
| ZENDNNL_BUILD_GTESTS | Build zendnnl gtests | BOOL | ON |
| ZENDNNL_BUILD_DOXYGEN | Build doxygen documantation | BOOL |OFF |
| ZENDNNL_BUILD_BENCHDNN | Build benchdnn benchmarking tool | BOOL |ON |
| ZENDNNL_LIB_BUILD_ARCHIVE | Build zendnnl archive (static) library | BOOL |ON |
| ZENDNNL_LIB_BUILD_SHARED | Build zendnnl shared library. Build should be configure to build at least one of the archive or shared library.| BOOL |OFF |

### Framework Integration Options

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_FWK_BUILD | Build ZenDNN* as a part of framework build. This option need to be ON if the library is being built as a part of a framework. If this option is OFF the library will be built as a standalone package. | BOOL | OFF |
| ZENDNNL_AMDBLIS_FWK_DIR | If the framework is building a ZenDNN* dependency (for example AMDBLIS in this case), it can inform the library where this dependency is installed, by populating this variable. If populated, ZenDNN* assumes that the dependency is already built before library build starts, the binaries of the dependency are kept in ${ZENDNNL_AMDBLIS_FWK_DIR}/lib, and the include files are kept in ${ZENDNNL_AMDBLIS_FWK_DIR}/include. The framework build will have to ascertain by creating proper CMake dependency tree that this dependency is built before the library starts building. ZenDNN* provides a CMake include file (fwk/ZenDNNLFwkIntergate.cmake) to enable it. If this variable is not populated, then ZenDNN* assumes that the dependency is not being provided by the framework, and builds it the way it does for standalone build. | PATH | NO DEFAULT |
| ZENDNNL_AOCLDLP_FWK_DIR | Its usage is same as that of ZENDNNL_AMDBLIS_FWK_DIR except it is for AOCLDLP dependency | PATH | NO DEFAULT |
| ZENDNNL_ONEDNN_FWK_DIR | Its usage is same as that of ZENDNNL_AMDBLIS_FWK_DIR except it is for ONEDNN dependency | PATH | NO DEFAULT |

### Dependencies Options

| Option | Description | Type | Default |
|--------|-------------|------|---------|
| ZENDNNL_DEPENDS_AOCLBLIS | aocl-blis is default BLAS backend for ZenDNN*. If this option is OFF, then aocl-dlp becomes default BLAS backend of ZenDNN* | BOOL | ON |
| ZENDNNL_DEPENDS_ONEDNN | ONEDNN is a backend of ZenDNN* | BOOL | OFF |
| ZENDNNL_LOCAL_AMDBLIS | Use a locally available code of amd-blis instead of downloading from a public repository. This option can be used by developers to test the library with the dependency version not yet publically available, or a varsion still unstable. In such a case the developer still need to copy (or provide soft link) the dependency code to ${ZENDNNL_SOURCE_DIR}/dependencies directory. | BOOL | OFF |
| ZENDNNL_LOCAL_ONEDNN | Its usage is same as that of ZENDNNL_LOCAL_AMDBLIS, except it is for ONEDNN | BOOL | OFF |
| ZENDNNL_LOCAL_AOCLDLP | Its usage is same as that of ZENDNNL_LOCAL_AMDBLIS, except it is for AOCLDLP | BOOL | OFF |

## The Build Process

### General Build Process

ZenDNN* follows CMake super-build structure, where all the dependencies, the library, and various other components (examples, documentation and benchdnn benchmarking tool) are built as independent external projects, and the top level CMake orchestrates these independent external builds. In general build process is as follows

- A dependency (for example amdblis or aoclutils) is downloaded from its public repository to ${ZENDNNL_SOURCE_DIR}/dependencies/ (or provided by the developer in the same folder in case of a local/unstable dependency source code to be used). This dependency is then source built using cmake external project, and installed. All dependencies get installed in ${ZENDNNL_INSTALL_PREFIX}/deps/<dependency name> directory.

- The library itself is also built as an cmake external project. While building the library tries to find all the dependencies installed in ${ZENDNNL_INSTALL_PREFIX}/zendnnl/ directory. If it finds any mandatory or enabled dependency missing, it reports it and exits the build.

- Library GTests, if enabled, are built as a part of the library.

- Post library build, other components (examples, benchdnn) are built as cmake external project. These components try to find the library in ${ZENDNNL_INSTALL_PREFIX}/[component] (eg. ${ZENDNNL_INSTALL_PREFIX}/examples) etc directories.

For each dependency (eg. amd-blis), the build adds a compiler option ZENDNNL_DEPENDS_[DEPENDENCY] (eg. ZENDNNL_DEPENDS_AMDBLIS). If the dependency is enabled, this is set to one, else set to zero. This compiler option can be used in c++  code to enable dependency specific code

```
#if ZENDNL_DEPENDS_AMDBLIS
 <amd-blis dependent code>
#else
 <alternative amd-blis independent code>
#endif
```

### Providing Local Dependencies

In the general build process of ZenDNN*, it downloads all dependencies source code from their public repositories, and source builds them. However there may be occasions, when the library need to be built with unreleased dependency source code. In such a case the following is needed to be done

- Enable build option ZENDNNL_LOCAL_[DEPENDENCY] (for example ZENDNNL_LOCAL_AMDBLIS).
- Copy (or create soft link of) the local dependency directory to ${ZENDNNL_SOURCE_DIR}/dependencies.

The build will not try to download dependency source code, and work with local source code.

### Framework Integration Build

ZenDNN* can be informed that it is being built as a part of an inference framework by enabling ZENDNNL_FWK_BUILD option.

It is possible that few of the ZenDNN* dependencies are being built by the framework, and rebuilding them again as a part of ZenDNN* may result in bloated binaries, or symbol name clashes. ZenDNN* provides a way to inject these dependencies being built by the framework into the build process of ZenDNN*. This is done by pointing ZENDNNL_<DEPENDENCY>_FWK_DIR (eg. ZENDNNL_AMDBLIS_FWK_DIR) to the install path of the dependency, and making ZenDNN* build dependent on the dependency target.

In order to assist ZenDNN* build integration to a framework, it provides CMake files for including it as an external project in a framework build. These files, kept in ${ZENDNL_SOURCE_DIR}/fwk/, are as follows

- ZenDnnlFwkMacros.cmake : This file defines few CMake macros needed by ZendnnlFwkIntegrate.cmake
- ZenDnnlFwkIntegrate.cmake : This file provides CMake code to include ZenDNN* as an external project in framework CMake build.

ZenDNN* build can be integrated to the framework build using these files as follows

- Put these files in CMAKE_MODULE_PATH of the framework build.
- ZenDnnlFwkIntegrate.cmake uses a macro called *zendnnl_add_option* to set ZenDNN* build options. Edit this file the set the ZenDNN*options as needed. In particular the following options are to be provided
  - ZENDNN_SOURCE_DIR : ZenDNN* source code.
  - ZENDNNL_BINARY_DIR : Where ZenDNN* will be built in the build tree. if unsure set ${CMAKE_CURRENT_BINARY_DIR}/zendnnl.
  - ZENDNNL_INSTALL_DIR : Where ZenDNN* will be built in the build tree. if unsure set ${CMAKE_INSTALL_PREFIX}/zendnnl.
  - ZENDNNL_AMDBLIS_FWK_DIR : Install path of amd-blis if framework is building it and wants to inject it to ZenDNN* build.
  - ZENDNNL_AOCLDLP_FWK_DIR : Install path of aocl-dlp if framework is building it and wants to inject it to ZenDNN* build.
  - ZENDNNL_ONEDNN_FWK_DIR : Install path of onednn if framework is building it and wants to inject it to ZenDNN* build.

- Make *fwk_zendnnl* target dependent on injected dependencies by editing its "add_dependencies(fwk_zendnnl...)" command.
- Include the edited ZenDnnlFwkIntegrate.cmake in the framework build flow, where ZenDNN* is to be built.
- Make any other targets that depend on ZenDNN* dependent on *fwk_zendnnl* target.

**ManyLinux Docker Hack:**

ManyLinux Docker toolchain builds aocl-utils (a ZenDNN* dependency) library binaries in ${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils/lib64, instead of ${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils/lib. This causes build to fail in ManyLinux Docker. ZenDnnlFwkIntegrate.cmake provides a hack, in the form of checking an environment variable ZENDNNL_MANYLINUX_BUILD. If this variable is set it uses aocl-utils library path suitable for ManyLinux Docker. ManyLinux Dockerfile need to set this environment variable.

### Standalone Build :

ZenDNN* standalone build is assisted by a shell script ${ZENDNNL_SOURCE_DIR}/scripts/zendnnl_build.sh. Going to scripts directory and command "source zendnnl_build.sh --help" will dispaly all the options available.