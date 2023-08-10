

Zen Deep Neural Network Library (ZenDNN)
========================================
ZenDNN (Zen Deep Neural Network) Library accelerates deep learning inference applications on AMD CPUs. This library, which includes APIs for basic neural network building blocks optimized for AMD CPUs, targets deep learning application and framework developers with the goal of improving inference performance on AMD CPUs across a variety of workloads, including computer vision, natural language processing (NLP), and recommender systems. ZenDNN leverages oneDNN/DNNL v2.6.3's basic infrastructure and APIs. ZenDNN optimizes several APIs and adds new APIs, which are currently integrated into TensorFlow, ONNXRT, and PyTorch.

ZenDNN depends on:
- AOCL-BLIS (BLAS Library) is a portable software framework for performing highperformance Basic Linear Algebra Subprograms (BLAS) functionality.
- AOCL-LibM (AMD Core Math Library) is a software library containing a collection of basic math functions optimized for x86-64 processor based machines.
- Composable Kernel for convolutions using an implicit GEMM algorithm

AOCL-BLIS and AOCL-LibM are required dependencies for ZenDNN, whereas AMD Composable
Kernel is an optional dependency.

# Table of Contents

- [Zen Deep Neural Network Library (ZenDNN)](#zen-deep-neural-network-library-zendnn)
- [Table of Contents](#table-of-contents)
- [Scope](#scope)
- [Release Highlights](#release-highlights)
- [Supported OS and Compilers](#supported-os-and-compilers)
  - [OS](#os)
  - [Compilers](#compilers)
- [Prerequisites](#prerequisites)
- [AOCL-BLIS and AOCL-libM Library Installation](#aocl_blis-and-aocl_libm-library-installation)
  - [General Convention](#general-convention)
  - [AOCL-BLIS Library Setup](#aocl_blis-library-setup)
  - [AOCL-LibM Library Setup](#aocl_libm-library-setup)
- [Composable Kernel Library Installation](#composable-kernel-library-installation)
  - [Composable Kernel Library Setup](#composable-kernel-library-setup)
    - [Prerequisites](#prerequisites-1)
    - [Download code](#download-code)
    - [Compile](#compile)
    - [Link from ZenDNN](#link-from-zendnn)
- [Runtime Dependencies](#runtime-dependencies)
- [Build from Source](#build-from-source)
  - [GCC compiler](#gcc-compiler)
  - [Validate the build](#validate-the-build)
- [Logs](#logs)
- [License](#license)
- [Technical Support](#technical-support)

# Scope
The scope of ZenDNN is to support AMD EPYC<sup>TM</sup> CPUs on the Linux® platform. ZenDNN v4.1 offers optimized primitives, such as Convolution, MatMul, Elementwise, and Pool (Max and Average), Gelu, LayerNorm that improve performance of many convolutional neural networks, recurrent neural networks, transformer-based models, and recommender system models. For the primitives not supported by ZenDNN, execution will fall back to the  native path of the framework.


# Release Highlights
Following are the highlights of this release:
* ZenDNN library is integrated with TensorFlow v2.12, ONNXRT v1.15.1, and PyTorch v1.13.
* Python v3.8-v3.11 has been used to generate the TensorFlow v2.12 wheel files (*.whl).
* Python v3.8-v3.11 has been used to generate the ONNXRT v1.15.1 wheel files (*.whl).
* Python v3.7-v3.10 has been used to generate the PyTorch v1.13 wheel files (*.whl).
* NHWC (default format), NHWC_BLOCKED and Blocked Format (NCHWc8) are supported.

ZenDNN library is intended to be used in conjunction with the frameworks mentioned above and cannot be used independently.

The latest information on the ZenDNN release and installers is available on AMD.com portal (https://www.amd.com/en/developer/zendnn.html).

# Supported OS and Compilers
This release of ZenDNN supports the following Operating Systems (OS) and compilers:
## OS
* Ubuntu® 22.04 LTS and later
* Red Hat® Enterprise Linux® (RHEL) 9.0 and later
* CentOS Stream 9 and later
## Compilers
* GCC 9.3 and later

Theoretically, for wheel files any Linux based OS with GLIBC version later than 2.17 could be supported.

For C++ interface binaries, any Linux based OS with GLIBC version later than 2.31 could be supported.

# Prerequisites
The following prerequisites must be met for this release of ZenDNN:
* AOCL-BLIS v4.1 and AOCL-LibM v3.1.0 must be installed for optimal performance of the ZenDNN library.


# AOCL-BLIS and AOCL-libM Library Installation

**AOCL-BLIS** is a portable open-source software framework for instantiating high-performance Basic Linear Algebra Subprograms (BLAS), such as, dense linear algebra libraries. AOCL-BLIS is part of AOCL and can be downloaded from AMD.com portal Developer Central (https://www.amd.com/en/developer/aocl/blis.html).

**AOCL-LibM** is a software library containing a collection of basic math functions optimized for x86-64 processor-based machines. It provides many routines from the list of standard C99 math functions. AOCL-LibM is part of AOCL and can be downloaded from AMD.com portal (https://www.amd.com/en/developer/aocl/libm.html).

Note: ZenDNN depends only on AOCL-BLIS amd AOCL-LibM and has no dependency on any other AOCL library.
## General Convention
The following points must be considered while installing AOCL-BLIS and AOCL-LibM:
* Change to the preferred directory where AOCL-BLIS and AOCL-LibM will be downloaded.
* This parent folder is referred to as folder `<compdir>` in the steps below.
* Assume that the parent folder for user setup follows this convention: `/home/<user-id>/my_work`.

## AOCL-BLIS Library Setup
Complete the following steps to setup the GCC compiled AOCL-BLIS library:
1. Execute the command `cd <compdir>`
2. Download aocl-blis-linux-gcc-4.1.0.tar.gz.
3. Execute the following commands:
    ```bash
    tar -xvf aocl-blis-linux-gcc-4.1.0.tar.gz
    cd amd-blis
	```
This will set up the environment for AOCL-BLIS path:
```bash
export ZENDNN_BLIS_PATH=$(pwd)
```
For example:
```bash
export ZENDNN_BLIS_PATH=/home/<user-id>/my_work/amd-blis
```
## AOCL-LibM Library Setup
Complete the following steps to setup the GCC compiled AOCL-LibM library:
1. Execute the command `cd <compdir>`.
2. Download aocl-linux-gcc-3.1.0.tar.gz.
3. Execute the following commands:
    ```bash
    tar -xvf aocl-linux-gcc-3.1.0.tar.gz
    cd aocl-linux-gcc-3.1.0
    tar -xvf aocl-libm-linux-gcc-3.1.0.tar.gz
    cd amd-libm
    ```
This will set up the environment for AOCL-LibM path:
```bash
export ZENDNN_LIBM_PATH=$(pwd)
```
For example:
```bash
export ZENDNN_LIBM_PATH=/home/<user-id>/my_work/aocl-linux-gcc-3.1.0/amd-libm
```


The bashrc file can be edited to setup ZENDNN_BLIS_PATH and ZENDNN_LIBM_PATH environment path.
For example, in the case of GCC compiled AOCL-BLIS and AOCL-LibM:
```bash
export ZENDNN_BLIS_PATH=/home/<user-id>/my_work/amd-blis
export ZENDNN_LIBM_PATH=/home/<user-id>/my_work/aocl-linux-gcc-3.1.0/amd-libm
```

# Composable Kernel Library Installation

**Composable Kernel** aims to provide a programming model for writing performance critical kernels for machine learning workloads across multiple architectures including GPUs, CPUs, etc, through general purpose kernel languages, like HIP C++.  Composable Kernel can be downloaded from the AMD ROCm Software Platform Repository (https://github.com/ROCmSoftwarePlatform/composable_kernel).

## Composable Kernel Library Setup

Composable Kernel (CK) for CPU is currently only on the `cpu_avx2` branch of the Composable Kernel repository and is at the experimental stage of development.

### Prerequisites
CK is suitable for these compilers:
1) hipclang: this is mainly used for compiling GPU hip kernels(require rocm environment), but also can be used for CPU. For a first trial use below compiler.
2) gcc: at least gcc-9 is needed, you may need manually install a gcc-9 if ubuntu default is not gcc-9.

### Download code
```
git clone https://github.com/ROCmSoftwarePlatform/composable_kernel.git
cd composable_kernel
git checkout origin/cpu_avx2 -b cpu_avx2
```

### Compile

From the root directory of Composable Kernel (CK) build CK libraries:
```
# if use gcc
sh script/cmake-avx2-gcc.sh
cd build
make -j`nproc` example_cpu_conv2d_fwd example_cpu_conv2d_fwd_bias_relu_add
```

### Link from ZenDNN

From the root directory of Composable Kernel (CK), this will set up the environment for including CK headers and linking to the CK libraries.
```bash
 export ZENDNN_CK_PATH=$(pwd)
```

The `Makefile` in this project contains a variable `DEPEND_ON_CK` which is set to `0` by default.  To enable CK use `DEPEND_ON_CK=1` when building the ZenDNN library.

Either modify DEPEND_ON_CK in the Makefile or pass DEPEND_ON_CK as an argument to
make by editing scripts/zendnn_build.sh gcc.

The `LD_LIBRARY_PATH` variable needs to be updated in order to run code that depends on CK.
```bash
export LD_LIBRARY_PATH=${ZENDNN_CK_PATH}/build/lib:${LD_LIBRARY_PATH}
```

# Runtime Dependencies
ZenDNN has the following runtime dependencies:
* GNU C library (glibc.so)
* GNU Standard C++ library (libstdc++.so)
* Dynamic linking library (libdl.so)
* POSIX Thread library (libpthread.so)
* C Math Library (libm.so)
* OpenMP (libomp.so)
* Python v3.8-v3.11 for TensorFlow v2.12
* Python v3.7-v3.10 for PyTorch v1.13
* Python v3.8-v3.11 for ONNXRT v1.15.1

Since ZenDNN is configured to use OpenMP, a C++ compiler with OpenMP 2.0 or later is required for runtime execution.

# Build from Source
Clone ZenDNN git:
```bash
git clone https://github.com/amd/ZenDNN.git
cd ZenDNN
```

## GCC compiler
**ZENDNN_BLIS_PATH** and **ZENDNN_LIBM_PATH** should be defined.
example:
```bash
export ZENDNN_BLIS_PATH=/home/<user-id>/my_work/amd-blis
export ZENDNN_LIBM_PATH=/home/<user-id>/my_work/aocl-linux-gcc-3.1.0/amd-libm
make clean
source scripts/zendnn_build.sh gcc
```
When new terminal is opened, user need to set up environment variables:
```bash
source scripts/zendnn_gcc_env_setup.sh
```
Please note above scripts must be sourced only from ZenDNN Folder.

## Validate the build
After the library is built on Linux host, user can run unit tests using:
```bash
source scripts/runApiTest.sh
```
Corresponding tests are located in the **tests/api_tests** directory. These unit tests don't produce any information/logs in the terminal. Library logs can be enabled with:
```bash
ZENDNN_LOG_OPTS=ALL:2 source scripts/runApiTest.sh
```

# Logs
Logging is disabled in the ZenDNN library by default. It can be enabled using the environment variable **ZENDNN_LOG_OPTS** before running any tests. Logging behavior can be specified by setting the environment variable **ZENDNN_LOG_OPTS** to a comma-delimited list of ACTOR:DBGLVL pairs.

The different ACTORS are as follows:
| ACTORS  | Usage
| :------ | :-----------
| ALGO    | Logs all algorithms executed
| CORE    | Logs all the core ZenDNN library operations
| API     | Logs all the ZenDNN API calls
| TEST    | Logs used in API tests, functionality tests and regression tests
| PROF    | Logs the performance of operations in millisecond
| FWK     | Logs all the framework (TensorFlow, ONNXRT, and PyTorch) specific calls

For example:
* To turn on info logging, use **ZENDNN_LOG_OPTS=ALL:2**
* To turn off all logging, use **ZENDNN_LOG_OPTS=ALL:-1**
* To only log errors, use **ZENDNN_LOG_OPTS=ALL:0**
* To only log info for ALGO, use **ZENDNN_LOG_OPTS=ALL:-1,ALGO:2**
* To only log info for CORE, use **ZENDNN_LOG_OPTS=ALL:-1,CORE:2**
* To only log info for API, use **ZENDNN_LOG_OPTS=ALL:-1,API:2**
* To only log info for PROF (profile), use **ZENDNN_LOG_OPTS=ALL:-1,PROF:2**
* To only log info for FWK, use **ZENDNN_LOG_OPTS=ALL:-1,FWK:2**



The Different Debug Levels (DBGLVL) are as follows:
```bash
enum LogLevel {
  LOG_LEVEL_DISABLED = -1,
  LOG_LEVEL_ERROR    =  0,
  LOG_LEVEL_WARNING  =  1,
  LOG_LEVEL_INFO     =  2,
  LOG_LEVEL_VERBOSE0 =  3,
  LOG_LEVEL_VERBOSE1 =  4,
  LOG_LEVEL_VERBOSE2 =  5
};
```
# License

ZenDNN is licensed under [Apache License Version 2.0](LICENSE). Refer to the "[LICENSE](LICENSE)" file for the full license text and copyright notice.

This distribution includes third party software governed by separate license terms.

3-clause BSD license:
* [Xbyak](https://github.com/herumi/xbyak)
* [Googletest](https://github.com/google/googletest)
* [Instrumentation and Tracing Technology API (ITT API)](https://github.com/intel/IntelSEAPI/tree/master/ittnotify)

Apache License Version 2.0:
* [oneDNN](https://github.com/oneapi-src/oneDNN)
* [Xbyak_aarch64](https://github.com/fujitsu/xbyak_aarch64)
* [TensorFlow](https://github.com/tensorflow/tensorflow)

Boost Software License, Version 1.0:
* [Boost C++ Libraries](https://www.boost.org/)

BSD 2-Clause license:
* [Caffe](https://github.com/BVLC/caffe)


This third party software, even if included with the distribution of the Advanced Micro Devices software, may be governed by separate license terms, including without limitation, third party license terms,  and open source software license terms. These separate license terms govern your use of the third party programs as set forth in the **THIRD-PARTY-PROGRAMS** file.

# Technical Support
Please email zendnnsupport@amd.com for questions, issues, and feedback on ZenDNN.

Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/amd/ZenDNN/issues) page.
