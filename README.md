


Zen Deep Neural Network Library (ZenDNN)
========================================
ZenDNN (Zen Deep Neural Network) Library accelerates deep learning inference applications on AMD CPUs. This library, which includes APIs for basic neural network building blocks optimized for AMD CPUs, targets deep learning application and framework developers with the goal of improving inference performance on AMD CPUs across a variety of workloads, including computer vision, natural language processing (NLP), and recommender systems. ZenDNN leverages oneDNN/DNNL v2.2's basic infrastructure and APIs. ZenDNN optimizes several APIs and adds new APIs, which are currently integrated into TensorFlow and ONNXRT. ZenDNN uses AMD BLIS (BLAS like Library Instantiation Software) library for its BLAS (Basic Linear Algebra Subprograms) API needs.

# Table of Contents

- [Scope](#scope)
- [Release Highlights](#release-highlights)
- [Supported OS and Compilers](#supported-os-and-compilers)
- [Prerequisites](#prerequisites)
- [AOCC and AMD-BLIS Library Installation](#aocc-and-amd-blis-library-installation)
- [Runtime Dependencies](#runtime-dependencies)
- [Build from Source](#build-from-source)
- [Logs](#logs)
- [License](#license)
- [Technical Support](#technical-support)

# Scope
The scope of ZenDNN is to support AMD EPYC CPUs on the Linux® platform. ZenDNN v3.1 offers optimized primitives, such as Convolution, MatMul, Elementwise, and Pool (Max and Average) that improve performance of many convolutional neural networks, recurrent neural networks, transformer-based models, and recommender system models. For the primitives not supported by ZenDNN, the execution will fall back to the native path of the framework.


# Release Highlights
Following are the highlights of this release:
* Python v3.8.10 has been used to generate the TensorFlow v2.5 wheel file (*.whl).
* Python v3.7.9 has been used to generate the TensorFlow v1.15 wheel file (*.whl).
* Python v3.8.5 has been used to generate the ONNXRT v1.5.1 wheel file (*.whl).
* NHWC (default format) and Blocked Format (NCHWc8) are supported.
* ZenDNN library is integrated with TensorFlow (v2.5, v1.15) and ONNXRT v1.5.1

ZenDNN library is intended to be used in conjunction with the frameworks mentioned above, you cannot use it independently.

The latest information on the ZenDNN release and installers is available on AMD Developer Central (https://developer.amd.com/zendnn/).

# Supported OS and Compilers
This release of ZenDNN supports the following Operating Systems (OS) and compilers:
## OS
* Ubuntu® 18.04 LTS and later
* Red Hat® Enterprise Linux® (RHEL) 8.0 and later
* CentOS 7.9 and later
## Compilers
* GCC 7.5 and later
* AOCC (AMD Optimizing C/C++ Compiler) 3.0 (https://developer.amd.com/amd-aocc/)

Theoretically, any Linux based OS with GLIBC version later than 2.17 could be supported.

# Prerequisites
The following prerequisites must be met for this release of ZenDNN:
* Conda must be installed and initialized with its path set properly so that the Conda env can be activated from the terminal.
  * Note: If Conda is not available, zendnn_release_setup.sh script will fail.
* The whl file must be generated with the same Python version for the following binaries:
	* Python 3.8.10 for TensorFlow v2.5
	* Python 3.7.9 for TensorFlow v1.15
	* Python 3.8.5 for ONNXRT v1.5.1
* AOCC 3.0 and AMD-BLIS 3.0.6 must be installed.
  * Note: While GCC 7.5 and later are also supported compilers, AOCC is recommended for optimal performance of the ZenDNN library.



# AOCC and AMD-BLIS Library Installation
**AOCC** is a high performance, production quality code generation tool. AOCC can be downloaded from AMD Developer Central (https://developer.amd.com/amd-aocc/).

ZenDNN compiled with AOCC may provide better performance as compared to the other open-source counterparts.

**AMD-BLIS** is a portable open-source software framework for instantiating high-performance Basic Linear Algebra Subprograms (BLAS), such as, dense linear algebra libraries. AMD-BLIS is part of AOCL and can be downloaded from AMD Developer Central (https://developer.amd.com/amd-aocl/).

Note: ZenDNN depends only on AMD-BLIS and has no dependency on any other AOCL library.
## General Convention
The following points must be considered while installing AOCC and AMD-BLIS:
* Change to the preferred directory where ZenDNN will be downloaded.
* This parent folder is referred to as folder `<compdir>` in the steps below.
* It is good practice to keep AOCC 3.0 and AMD-BLIS 3.0.6 downloads in the same parent folder.
* Assume that the parent folder for user setup follows this convention: `/home/<user-id>/my_work`.

## AMD-BLIS Library Setup
Complete the following steps to setup the AOOC compiled BLIS library:
1. Execute the command `cd <compdir>`
2.  Download aocl-linux-aocc-3.0-6.tar.gz.
3. Execute the following commands:
`tar -xvf aocl-linux-aocc-3.0-6.tar.gz`
  `cd aocl-linux-aocc-3.0-6`
  `tar -xvf aocl-blis-linux-aocc-3.0-6.tar.gz`
  `cd amd-blis`
This will set up the environment for BLIS AOCC path:
`export ZENDNN_BLIS_PATH=$(pwd)`
For example:
`export ZENDNN_BLIS_PATH=/home/<user-id>/my_work/aocl-linux-aocc-3.0-6/amd-blis`

Complete the following steps to setup the GCC compiled BLIS library:
1. Execute the command `cd <compdir>`.
2. Download aocl-linux-gcc-3.0-6.tar.gz.
3. Execute the following commands:
`tar -xvf aocl-linux-gcc-3.0-6.tar.gz`
`cd aocl-linux-gcc-3.0-6`
`tar -xvf aocl-blis-linux-gcc-3.0-6.tar.gz`
`cd amd-blis`
This will set up the environment for BLIS GCC path:
`export ZENDNN_BLIS_PATH=$(pwd)`
For example:
`export ZENDNN_BLIS_PATH=/home/<user-id>/my_work/aocl-linux-gcc-3.0-6/amd-blis`

## AOCC Installation
Complete the following steps to install AOCC:
1. Execute the command `cd <compdir>`.
2. Download aocc-compiler-3.0.0.tar from the AMD Developer Central (https://developer.amd.com/
amd-aocc/).
3. Execute the command `tar -xvf aocc-compiler-3.0.0.tar`.
4. Execute the command `cd aocc-compiler-3.0.0`.
This will install the compiler and display the AOCC set up instructions.
5. Execute the command `bash install.sh`.
This will set up the environment for the AOCC path:
 `export ZENDNN_AOCC_COMP_PATH=$(pwd)`
For example:
`export ZENDNN_AOCC_COMP_PATH=/home/<user-id>/my_work/aocc-compiler-3.0.0`
The bashrc file can be edited to setup ZENDNN_AOCC_COMP_PATH environment path.
For example, in the case of AOCC compiled AMD-BLIS:
`export ZENDNN_AOCC_COMP_PATH=/home/<user-id>/my_work/aocc-compiler-3.0.0`
`export ZENDNN_BLIS_PATH=/home/<user-id>/my_work/aocl-linux-aocc-3.0-6/amd-blis`
For example, in the case of GCC compiled AMD-BLIS:
`export ZENDNN_BLIS_PATH=/home/<user-id>/my_work/aocl-linux-gcc-3.0-6/amd-blis`


# Runtime Dependencies
To use ZenDNN, the following runtime libraries must be installed:
* GNU C library (glibc.so)
* GNU Standard C++ library (libstdc++.so)
* Dynamic linking library (libdl.so)
* POSIX Thread library (libpthread.so)
* C Math Library (libm.so)
* OpenMP (libomp.so)
* Python 3.8.10 for TensorFlow v2.5
* Python 3.7.9 for TensorFlow v1.15
* Python 3.8.5 for ONNXRT v1.5.1

Since ZenDNN is configured to use OpenMP, a C++ compiler with OpenMP 2.0 or later is required for runtime execution.

# Build from Source
Clone ZenDNN git:
`git clone https://github.com/amd/ZenDNN.git`
`cd ZenDNN`

## AOCC compiler
**ZENDNN_AOCC_COMP_PATH** and **ZENDNN_BLIS_PATH** should be defined.
example:
`export ZENDNN_AOCC_COMP_PATH=/home/<user-id>/my_work/aocc-compiler-3.0.0`
`export ZENDNN_BLIS_PATH=/home/<user-id>/my_work/aocl-linux-aocc-3.0-6/amd-blis`
`make clean`
`source scripts/zendnn_aocc_build.sh`
When new terminal is opened, user need to set up environment variables:
`source scripts/zendnn_aocc_env_setup.sh`

## GCC compiler
**ZENDNN_BLIS_PATH** should be defined.
example:
`export ZENDNN_BLIS_PATH=/home/<user-id>/my_work/aocl-linux-gcc-3.0-6/amd-blis`
`make clean`
`source scripts/zendnn_gcc_build.sh`
When new terminal is opened, user need to set up environment variables:
`source scripts/zendnn_gcc_env_setup.sh`

Please note above scripts must be sourced only from ZenDNN Folder.

## Validate the build
After the library is built on Linux host, user can run unit tests using:
`source scripts/runApiTest.sh`
Corresponding tests are located in the **tests/api_tests** directory. These unit tests don't produce any information/logs in the terminal. Library logs can be enabled with:
`ZENDNN_LOG_OPTS=ALL:2 source scripts/runApiTest.sh`

# Logs
Logging is disabled in the ZenDNN library by default. It can be enabled using the environment variable **ZENDNN_LOG_OPTS** before running any tests. Logging behavior can be specified by setting the environment variable **ZENDNN_LOG_OPTS** to a comma-delimited list of ACTOR:DBGLVL pairs.

For example:
* To turn on info logging, use **ZENDNN_LOG_OPTS=ALL:2**
* To turn off all logging, use **ZENDNN_LOG_OPTS=ALL:-1**
* To only log errors, use **ZENDNN_LOG_OPTS=ALL:0**
* To only log info for ALGO, use **ZENDNN_LOG_OPTS=ALL:-1,ALGO:2**
* To only log info for CORE, use **ZENDNN_LOG_OPTS=ALL:-1,CORE:2**
* To only log info for API, use **ZENDNN_LOG_OPTS=ALL:-1,API:2**
* To only log info for PROF (profile), use **ZENDNN_LOG_OPTS=ALL:-1,PROF:2**
* To only log info for FWK, use **ZENDNN_LOG_OPTS=ALL:-1,FWK:2**

The different ACTORS are as follows:
| ACTORS  | Usage
| :------ | :-----------
| ALGO    | Logs all algorithms executed
| CORE    | Logs all the core ZenDNN library operations
| API     | Logs all the ZenDNN API calls
| TEST    | Logs used in API tests, functionality tests and regression tests
| PROF    | Logs the performance of operations in millisecond
| FWK     | Logs all the framework (TensorFlow, ONNXRT, and PyTorch) specific calls

The Different Debug Levels (DBGLVL) are as follows:
```
enum LogLevel {
	LOG_LEVEL_DISABLED = -1,
	LOG_LEVEL_ERROR    =  0,
	LOG_LEVEL_WARNING  =  1,
	LOG_LEVEL_INFO     =  2,
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
[GitHub issues](https://https://github.com/amd/ZenDNN/issues) page.