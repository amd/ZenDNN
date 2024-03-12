::*******************************************************************************
:: Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::
::*******************************************************************************

::-----------------------------------------------------------------------------
::   zendnn_env_setup_win.bat
::   Prerequisite: This script needs to run first to setup environment variables
::                before any ZenDNN build in Windows.
::
::   This script does following:
::   -sets the environment variables , if they are present they will bre replaced
::   -Sets important environment variables for benchmarking:
::       -OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PROC_BIND
::----------------------------------------------------------------------------

@echo off

if not defined ZENDNN_LOG_OPTS set ZENDNN_LOG_OPTS=ALL:0
echo "ZENDNN_LOG_OPTS=%ZENDNN_LOG_OPTS%"

if not defined OMP_NUM_THREADS set OMP_NUM_THREADS=64
echo "OMP_NUM_THREADS=%OMP_NUM_THREADS%"

if not defined OMP_WAIT_POLICY set OMP_WAIT_POLICY=ACTIVE
echo "OMP_WAIT_POLICY=%OMP_WAIT_POLICY%"

if not defined OMP_PROC_BIND set OMP_PROC_BIND=FALSE
echo "OMP_PROC_BIND=%OMP_PROC_BIND%"

:: If the environment variable OMP_DYNAMIC is set to true, the OpenMP implementation
:: may adjust the number of threads to use for executing parallel regions in order
:: to optimize the use of system resources. ZenDNN depend on a number of threads
:: which should not be modified by runtime, doing so can cause incorrect execution
set OMP_DYNAMIC=FALSE
echo "OMP_DYNAMIC=%OMP_DYNAMIC%"

::Disable ONNXRT check for training ops and stop execution if any training ops
::found in ONNXRT graph. By default, its enabled
set ZENDNN_INFERENCE_ONLY=1
echo "ZENDNN_INFERENCE_ONLY=%ZENDNN_INFERENCE_ONLY%"

::Disable TF memory pool optimization, By default, its enabled
set ZENDNN_ENABLE_MEMPOOL=1
echo "ZENDNN_ENABLE_MEMPOOL=%ZENDNN_ENABLE_MEMPOOL%"

::Set the max no. of tensors that can be used inside TF memory pool, Default is
::set to 64
set ZENDNN_TENSOR_POOL_LIMIT=64
echo "ZENDNN_TENSOR_POOL_LIMIT=%ZENDNN_TENSOR_POOL_LIMIT"%

::Enable fixed max size allocation for Persistent tensor with TF memory pool
::optimization, By default, its disabled
set ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=0
echo "ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=%ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE%"

::Convolution GEMM Algo path is default
set ZENDNN_CONV_ALGO=1
echo "ZENDNN_CONV_ALGO=%ZENDNN_CONV_ALGO%"

:: INT8 support  is disabled by default
set ZENDNN_INT8_SUPPORT=0
echo "ZENDNN_INT8_SUPPORT=%ZENDNN_INT8_SUPPORT%"

:: INT8 Relu6 fusion support is disabled by default
set ZENDNN_RELU_UPPERBOUND=0
echo "ZENDNN_RELU_UPPERBOUND=%ZENDNN_RELU_UPPERBOUND%"

:: ZENDNN_MATMUL_ALGO is set to FP32:4 and BF16:3 by default
set ZENDNN_MATMUL_ALGO=FP32:4,BF16:3
echo "ZENDNN_MATMUL_ALGO=%ZENDNN_MATMUL_ALGO%"

:: Switch to enable Conv, Add fusion on users discretion. Currently it is
:: safe to enable this switch for resnet50v1_5, resnet101, and
:: inception_resnet_v2 models only. By default the switch is disabled.
set ZENDNN_TF_CONV_ADD_FUSION_SAFE=0
echo "ZENDNN_TF_CONV_ADD_FUSION_SAFE=%ZENDNN_TF_CONV_ADD_FUSION_SAFE%"

:: Primitive reuse is disabled by default
set TF_ZEN_PRIMITIVE_REUSE_DISABLE=FALSE
echo "TF_ZEN_PRIMITIVE_REUSE_DISABLE=%TF_ZEN_PRIMITIVE_REUSE_DISABLE%"

:: Set the no. of InterOp threads, Default is set to 1
set ZENDNN_TF_INTEROP_THREADS=1
echo "ZENDNN_TF_INTEROP_THREADS=%ZENDNN_TF_INTEROP_THREADS%"

::Check if below declaration of ZENDNN_GIT_ROOT is correct
set ZENDNN_GIT_ROOT=%cd%
if not defined ZENDNN_GIT_ROOT (
    echo "Error: Environment variable ZENDNN_GIT_ROOT needs to be set"
    echo "Error: \ZENDNN_GIT_ROOT points to root of ZENDNN repo"
    exit
) else (
    if exist "%ZENDNN_GIT_ROOT%\" (
        echo "Directory ZenDNN exists!"
    ) else (
        echo "Directory ZenDNN DOES NOT exists!"
    )
    echo "ZENDNN_GIT_ROOT=%ZENDNN_GIT_ROOT%"
)

::Change ZENDNN_UTILS_GIT_ROOT as per need in future
cd ..
set ZENDNN_UTILS_GIT_ROOT=%cd%\ZenDNN_utils
if not defined ZENDNN_UTILS_GIT_ROOT (
    echo "Error: Environment variable ZENDNN_UTILS_GIT_ROOT needs to be set"
    echo "Error: \ZENDNN_UTILS_GIT_ROOT points to root of ZENDNN repo"
    pause
    exit
)
else (
    if exist "%ZENDNN_UTILS_GIT_ROOT%\" (
        echo "Directory ZenDNN_utils exists!"
    ) else (
        echo "Directory ZenDNN_utils DOES NOT exists!"
    )
    echo "ZENDNN_UTILS_GIT_ROOT=%ZENDNN_UTILS_GIT_ROOT%"
)

::Change ZENDNN_TOOLS_GIT_ROOT as per need in future

set ZENDNN_TOOLS_GIT_ROOT=%cd%\ZenDNN_tools
if not defined ZENDNN_TOOLS_GIT_ROOT (
    echo "Error: Environment variable ZENDNN_TOOLS_GIT_ROOT needs to be set"
    echo "Error: \ZENDNN_TOOLS_GIT_ROOT points to root of ZENDNN repo"
    pause
    exit
)
else (
    if exist "%ZENDNN_TOOLS_GIT_ROOT%\" (
        echo "Directory ZenDNN_tools exists!"
    ) else (
        echo "Directory ZenDNN_tools DOES NOT exists!"
    )
    echo "ZENDNN_TOOLS_GIT_ROOT=%ZENDNN_TOOLS_GIT_ROOT%"
)

::Change ZENDNN_PARENT_FOLDER as per need in future
::Current assumption, ONNXRT is located parallel to ZenDNN
set ZENDNN_PARENT_FOLDER=%cd%
if defined ZENDNN_PARENT_FOLDER (
    echo "ZENDNN_PARENT_FOLDER=%ZENDNN_PARENT_FOLDER%"
) else (
    set ZENDNN_PARENT_FOLDER=%cd%
    echo "ZENDNN_PARENT_FOLDER=%ZENDNN_PARENT_FOLDER%"
)

:: Use local copy of ZenDNN library source code when building ONNXRT
:: Default is build from local source for development and verification.
:: For release, set ZENDNN_ONNXRT_USE_LOCAL_ZENDNN=0
if defined ZENDNN_PARENT_FOLDER (
    set ZENDNN_ONNXRT_USE_LOCAL_ZENDNN=1
)

if defined ZENDNN_BLIS_PATH (
    echo "Error: Environment variable ZENDNN_BLIS_PATH needs to be set"
) else (
    echo "ZENDNN_BLIS_PATH: %ZENDNN_BLIS_PATH%"
)

set TF_GIT_ROOT=%ZENDNN_PARENT_FOLDER%/tensorflow
echo "TF_GIT_ROOT: %TF_GIT_ROOT%"

set BENCHMARKS_GIT_ROOT=%ZENDNN_PARENT_FOLDER%/benchmarks
echo "BENCHMARKS_GIT_ROOT: %BENCHMARKS_GIT_ROOT%"

set PYTORCH_GIT_ROOT=%ZENDNN_PARENT_FOLDER%/pytorch
echo "PYTORCH_GIT_ROOT: %PYTORCH_GIT_ROOT%"

set PYTORCH_BENCHMARK_GIT_ROOT=%ZENDNN_PARENT_FOLDER%/pytorch-benchmarks
echo "PYTORCH_BENCHMARK_GIT_ROOT: %PYTORCH_BENCHMARK_GIT_ROOT%"

set ONNXRUNTIME_GIT_ROOT=%ZENDNN_PARENT_FOLDER%/onnxruntime
echo "ONNXRUNTIME_GIT_ROOT: %ONNXRUNTIME_GIT_ROOT%"

:: Primitive Caching Capacity
set ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
echo "ZENDNN_PRIMITIVE_CACHE_CAPACITY: %ZENDNN_PRIMITIVE_CACHE_CAPACITY%"

:: MAX_CPU_ISA
:: MAX_CPU_ISA is disabld at build time. When feature is enabled, uncomment the
:: below 2 lines
::export ZENDNN_MAX_CPU_ISA=ALL
::echo "ZENDNN_MAX_CPU_ISA: %ZENDNN_MAX_CPU_ISA%"

:: Enable primitive create and primitive execute logs. By default it is disabled
set ZENDNN_PRIMITIVE_LOG_ENABLE=0
echo "ZENDNN_PRIMITIVE_LOG_ENABLE: %ZENDNN_PRIMITIVE_LOG_ENABLE%"

::-------------------------------------------------------------------------------
:: Go to ZENDNN_GIT_ROOT
cd %ZENDNN_GIT_ROOT%
echo :
echo "Please set below environment variables explicitly as per the platform you are using!!"
echo :
echo "      OMP_NUM_THREADS"
echo "Please set below environment variables explicitly for better performance!!"
echo "      OMP_PROC_BIND=CLOSE"
echo "      OMP_PLACES=CORES"
echo :
