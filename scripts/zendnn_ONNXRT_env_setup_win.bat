::*******************************************************************************
:: Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
::*******************************************************************************

::!/bin/bash

::-----------------------------------------------------------------------------
::   zendnn_ONNXRT_env_setup_win.bat
::   Prerequisite: This script needs to run first to setup environment variables
::                 before ONNXRT setup
::
::   This script does following:
::   -sets the environment variables , if they are present they will bre replaced
::   -Sets important environment variables for benchmarking:
::       -OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PROC_BIND
::----------------------------------------------------------------------------

if defined %ZENDNN_LOG_OPTS%(
    echo "ZENDNN_LOG_OPTS=%ZENDNN_LOG_OPTS%"
) else (
    set ZENDNN_LOG_OPTS=ALL:0
    echo "ZENDNN_LOG_OPTS=%ZENDNN_LOG_OPTS%"
)

if defined %OMP_NUM_THREADS%(
    echo "OMP_NUM_THREADS=%OMP_NUM_THREADS%"
) else (
    set OMP_NUM_THREADS=64
    echo "OMP_NUM_THREADS=%OMP_NUM_THREADS%"
)

if defined %OMP_WAIT_POLICY%(
    echo "OMP_WAIT_POLICY=%OMP_WAIT_POLICY%"
) else (
    set OMP_WAIT_POLICY=ACTIVE
    echo "OMP_WAIT_POLICY=%OMP_WAIT_POLICY%"
)

if defined %OMP_PROC_BIND%(
    echo "OMP_PROC_BIND=%OMP_PROC_BIND%"
) else (
    set OMP_PROC_BIND=FALSE
    echo "OMP_PROC_BIND=%OMP_PROC_BIND%"
)

if defined %OMP_DYNAMIC%(
    echo "OMP_DYNAMIC=%OMP_DYNAMIC%"
) else (
    set OMP_DYNAMIC=FALSE
    echo "OMP_DYNAMIC=%OMP_DYNAMIC%"
)

if defined %ZENDNN_INFERENCE_ONLY%(
    echo "ZENDNN_INFERENCE_ONLY=%ZENDNN_INFERENCE_ONLY%"
) else (
    set ZENDNN_INFERENCE_ONLY=1
    echo 
)

if defined %ZENDNN_NHWC_BLOCKED%(
    echo "ZENDNN_NHWC_BLOCKED=%ZENDNN_NHWC_BLOCKED%"
) else (
    set ZENDNN_NHWC_BLOCKED=0
    echo "ZENDNN_NHWC_BLOCKED=%ZENDNN_NHWC_BLOCKED%"
)

if defined %ZENDNN_INT8_SUPPORT%(
    echo "ZENDNN_INT8_SUPPORT=%ZENDNN_INT8_SUPPORT%"
) else (
    set ZENDNN_INT8_SUPPORT=0
    echo "ZENDNN_INT8_SUPPORT=%ZENDNN_INT8_SUPPORT%"
)

if defined %ZENDNN_RELU_UPPERBOUND%(
    echo "ZENDNN_RELU_UPPERBOUND=%ZENDNN_RELU_UPPERBOUND%"
) else (
    set ZENDNN_RELU_UPPERBOUND=0
    echo "ZENDNN_RELU_UPPERBOUND=%ZENDNN_RELU_UPPERBOUND%"
)

if defined %ZENDNN_GEMM_ALGO%(
    echo "ZENDNN_GEMM_ALGO=%ZENDNN_GEMM_ALGO%"
) else (
    set ZENDNN_GEMM_ALGO=0
    echo "ZENDNN_GEMM_ALGO=%ZENDNN_GEMM_ALGO%"
)

if defined %ZENDNN_GIT_ROOT%(
    echo "ZENDNN_GIT_ROOT=%ZENDNN_GIT_ROOT%"
) else (
    set ZENDNN_GIT_ROOT=%cd%
    echo "ZENDNN_GIT_ROOT=%ZENDNN_GIT_ROOT%"
)

cd .. 

if defined %ZENDNN_UTILS_GIT_ROOT%(
    echo "ZENDNN_UTILS_GIT_ROOT=%ZENDNN_UTILS_GIT_ROOT%"
) else (
    set ZENDNN_UTILS_GIT_ROOT=%cd%\ZenDNN_utils
    echo "ZENDNN_UTILS_GIT_ROOT=%ZENDNN_UTILS_GIT_ROOT%"
)
if defined %ZENDNN_PARENT_FOLDER%(
    echo "ZENDNN_PARENT_FOLDER=%ZENDNN_PARENT_FOLDER%"
) else (
    set ZENDNN_PARENT_FOLDER=%cd%
    echo "ZENDNN_PARENT_FOLDER=%ZENDNN_PARENT_FOLDER%"
)

echo "Set the below path to ZENDNN_GIT_ROOT,ZENDNN_UTILS_GIT_ROOT,ZENDNN_PARENT_FOLDER"
echo %ZENDNN_GIT_ROOT%
echo %ZENDNN_UTILS_GIT_ROOT%
echo %ZENDNN_PARENT_FOLDER%

set ZENDNN_ONNXRT_USE_LOCAL_ZENDNN=1
echo "ZENDNN_ONNXRT_USE_LOCAL_ZENDNN:%ZENDNN_ONNXRT_USE_LOCAL_ZENDNN%"
set ZENDNN_ONNXRT_USE_LOCAL_BLIS=0
echo "ZENDNN_ONNXRT_USE_LOCAL_BLIS:%ZENDNN_ONNXRT_USE_LOCAL_BLIS%"

set ONNXRUNTIME_GIT_ROOT=%ZENDNN_PARENT_FOLDER%\onnxruntime
echo "ONNXRUNTIME_GIT_ROOT:%ONNXRUNTIME_GIT_ROOT%"
set ZENDNN_ONNXRT_VERSION=1.10.0
echo "ZENDNN_ONNXRT_VERSION:%ZENDNN_ONNXRT_VERSION%"
set ZENDNN_ONNX_VERSION=1.9.0
echo "ZENDNN_ONNX_VERSION:%ZENDNN_ONNX_VERSION%"

set ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
echo "ZENDNN_PRIMITIVE_CACHE_CAPACITY:%ZENDNN_PRIMITIVE_CACHE_CAPACITY%"
set ZENDNN_PRIMITIVE_LOG_ENABLE=0
echo "ZENDNN_PRIMITIVE_LOG_ENABLE:%ZENDNN_PRIMITIVE_LOG_ENABLE%"
set ZENDNN_ENABLE_LIBM=0
echo "ZENDNN_ENABLE_LIBM:%ZENDNN_ENABLE_LIBM%"

REM Flags for optimized execution of ONNXRT model

set ZENDNN_BLOCKED_FORMAT=1
echo " ZENDNN_BLOCKED_FORMAT:% ZENDNN_BLOCKED_FORMAT%"
set ZENDNN_CONV_ADD_FUSION_ENABLE=1
echo "ZENDNN_CONV_ADD_FUSION_ENABLE:%ZENDNN_CONV_ADD_FUSION_ENABLE%"
set ZENDNN_RESNET_STRIDES_OPT1_ENABLE=1
echo "ZENDNN_RESNET_STRIDES_OPT1_ENABLE:%ZENDNN_RESNET_STRIDES_OPT1_ENABLE%"
set ORT_ZENDNN_ENABLE_INPLACE_CONCAT=1
echo "ORT_ZENDNN_ENABLE_INPLACE_CONCAT:%ORT_ZENDNN_ENABLE_INPLACE_CONCAT%"

cd %ZENDNN_GIT_ROOT%
