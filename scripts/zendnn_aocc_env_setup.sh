#*******************************************************************************
# Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#*******************************************************************************

#!/bin/bash

#-----------------------------------------------------------------------------
#   zendnn_aocc_env_setup.sh
#   Prerequisite: This script needs to run first to setup environment variables
#                 before before any AOCC build or TF setup
#
#   This script does following:
#   -Checks if important env variables are declared
#   -Checks and print version informations for following:
#       -make, gcc, g++, ld, python
#   -Sets important environment variables for benchmarking:
#       -OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PROC_BIND
#   -Calls script to gather HW, OS, Kernel, Bios information
#   -exports LD_LIBRARY_PATH
#----------------------------------------------------------------------------

#Function to check mandatory prerequisites
function check_mandatory_prereqs() {
    if type -P make >/dev/null 2>&1;
    then
        echo "make is installed"
        echo `make -v | grep "GNU Make"`
    else
        echo "make is not installed, install make"
        return
    fi

    if type -P gcc >/dev/null 2>&1;
    then
        echo "gcc is installed"
        echo `gcc --version | grep "gcc "`
    else
        echo "gcc is not installed, install gcc"
        return
    fi

    if type -P g++ >/dev/null 2>&1;
    then
        echo "g++ is installed"
        echo `g++ --version | grep "g++ "`
    else
        echo "g++ is not installed, install g++"
        return
    fi

    if type -P ld >/dev/null 2>&1;
    then
        echo "ld is installed"
        echo `ld --version | grep "GNU ld "`
    else
        echo "ld is not installed, install ld"
        return
    fi

    if type -P python3 >/dev/null 2>&1;
    then
        echo "python3 is installed"
        echo `python3 --version`
    else
        echo "python3 is not installed, install python3"
        return
    fi
}

#Function to check optional prerequisites
function check_optional_prereqs() {
    if type -P lscpu >/dev/null 2>&1;
    then
        echo "lscpu is installed"
        echo `lscpu --version`
    else
        echo "lscpu is not installed, install lscpu"
    fi

    # Check if hwloc/lstopo-no-graphics is installed
    if type -P lstopo-no-graphics >/dev/null 2>&1;
    then
        echo "lstopo-no-graphics is installed"
        echo `lstopo-no-graphics --version`
    else
        echo "lstopo-no-graphics is not installed, install hwloc/lstopo-no-graphics"
    fi

    # Check if uname is installed
    if type -P uname >/dev/null 2>&1;
    then
        echo "uname is installed"
        echo `uname --version`
    else
        echo "uname is not installed, install uname"
    fi

    # Check if dmidecode is installed
    if type -P dmidecode >/dev/null 2>&1;
    then
        echo "dmidecode is installed"
        echo `dmidecode --version`
    else
        echo "dmidecode is not installed, install dmidecode"
    fi
}

#------------------------------------------------------------------------------
# Check if mandatory prerequisites are installed
echo "Checking mandatory prerequisites"
check_mandatory_prereqs

echo "Checking optional prerequisites"
# Check if optional prerequisites are installed
check_optional_prereqs
echo""

#------------------------------------------------------------------------------
if [ -z "$ZENDNN_LOG_OPTS" ];
then
    export ZENDNN_LOG_OPTS=ALL:0
    echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"
else
    echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"
fi

if [ -z "$OMP_NUM_THREADS" ];
then
    export OMP_NUM_THREADS=64
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
else
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
fi

if [ -z "$OMP_WAIT_POLICY" ];
then
    export OMP_WAIT_POLICY=ACTIVE
    echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
else
    echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
fi

if [ -z "$OMP_PROC_BIND" ];
then
    export OMP_PROC_BIND=FALSE
    echo "OMP_PROC_BIND=$OMP_PROC_BIND"
else
    echo "OMP_PROC_BIND=$OMP_PROC_BIND"
fi

#By default setting ZENDNN_TF_VERSION as 1.15
export ZENDNN_TF_VERSION="1.15"
echo "ZENDNN_TF_VERSION=$ZENDNN_TF_VERSION"

#By default setting ZENDNN_PT_VERSION as 1.9.0
export ZENDNN_PT_VERSION="1.9.0"
echo "ZENDNN_PT_VERSION=$ZENDNN_PT_VERSION"

#If the environment variable OMP_DYNAMIC is set to true, the OpenMP implementation
#may adjust the number of threads to use for executing parallel regions in order
#to optimize the use of system resources. ZenDNN depend on a number of threads
#which should not be modified by runtime, doing so can cause incorrect execution
export OMP_DYNAMIC=FALSE
echo "OMP_DYNAMIC=$OMP_DYNAMIC"

#Disable TF check for training ops and stop execution if any training ops
#found in TF graph. By default, its enabled
export ZENDNN_INFERENCE_ONLY=1
echo "ZENDNN_INFERENCE_ONLY=$ZENDNN_INFERENCE_ONLY"

#Disable TF memory pool optimization, By default, its enabled
export ZENDNN_ENABLE_MEMPOOL=1
echo "ZENDNN_ENABLE_MEMPOOL=$ZENDNN_ENABLE_MEMPOOL"

#Set the max no. of tensors that can be used inside TF memory pool, Default is
#set to 1024
export ZENDNN_TENSOR_POOL_LIMIT=1024
echo "ZENDNN_TENSOR_POOL_LIMIT=$ZENDNN_TENSOR_POOL_LIMIT"

#Enable fixed max size allocation for Persistent tensor with TF memory pool
#optimization, By default, its disabled
export ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=0
echo "ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE=$ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE"

# Convolution algo path.
#   0. AUTO
#   1. GEMM
#   2. WINOGRAD
#   3. NHWC BLOCKED
# If env variable is not set, GEMM will be selected as default path.
# We recommend NHWC BLOCKED format for the best possible performance.
export ZENDNN_CONV_ALGO=3
echo "ZENDNN_CONV_ALGO=$ZENDNN_CONV_ALGO"

# INT8 support  is disabled by default
export ZENDNN_INT8_SUPPORT=0
echo "ZENDNN_INT8_SUPPORT=$ZENDNN_INT8_SUPPORT"

# INT8 Relu6 fusion support is disabled by default
export ZENDNN_RELU_UPPERBOUND=0
echo "ZENDNN_RELU_UPPERBOUND=$ZENDNN_RELU_UPPERBOUND"

# ZENDNN_MATMUL_ALGO is set to FP32:4 and BF16:3 by default
export ZENDNN_MATMUL_ALGO=FP32:4,BF16:3
echo "ZENDNN_MATMUL_ALGO=$ZENDNN_MATMUL_ALGO"

# Switch to enable Conv, Add fusion on users discretion. Currently it is
# safe to enable this switch for resnet50v1_5, resnet101, and
# inception_resnet_v2 models only. By default the switch is disabled.
export ZENDNN_TF_CONV_ADD_FUSION_SAFE=0
echo "ZENDNN_TF_CONV_ADD_FUSION_SAFE=$ZENDNN_TF_CONV_ADD_FUSION_SAFE"

# Primitive reuse is disabled by default
export TF_ZEN_PRIMITIVE_REUSE_DISABLE=FALSE
echo "TF_ZEN_PRIMITIVE_REUSE_DISABLE=$TF_ZEN_PRIMITIVE_REUSE_DISABLE"

# Enable build of ZenDNN standlone library
export ZENDNN_STANDALONE_BUILD="${ZENDNN_STANDALONE_BUILD:-1}"
echo "ZENDNN_STANDALONE_BUILD=$ZENDNN_STANDALONE_BUILD"

#Set the no. of InterOp threads, Default is set to 1
export ZENDNN_TF_INTEROP_THREADS=1
echo "ZENDNN_TF_INTEROP_THREADS=$ZENDNN_TF_INTEROP_THREADS"

#Check if below declaration of ZENDNN_GIT_ROOT is correct
export ZENDNN_GIT_ROOT=$(pwd)
if [ -z "$ZENDNN_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_GIT_ROOT needs to be set"
    echo "Error: \$ZENDNN_GIT_ROOT points to root of ZENDNN repo"
    return
else
    [ ! -d "$ZENDNN_GIT_ROOT" ] && echo "Directory ZenDNN DOES NOT exists!"
    echo "ZENDNN_GIT_ROOT: $ZENDNN_GIT_ROOT"
fi

#Change ZENDNN_UTILS_GIT_ROOT as per need in future
cd ..
export ZENDNN_UTILS_GIT_ROOT=$(pwd)/ZenDNN_utils
cd -
if [ -z "$ZENDNN_UTILS_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_UTILS_GIT_ROOT needs to be set"
else
    [ ! -d "$ZENDNN_UTILS_GIT_ROOT" ] && echo "Directory ZenDNN_utils DOES NOT exists!"
    echo "ZENDNN_UTILS_GIT_ROOT: $ZENDNN_UTILS_GIT_ROOT"
fi

#Change ZENDNN_TOOLS_GIT_ROOT as per need in future
cd ..
export ZENDNN_TOOLS_GIT_ROOT=$(pwd)/ZenDNN_tools
cd -
if [ -z "$ZENDNN_TOOLS_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_TOOLS_GIT_ROOT needs to be set"
else
    [ ! -d "$ZENDNN_TOOLS_GIT_ROOT" ] && echo "Directory ZenDNN_tools DOES NOT exists!"
    echo "ZENDNN_TOOLS_GIT_ROOT: $ZENDNN_TOOLS_GIT_ROOT"
fi

#Change ZENDNN_PARENT_FOLDER as per need in future
#Current assumption, TF is located parallel to ZenDNN
cd ..
export ZENDNN_PARENT_FOLDER=$(pwd)
cd -

if [ -z "$ZENDNN_PARENT_FOLDER" ];
then
    echo "Error: Environment variable ZENDNN_PARENT_FOLDER needs to be set"
    echo "Error: \$ZENDNN_PARENT_FOLDER points to parent of TF repo"
else
    echo "ZENDNN_PARENT_FOLDER: $ZENDNN_PARENT_FOLDER"
fi

if [ -z "$ZENDNN_AOCC_COMP_PATH" ];
then
    echo "Error: Environment variable ZENDNN_AOCC_COMP_PATH needs to be set"
    return
else
    echo "ZENDNN_AOCC_COMP_PATH: $ZENDNN_AOCC_COMP_PATH"
fi

if [ -z "$ZENDNN_BLIS_PATH" ];
then
    echo "Error: Environment variable ZENDNN_BLIS_PATH needs to be set"
    return
else
    echo "ZENDNN_BLIS_PATH: $ZENDNN_BLIS_PATH"
fi

export TF_GIT_ROOT=$ZENDNN_PARENT_FOLDER/tensorflow
echo "TF_GIT_ROOT: $TF_GIT_ROOT"

export BENCHMARKS_GIT_ROOT=$ZENDNN_PARENT_FOLDER/benchmarks
echo "BENCHMARKS_GIT_ROOT: $BENCHMARKS_GIT_ROOT"

export PYTORCH_GIT_ROOT=$ZENDNN_PARENT_FOLDER/pytorch
echo "PYTORCH_GIT_ROOT: $PYTORCH_GIT_ROOT"

export PYTORCH_BENCHMARK_GIT_ROOT=$ZENDNN_PARENT_FOLDER/pytorch-benchmarks
echo "PYTORCH_BENCHMARK_GIT_ROOT: $PYTORCH_BENCHMARK_GIT_ROOT"

export ONNXRUNTIME_GIT_ROOT=$ZENDNN_PARENT_FOLDER/onnxruntime
echo "ONNXRUNTIME_GIT_ROOT: $ONNXRUNTIME_GIT_ROOT"

export ZENDNN_ONNXRT_VERSION="1.8.0"
echo "ZENDNN_ONNXRT_VERSION: $ZENDNN_ONNXRT_VERSION"

export ZENDNN_ONNX_VERSION="1.9.0"
echo "ZENDNN_ONNX_VERSION: $ZENDNN_ONNX_VERSION"

# Primitive Caching Capacity
export ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
echo "ZENDNN_PRIMITIVE_CACHE_CAPACITY: $ZENDNN_PRIMITIVE_CACHE_CAPACITY"

# MAX_CPU_ISA
# MAX_CPU_ISA is disabld at build time. When feature is enabled, uncomment the
# below 2 lines
#export ZENDNN_MAX_CPU_ISA=ALL
#echo "ZENDNN_MAX_CPU_ISA: $ZENDNN_MAX_CPU_ISA"

# Enable primitive create and primitive execute logs. By default it is disabled
export ZENDNN_PRIMITIVE_LOG_ENABLE=0
echo "ZENDNN_PRIMITIVE_LOG_ENABLE: $ZENDNN_PRIMITIVE_LOG_ENABLE"

# libxsmm integration(retained for future use)
export ZENDNN_ENABLE_LIBXSMM=0
echo "ZENDNN_ENABLE_LIBXSMM: $ZENDNN_ENABLE_LIBXSMM"

if [ "$ZENDNN_ENABLE_LIBXSMM" = "1" ];
then
    if [ -z ${ZENDNN_LIBXSMM_PATH+x} ];
    then
        echo "ZENDNN_LIBXSMM_PATH is not set. (TPP requires libxsmm)"
    else
        echo "ZENDNN_LIBXSMM_PATH : $ZENDNN_LIBXSMM_PATH"
    fi
fi

# Default location for benchmark data.
export ZENDNN_TEST_USE_COMMON_BENCHMARK_LOC=FALSE
echo "ZENDNN_TEST_USE_COMMON_BENCHMARK_LOC: $ZENDNN_TEST_USE_COMMON_BENCHMARK_LOC"
export ZENDNN_TEST_COMMON_BENCHMARK_LOC=/home/amd/benchmark_data
echo "ZENDNN_TEST_COMMON_BENCHMARK_LOC: $ZENDNN_TEST_COMMON_BENCHMARK_LOC"

#----------------------------------------------------------------------------
# Source <compdir>/setenv_AOCC.sh from AOCC installer
# FIXME: This needs to fixed with AOCC 2.2
#chmod u+x $ZENDNN_AOCC_COMP_PARENT_FOLDER/setenv_AOCC.sh
#source $ZENDNN_AOCC_COMP_PARENT_FOLDER/setenv_AOCC.sh

#----------------------------------------------------------------------------
# Export PATH and LD_LIBRARY_PATH for AOCC Compiler
#FIXME: Use this for AOCC compiler path since 'source <compdir>/setenv_AOCC.sh' has issues
export PATH=$ZENDNN_AOCC_COMP_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ZENDNN_AOCC_COMP_PATH/lib:$ZENDNN_AOCC_COMP_PATH/lib32:$LD_LIBRARY_PATH

# Export PATH and LD_LIBRARY_PATH for ZenDNN
export PATH=$PATH:$ZENDNN_GIT_ROOT/_out/tests
# BLIS include path and lib path is not same when BLIS is build from source vs
# BLIS official release is used
if [ "$ZENDNN_STANDALONE_BUILD" = "1" ];
then
    export LD_LIBRARY_PATH=$ZENDNN_BLIS_PATH/lib/LP64:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$ZENDNN_BLIS_PATH/lib:$LD_LIBRARY_PATH
fi

if [ "$ZENDNN_ENABLE_LIBXSMM" = "1" ];
then
    echo -n "checking and adding ZENDNN_LIBXSMM path..."
    if [ ! -z ${ZENDNN_LIBXSMM_PATH+x} ];
    then
        export LD_LIBRARY_PATH=$ZENDNN_LIBXSMM_PATH/lib/:$LD_LIBRARY_PATH
    else
        echo "ZENNDNN_LIBXSMM_PATH not set."
    fi
fi

export LD_LIBRARY_PATH=$ZENDNN_GIT_ROOT/_out/lib/:$ZENDNN_GIT_ROOT/external/googletest/lib:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH: "$LD_LIBRARY_PATH

#FIXME: create softlink for libomp.so.5, find a better solution
cd $ZENDNN_AOCC_COMP_PATH/lib
ln -sf libomp.so libomp.so.5

#-------------------------------------------------------------------------------
# HW, HW architecture, Cache, OS, Kernel details
#-----------------------------------------------------------------------------
# Go to ZENDNN_GIT_ROOT
cd $ZENDNN_GIT_ROOT

chmod u+x scripts/gather_hw_os_kernel_bios_info.sh
echo "scripts/gather_hw_os_kernel_bios_info.sh"
source scripts/gather_hw_os_kernel_bios_info.sh true > system_hw_os_kernel_bios_info.txt

#-------------------------------------------------------------------------------
# Go to ZENDNN_GIT_ROOT
cd $ZENDNN_GIT_ROOT
echo -e "\n"
echo "Please set below environment variables explicitly as per the platform you are using!!"
echo -e "\tOMP_NUM_THREADS, GOMP_CPU_AFFINITY"
echo -e "\n"
