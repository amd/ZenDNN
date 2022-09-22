/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef ZENDNN_CONFIG_H
#define ZENDNN_CONFIG_H

#include "zendnn_types.h"

/// @cond DO_NOT_DOCUMENT_THIS

// All symbols shall be internal unless marked as ZENDNN_API
#if defined _WIN32 || defined __CYGWIN__
#define ZENDNN_HELPER_DLL_IMPORT __declspec(dllimport)
#define ZENDNN_HELPER_DLL_EXPORT __declspec(dllexport)
#else
#if __GNUC__ >= 4
#define ZENDNN_HELPER_DLL_IMPORT __attribute__((visibility("default")))
#define ZENDNN_HELPER_DLL_EXPORT __attribute__((visibility("default")))
#else
#define ZENDNN_HELPER_DLL_IMPORT
#define ZENDNN_HELPER_DLL_EXPORT
#endif
#endif

#ifdef ZENDNN_DLL
#ifdef ZENDNN_DLL_EXPORTS
#define ZENDNN_API ZENDNN_HELPER_DLL_EXPORT
#else
#define ZENDNN_API ZENDNN_HELPER_DLL_IMPORT
#endif
#else
#define ZENDNN_API
#endif

#if defined(__GNUC__)
#define ZENDNN_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
#define ZENDNN_DEPRECATED __declspec(deprecated)
#else
#define ZENDNN_DEPRECATED
#endif

/// @endcond

// clang-format off

// ZENDNN CPU threading runtime
#define ZENDNN_CPU_THREADING_RUNTIME ZENDNN_RUNTIME_OMP

// ZENDNN CPU engine runtime
#define ZENDNN_CPU_RUNTIME ZENDNN_RUNTIME_OMP

// ZENDNN GPU engine runtime
#define ZENDNN_GPU_RUNTIME ZENDNN_RUNTIME_NONE

// clang-format on

#if defined(ZENDNN_CPU_RUNTIME) && defined(ZENDNN_GPU_RUNTIME)
#if (ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_OCL)
#error "Unexpected ZENDNN_CPU_RUNTIME"
#endif
#if (ZENDNN_GPU_RUNTIME != ZENDNN_RUNTIME_NONE) \
        && (ZENDNN_GPU_RUNTIME != ZENDNN_RUNTIME_OCL) \
        && (ZENDNN_GPU_RUNTIME != ZENDNN_RUNTIME_SYCL)
#error "Unexpected ZENDNN_GPU_RUNTIME"
#endif
#if (ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_NONE \
        && ZENDNN_GPU_RUNTIME == ZENDNN_RUNTIME_NONE)
#error "At least one runtime must be specified"
#endif
#else
#error "BOTH ZENDNN_CPU_RUNTIME and ZENDNN_GPU_RUNTIME must be defined"
#endif

// For SYCL CPU, a primitive may be created and executed in different threads
// hence the global scratchpad does not work. This enables concurrent execution
// when CPU runtime is SYCL to avoid the issue.
#if ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_SYCL
#ifndef ZENDNN_ENABLE_CONCURRENT_EXEC
#define ZENDNN_ENABLE_CONCURRENT_EXEC
#endif
#endif

// When defined, primitive cache stores runtime objects.
#undef ZENDNN_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE

// When defined, DPCPP is supported.
#undef ZENDNN_WITH_SYCL

// When defined, Level Zero is supported.
#undef ZENDNN_WITH_LEVEL_ZERO

// When defined, SYCL CUDA backend is used.
#undef ZENDNN_SYCL_CUDA

// When defined, stack checker is enabled.
#undef ZENDNN_ENABLE_STACK_CHECKER

// When defined, experimental features are enabled.
#undef ZENDNN_EXPERIMENTAL

// List of configurating build controls
// Workload controls
#define BUILD_TRAINING 0
#define BUILD_INFERENCE 1
// Primitive controls
#define BUILD_PRIMITIVE_ALL 1
#define BUILD_BATCH_NORMALIZATION 0
#define BUILD_BINARY 0
#define BUILD_CONCAT 0
#define BUILD_CONVOLUTION 0
#define BUILD_DECONVOLUTION 0
#define BUILD_ELTWISE 0
#define BUILD_INNER_PRODUCT 0
#define BUILD_LAYER_NORMALIZATION 0
#define BUILD_LRN 0
#define BUILD_MATMUL 0
#define BUILD_POOLING 0
#define BUILD_PRELU 0
#define BUILD_REDUCTION 0
#define BUILD_REORDER 0
#define BUILD_RESAMPLING 0
#define BUILD_RNN 0
#define BUILD_SHUFFLE 0
#define BUILD_SOFTMAX 0
#define BUILD_SUM 0
// Primitives CPU ISA controls
#define BUILD_PRIMITIVE_CPU_ISA_ALL 1
#define BUILD_SSE41 0
#define BUILD_AVX2 0
#define BUILD_AVX512 0
#define BUILD_AMX 0
// Primitives GPU ISA controls
#define BUILD_PRIMITIVE_GPU_ISA_ALL 0
#define BUILD_GEN9 0
#define BUILD_GEN11 0
#define BUILD_XELP 0
#define BUILD_XEHP 0
#define BUILD_XEHPG 0
#define BUILD_XEHPC 0
//ZenDNN core specific control
//Few needs to move to frameworks specific build files
#define ZENDNN_ENABLE_PRIMITIVE_CACHE 1
//#define ZENDNN_DISABLE_PRIMITIVE_CACHE  1
#define ZENDNN_ENABLE_MAX_CPU_ISA       0
#define ZENDNN_ENABLE_CPU_ISA_HINTS     0
#define ZENDNN_ENABLE_CONCURRENT_EXEC   0

#endif
