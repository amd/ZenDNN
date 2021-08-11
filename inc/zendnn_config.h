/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

// ZenDNN CPU threading runtime
#define ZENDNN_CPU_THREADING_RUNTIME ZENDNN_RUNTIME_OMP

// ZenDNN CPU engine runtime
#define ZENDNN_CPU_RUNTIME ZENDNN_RUNTIME_OMP

// ZenDNN GPU engine runtime
#define ZENDNN_GPU_RUNTIME ZENDNN_RUNTIME_NONE

// clang-format on

/// Build Options
/// Enable Primitive Cache. To mitigate primitive creation overhead, ZenDNN
/// provides the primitive cache which automatically caches created primitives
/// to avoid repeating JIT compilation for the primitives with identical
/// operation descriptors, attributes.
/// To diable this feature, change the below define to 0
/// Corresponding run time environment variable is
/// ZENDNN_PRIMITIVE_CACHE_CAPACITY whose default value is 1024
#define ZENDNN_ENABLE_PRIMITIVE_CACHE   1

/// When the feature is enabled at build-time, the ZENDNN_MAX_CPU_ISA
/// environment variable can be used to limit processor features ZenDNN is
/// able to detect to certain Instruction Set Architecture (ISA) and older
/// instruction sets.
/// This feature is disabled at build time. To enable this feature change the
/// below define to 1 and use corresponding environment variable mentioned here.
/// Corresponding run time environment variable is
/// ZENDNN_MAX_CPU_ISA whose default value is ALL.
/// Possible values for ZENDNN_MAX_CPU_ISA: SSE41, AVX, AVX2, AVX512, ALL
#define ZENDNN_ENABLE_MAX_CPU_ISA       0

/// For performance reasons, extra hints may be provided to ZenDNN which enable
/// the just-in-time (JIT) code generation to prefer or avoid certain CPU ISA
/// features.
/// This feature is disabled at build time. To enable this feature change the
/// below define to 1 and use corresponding environment variable mentioned here.
/// Corresponding run time environment variable is
/// ZENDNN_CPU_ISA_HINTS whose default value is NO_HINTS.
/// Possible values for ZENDNN_CPU_ISA_HINTS: NO_HINTS, PREFER_YMM
#define ZENDNN_ENABLE_CPU_ISA_HINTS        0

/// Disables sharing a common scratchpad between primitives in
/// zendnn::scratchpad_mode::library mode
#define ZENDNN_ENABLE_CONCURRENT_EXEC   0

#if defined(ZENDNN_CPU_RUNTIME) && defined(ZENDNN_GPU_RUNTIME)
#if (ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_NONE) \
        || (ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_OCL)
#error "Unexpected ZENDNN_CPU_RUNTIME"
#endif
#if (ZENDNN_GPU_RUNTIME != ZENDNN_RUNTIME_NONE) \
        && (ZENDNN_GPU_RUNTIME != ZENDNN_RUNTIME_OCL) \
        && (ZENDNN_GPU_RUNTIME != ZENDNN_RUNTIME_SYCL)
#error "Unexpected ZENDNN_GPU_RUNTIME"
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

// When defined, DPCPP is supported.
/* #undef ZENDNN_WITH_SYCL */

// When defined, Level Zero is supported.
/* #undef ZENDNN_WITH_LEVEL_ZERO */

// When defined, SYCL CUDA backend is used.
/* #undef ZENDNN_SYCL_CUDA */

#endif
