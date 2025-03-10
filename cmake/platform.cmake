#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
#******************************************************************************/

#===============================================================================
# Copyright 2016-2022 Intel Corporation
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
#===============================================================================

# Manage platform-specific quirks
#===============================================================================

if(platform_cmake_included)
    return()
endif()
set(platform_cmake_included true)

include("cmake/utils.cmake")

if (ZENDNN_LIBRARY_TYPE STREQUAL "SHARED")
    add_definitions_with_host_compiler(-DZENDNN_DLL)
endif()

# Specify the target architecture
add_definitions_with_host_compiler(-DZENDNN_${ZENDNN_TARGET_ARCH}=1)

# UNIT8_MAX-like macros are a part of the C99 standard and not a part of the
# C++ standard (see C99 standard 7.18.2 and 7.18.4)
add_definitions_with_host_compiler(-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)

set(CMAKE_CCXX_FLAGS)
set(CMAKE_CCXX_NOWARN_FLAGS)
set(CMAKE_CCXX_NOEXCEPT_FLAGS)
set(DEF_ARCH_OPT_FLAGS)

# Compatibility with ZENDNN
if($ENV{ZENDNN_WERROR})
    set(ZENDNN_WERROR $ENV{ZENDNN_WERROR})
endif()

if($ENV{ZENDNN_WERROR})
    set(ZENDNN_WERROR $ENV{ZENDNN_WERROR})
endif()

if (ZENDNN_PATH_ENABLE)
    set(ZENDNN_ENABLE TRUE)
    #add_definitions_with_host_compiler(-DZENDNN_ENABLE=1)
endif()

if (ZENDNN_BIAS_ENABLE)
    set(BIAS_ENABLED TRUE)
    #add_definitions_with_host_compiler(-DBIAS_ENABLED=1)
endif()

if (ZENDNN_BLIS_EXPERT_ENABLE)
    set(BLIS_EXPERT TRUE)
    #add_definitions_with_host_compiler(-DBLIS_EXPERT=1)
endif()

if (ZENDNN_DIRECT_CONV_ENABLE)
    set(ZENDNN_DIRECT_CONV TRUE)
    #add_definitions_with_host_compiler(-DZENDNN_DIRECT_CONV=1)
endif()

if (ZENDNN_BLOCKED_POOLING_ENABLE)
    set(ZENDNN_BLOCKED_POOLING TRUE)
    #add_definitions_with_host_compiler(-DZENDNN_BLOCKED_POOLING=1)
endif()

if (ZENDNN_ENABLE_LPGEMM)
	set(ZENDNN_ENABLE_LPGEMM_CONV FALSE)
	add_definitions_with_host_compiler(-DZENDNN_ENABLE_LPGEMM_CONV=0)
endif()

if (ZENDNN_LPGEMM_V5_0_ENABLE)
       add_definitions_with_host_compiler(-DZENDNN_ENABLE_LPGEMM_V5_0=1)
endif()

if (ZENDNN_USE_LOCAL_BLIS)
    set(BLIS_PATH $ENV{ZENDNN_BLIS_PATH})
    if(UNIX)
        include_directories_with_host_compiler(${BLIS_PATH}/include)
        list(APPEND EXTRA_SHARED_LIBS ${BLIS_PATH}/lib/libblis-mt.so)
    elseif(WIN32)
        cmake_path(SET BLIS_PATH_WIN "$ENV{ZENDNN_BLIS_PATH}")
        message(STATUS "BLIS PATH : ${BLIS_PATH_WIN}")
        include_directories_with_host_compiler(${BLIS_PATH_WIN}/include/ILP64)
        list(APPEND EXTRA_SHARED_LIBS ${BLIS_PATH_WIN}/lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib)
    endif()
endif()

# The flags that can be used for the main and host compilers should be moved to
# the macros to avoid code duplication and ensure consistency.
macro(platform_unix_and_mingw_common_ccxx_flags var)
    append(${var} "-Wall -Wno-unknown-pragmas")
    append_if(ZENDNN_WERROR ${var} "-Werror")
    append(${var} "-fvisibility=internal")
endmacro()

macro(platform_unix_and_mingw_common_cxx_flags var)
    append(${var} "-fvisibility-inlines-hidden")
endmacro()

macro(platform_unix_and_mingw_noexcept_ccxx_flags var)
    append(${var} "-fno-exceptions")
endmacro()

macro(platform_gnu_x64_arch_ccxx_flags var)
    set(${var} "-msse4.1")
endmacro()

macro(platform_gnu_nowarn_ccxx_flags var gnu_version)
    # suppress warning on assumptions made regarding overflow (#146)
    append(${var} "-Wno-strict-overflow")
    # suppress false positive warnings about uninitialized variables
    append(${var} "-Wno-maybe-uninitialized")
    # suppress false positive warnings with 10.x: GCC Bugzilla – Bug 96963
    # assume 0.0 is unknown version - always suppress the warning
    if(${gnu_version} VERSION_EQUAL 0.0 OR
        (${gnu_version} VERSION_GREATER 10.0 AND ${gnu_version} VERSION_LESS 11.0))
        append(${var} "-Wno-stringop-overflow")
    endif()
endmacro()

macro(platform_unix_and_mingw_common_ccxx_hw_flags gnu_version)
    # Execute cat /proc/cpuinfo | grep "model name" -m1 | awk '{print (substr($6, 1)); }'
    # Should print 7742, 7743, Threadripper
    # Execute cat /proc/cpuinfo | grep "model name" -m1 | awk '{print (substr($6, 1)); }'
    # Should print 2 (7742), 3 (7743), e (Threadripper)
    execute_process(COMMAND cat /proc/cpuinfo
        COMMAND grep "model name" -m1
        COMMAND awk "{print (substr($6, 4)); }"
        OUTPUT_VARIABLE EPYC_MODEL_NAME)
    if (NOT EPYC_MODEL_NAME)
        message(STATUS "Unable to determine EPYC_MODEL_NAME: ${EPYC_MODEL_NAME}")
    endif()

    #This strips terminating newline in the variable <varname>:
    string(REGEX REPLACE "\n$" "" EPYC_MODEL_NAME "${EPYC_MODEL_NAME}")

    if(${gnu_version} VERSION_GREATER 11.0)
        if("${EPYC_MODEL_NAME}" STREQUAL "3")
            append(CMAKE_CCXX_FLAGS "-march=znver3")
            message(STATUS "Platform: GNU VERSION: ${gnu_version}, 3rd Gen Epic (Milan)")
        elseif("${EPYC_MODEL_NAME}" STREQUAL "2")
            append(CMAKE_CCXX_FLAGS "-march=znver2")
            message(STATUS "Platform: GNU VERSION: ${gnu_version}, 2nd Gen Epic (Rome)")
        else()
            append(CMAKE_CCXX_FLAGS "-march=znver1")
        endif()
    elseif(${gnu_version} VERSION_GREATER 9.0)
        if("${EPYC_MODEL_NAME}" STREQUAL "3")
            append(CMAKE_CCXX_FLAGS "-march=znver2")
            message(STATUS "Platform: GNU VERSION: ${gnu_version}, 3rd Gen Epic (Milan)")
        elseif("${EPYC_MODEL_NAME}" STREQUAL "2")
            append(CMAKE_CCXX_FLAGS "-march=znver2")
            message(STATUS "Platform: GNU VERSION: ${gnu_version}, 2nd Gen Epic (Rome)")
        else()
            append(CMAKE_CCXX_FLAGS "-march=znver1")
        endif()
    else()
        append(CMAKE_CCXX_FLAGS "-march=znver1")
    endif()
    message(STATUS "GNU VERSION: ${gnu_version}")
    message(STATUS "CMAKE_CCXX_FLAGS: ${CMAKE_CCXX_FLAGS}")
endmacro()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32)
    #set(NOMINMAX TRUE)
    #add_definitions_with_host_compiler(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions_with_host_compiler(-DZENDNN_ENABLE)
    append(CMAKE_CCXX_FLAGS "/arch:AVX2")
    append(CMAKE_CCXX_FLAGS "/MP")
endif()

if(ZENDNN_WITH_SYCL)
    # Suppress self-comparison warning to avoid false positives in macros such
    # as JIT_IMPL_NAME_HELPER.
    append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-tautological-compare")
    if(WIN32)
        # XXX: SYCL does not like __thiscall convention coming from TBB,
        # suppress warnings for now.
        append(CMAKE_CCXX_FLAGS "-Wno-ignored-attributes")
        # XXX: compiler always pulls in release C++ runtime by default, until
        # this is fixed we have to explicitly drop release C++ runtime for
        # debug build types.
        string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)
        if(UPPERCASE_CMAKE_BUILD_TYPE MATCHES "(DEBUG|RELWITHMDD)" AND NOT CMAKE_BASE_NAME MATCHES "(icx|icpx)")
            append(CMAKE_EXE_LINKER_FLAGS "-Xlinker /NODEFAULTLIB:msvcrt")
            append(CMAKE_SHARED_LINKER_FLAGS "-Xlinker /NODEFAULTLIB:msvcrt")
        endif()
    endif()
endif()

if(MSVC)
    set(USERCONFIG_PLATFORM "x64")
    append_if(ZENDNN_WERROR CMAKE_CCXX_FLAGS "/WX")
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        append(CMAKE_CCXX_FLAGS "/MP")
        # increase number of sections in obj file
        append(CMAKE_CCXX_FLAGS "/bigobj")
        # int -> bool
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4800")
        # unknown pragma
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4068")
        # double -> float
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4305")
        # UNUSED(func)
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4551")
        # int64_t -> int (tent)
        append(CMAKE_CCXX_NOWARN_FLAGS "/wd4244")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        append(CMAKE_CCXX_FLAGS "/MP")
        # increase number of sections in obj file
        append(CMAKE_CCXX_FLAGS "/bigobj")
        set(DEF_ARCH_OPT_FLAGS "-QxSSE4.1")
        # disable: loop was not vectorized with "simd"
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:13379")
        # disable: loop was not vectorized with "simd"
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:15552")
        # disable: unknown pragma
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:3180")
        # disable: foo has been targeted for automatic cpu dispatch
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:15009")
        # disable: disabling user-directed function packaging (COMDATs)
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:11031")
        # disable: decorated name length exceeded, name was truncated
        append(CMAKE_CCXX_NOWARN_FLAGS "-Qdiag-disable:2586")
        # disable: disabling optimization; runtime debug checks enabled
        append(CMAKE_CXX_FLAGS_DEBUG "-Qdiag-disable:10182")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        append(CMAKE_CCXX_NOEXCEPT_FLAGS "-fno-exceptions")
        # Clang cannot vectorize some loops with #pragma omp simd and gets
        # very upset. Tell it that it's okay and that we love it
        # unconditionally.
        append(CMAKE_CCXX_FLAGS "-Wno-pass-failed")
        # Clang doesn't like the idea of overriding optimization flags.
        # We don't want to optimize jit gemm kernels to reduce compile time
        append(CMAKE_CCXX_FLAGS "-Wno-overriding-t-option")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR ZENDNN_WITH_SYCL OR
            CMAKE_BASE_NAME STREQUAL "icx" OR CMAKE_BASE_NAME STREQUAL "icpx")
        # Default fp-model in icx and dpcpp (unlike clang) may be precise or
        # fast=1 depending on the version.
        append(CMAKE_CCXX_FLAGS "/fp:precise")
        # icx and dpcpp cannot vectorize some loops with #pragma omp simd and
        # gets very upset. Tell it that it's okay and that we love it
        # unconditionally.
        append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-pass-failed")
    endif()
elseif(UNIX OR MINGW)
    if(ZENDNN_WITH_SYCL OR CMAKE_BASE_NAME STREQUAL "icx" OR CMAKE_BASE_NAME STREQUAL "icpx")
        # Default fp-model in icx and dpcpp (unlike clang) may be precise or
        # fast=1 depending on the version.
        append(CMAKE_CCXX_FLAGS "-ffp-model=precise -fno-reciprocal-math")
        # icx and dpcpp cannot vectorize some loops with #pragma omp simd and
        # gets very upset. Tell it that it's okay and that we love it
        # unconditionally.
        append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-pass-failed")
    endif()

    platform_unix_and_mingw_common_ccxx_flags(CMAKE_CCXX_FLAGS)
    platform_unix_and_mingw_common_cxx_flags(CMAKE_CXX_FLAGS)
    platform_unix_and_mingw_noexcept_ccxx_flags(CMAKE_CMAKE_CCXX_NOEXCEPT_FLAGS)
    platform_unix_and_mingw_common_ccxx_hw_flags(${CMAKE_CXX_COMPILER_VERSION})

    # compiler specific settings
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(ZENDNN_TARGET_ARCH STREQUAL "AARCH64")
             if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
                 set(DEF_ARCH_OPT_FLAGS "-O3")
             endif()
             # For native compilation tune for the host processor
             if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                 append(DEF_ARCH_OPT_FLAGS "-mcpu=native")
             endif()
        elseif(ZENDNN_TARGET_ARCH STREQUAL "PPC64")
             if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
                 set(DEF_ARCH_OPT_FLAGS "-O3")
             endif()
             # For native compilation tune for the host processor
             if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                 append(DEF_ARCH_OPT_FLAGS "-mcpu=native")
             endif()
        elseif(ZENDNN_TARGET_ARCH STREQUAL "S390X")
             if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
                 set(DEF_ARCH_OPT_FLAGS "-O3")
             endif()
             # For native compilation tune for the host processor
             if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                 append(DEF_ARCH_OPT_FLAGS "-march=native")
             endif()
        elseif(ZENDNN_TARGET_ARCH STREQUAL "X64")
             set(DEF_ARCH_OPT_FLAGS "-msse4.1")
        endif()
        # Clang cannot vectorize some loops with #pragma omp simd and gets
        # very upset. Tell it that it's okay and that we love it
        # unconditionally.
        append(CMAKE_CCXX_NOWARN_FLAGS "-Wno-pass-failed")
        if(ZENDNN_USE_CLANG_SANITIZER MATCHES "Memory(WithOrigin)?")
            if(NOT ZENDNN_CPU_THREADING_RUNTIME STREQUAL "SEQ")
                message(WARNING "Clang OpenMP is not compatible with MSan! "
                    "Expect a lot of false positives!")
            endif()
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=memory")
            if(ZENDNN_USE_CLANG_SANITIZER STREQUAL "MemoryWithOrigin")
                append(CMAKE_CCXX_SANITIZER_FLAGS
                    "-fsanitize-memory-track-origins=2")
                append(CMAKE_CCXX_SANITIZER_FLAGS
                    "-fno-omit-frame-pointer")
            endif()
            set(ZENDNN_ENABLED_CLANG_SANITIZER "${ZENDNN_USE_CLANG_SANITIZER}")
        elseif(ZENDNN_USE_CLANG_SANITIZER STREQUAL "Undefined")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=undefined")
            append(CMAKE_CCXX_SANITIZER_FLAGS
                "-fno-sanitize=function,vptr")  # work around linking problems
            set(ZENDNN_ENABLED_CLANG_SANITIZER "${ZENDNN_USE_CLANG_SANITIZER}")
        elseif(ZENDNN_USE_CLANG_SANITIZER STREQUAL "Address")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=address")
            set(ZENDNN_ENABLED_CLANG_SANITIZER "${ZENDNN_USE_CLANG_SANITIZER}")
        elseif(ZENDNN_USE_CLANG_SANITIZER STREQUAL "Thread")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=thread")
            set(ZENDNN_ENABLED_CLANG_SANITIZER "${ZENDNN_USE_CLANG_SANITIZER}")
        elseif(ZENDNN_USE_CLANG_SANITIZER STREQUAL "Leak")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-fsanitize=leak")
            set(ZENDNN_ENABLED_CLANG_SANITIZER "${ZENDNN_USE_CLANG_SANITIZER}")
        elseif(NOT ZENDNN_USE_CLANG_SANITIZER STREQUAL "")
            message(FATAL_ERROR
                "Unsupported Clang sanitizer '${ZENDNN_USE_CLANG_SANITIZER}'")
        endif()
        if(ZENDNN_ENABLED_CLANG_SANITIZER)
            message(STATUS
                "Using Clang ${ZENDNN_ENABLED_CLANG_SANITIZER} "
                "sanitizer (experimental!)")
            append(CMAKE_CCXX_SANITIZER_FLAGS "-g -fno-omit-frame-pointer")
            # Blacklist to ignore false-positive cases. Each case may be
            # assigned to a specific sanitizer. See online doc for help.
            append(CMAKE_CCXX_SANITIZER_FLAGS
                   "-fsanitize-blacklist=${PROJECT_SOURCE_DIR}/.clang-ignorelist")
        endif()

        if (ZENDNN_USE_CLANG_TIDY MATCHES "(CHECK|FIX)" AND ${CMAKE_VERSION} VERSION_LESS "3.6.0")
            message(FATAL_ERROR "Using clang-tidy requires CMake 3.6.0 or newer")
        elseif(ZENDNN_USE_CLANG_TIDY MATCHES "(CHECK|FIX)")
            find_program(CLANG_TIDY NAMES clang-tidy)
            if(NOT CLANG_TIDY)
                message(FATAL_ERROR "Clang-tidy not found")
            else()
                if(ZENDNN_USE_CLANG_TIDY STREQUAL "CHECK")
                    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY})
                    message(STATUS "Using clang-tidy to run checks")
                elseif(ZENDNN_USE_CLANG_TIDY STREQUAL "FIX")
                    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY} -fix)
                    message(STATUS "Using clang-tidy to run checks and fix found issues")
                endif()
            endif()
        endif()

    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # XXX: Suppress a warning that pops up when using a function pointer
        # to an OpenCL function as a template argument (GCC Bugzilla – Bug 71463).
        if (ZENDNN_GPU_RUNTIME STREQUAL "OCL")
            append(CMAKE_CCXX_FLAGS "-Wno-ignored-attributes")
        endif()

        if(ZENDNN_TARGET_ARCH STREQUAL "AARCH64")
            if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
                set(DEF_ARCH_OPT_FLAGS "-O3")
            endif()
            # For native compilation tune for the host processor
            if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                append(DEF_ARCH_OPT_FLAGS "-mcpu=native")
            endif()
        elseif(ZENDNN_TARGET_ARCH STREQUAL "PPC64")
            if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
                set(DEF_ARCH_OPT_FLAGS "-O3")
            endif()
            # In GCC, -ftree-vectorize is turned on under -O3 since 2007.
            # For native compilation tune for the host processor
            if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                append(DEF_ARCH_OPT_FLAGS "-mcpu=native")
            endif()
        elseif(ZENDNN_TARGET_ARCH STREQUAL "S390X")
            if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
                set(DEF_ARCH_OPT_FLAGS "-O3")
            endif()
            # In GCC, -ftree-vectorize is turned on under -O3 since 2007.
            # For native compilation tune for the host processor
            if (CMAKE_SYSTEM_PROCESSOR STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
                append(DEF_ARCH_OPT_FLAGS "-march=native")
            endif()
        elseif(ZENDNN_TARGET_ARCH STREQUAL "RV64")
            # G = General-purpose extensions, C = Compression extension (very common).
            append(DEF_ARCH_OPT_FLAGS "-march=rv64gc")
        elseif(ZENDNN_TARGET_ARCH STREQUAL "X64")
            platform_gnu_x64_arch_ccxx_flags(DEF_ARCH_OPT_FLAGS)
        endif()
        platform_gnu_nowarn_ccxx_flags(CMAKE_CCXX_NOWARN_FLAGS ${CMAKE_CXX_COMPILER_VERSION})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(DEF_ARCH_OPT_FLAGS "-xSSE4.1")
        # workaround for Intel Compiler that produces error caused
        # by pragma omp simd collapse(..)
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:13379")
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:15552")
        # disable `was not vectorized: vectorization seems inefficient` remark
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:15335")
        # disable: foo has been targeted for automatic cpu dispatch
        append(CMAKE_CCXX_NOWARN_FLAGS "-diag-disable:15009")
    endif()
endif()

if(MSVC)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # There's no opportunity for icl to link its libraries statically
        # into the library. That's why removing them when searching symbols.
        # libm symbols will be taken from ucrt.lib, otherwise, linker will
        # complain about duplicated symbols being linked to the library.
        append(NO_DYNAMIC_LIBS "/NODEFAULTLIB:libmmd.lib")
        append(NO_DYNAMIC_LIBS "/NODEFAULTLIB:svml_dispmd.lib svml_dispmt.lib")
        append(CMAKE_SHARED_LINKER_FLAGS "${NO_DYNAMIC_LIBS}")
    endif()
elseif(UNIX OR MINGW)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # Link Intel libraries statically (except for iomp5)
        if ("${ZENDNN_CPU_THREADING_RUNTIME}" STREQUAL "OMP")
            append(CMAKE_SHARED_LINKER_FLAGS "-liomp5")
        endif()
        append(CMAKE_SHARED_LINKER_FLAGS "-static-intel")
        # Tell linker to not complain about missing static libraries
        append(CMAKE_SHARED_LINKER_FLAGS "-diag-disable:10237")
    endif()
endif()

if(ZENDNN_ARCH_OPT_FLAGS STREQUAL "HostOpts")
    set(ZENDNN_ARCH_OPT_FLAGS "${DEF_ARCH_OPT_FLAGS}")
endif()

append(CMAKE_C_FLAGS "${CMAKE_CCXX_FLAGS} ${ZENDNN_ARCH_OPT_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_FLAGS} ${ZENDNN_ARCH_OPT_FLAGS}")

if(APPLE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    # FIXME: this is ugly but required when compiler does not add its library
    # paths to rpath (like Intel compiler...)
    foreach(_ ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES})
        set(_rpath "-Wl,-rpath,${_}")
        append(CMAKE_SHARED_LINKER_FLAGS "${_rpath}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${_rpath}")
    endforeach()
endif()
