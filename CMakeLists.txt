# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.22)

# Project name
project(amdZenDNN)

include(CMakePrintHelpers)

find_package(Git REQUIRED)

# Set C++ standard
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_COMPILER "g++")

set(ZENDNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${ZENDNN_DIR}/cmake2025")

option(BUILD_SHARED_LIBS "Build SHARED/STATIC" ON)

if(NOT DEFINED AOCC)
  set(AOCC 0)
endif()

if(NOT DEFINED BLIS_API)
  set(BLIS_API 0)
endif()

if(BLIS_API EQUAL 1)
  set(AMDBLIS_ENABLE_CBLAS OFF)
elseif(BLIS_API EQUAL 0)
  set(AMDBLIS_ENABLE_CBLAS ON)
else()
   message(FATAL_ERROR, "Error, possible values for BLIS_API 0/1, received ${BLIS_API}")
endif()

if(NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
  set(_GLIBCXX_USE_CXX11_ABI 0)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release")
endif()

#FBGEMM options and default values
if(NOT DEFINED FBGEMM_TAG)
  set(FBGEMM_TAG v0.6.0)
endif()

set(FBGEMM_BUILD_TESTS 0)

set(FBGEMM_CMAKE_C_FLAGS   "-Wno-error=maybe-uninitialized -Wno-error=uninitialized")
set(FBGEMM_CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI} -Wno-error=maybe-uninitialized -Wno-error=uninitialized")

#AMD-BLIS options and default values
if(NOT DEFINED AMDBLIS_TAG)
  set(AMDBLIS_TAG AOCL-LPGEMM-012925)
endif()

option(AMDBLIS_ENABLE_BLAS "AMD-BLIS, ENABLE_BLAS" ON)

if(NOT DEFINED AMDBLIS_ENABLE_ADDON)
    set(AMDBLIS_ENABLE_ADDON aocl_gemm)
endif()

if(NOT DEFINED AMDBLIS_BLIS_CONFIG_FAMILY)
    set(AMDBLIS_BLIS_CONFIG_FAMILY amdzen)
endif()

if(NOT DEFINED AMDBLIS_ENABLE_THREADING)
    set(AMDBLIS_ENABLE_THREADING openmp)
endif()

cmake_host_system_information(RESULT NUM_CORES QUERY NUMBER_OF_PHYSICAL_CORES)
string(REGEX MATCH "^[0-9]+" GCC_VERSION_MAJOR ${CMAKE_CXX_COMPILER_VERSION})

if(GCC_VERSION_MAJOR GREATER_EQUAL 12)
    set(CPP_STD 17)
    set(AVX512_BF16_EN 1)
else()
    set(CPP_STD 14)
    set(AVX512_BF16_EN 0)
endif()

# Set the C++ standard
set(CMAKE_CXX_STANDARD ${CPP_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#if AMDBLIS Package is not found
#First check if AMDBLIS_LOCAL_SOURCE is defined and use the source code for build/install
#and AMDBLIS_TAG if defined also, not used

#AMDBLIS_LOCAL_SOURCE is not defined then download frm Github repositiory with the defined tag

find_package(AMDBLIS)
if(NOT AMDBLIS_FOUND)

    if(AMDBLIS_LOCAL_SOURCE)
        message(STATUS "AMDBLIS not found, using ${AMDBLIS_LOCAL_SOURCE}")
        execute_process(
            COMMAND bash -c "mkdir -p ${ZENDNN_DIR}/third_party && cd ${ZENDNN_DIR}/third_party && rm -rf amdblis-local-src && cp -R ${AMDBLIS_LOCAL_SOURCE} amdblis-local-src"
            RESULT_VARIABLE EXEC_PROCESS_RESULT)
        if(NOT EXEC_PROCESS_RESULT EQUAL 0)
            message(FATAL_ERROR, "Error, cp ${AMDBLIS_LOCAL_SOURCE} failed to ${ZENDNN_DIR}/third_party/amdblis-local-src")
        else()
            message(STATUS, "INFO, cp ${AMDBLIS_LOCAL_SOURCE} successful")
        endif()
        set(AMDBLIS_LOCAL_OR_GIT_SOURCE_DIR ${ZENDNN_DIR}/third_party/amdblis-local-src)
    else()
        message(STATUS "AMDBLIS not found, downloading")
        include(cmake2025/download_amdblis.cmake)
        set(AMDBLIS_LOCAL_OR_GIT_SOURCE_DIR ${amdblis_SOURCE_DIR})
    endif()

    set(BLIS_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party/amdblis-build")

    execute_process(
        COMMAND bash -c "rm -rf ${BLIS_BINARY_DIR} && mkdir -p ${BLIS_BINARY_DIR} && export CXXFLAGS=\"-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}\" && cmake -B ${BLIS_BINARY_DIR} -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBLIS_CONFIG_FAMILY=${AMDBLIS_BLIS_CONFIG_FAMILY} -DENABLE_ADDON=${AMDBLIS_ENABLE_ADDON} -DENABLE_THREADING=${AMDBLIS_ENABLE_THREADING} -DENABLE_BLAS=${AMDBLIS_ENABLE_BLAS} -DENABLE_CBLAS=${AMDBLIS_ENABLE_CBLAS} -DCMAKE_INSTALL_PREFIX=$ENV{ZENDNN_BLIS_PATH} ."
        WORKING_DIRECTORY ${AMDBLIS_LOCAL_OR_GIT_SOURCE_DIR}
        RESULT_VARIABLE BASH_COMMAND_RESULT)

    if(NOT BASH_COMMAND_RESULT EQUAL 0)
        message(FATAL_ERROR "AMDBLIS BUILD, CONFIG failed")
    endif()

    execute_process(
        COMMAND bash -c "cmake --build ${BLIS_BINARY_DIR} --parallel=${NUM_CORES}"
        WORKING_DIRECTORY ${AMDBLIS_LOCAL_OR_GIT_SOURCE_DIR}
        RESULT_VARIABLE BASH_COMMAND_RESULT)

    if(NOT BASH_COMMAND_RESULT EQUAL 0)
        message(FATAL_ERROR "AMDBLIS BUILD, BUILD failed")
    endif()

    execute_process(
        COMMAND bash -c "cmake --install ${BLIS_BINARY_DIR}"
        WORKING_DIRECTORY ${AMDBLIS_LOCAL_OR_GIT_SOURCE_DIR}
        RESULT_VARIABLE BASH_COMMAND_RESULT)

    if(NOT BASH_COMMAND_RESULT EQUAL 0)
        message(FATAL_ERROR "AMDBLIS BUILD, INSTALL failed")
    endif()

    execute_process(
            COMMAND bash -c "mkdir -p $ENV{ZENDNN_BLIS_PATH}/include/LP64 && cp $ENV{ZENDNN_BLIS_PATH}/include/*.h $ENV{ZENDNN_BLIS_PATH}/include/LP64"
            WORKING_DIRECTORY ${ZENDNN_DIR}
            RESULT_VARIABLE BASH_COMMAND_RESULT)

    if(NOT BASH_COMMAND_RESULT EQUAL 0)
        message(FATAL_ERROR "AMDBLIS, POST INSTALL HEADERS-1 failed")
    endif()

    execute_process(
            COMMAND bash -c "cp $ENV{ZENDNN_BLIS_PATH}/include/*.hh $ENV{ZENDNN_BLIS_PATH}/include/LP64"
            WORKING_DIRECTORY ${ZENDNN_DIR}
            RESULT_VARIABLE BASH_COMMAND_RESULT)

    if(NOT BASH_COMMAND_RESULT EQUAL 0)
        message(FATAL_ERROR "AMDBLIS, POST INSTALL HEADERS-2 failed")
    endif()

    execute_process(
            COMMAND bash -c "mkdir -p $ENV{ZENDNN_BLIS_PATH}/lib/LP64 && cp $ENV{ZENDNN_BLIS_PATH}/lib/libblis* $ENV{ZENDNN_BLIS_PATH}/lib/LP64"
            WORKING_DIRECTORY ${ZENDNN_DIR}
            RESULT_VARIABLE BASH_COMMAND_RESULT)

    if(NOT BASH_COMMAND_RESULT EQUAL 0)
        message(FATAL_ERROR "AMDBLIS, POST INSTALL LIB failed")
    endif()

    find_package(AMDBLIS REQUIRED)
    message(STATUS "AMDBLIS, found with current build/install")
else()
    message(STATUS "AMDBLIS, found with pre-compiled binaries or previous build/install ")
endif()
message(STATUS "AMDBLIS, INCLUDES/LIBS,  ${AMDBLIS_INCLUDE_DIR}, ${AMDBLIS_LIB}")

#if FBGEMM_ENABLE is 1, then only check, build
if(FBGEMM_ENABLE EQUAL 1)
    find_package(FBGEMM)
    if(NOT FBGEMM_FOUND)
        message(STATUS "FBGEMM not found, downloading")
        include(cmake2025/download_fbgemm.cmake)
        set(CMAKE_C_FLAGS ${FBGEMM_CMAKE_C_FLAGS})   
        set(CMAKE_CXX_FLAGS ${FBGEMM_CMAKE_CXX_FLAGS})
        set(CMAKE_POSITION_INDEPENDENT_CODE ON)
        add_subdirectory(${ZENDNN_DIR}/third_party/fbgemm ${CMAKE_BINARY_DIR}/third_party/fbgemm)
        message(STATUS "FBGEMM, found with current build")
    else()
        message(STATUS "FBGEMM, found with pre-compiled binaries")
    endif()
else()
    message(STATUS "FBGEMM, not included")
endif()
message(STATUS "FBGEMM, INCLUDE_DIRS/LIBRARIES, ${FBGEMM_INCLUDE_DIRS}, ${FBGEMM_LIBRARIES}")

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Set Composable Kernel paths
# Check if DEPEND_ON_CK is set to 1
if(DEPEND_ON_CK EQUAL 1)
    set(CK_PATH ${ZENDNN_CK_PATH})

    # Set link flags
    set(CK_LINK_FLAGS "-L${CK_PATH}/build/lib -lck_cpu_instance -lhost_tensor")

    # Set defines
    add_definitions(-DCK_NOGPU -DENABLE_CK)

    # Set common flags
    set(CK_COMMON_FLAGS "-Wno-attributes -Wno-ignored-attributes -Wno-write-strings")

    # Set include directories
    include_directories(${CK_PATH} ${CK_PATH}/include ${CK_PATH}/library/include)
endif()

# Check if uProf is enabled
if(ZENDNN_ENABLE_UPROF)
    set(UPROF_PATH "${UPROF_INSTALL_PATH}")
    set(UPROF_INCLUDE_PATH "${UPROF_PATH}/include")
    set(UPROF_LIB_PATH "${UPROF_PATH}/lib/x64")
    add_definitions(-DUPROF_ENABLE=1)
    set(UPROF_LINK "AMDProfileController")
    set(CXX_UPROF_LINK "-Wl,--whole-archive ${UPROF_PATH}/lib/x64/libAMDProfileController.a -Wl,--no-whole-archive")
else()
    set(UPROF_PATH "")
    set(UPROF_INCLUDE_PATH "")
    set(UPROF_LIB_PATH "")
    add_definitions(-DUPROF_ENABLE=0)
    set(CXX_UPROF_LINK "")
endif()

if(ZENDNN_ENABLE_LIBXSMM)
  set(LIBXSMM_PATH "${ZENDNN_LIBXSMM_PATH}")
  set(LIBXSMM_INCLUDE_PATH "${LIBXSMM_PATH}/include")
  set(LIBXSMM_LIB_PATH "${LIBXSMM_PATH}/lib")
  set(LIBXSMM_LIBS dl xsmm xsmmnoblas xsmmext)
else()
  set(LIBXSMM_PATH "")
  set(LIBXSMM_INCLUDE_PATH "")
  set(LIBXSMM_LIB_PATH "")
  set(LIBXSMM_LIBS "")
endif()

#zendnn source files
file(GLOB SRCS
    "src/common/*.cpp"
    "src/cpu/*.cpp"
    "src/cpu/gemm/*.cpp"
    "src/cpu/gemm/f32/*.cpp"
    "src/cpu/gemm/s8x8s32/*.cpp"
    "src/cpu/matmul/*.cpp"
    "src/cpu/reorder/*.cpp"
    "src/cpu/rnn/*.cpp"
    "src/cpu/x64/*.cpp"
    "src/cpu/x64/brgemm/*.cpp"
    "src/cpu/x64/gemm/*.cpp"
    "src/cpu/x64/gemm/amx/*.cpp"
    "src/cpu/x64/gemm/bf16/*.cpp"
    "src/cpu/x64/gemm/f32/*.cpp"
    "src/cpu/x64/gemm/s8x8s32/*.cpp"
    "src/cpu/x64/injectors/*.cpp"
    "src/cpu/x64/lrn/*.cpp"
    "src/cpu/x64/matmul/*.cpp"
    "src/cpu/x64/prelu/*.cpp"
    "src/cpu/x64/rnn/*.cpp"
    "src/cpu/x64/shuffle/*.cpp"
    "src/cpu/x64/utils/*.cpp"
)

add_library(amdZenDNN ${SRCS})

set_target_properties(amdZenDNN PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(FBGEMM_ENABLE EQUAL 1)
    target_compile_definitions(amdZenDNN PUBLIC FBGEMM_ENABLE=1)
else()
    target_compile_definitions(amdZenDNN PUBLIC FBGEMM_ENABLE=0)
endif()

if(ZENDNN_TF_USE_CUSTOM_BLIS EQUAL 1)
    target_compile_definitions(amdZenDNN PUBLIC USE_CUSTOM_BLIS=1)
else()
    target_compile_definitions(amdZenDNN PUBLIC USE_CUSTOM_BLIS=0)
endif()

if(BLIS_API EQUAL 1)
    target_compile_definitions(amdZenDNN PUBLIC ZENDNN_USE_AOCL_BLIS_API=1)
endif()

# set LPGEMM to 0 if it's not already set (default)
if(NOT DEFINED LPGEMM)
  set(LPGEMM 0)
endif()

if(NOT DEFINED LPGEMM_V4_2)
  set(LPGEMM_V4_2 0)
endif()

if(LPGEMM_V5_0 EQUAL 1)
    target_compile_definitions(amdZenDNN PUBLIC ZENDNN_ENABLE_LPGEMM_V5_0=1)
elseif(LPGEMM_V4_2 EQUAL 1)
    target_compile_definitions(amdZenDNN PUBLIC ZENDNN_ENABLE_LPGEMM=1)
    target_compile_definitions(amdZenDNN PUBLIC ZENDNN_ENABLE_LPGEMM_V4_2=1)
elseif(LPGEMM EQUAL 1)
    target_compile_definitions(amdZenDNN PUBLIC ZENDNN_ENABLE_LPGEMM=1)
endif()

target_compile_definitions(amdZenDNN PUBLIC BIAS_ENABLED=1)
target_compile_definitions(amdZenDNN PUBLIC ZENDNN_ENABLE=1)
target_compile_definitions(amdZenDNN PUBLIC ZENDNN_X64=1)
target_compile_definitions(amdZenDNN PUBLIC AVX512_BF16_EN=${AVX512_BF16_EN})
target_compile_definitions(amdZenDNN PUBLIC _GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    target_compile_definitions(amdZenDNN PUBLIC NDEBUG)
endif()

if(AOCC EQUAL 0)
    if(GCC_VERSION_MAJOR GREATER_EQUAL 12)
        target_compile_options(amdZenDNN PUBLIC -march=znver2)
    else()
        target_compile_options(amdZenDNN PUBLIC -march=znver1)
    endif()
else()
    target_compile_options(amdZenDNN PUBLIC -march=znver2)
endif()
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_options(amdZenDNN PUBLIC -ggdb)
endif()

target_compile_options(amdZenDNN PUBLIC -fopenmp -fconcepts)
target_compile_options(amdZenDNN PUBLIC -Wno-format-zero-length -Wno-format-truncation)
target_compile_options(amdZenDNN PUBLIC -Wno-unused-result -Wno-stringop-overflow)
target_compile_options(amdZenDNN PUBLIC -Wno-format -Wno-narrowing)
target_compile_options(amdZenDNN PUBLIC -mssse3)

target_include_directories(amdZenDNN PUBLIC ${AMDBLIS_INCLUDE_DIR})
target_include_directories(amdZenDNN PUBLIC ${FBGEMM_INCLUDE_DIRS})
target_include_directories(amdZenDNN PUBLIC ${ZENDNN_DIR})
target_include_directories(amdZenDNN PUBLIC inc)
target_include_directories(amdZenDNN PUBLIC src)
target_include_directories(amdZenDNN PUBLIC src/common)
target_include_directories(amdZenDNN PUBLIC src/cpu)
target_include_directories(amdZenDNN PUBLIC src/tpp)

target_link_libraries(amdZenDNN PUBLIC Threads::Threads)
target_link_libraries(amdZenDNN PUBLIC OpenMP::OpenMP_CXX OpenMP::OpenMP_C)

set(dep_target "amdblis::amdblis")
if(TARGET ${dep_target})
   message(STATUS "TARGET ${dep_target}, exist")
   target_link_libraries(amdZenDNN PUBLIC ${dep_target})
else()
   message(STATUS "TARGET ${dep_target}, doesn't exist")
endif()

if(FBGEMM_ENABLE EQUAL 1)
    set(dep_target_list fbgemm fbgemm::fbgemm fbgemm::asmjit fbgemm::cpuinfo fbgemm::clog)
    foreach(dep_target IN LISTS dep_target_list)
        if(TARGET ${dep_target} EQUAL 0)
            message(STATUS "TARGET ${dep_target}, doesn't exist")
        else()
            message(STATUS "TARGET ${dep_target}, exist")
            target_link_libraries(amdZenDNN PUBLIC ${dep_target})
        endif()
    endforeach()
endif()

#get ZENDNN Version Hash
execute_process(COMMAND ${GIT_EXECUTABLE} -c log.showSignature=false log --no-abbrev-commit --oneline -1 --format=%H
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE ZENDNN_LIB_VERSION_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "ZENDNN, git log, failed")
endif()
set(ZENDNN_LIB_VERSION_HASH "${ZENDNN_LIB_VERSION_HASH}" CACHE STRING "ZENDNN_LIB_VERSION_HASH" FORCE)

add_subdirectory(tests)

cmake_print_variables(CMAKE_CXX_COMPILER_VERSION)
cmake_print_variables(CMAKE_CXX_FLAGS)
cmake_print_variables(CMAKE_MODULE_PATH)
cmake_print_variables(CMAKE_CXX_STANDARD)
cmake_print_variables(CMAKE_BUILD_TYPE)
cmake_print_variables(BUILD_SHARED_LIBS)

cmake_print_variables(_GLIBCXX_USE_CXX11_ABI)

cmake_print_variables(NUM_CORES)
cmake_print_variables(GCC_VERSION_MAJOR)
cmake_print_variables(AVX512_BF16_EN)

cmake_print_variables(FBGEMM_TAG)
cmake_print_variables(FBGEMM_CMAKE_C_FLAGS)
cmake_print_variables(FBGEMM_CMAKE_CXX_FLAGS)
cmake_print_variables(FBGEMM_ENABLE)
cmake_print_variables(FBGEMM_FOUND)
cmake_print_variables(FBGEMM_VERSION_HASH)
cmake_print_variables(FBGEMM_LIBRARIES)
cmake_print_variables(FBGEMM_INCLUDE_DIRS)

cmake_print_variables(AMDBLIS_TAG)
cmake_print_variables(AMDBLIS_ENABLE_BLAS)
cmake_print_variables(AMDBLIS_ENABLE_CBLAS)
cmake_print_variables(AMDBLIS_ENABLE_ADDON)
cmake_print_variables(AMDBLIS_BLIS_CONFIG_FAMILY)
cmake_print_variables(AMDBLIS_ENABLE_THREADING)
cmake_print_variables(AMDBLIS_LOCAL_SOURCE)
cmake_print_variables(AMDBLIS_FOUND)
cmake_print_variables(AMDBLIS_VERSION_HASH)
cmake_print_variables(AMDBLIS_LIB)
cmake_print_variables(AMDBLIS_INCLUDE_DIR)

cmake_print_variables(AOCC)
cmake_print_variables(BLIS_API)
cmake_print_variables(LPGEMM)
cmake_print_variables(LPGEMM_V4_2)
cmake_print_variables(LPGEMM_V5_0)
cmake_print_variables(ZENDNN_TF_USE_CUSTOM_BLIS)
cmake_print_variables(ZENDNN_LIB_VERSION_HASH)

cmake_print_properties(TARGETS amdZenDNN PROPERTIES INTERFACE_INCLUDE_DIRECTORIES)
cmake_print_properties(TARGETS amdblis::amdblis PROPERTIES INTERFACE_INCLUDE_DIRECTORIES)
cmake_print_properties(TARGETS fbgemm::fbgemm PROPERTIES INTERFACE_INCLUDE_DIRECTORIES)
cmake_print_properties(TARGETS fbgemm::asmjit PROPERTIES INTERFACE_INCLUDE_DIRECTORIES)
cmake_print_properties(TARGETS fbgemm::cpuinfo PROPERTIES INTERFACE_INCLUDE_DIRECTORIES)
cmake_print_properties(TARGETS fbgemm::clog amdZenDNN PROPERTIES INTERFACE_INCLUDE_DIRECTORIES)
