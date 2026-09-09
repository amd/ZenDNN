# *******************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

cmake_minimum_required(VERSION 3.26)

# project information
project(ZENDNNL_EXAMPLES)
#set(ZENDNNL_EXAMPLES_VERSION "${PROJECT_VERSION}")
set(ZENDNNL_EXAMPLES_VERSION "1.0.0")

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# cmake verbose mode
set(CMAKE_VERBOSE_MAKEFILE ON)

# cmake module path
set(CMAKE_MODULE_PATH "${ZENDNNL_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")

# display cmake variables
message(DEBUG "EXAMPLES CMAKE_MODULE_PATH = ${CMAKE_MODULE_PATH}")
message(DEBUG "EXAMPLES CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
if(MSVC)
  # cl does not accept the GCC/Clang -Wall -Werror spellings.
  set(CMAKE_CXX_FLAGS "/W3 ${CMAKE_CXX_FLAGS}")
  # ZenDNN's installed public headers contain AVX-512 code and use
  # std::min/std::max and std::numeric_limits<>::max(); a consumer on MSVC must
  # therefore compile them with the same baseline the library uses: NOMINMAX
  # (so <windows.h>'s min/max macros do not clobber the standard ones),
  # _USE_MATH_DEFINES (M_* constants), /arch:AVX512, /bigobj and /EHsc.
  add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
  add_compile_options(/EHsc /arch:AVX512 /bigobj /wd4141)
else()
  set(CMAKE_CXX_FLAGS "-Wall -Werror ${CMAKE_CXX_FLAGS}")
endif()

# find common packages
# On MSVC, request the LLVM OpenMP runtime (/openmp:llvm, OpenMP 3.1) so the
# examples match the zendnnl library and AOCL-DLP (which are built /openmp:llvm)
# and resolve OpenMP 3.x symbols. Requires CMake >= 3.30. No effect elsewhere.
if(MSVC)
  if(CMAKE_VERSION VERSION_LESS "3.30")
    message(FATAL_ERROR
      "ZenDNN MSVC builds require CMake >= 3.30: OpenMP_RUNTIME_MSVC (which "
      "selects the LLVM OpenMP runtime /openmp:llvm) is only honored by CMake "
      ">= 3.30. On older CMake it is silently ignored and MSVC falls back to "
      "/openmp (vcomp), which breaks OpenMP 3.x linking and mixes two OpenMP "
      "runtimes with the zendnnl library and AOCL-DLP. Use CMake >= 3.30.")
  endif()
  set(OpenMP_RUNTIME_MSVC "llvm")
endif()
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# find zendnnl
cmake_path(GET CMAKE_INSTALL_PREFIX PARENT_PATH ZENDNNL_INSTALL_PREFIX)
set(ZENDNNL_INSTALL_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl")
set(zendnnl_ROOT "${ZENDNNL_INSTALL_DIR}")
set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
find_package(zendnnl REQUIRED)
if(zendnnl_FOUND)
  message(STATUS "zendnnl forund at ${zendnnl_ROOT}")
endif()

 set(examples_sources
   examples.cpp
   example_utils.cpp
   tensor_example.cpp
   sample_example.cpp
   matmul_example.cpp
   lowoha_matmul_example.cpp
   lowoha_group_matmul_example.cpp
   lowoha_conv2d_example.cpp
   lowoha_reorder_example.cpp
   lowoha_softmax_example.cpp
   lowoha_pooling_example.cpp
   lowoha_normalization_example.cpp
   batchmatmul_example.cpp
   compare_op_example.cpp
   reorder_example.cpp
   embedding_bag_example.cpp
   sdpa_example.cpp)

 set(example_headers
   example_utils.hpp
   tensor_example.hpp
   sample_example.hpp
   matmul_example.hpp
   lowoha_matmul_example.hpp
   lowoha_group_matmul_example.hpp
   lowoha_conv2d_example.hpp
   lowoha_reorder_example.hpp
   lowoha_softmax_example.hpp
   lowoha_pooling_example.hpp
   lowoha_normalization_example.hpp
   batchmatmul_example.hpp
   compare_op_example.hpp
   reorder_example.hpp
   embedding_bag_example.hpp
   sdpa_example.hpp)

add_executable(examples ${examples_sources})

# The tensor factory is an in-tree support helper for examples/benchdnn, not an
# installed public ZenDNN package target.
add_subdirectory("${ZENDNNL_SOURCE_DIR}/zendnnl/tensor_helper" tensor_helper)
target_link_libraries(examples
  PUBLIC OpenMP::OpenMP_CXX
  PRIVATE tensor_helper)
# target_link_libraries(examples
#   OpenMP::OpenMP_CXX
#   zendnnl::zendnnl_archive)

# Link to whichever zendnnl library variant was built
if(TARGET zendnnl::zendnnl_archive)
  # Prefer static library if available
  target_link_libraries(examples
    PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,zendnnl::zendnnl_archive>")
  message(STATUS "Examples linking to static library (zendnnl_archive)")
elseif(TARGET zendnnl::zendnnl)
  # Fall back to shared library
  target_link_libraries(examples
    PRIVATE zendnnl::zendnnl)
  message(STATUS "Examples linking to shared library (zendnnl)")
  # The installed examples binary (install/examples/bin/examples) must be able
  # to find libzendnnl.so (installed to install/zendnnl/lib) at runtime. Embed
  # an $ORIGIN-relative RUNPATH so it works without LD_LIBRARY_PATH.
  set_target_properties(examples PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../zendnnl/lib")
else()
  message(FATAL_ERROR "No zendnnl library target found! Build with at least one of ZENDNNL_LIB_BUILD_ARCHIVE=ON or ZENDNNL_LIB_BUILD_SHARED=ON")
endif()

if(ZENDNNL_CODE_COVERAGE)
  message(STATUS "code-coverage enabled for zendnnl examples")
  target_compile_options(examples PRIVATE --coverage -O0 -g)
  target_link_options(examples PRIVATE --coverage)
endif()

if(ZENDNNL_BUILD_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "ASAN enabled for zendnnl examples")
  set(SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer -O1)
  target_compile_options(examples PRIVATE ${SANITIZER_FLAGS})
  target_link_options(examples PRIVATE "-fsanitize=address")
endif()
install(TARGETS examples)
