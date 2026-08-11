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
project(ZENDNNL_LIBRARY VERSION ${ZENDNNL_PROJECT_VERSION})

# library version
set(ZENDNNL_LIBRARY_VERSION ${PROJECT_VERSION})
set(ZENDNNL_LIBRARY_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(ZENDNNL_LIBRARY_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(ZENDNNL_LIBRARY_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# specify c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# library components
set(ZENDNNL_LIB_BUILD_ARCHIVE ON CACHE BOOL "build zendnnl archive")
set(ZENDNNL_LIB_BUILD_SHARED OFF CACHE BOOL "build zendnnl shared lib")

# build integration mode
set(ZENDNNL_FWK_BUILD OFF CACHE BOOL "framework build mode")

# library dependencies
set(ZENDNNL_AOCLDLP_INJECTED OFF CACHE BOOL "aocldlp injected")
set(ZENDNNL_ONEDNN_INJECTED  OFF CACHE BOOL "onednn injected")
set(ZENDNNL_LIBXSMM_INJECTED  OFF CACHE BOOL "libxsmm injected")
set(ZENDNNL_PARLOOPER_INJECTED  OFF CACHE BOOL "parlooper injected")
set(ZENDNNL_FBGEMM_INJECTED  OFF CACHE BOOL "fbgemm injected")

# cmake variables
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "verbose makefile")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_MESSAGE_LOG_LEVEL "DEBUG")
set(CMAKE_MODULE_PATH "${ZENDNNL_SOURCE_DIR}/cmake")
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")

# informative messages
message(STATUS
  "${ZENDNNL_MSG_PREFIX} ZENDNNL_LIBRARY_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
message(STATUS "ZENDNNL_LIBRARY_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  message(STATUS "building zendnnl library archive")
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  message(STATUS "building zendnnl library shared lib")
endif()

# include zendnnl files
include(CheckCXXSourceCompiles)
include(ZenDnnlMacros)
include(ZenDnnlDependenciesInfo)

# global cxx flags (split per compiler). Keeping the GCC/Clang-only flags out
# of the MSVC build also keeps them out of CMake's OpenMP probe compile test,
# which otherwise fails when GCC-style flags are fed to cl.
if(MSVC)
  # Whole-translation-unit AVX-512 (Zen 4+ is the baseline; MSVC has no
  # per-function ISA targeting), large object files for the heavily-templated
  # TUs, and suppress C4141 ('inline' used more than once). The Windows
  # source-portability defines: NOMINMAX (suppress the min/max macros) and
  # _USE_MATH_DEFINES (expose M_PI and friends that MSVC otherwise withholds).
  # /FI force-includes the portability shim into every TU so its MSVC fixups are
  # active before any code — in particular the `#define collapse(x)` that
  # neutralizes the OpenMP `collapse` clause MSVC's libomp crashes on. An
  # absolute path is used so it also resolves in CMake's find_package(OpenMP) /
  # check_cxx_source_compiles probe TUs, which do not carry the src include dir.
  # The /FI path is double-quoted so it survives spaces in the source tree --
  # MSVC otherwise treats /FI<path with spaces> as multiple arguments.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512 /bigobj /wd4141 /DNOMINMAX /D_USE_MATH_DEFINES /FI\"${CMAKE_CURRENT_SOURCE_DIR}/src/common/zendnnl_compat.hpp\"")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mavx2 -mavx512f -mavx512vl -mavx512bf16")
  endif()
endif()

check_cxx_source_compiles("
  #include <immintrin.h>
  __attribute__((target(\"avx512f,avx512vl,avx512bw,avx512fp16\")))
  void test(const void *src, void *dst, __mmask32 k, __m512h v) {
    (void)_mm512_maskz_loadu_ph(k, src);
    _mm512_mask_storeu_ph(dst, k, v);
  }
  int main() { return 0; }
" ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS)

if(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS)
  add_compile_definitions(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS)
endif()

# find common packages
zendnnl_required_packages()

# find required dependencies
find_build_dependencies(${CMAKE_INSTALL_PREFIX})

# build library
add_subdirectory(src)

# build gtests
add_subdirectory(gtests)

# create version file
set(ZENDNNL_VERSION_FILE "${CMAKE_BINARY_DIR}/zendnnl_version.hpp")
create_version_header_file(${ZENDNNL_VERSION_FILE}
  ${ZENDNNL_LIBRARY_VERSION_MAJOR}
  ${ZENDNNL_LIBRARY_VERSION_MINOR}
  ${ZENDNNL_LIBRARY_VERSION_PATCH})

# install config file
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/config/zendnnl_user_config.json"
  DESTINATION zendnnl/config)
# install version file
install(FILES "${ZENDNNL_VERSION_FILE}"
  DESTINATION zendnnl/include)
