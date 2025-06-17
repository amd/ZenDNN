
# *******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
include_guard(GLOBAL)
include(ExternalProject)

# !!!
# ZENDNNL_SOURCE_DIR is commented out.
# point ZENDNNL_SOURCE_DIR to top level ZenDNN folder.This should be absolute path.
# UNCOMMENT to make the script work

set(ZENDNNL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/dependencies/ZenDNN")

# !!!

set(ZENDNNL_BUILD_DIR "${ZENDNNL_SOURCE_DIR}/build")
set(ZENDNNL_INSTALL_DIR "${ZENDNNL_BUILD_DIR}/install")

# find required packages
find_package(OpenMP REQUIRED)

# try to find pre-built package
set(zendnnl_ROOT "${ZENDNNL_INSTALL_DIR}/zendnnl")
set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
find_package(zendnnl QUIET)
if(zendnnl_FOUND)
  message(STATUS "zendnnl found at ${zendnnl_ROOT}")
else()
  message(STATUS "zendnnl not found... building as an external project")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_BUILD_EXAMPLES:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_BUILD_GTEST:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_BUILD_DOXYGEN:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  # !!!
  # build zendnnl as external project
  # The following external project setup assumes ZenDNNL is already
  # download to a local repo. comment this out if ZenDNNL is needed
  # to be downloaded.


  ExternalProject_ADD(fwk_zendnnl
    SOURCE_DIR  "${ZENDNNL_SOURCE_DIR}"
    BINARY_DIR  "${ZENDNNL_BUILD_DIR}"
    INSTALL_DIR "${ZENDNNL_INSTALL_DIR}"
    CMAKE_ARGS  "${ZNL_CMAKE_ARGS}"
    INSTALL_COMMAND cmake --build . --target all -j
    BUILD_BYPRODUCTS <INSTALL_DIR>/deps/amdblis/lib/libblis-mt.a
                     <INSTALL_DIR>/deps/aoclutils/lib/libaoclutils.a
                     <INSTALL_DIR>/deps/aoclutils/lib/libau_cpuid.a
                     <INSTALL_DIR>/zendnnl/lib/libzendnnl_archive.a )

  # !!!

  # !!!
  # build zendnnl as external project
  # uncomment and put appropriate git information if ZenDNNL is to
  # be downloaded.

  # ExternalProject_ADD(fwk_zendnnl
  #   SOURCE_DIR  "${ZENDNNL_SOURCE_DIR}"
  #   BINARY_DIR  "${ZENDNNL_BUILD_DIR}"
  #   INSTALL_DIR "${ZENDNNL_INSTALL_DIR}"
  #   GIT_REPOSITORY <ZenDNNL Git Repo>
  #   GIT_TAG        <ZenDNNL Git Tag>
  #   CMAKE_ARGS  "${ZNL_CMAKE_ARGS}"
  #   INSTALL_COMMAND cmake --build . --target install -j)

  # !!!


  # !!!
  # HACK cmake targets instead of depending on package information
  # This kind of hacking can be error-prone.

  # amd blis
  set(ZENDNNL_AMDBLIS_INC_DIR "${ZENDNNL_INSTALL_DIR}/deps/amdblis/include")
  set(ZENDNNL_AMDBLIS_LIB_DIR "${ZENDNNL_INSTALL_DIR}/deps/amdblis/lib")

  file(MAKE_DIRECTORY ${ZENDNNL_AMDBLIS_INC_DIR})
  add_library(zendnnl_amdblis_deps STATIC IMPORTED GLOBAL)
  add_dependencies(zendnnl_amdblis_deps fwk_zendnnl)
  set_target_properties(zendnnl_amdblis_deps
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AMDBLIS_LIB_DIR}/libblis-mt.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_AMDBLIS_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AMDBLIS_INC_DIR}")

  add_library(amdblis::amdblis_archive ALIAS zendnnl_amdblis_deps)

  list(APPEND ZENDNNL_LINK_LIBS "amdblis::amdblis_archive")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_AMDBLIS_INC_DIR})

  # aocl utils
  set(ZENDNNL_AOCLUTILS_INC_DIR "${ZENDNNL_INSTALL_DIR}/deps/aoclutils/include")
  set(ZENDNNL_AOCLUTILS_LIB_DIR "${ZENDNNL_INSTALL_DIR}/deps/aoclutils/lib")

  file(MAKE_DIRECTORY ${ZENDNNL_AOCLUTILS_INC_DIR})
  add_library(zendnnl_aoclutils_deps STATIC IMPORTED GLOBAL)
  add_dependencies(zendnnl_aoclutils_deps fwk_zendnnl)
  set_target_properties(zendnnl_aoclutils_deps
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AOCLUTILS_LIB_DIR}/libaoclutils.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}")

  add_library(au::aoclutils ALIAS zendnnl_aoclutils_deps)
  list(APPEND ZENDNNL_LINK_LIBS "au::aoclutils")


  add_library(zendnnl_aucpuid_deps STATIC IMPORTED GLOBAL)
  add_dependencies(zendnnl_aucpuid_deps fwk_zendnnl)
  set_target_properties(zendnnl_aucpuid_deps
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AOCLUTILS_LIB_DIR}/libau_cpuid.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}")

  add_library(au::au_cpuid ALIAS zendnnl_aucpuid_deps)
  list(APPEND ZENDNNL_LINK_LIBS "au::au_cpuid")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_AOCLUTILS_INC_DIR})

  # zendnnl
  set(ZENDNNL_LIBRARY_INC_DIR "${ZENDNNL_INSTALL_DIR}/zendnnl/include")
  set(ZENDNNL_LIBRARY_LIB_DIR "${ZENDNNL_INSTALL_DIR}/zendnnl/lib")

  file(MAKE_DIRECTORY ${ZENDNNL_LIBRARY_INC_DIR})
  add_library(zendnnl_library STATIC IMPORTED GLOBAL)
  add_dependencies(zendnnl_library fwk_zendnnl)
  set_target_properties(zendnnl_library
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")

  target_link_libraries(zendnnl_library
    INTERFACE ${CMAKE_DL_LIBS}
    INTERFACE OpenMP::OpenMP_CXX
    INTERFACE au::au_cpuid
    INTERFACE au::aoclutils
    INTERFACE amdblis::amdblis_archive)

  target_link_options(zendnnl_library INTERFACE "-fopenmp")

  add_library(zendnnl::zendnnl_archive ALIAS zendnnl_library)

  list(APPEND ZENDNNL_LINK_LIBS "zendnnl_library")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_LIBRARY_INC_DIR})

  # !!!

endif()
