#  *******************************************************************************
#  * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  *******************************************************************************
include_guard(GLOBAL)

include(ZenDnnlOptions)

find_package(OpenMP REQUIRED QUIET)

message(DEBUG "${ZENDNNL_MSG_PREFIX}Configuring ZENDNNL LIBRARY")

# project options
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_SOURCE_DIR=${ZENDNNL_SOURCE_DIR}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_PROJECT_VERSION=${ZENDNNL_PROJECT_VERSION}")

# dependencies
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_DEPENDS_AOCLUTILS=${ZENDNNL_DEPENDS_AOCLUTILS}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_DEPENDS_JSON=${ZENDNNL_DEPENDS_JSON}")

list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_DEPENDS_AOCLDLP=${ZENDNNL_DEPENDS_AOCLDLP}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_DEPENDS_AMDBLIS=${ZENDNNL_DEPENDS_AMDBLIS}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_DEPENDS_ONEDNN=${ZENDNNL_DEPENDS_ONEDNN}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_DEPENDS_LIBXSMM=${ZENDNNL_DEPENDS_LIBXSMM}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_DEPENDS_PARLOOPER=${ZENDNNL_DEPENDS_PARLOOPER}")

list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_AOCLDLP_INJECTED=${ZENDNNL_AOCLDLP_INJECTED}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_AMDBLIS_INJECTED=${ZENDNNL_AMDBLIS_INJECTED}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_ONEDNN_INJECTED=${ZENDNNL_ONEDNN_INJECTED}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_LIBXSMM_INJECTED=${ZENDNNL_LIBXSMM_INJECTED}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_PARLOOPER_INJECTED=${ZENDNNL_PARLOOPER_INJECTED}")

# library components
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_LIB_BUILD_ARCHIVE=${ZENDNNL_LIB_BUILD_ARCHIVE}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_LIB_BUILD_SHARED=${ZENDNNL_LIB_BUILD_SHARED}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_BUILD_GTEST=${ZENDNNL_BUILD_GTEST}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_BUILD_AI_GTESTS=${ZENDNNL_BUILD_AI_GTESTS}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_CODE_COVERAGE=${ZENDNNL_CODE_COVERAGE}")
list(APPEND ZL_CMAKE_ARGS "-DZENDNNL_BUILD_ASAN=${ZENDNNL_BUILD_ASAN}")

# cmake variables
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_MESSAGE_LOG_LEVEL}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")
list(APPEND ZL_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

message(DEBUG "${ZENDNNL_MSG_PREFIX}ZENDNNL_LIB_CMAKE_ARGS = ${ZL_CMAKE_ARGS}")

set(ZENDNNL_ROOT ${ZENDNNL_SOURCE_DIR}/zendnnl)
set(NPROC ${ZENDNNL_BUILD_SYS_NPROC})
ExternalProject_ADD(zendnnl
  DEPENDS "zendnnl-deps"
  SOURCE_DIR "${ZENDNNL_ROOT}"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/zendnnl"
  INSTALL_DIR "${CMAKE_INSTALL_PREFIX}"
  CMAKE_ARGS "${ZL_CMAKE_ARGS}"
  BUILD_COMMAND cmake --build . --target all -- -j${NPROC}
  INSTALL_COMMAND cmake --build .  --target install
  BUILD_BYPRODUCTS <INSTALL_DIR>/zendnnl/lib/libzendnnl_archive.a
  BUILD_ALWAYS TRUE
  CONFIGURE_HANDLED_BY_BUILD TRUE)

list(APPEND ZENDNNL_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/zendnnl")
list(APPEND ZENDNNL_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/zendnnl")
list(APPEND ZENDNNL_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/gtests")

set_target_properties(zendnnl
  PROPERTIES
  ADDITIONAL_CLEAN_FILES "${ZENDNNL_CLEAN_FILES}")

# !!!
# HACK to make zendnnl a sub-project using add_directory() !
# ZenDNNL packages the exported information of this dependency in its own
# package config file. However to use this information, ZenDNNL package
# need to be installed.
# when ZenDNNL is included as a sub-project using add_subdirectory it
# can not be installed before the super-project of which it is a sub-project
# is built. Thus super-project can not use its package information. This
# makes this hack of manually interfacing the libraries this dependency has
# built necessary.
# Since we do not know what kind of information and targets this package exports
# this kind of manual interface could be error-prone.
#
# UNCOMMENT the code below for manual interface.

# if(NOT ZENDNNL_STANDALONE_BUILD)
#   set(ZENDNNL_LIBRARY_INC_DIR "${CMAKE_INSTALL_PREFIX}/zendnnl/include")
#   set(ZENDNNL_LIBRARY_LIB_DIR "${CMAKE_INSTALL_PREFIX}/zendnnl/lib")
#   set(ZENDNNL_JSON_INC_DIR "${CMAKE_INSTALL_PREFIX}/deps/json/include")

#   file(MAKE_DIRECTORY ${ZENDNNL_LIBRARY_INC_DIR})
#   add_library(zendnnl_library STATIC IMPORTED GLOBAL)

#   add_dependencies(zendnnl_library
#     zendnnl zendnnl-deps)

#   set_target_properties(zendnnl_library
#     PROPERTIES
#     IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a"
#     INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
#     INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR};${ZENDNNL_INCLUDE_DIRECTORIES}")

#   target_link_options(zendnnl_library INTERFACE "-fopenmp")

#   target_link_libraries(zendnnl_library
#     INTERFACE ${CMAKE_DL_LIBS}
#     INTERFACE OpenMP::OpenMP_CXX
#     INTERFACE nlohmann_json::nlohmann_json)

#   target_link_libraries(zendnnl_library
#     INTERFACE au::aoclutils
#     INTERFACE au::au_cpuid)

#   if(ZENDNNL_DEPENDS_AMDBLIS)
#     if(NOT ZENDNNL_AMDBLIS_INJECTED)
#       target_link_libraries(zendnnl_library
#         INTERFACE amdblis::amdblis_archive)
#     endif()
#   endif()

#   if(ZENDNNL_DEPENDS_AOCLDLP)
#     if(NOT ZENDNNL_AOCLDLP_INJECTED)
#       target_link_libraries(zendnnl_library
#         INTERFACE aocldlp::aocl_dlp_static)
#     endif()
#   endif()

#   if(ZENDNNL_DEPENDS_ONEDNN)
#     if(NOT ZENDNNL_ONEDNN_INJECTED)
#       target_link_libraries(zendnnl_library
#         INTERFACE DNNL::dnnl)
#     endif()
#   endif()

#   add_library(zendnnl::zendnnl_archive ALIAS zendnnl_library)

#   list(APPEND ZENDNNL_LINK_LIBS "zendnnl_library")
#   list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_LIBRARY_INC_DIR})
# endif()
# !!!
