# *******************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

# set(ZENDNNL_SOURCE_DIR "<Top level ZenDNN folder>")

# !!!

IF(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL)
    IF(EXISTS ${PLUGIN_PARENT_DIR}/ZenDNNL)
        file(COPY ${PLUGIN_PARENT_DIR}/ZenDNNL DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
    ELSE()
        message( FATAL_ERROR "Copying of ZenDNNL library from local failed, CMake will exit." )
    ENDIF()
ENDIF()
# execute_process(COMMAND git pull WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL)

set(ZENDNNL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL/build")
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
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_BUILD_BENCHDNN:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DZENDNNL_CODE_COVERAGE:BOOL=OFF")
  list(APPEND ZNL_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  # !!!
  # build zendnnl as external project
  # The following external project setup assumes ZenDNNL is already
  # download to a local repo. comment this out if ZenDNNL is needed
  # to be downloaded.


  ExternalProject_ADD(fwk_zendnnl
    SOURCE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ZenDNNL"
    BINARY_DIR  "${ZENDNNL_BUILD_DIR}"
    INSTALL_DIR "${ZENDNNL_INSTALL_DIR}"
    CMAKE_ARGS  "${ZNL_CMAKE_ARGS}"
    INSTALL_COMMAND cmake --build . --target all -j)

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
  add_library(zendnnl_amdblis_deps STATIC IMPORTED)
  add_dependencies(zendnnl_amdblis_deps fwk_zendnnl)  
  set_target_properties(zendnnl_amdblis_deps
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AMDBLIS_LIB_DIR}/libblis-mt.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_AMDBLIS_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AMDBLIS_INC_DIR}")

  add_library(amdblis::amdblis_archive ALIAS zendnnl_amdblis_deps)

  list(APPEND ZENDNNL_LINK_LIBS "zendnnl_amdblis_deps")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_AMDBLIS_INC_DIR})

  # aocl utils
  set(ZENDNNL_AOCLUTILS_INC_DIR "${ZENDNNL_INSTALL_DIR}/deps/aoclutils/include")
  set(ZENDNNL_AOCLUTILS_LIB_DIR "${ZENDNNL_INSTALL_DIR}/deps/aoclutils/lib")

  file(MAKE_DIRECTORY ${ZENDNNL_AOCLUTILS_INC_DIR})
  add_library(zendnnl_aoclutils_deps STATIC IMPORTED)
  add_dependencies(zendnnl_aoclutils_deps fwk_zendnnl)
  set_target_properties(zendnnl_aoclutils_deps
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AOCLUTILS_LIB_DIR}/libaoclutils.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}")

  add_library(au::aoclutils ALIAS zendnnl_aoclutils_deps)
  list(APPEND ZENDNNL_LINK_LIBS "zendnnl_aoclutils_deps")


  add_library(zendnnl_aucpuid_deps STATIC IMPORTED)
  add_dependencies(zendnnl_aucpuid_deps fwk_zendnnl)
  set_target_properties(zendnnl_aucpuid_deps
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AOCLUTILS_LIB_DIR}/libau_cpuid.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}")

  add_library(au::au_cpuid ALIAS zendnnl_aucpuid_deps)
  list(APPEND ZENDNNL_LINK_LIBS "zendnnl_aucpuid_deps")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_AOCLUTILS_INC_DIR})

  # For Json
  set(ZENDNNL_JSON_INC_DIR "${ZENDNNL_INSTALL_DIR}/deps/json/include")

  file(MAKE_DIRECTORY ${ZENDNNL_JSON_INC_DIR})
  add_library(zendnnl_json_deps INTERFACE IMPORTED)
  set_target_properties(zendnnl_json_deps
  PROPERTIES INTERFACE_COMPILE_DEFINITIONS "\$<\$<NOT:\$<BOOL:ON>>:JSON_USE_GLOBAL_UDLS=0>;\$<\$<NOT:\$<BOOL:ON>>:JSON_USE_IMPLICIT_CONVERSIONS=0>;\$<\$<BOOL:OFF>:JSON_DISABLE_ENUM_SERIALIZATION=1>;\$<\$<BOOL:OFF>:JSON_DIAGNOSTICS=1>;\$<\$<BOOL:OFF>:JSON_DIAGNOSTIC_POSITIONS=1>;\$<\$<BOOL:OFF>:JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON=1>"
             INTERFACE_COMPILE_FEATURES "cxx_std_11"
             INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_JSON_INC_DIR}")

  add_library(nlohmann_json::nlohmann_json ALIAS zendnnl_json_deps)

  list(APPEND ZENDNNL_LINK_LIBS "zendnnl_json_deps")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_JSON_INC_DIR})

  # OneDNN
  set(ZENDNNL_ONEDNN_INC_DIR "${ZENDNNL_INSTALL_DIR}/deps/onednn/include")
  set(ZENDNNL_ONEDNN_LIB_DIR "${ZENDNNL_INSTALL_DIR}/deps/onednn/lib")

  file(MAKE_DIRECTORY ${ZENDNNL_ONEDNN_INC_DIR})
  add_library(zendnnl_onednn_deps STATIC IMPORTED)
  add_dependencies(zendnnl_onednn_deps fwk_zendnnl)
  set_target_properties(zendnnl_onednn_deps
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_ONEDNN_LIB_DIR}/libdnnl.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_ONEDNN_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_ONEDNN_INC_DIR}")

  add_library(DNNL::dnnl ALIAS zendnnl_onednn_deps)

  list(APPEND ZENDNNL_LINK_LIBS "zendnnl_onednn_deps")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_ONEDNN_INC_DIR})

  # zendnnl
  set(ZENDNNL_LIBRARY_INC_DIR "${ZENDNNL_INSTALL_DIR}/zendnnl/include")
  set(ZENDNNL_LIBRARY_LIB_DIR "${ZENDNNL_INSTALL_DIR}/zendnnl/lib")

  file(MAKE_DIRECTORY ${ZENDNNL_LIBRARY_INC_DIR})
  add_library(zendnnl_library STATIC IMPORTED)
  add_dependencies(zendnnl_library fwk_zendnnl)
  set_target_properties(zendnnl_library
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")

  # UNCOMMENT if amdblis is also to be included in interface libs
  # target_link_libraries(zendnnl_library
  #   INTERFACE ${CMAKE_DL_LIBS}
  #   INTERFACE OpenMP::OpenMP_CXX
  #   INTERFACE nlohmann_json::nlohmann_json
  #   INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::aoclutils>"
  #   INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,DNNL::dnnl>"
  #   INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,amdblis::amdblis_archive>")

  # amdblis is removeed temporarily as it is being supplied by old zendnn
  target_link_libraries(zendnnl_library
    INTERFACE ${CMAKE_DL_LIBS}
    INTERFACE OpenMP::OpenMP_CXX
    INTERFACE nlohmann_json::nlohmann_json
    INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::aoclutils>"
    INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,DNNL::dnnl>")
  target_link_options(zendnnl_library INTERFACE "-fopenmp")

  add_library(zendnnl::zendnnl_archive ALIAS zendnnl_library)

  list(APPEND ZENDNNL_LINK_LIBS "zendnnl_library")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_LIBRARY_INC_DIR})
  # !!!

endif()
