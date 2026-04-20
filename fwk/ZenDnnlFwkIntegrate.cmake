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
include_guard(GLOBAL)
include(ExternalProject)
include(ZenDnnlFwkMacros)

message(AUTHOR_WARNING "(ZENDNNL) please ensure all zendnnl variables are set properly.")

# find openmp
find_package(OpenMP REQUIRED QUIET)

# set zendnnl source dir, where zendnnl has been downloaded.
zendnnl_add_option(NAME ZENDNNL_SOURCE_DIR
  VALUE <zendnnl source dir>
  TYPE PATH
  CACHE_STRING "zendnnl_source_dir"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl binary dir, if unsure set ${CMAKE_CURRENT_BINARY_DIR}/zendnnl.
zendnnl_add_option(NAME ZENDNNL_BINARY_DIR
  VALUE <zendnnl binary dir>
  TYPE PATH
  CACHE_STRING "zendnnl_binary_dir"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl install dir, if unsure set ${CMAKE_INSTALL_PREFIX}/zendnnl.
zendnnl_add_option(NAME ZENDNNL_INSTALL_PREFIX
  VALUE <zendnnl install prefix>
  TYPE PATH
  CACHE_STRING "zendnnl_install_dir"
  COMMAND_LIST ZNL_CMAKE_ARGS)

## general zendnnl options
# set ZenDNNL framework build, this should on ON to avoid standalone build.
zendnnl_add_option(NAME ZENDNNL_FWK_BUILD
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl framework build"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl build option, default is Release.
zendnnl_add_option(NAME ZENDNNL_BUILD_TYPE
  VALUE "Release"
  TYPE STRING
  CACHE_STRING "zendnnl build type"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl log level.
zendnnl_add_option(NAME ZENDNNL_MESSAGE_LOG_LEVEL
  VALUE "DEBUG"
  TYPE STRING
  CACHE_STRING "zendnnl message log level"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl verbose makefile option.
zendnnl_add_option(NAME ZENDNNL_VERBOSE_MAKEFILE
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl verbose makefile"
  COMMAND_LIST ZNL_CMAKE_ARGS)

## components options
# set building zendnnl examples, default os OFF.
zendnnl_add_option(NAME ZENDNNL_BUILD_EXAMPLES
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl examples"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set building zendnnl gtests, default os OFF.
zendnnl_add_option(NAME ZENDNNL_BUILD_GTEST
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl gtests"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set building zendnnl doxygen documentation, default os OFF.
zendnnl_add_option(NAME ZENDNNL_BUILD_DOXYGEN
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl doxygen documentation"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set building zendnnl benchmarking tool, default os OFF.
zendnnl_add_option(NAME ZENDNNL_BUILD_BENCHDNN
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl benchdnn"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set zendnnl code coverage option, default os OFF.
zendnnl_add_option(NAME ZENDNNL_CODE_COVERAGE
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "build zendnnl code coverage"
  COMMAND_LIST ZNL_CMAKE_ARGS)

## dependencies
# set if zendnnl depends on amdblis. this should bf OFF only if
# aocldlp dependency is ON.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_AMDBLIS
  VALUE ON
  TYPE BOOL
  CACHE_STRING "zendnnl amdblis dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set if zendnnl depends on aocldlp. this should bf ON only if
# amdblis dependency is OFF.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_AOCLDLP
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "zendnnl aocldlp dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set if zendnnl depends on onednn, default is OFF.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_ONEDNN
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "zendnnl onednn dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set if zendnnl depends on libxsmm, default is OFF.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_LIBXSMM
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "zendnnl libxsmm dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set if zendnnl depends on parlooper default is OFF.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_PARLOOPER
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "zendnnl parlooper dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set if zendnnl depends on fbgemm, default is OFF.
zendnnl_add_option(NAME ZENDNNL_DEPENDS_FBGEMM
  VALUE OFF
  TYPE BOOL
  CACHE_STRING "zendnnl fbgemm dependency"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set path of amdblis if amdblis is injected. if the framework
# does not inject it, set it to "" (empty string).
zendnnl_add_option(NAME ZENDNNL_AMDBLIS_INJECT_DIR
  VALUE <amd blis install path>
  TYPE PATH
  CACHE_STRING "zendnnl amdblis injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set path of aocldlp if aocldlp is injected. if the framework
# does not inject it, set it to "" (empty string).
zendnnl_add_option(NAME ZENDNNL_AOCLDLP_INJECT_DIR
  VALUE <aocldlp install path>
  TYPE PATH
  CACHE_STRING "zendnnl aocldlp injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set path of onednn if onednn is injected. if the framework
# does not inject it, set it to "" (empty string).
zendnnl_add_option(NAME ZENDNNL_ONEDNN_INJECT_DIR
  VALUE <onednn install path>
  TYPE PATH
  CACHE_STRING "zendnnl onednn injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set path of libxsmm if libxsmm is injected. if the framework
# does not inject it, set it to "" (empty string).
zendnnl_add_option(NAME ZENDNNL_LIBXSMM_INJECT_DIR
  VALUE <libxsmm install path>
  TYPE PATH
  CACHE_STRING "zendnnl libxsmm injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set path of parlooper if parlooper is injected. if the framework
# does not inject it, set it to "" (empty string).
zendnnl_add_option(NAME ZENDNNL_PARLOOPER_INJECT_DIR
  VALUE <parlooper install path>
  TYPE PATH
  CACHE_STRING "zendnnl parlooper injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# set path of fbgemm if fbgemm is injected. if the framework
# does not inject it, set it to "" (empty string).
zendnnl_add_option(NAME ZENDNNL_FBGEMM_INJECT_DIR
  VALUE <fbgemm install path>
  TYPE PATH
  CACHE_STRING "zendnnl fbgemm injection path"
  COMMAND_LIST ZNL_CMAKE_ARGS)

# try to find pre-built package
set(zendnnl_ROOT "${ZENDNNL_INSTALL_PREFIX}/zendnnl")
set(zendnnl_DIR "${zendnnl_ROOT}/lib/cmake")
find_package(zendnnl QUIET)
if(zendnnl_FOUND)
  message(STATUS "(ZENDNNL) ZENDNNL FOUND AT ${zendnnl_ROOT}")
  message(STATUS "(ZENDNNL) if zendnnl options are changed from previous build,")
  message(STATUS "(ZENDNNL) they will not be reflected")
  message(STATUS "(ZENDNNL) If options are changed, please do a clean build.")
  if(TARGET zendnnl::zendnnl_archive)
    set_target_properties(zendnnl::zendnnl_archive
      PROPERTIES IMPORTED_GLOBAL ON)
  endif()
  if(TARGET zendnnl::zendnnl)
    set_target_properties(zendnnl::zendnnl
      PROPERTIES IMPORTED_GLOBAL ON)
  endif()
else()
  message(STATUS "(ZENDNNL) ZENDNNL NOT FOUND, will be built as an external project.")

  # declare zendnnl library
  set(ZENDNNL_LIBRARY_INC_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl/include")
  set(ZENDNNL_LIBRARY_LIB_DIR "${ZENDNNL_INSTALL_PREFIX}/zendnnl/lib")

  if(NOT EXISTS ${ZENDNNL_LIBRARY_INC_DIR})
    file(MAKE_DIRECTORY ${ZENDNNL_LIBRARY_INC_DIR})
  endif()

  if(ZENDNNL_LIB_BUILD_ARCHIVE)
    add_library(zendnnl_library STATIC IMPORTED GLOBAL)
    add_dependencies(zendnnl_library fwk_zendnnl)
    set_target_properties(zendnnl_library
      PROPERTIES
      IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a"
      INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")

    target_link_options(zendnnl_library INTERFACE "-fopenmp")
    target_link_libraries(zendnnl_library
      INTERFACE OpenMP::OpenMP_CXX
      INTERFACE ${CMAKE_DL_LIBS})

    add_library(zendnnl::zendnnl_archive ALIAS zendnnl_library)

    list(APPEND ZNL_BYPRODUCTS "${ZENDNNL_LIBRARY_LIB_DIR}/libzendnnl_archive.a")
  endif()

  if(ZENDNNL_LIB_BUILD_SHARED)
    add_library(zendnnl_shared_library SHARED IMPORTED GLOBAL)
    add_dependencies(zendnnl_shared_library fwk_zendnnl)
    set_target_properties(zendnnl_shared_library
      PROPERTIES
      IMPORTED_LOCATION "${ZENDNNL_LIBRARY_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}zendnnl${CMAKE_SHARED_LIBRARY_SUFFIX}"
      INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_LIBRARY_INC_DIR}")

    target_link_options(zendnnl_shared_library INTERFACE "-fopenmp")
    target_link_libraries(zendnnl_shared_library
      INTERFACE OpenMP::OpenMP_CXX
      INTERFACE ${CMAKE_DL_LIBS})

    add_library(zendnnl::zendnnl ALIAS zendnnl_shared_library)

    list(APPEND ZNL_BYPRODUCTS "${ZENDNNL_LIBRARY_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}zendnnl${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()

  if(NOT ZENDNNL_LIB_BUILD_ARCHIVE AND NOT ZENDNNL_LIB_BUILD_SHARED)
    message(FATAL_ERROR "At least one of ZENDNNL_LIB_BUILD_ARCHIVE or ZENDNNL_LIB_BUILD_SHARED must be ON")
  endif()

  # declare all dependencies

  # json dependency
  zendnnl_add_dependency(NAME json
    PATH "${ZENDNNL_INSTALL_PREFIX}/deps/json"
    ALIAS "nlohmann_json::nlohmann_json"
    INCLUDE_ONLY)

  if(ZENDNNL_LIB_BUILD_ARCHIVE)
    target_link_libraries(zendnnl_library INTERFACE nlohmann_json::nlohmann_json)
  endif()
  if(ZENDNNL_LIB_BUILD_SHARED)
    target_link_libraries(zendnnl_shared_library INTERFACE nlohmann_json::nlohmann_json)
  endif()

  if (DEFINED ENV{ZENDNNL_MANYLINUX_BUILD})

    # zendnnl_add_dependency(NAME onednn
    #   PATH "${ZENDNNL_INSTALL_PREFIX}/deps/onednn"
    #   LIB_SUFFIX lib64
    #   ARCHIVE_FILE "libdnnl.a"
    #   ALIAS "DNNL::dnnl")

    # target_link_libraries(zendnnl_library INTERFACE DNNL::dnnl)

    zendnnl_add_dependency(NAME aoclutils
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
      LIB_SUFFIX lib64
      ARCHIVE_FILE "libaoclutils.a"
      ALIAS "au::aoclutils")

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE au::aoclutils)
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE au::aoclutils)
    endif()

    zendnnl_add_dependency(NAME aucpuid
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
      LIB_SUFFIX lib64
      ARCHIVE_FILE "libau_cpuid.a"
      ALIAS "au::au_cpuid")

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE au::au_cpuid)
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE au::au_cpuid)
    endif()

    zendnnl_add_dependency(NAME onednn
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/onednn"
        LIB_SUFFIX lib64
        ARCHIVE_FILE "libdnnl.a"
        ALIAS "DNNL::dnnl")

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE DNNL::dnnl)
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE DNNL::dnnl)
    endif()

  else()
    # zendnnl_add_dependency(NAME onednn
    #   PATH "${ZENDNNL_INSTALL_PREFIX}/deps/onednn"
    #   ARCHIVE_FILE "libdnnl.a"
    #   ALIAS "DNNL::dnnl")

    # if(ZENDNNL_LIB_BUILD_ARCHIVE)
    #   target_link_libraries(zendnnl_library INTERFACE DNNL::dnnl)
    # endif()
    # if(ZENDNNL_LIB_BUILD_SHARED)
    #   target_link_libraries(zendnnl_shared_library INTERFACE DNNL::dnnl)
    # endif()

    zendnnl_add_dependency(NAME aoclutils
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
      ARCHIVE_FILE "libaoclutils.a"
      ALIAS "au::aoclutils")

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE au::aoclutils)
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE au::aoclutils)
    endif()

    zendnnl_add_dependency(NAME aucpuid
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aoclutils"
      ARCHIVE_FILE "libau_cpuid.a"
      ALIAS "au::au_cpuid")

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE au::au_cpuid)
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE au::au_cpuid)
    endif()

    zendnnl_add_dependency(NAME onednn
      PATH "${ZENDNNL_INSTALL_PREFIX}/deps/onednn"
      ARCHIVE_FILE "libdnnl.a"
      ALIAS "DNNL::dnnl")

    if(ZENDNNL_LIB_BUILD_ARCHIVE)
      target_link_libraries(zendnnl_library INTERFACE DNNL::dnnl)
    endif()
    if(ZENDNNL_LIB_BUILD_SHARED)
      target_link_libraries(zendnnl_shared_library INTERFACE DNNL::dnnl)
    endif()

  endif()

  # amdblis dependency
  if (ZENDNNL_DEPENDS_AMDBLIS)
      zendnnl_add_dependency(NAME amdblis
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/amdblis"
        ARCHIVE_FILE "libblis-mt.a"
        ALIAS "amdblis::amdblis_archive")

      if(ZENDNNL_LIB_BUILD_ARCHIVE)
        target_link_libraries(zendnnl_library INTERFACE amdblis::amdblis_archive)
      endif()
      if(ZENDNNL_LIB_BUILD_SHARED)
        target_link_libraries(zendnnl_shared_library INTERFACE amdblis::amdblis_archive)
      endif()
  endif()

  if (ZENDNNL_DEPENDS_AOCLDLP)
      zendnnl_add_dependency(NAME aocldlp
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/aocldlp"
        ARCHIVE_FILE "libaocl-dlp.a"
        ALIAS "aocldlp::aocl_dlp_static")

      if(ZENDNNL_LIB_BUILD_ARCHIVE)
        target_link_libraries(zendnnl_library INTERFACE aocldlp::aocl_dlp_static)
      endif()
      if(ZENDNNL_LIB_BUILD_SHARED)
        target_link_libraries(zendnnl_shared_library INTERFACE aocldlp::aocl_dlp_static)
      endif()
  endif()

  # libxsmm dependency
  if (ZENDNNL_DEPENDS_LIBXSMM)
      zendnnl_add_dependency(NAME libxsmm
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/libxsmm"
        ARCHIVE_FILE "libxsmm.a"
        ALIAS "libxsmm::libxsmm_archive")

      if(ZENDNNL_LIB_BUILD_ARCHIVE)
        target_link_libraries(zendnnl_library INTERFACE libxsmm::libxsmm_archive)
      endif()
      if(ZENDNNL_LIB_BUILD_SHARED)
        target_link_libraries(zendnnl_shared_library INTERFACE libxsmm::libxsmm_archive)
      endif()
  endif()

  # parlooper dependency
  if (ZENDNNL_DEPENDS_PARLOOPER)
      zendnnl_add_dependency(NAME parlooper
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/parlooper"
        ARCHIVE_FILE "libparlooper.a"
        ALIAS "parlooper::parlooper_archive")

      if(ZENDNNL_LIB_BUILD_ARCHIVE)
        target_link_libraries(zendnnl_library INTERFACE parlooper::parlooper_archive)
      endif()
      if(ZENDNNL_LIB_BUILD_SHARED)
        target_link_libraries(zendnnl_shared_library INTERFACE parlooper::parlooper_archive)
      endif()
  endif()

  # fbgemm dependency
  if (ZENDNNL_DEPENDS_FBGEMM)
      zendnnl_add_dependency(NAME fbgemm
        PATH "${ZENDNNL_INSTALL_PREFIX}/deps/fbgemm"
        ARCHIVE_FILE "libfbgemm.a"
        ALIAS "fbgemm::fbgemm_archive")

      if(ZENDNNL_LIB_BUILD_ARCHIVE)
        target_link_libraries(zendnnl_library INTERFACE fbgemm::fbgemm_archive)
      endif()
      if(ZENDNNL_LIB_BUILD_SHARED)
        target_link_libraries(zendnnl_shared_library INTERFACE fbgemm::fbgemm_archive)
      endif()
  endif()

  message(STATUS "(ZENDNNL) ZNL_BYPRODUCTS=${ZNL_BYPRODUCTS}")
  message(STATUS "(ZENDNNL) ZNL_CMAKE_ARGS=${ZNL_CMAKE_ARGS}")

  ExternalProject_ADD(fwk_zendnnl
    SOURCE_DIR  "${ZENDNNL_SOURCE_DIR}"
    BINARY_DIR  "${ZENDNNL_BINARY_DIR}"
    CMAKE_ARGS  "${ZNL_CMAKE_ARGS}"
    BUILD_COMMAND cmake --build . --target all -j
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${ZNL_BYPRODUCTS})

  list(APPEND ZENDNNL_CLEAN_FILES "${ZENDNNL_BINARY_DIR}")
  list(APPEND ZENDNNL_CLEAN_FILES "${ZENDNNL_INSTALL_PREFIX}")
  set_target_properties(fwk_zendnnl
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${ZENDNNL_CLEAN_FILES}")

  # framwork dependencies
  # add_dependencies(fwk_zendnnl <injected dependency targets>)
  get_target_property(FWK_ZENDNNL_DEPENDS fwk_zendnnl MANUALLY_ADDED_DEPENDENCIES)
  if(${FWK_ZENDNNL_DEPENDS} STREQUAL "FWK_ZENDNNL_DEPENDS-NOTFOUND")
    message(AUTHOR_WARNING "(ZENDNNL) please ensure fwk_zendnnl depends on injected dependencies targets")
  else()
    message(STATUS "fwk_zendnnl dependencies : ${FWK_ZENDNNL_DEPENDS}")
  endif()

  # make library and its dependencies depend on fwk_zendnnl
  if(ZENDNNL_LIB_BUILD_ARCHIVE)
    add_dependencies(zendnnl_library fwk_zendnnl)
  endif()
  if(ZENDNNL_LIB_BUILD_SHARED)
    add_dependencies(zendnnl_shared_library fwk_zendnnl)
  endif()
  add_dependencies(zendnnl_json_deps fwk_zendnnl)
  add_dependencies(zendnnl_aoclutils_deps fwk_zendnnl)
  add_dependencies(zendnnl_aucpuid_deps fwk_zendnnl)

  if(ZENDNNL_DEPENDS_AMDBLIS)
    add_dependencies(zendnnl_amdblis_deps fwk_zendnnl)
  endif()

  if(ZENDNNL_DEPENDS_AOCLDLP)
    add_dependencies(zendnnl_aocldlp_deps fwk_zendnnl)
  endif()

  if(ZENDNNL_DEPENDS_ONEDNN)
    add_dependencies(zendnnl_onednn_deps fwk_zendnnl)
  endif()

  if(ZENDNNL_DEPENDS_LIBXSMM)
    add_dependencies(zendnnl_libxsmm_deps fwk_zendnnl)
  endif()

  if(ZENDNNL_DEPENDS_PARLOOPER)
    add_dependencies(zendnnl_parlooper_deps fwk_zendnnl)
  endif()

  if(ZENDNNL_DEPENDS_FBGEMM)
    add_dependencies(zendnnl_fbgemm_deps fwk_zendnnl)
  endif()

endif()
