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

if(ZENDNNL_DEPENDS_AOCLUTILS)

  message(DEBUG "${ZENDNNL_MSG_PREFIX}Configurig AOCL-UTILS...")

  list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_DOCS=OFF")
  list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_TESTS=OFF")
  list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_STATIC_LIBS=ON")
  list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_SHARED_LIBS=ON")
  list(APPEND AU_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
  list(APPEND AU_CMAKE_ARGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
  list(APPEND AU_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  if(WIN32)
    # Windows/MSVC: no -lpthread (a Linux-ism cl rejects); disable examples;
    # silence CRT deprecation warnings; forward the build type so the sub-build
    # does not silently default to Debug (/MTd) and mismatch the Release main
    # build's C runtime.
    #
    # NOTE: this aocl-utils source accesses the Windows SDK processor-info union
    # members directly (anonymous union, e.g. `.Processor`), so it must NOT be
    # built with /DNONAMELESSUNION (that names the union and breaks the direct
    # access with C2039). Older aocl-utils (used `pInfo->u.Processor`) required
    # the flag; this version fixed the access, so the flag is intentionally
    # absent. clang-tidy auto-enable is disabled in the aocl-utils source
    # (its CMakeLists.txt), matching the approach that avoids tidy erroring on
    # the cl build.
    list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_EXAMPLES=OFF")

    # Build aocl-utils static-only on Windows: this source's Cpuid CMake gives
    # its shared and static targets the same OUTPUT_NAME (aoclutils_c), so both
    # emit aoclutils_c.lib -> ninja "multiple rules generate" collision. Disable
    # via BUILD_SHARED_LIBS (au_options.cmake force-syncs AU_BUILD_SHARED_LIBS
    # from it). ZenDNNL only whole-archives the static aocl-utils anyway.
    list(APPEND AU_CMAKE_ARGS "-DBUILD_SHARED_LIBS=OFF")

    list(APPEND AU_CMAKE_ARGS "-DCMAKE_CXX_FLAGS=/D_CRT_SECURE_NO_WARNINGS")
    list(APPEND AU_CMAKE_ARGS "-DCMAKE_C_FLAGS=/D_CRT_SECURE_NO_WARNINGS")
    list(APPEND AU_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
    # Force aocl-utils' STATIC libraries onto the dynamic CRT (/MD) so they
    # match ZenDNNL's Windows build (which links /MD). Without this the static
    # libs default to the static CRT (/MT) and folding them into the /MD
    # zendnnl archive/DLL triggers CRT-mismatch link errors (LNK2038/LNK4098).
    # Requires an aocl-utils that provides this option (the CRT-MD-capable
    # source); ignored with a harmless warning on sources that lack it.
    list(APPEND AU_CMAKE_ARGS "-DAU_STATIC_FORCE_CRT_MD=ON")
  else()
    # adding pthread to cxx flags is a manylinux docker requirement.
    list(APPEND AU_CMAKE_ARGS "-DAU_BUILD_EXAMPLES=ON")
    list(APPEND AU_CMAKE_ARGS "-DCMAKE_CXX_FLAGS=-lpthread")
    # The C examples need it too: aocl-utils does not link Threads on those
    # targets, and thread_pinning_example_linux.c calls pthread_join as of
    # 5.3.2, so without this the C example fails to link (DSO missing from
    # command line) even though the CXX example above builds.
    list(APPEND AU_CMAKE_ARGS "-DCMAKE_C_FLAGS=-lpthread")
  endif()

  # Windows produces .lib (with build-configuration-dependent names/paths), so
  # do not pin the .a byproducts there; keep them on Unix for Ninja tracking.
  if(WIN32)
    set(AU_BYPRODUCTS "")
  else()
    set(AU_BYPRODUCTS
      <INSTALL_DIR>/lib/libaoclutils.a
      <INSTALL_DIR>/lib/libau_cpuid.a)
  endif()

  message(DEBUG "${ZENDNNL_MSG_PREFIX}AU_CMAKE_ARGS=${AU_CMAKE_ARGS}")

  set(NPROC ${ZENDNNL_BUILD_SYS_NPROC})
  if (ZENDNNL_LOCAL_AOCLUTILS)

    message(DEBUG "${ZENDNNL_MSG_PREFIX}Will use local AOCL-UTILS from ${AOCLUTILS_ROOT_DIR}")

    ExternalProject_ADD(zendnnl-deps-aoclutils
      SOURCE_DIR "${AOCLUTILS_ROOT_DIR}"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/aoclutils"
      INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/aoclutils"
      CMAKE_ARGS ${AU_CMAKE_ARGS}
      BUILD_COMMAND cmake --build . --config Release --target all -- -j${NPROC}
      INSTALL_COMMAND cmake --build . --config Release --target install
      BUILD_BYPRODUCTS ${AU_BYPRODUCTS})
  else()

    message(DEBUG "${ZENDNNL_MSG_PREFIX}Will download AOCL-UTILS with tag ${AOCLUTILS_GIT_TAG}")

    ExternalProject_ADD(zendnnl-deps-aoclutils
      SOURCE_DIR "${AOCLUTILS_ROOT_DIR}"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/aoclutils"
      INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/aoclutils"
      GIT_REPOSITORY ${AOCLUTILS_GIT_REPO}
      GIT_TAG ${AOCLUTILS_GIT_TAG}
      GIT_PROGRESS ${AOCLUTILS_GIT_PROGRESS}
      CMAKE_ARGS ${AU_CMAKE_ARGS}
      BUILD_COMMAND cmake --build . --config Release --target all -- -j${NPROC}
      INSTALL_COMMAND cmake --build . --config Release --target install
      BUILD_BYPRODUCTS ${AU_BYPRODUCTS}
      UPDATE_DISCONNECTED TRUE)
  endif()

  list(APPEND AOCLUTILS_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/aoclutils")
  list(APPEND AOCLUTILS_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/aoclutils")

  set_target_properties(zendnnl-deps-aoclutils
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${AOCLUTILS_CLEAN_FILES}")

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-aoclutils")

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

  # set(ZENDNNL_AOCLUTILS_INC_DIR "${CMAKE_INSTALL_PREFIX}/deps/aoclutils/include")
  # set(ZENDNNL_AOCLUTILS_LIB_DIR "${CMAKE_INSTALL_PREFIX}/deps/aoclutils/lib")

  # if(NOT EXISTS ${ZENDNNL_AOCLUTILS_INC_DIR})
  #   file(MAKE_DIRECTORY ${ZENDNNL_AOCLUTILS_INC_DIR})
  # endif()

  # add_library(zendnnl_aoclutils_deps STATIC IMPORTED GLOBAL)
  # add_dependencies(zendnnl_aoclutils_deps zendnnl-deps-aoclutils)
  # set_target_properties(zendnnl_aoclutils_deps
  #   PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AOCLUTILS_LIB_DIR}/libaoclutils.a"
  #              INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}"
  #              INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}")

  # add_library(au::aoclutils ALIAS zendnnl_aoclutils_deps)
  # list(APPEND ZENDNNL_LINK_LIBS "au::aoclutils")

  # add_library(zendnnl_aucpuid_deps STATIC IMPORTED GLOBAL)
  # add_dependencies(zendnnl_aucpuid_deps zendnnl-deps-aoclutils)
  # set_target_properties(zendnnl_aucpuid_deps
  #   PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AOCLUTILS_LIB_DIR}/libau_cpuid.a"
  #              INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}"
  #              INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AOCLUTILS_INC_DIR}")

  # add_library(au::au_cpuid ALIAS zendnnl_aucpuid_deps)
  # list(APPEND ZENDNNL_LINK_LIBS "au::au_cpuid")
  # list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_AOCLUTILS_INC_DIR})

  # !!!

else()
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Building AOCL-UTILS will be skipped.")
endif()


