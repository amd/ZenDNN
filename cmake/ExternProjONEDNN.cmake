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
if (ZENDNNL_DEPENDS_ONEDNN)
  # check if zendnnl needs to build this dependency
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Configurig ONEDNN...")

  if (NOT ZENDNNL_ONEDNN_INJECTED)

    get_property(ONEDNN_INSTALL_DIR GLOBAL PROPERTY ONEDNNROOT)

    list(APPEND ONEDNN_CMAKE_ARGS "-DONEDNN_BUILD_GRAPH=OFF")
    list(APPEND ONEDNN_CMAKE_ARGS "-DONEDNN_BUILD_TESTS=OFF")
    list(APPEND ONEDNN_CMAKE_ARGS "-DONEDNN_BUILD_EXAMPLES=OFF")
    list(APPEND ONEDNN_CMAKE_ARGS "-DONEDNN_LIBRARY_TYPE=STATIC")
    list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_VERBOSE_MAKEFILE=OFF")
    list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=Release")
    list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")
    # Not propagating parent CMAKE_C/CXX_FLAGS to oneDNN: oneDNN is a mature standalone
    # library that manages its own compilation flags; avoids flag conflicts with TensorFlow
    # builds (e.g. TensorFlow's -D_FORTIFY_SOURCE=1 vs oneDNN's -D_FORTIFY_SOURCE=2).
    # list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}")
    # list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -DXbyak=XbyakoneDNN")
    list(APPEND ONEDNN_CMAKE_ARGS "-DCMAKE_CXX_FLAGS=-DXbyak=XbyakoneDNN")

    message(DEBUG "${ZENDNNL_MSG_PREFIX}ONEDNN_CMAKE_ARGS=${ONEDNN_CMAKE_ARGS}")

    set(NPROC ${ZENDNNL_BUILD_SYS_NPROC})
    # MSBuild rejects "-j", has no "all" target and builds Debug without --config.
    if(CMAKE_GENERATOR MATCHES "^Visual Studio")
      set(ONEDNN_BUILD_COMMAND cmake --build . --config Release --parallel ${NPROC})
      set(ONEDNN_INSTALL_COMMAND cmake --build . --config Release --target install)
    else()
      set(ONEDNN_BUILD_COMMAND cmake --build . --target all -- -j${NPROC})
      set(ONEDNN_INSTALL_COMMAND cmake --build . --target install)
    endif()
    if (ZENDNNL_LOCAL_ONEDNN)

      message(WARNING "${ZENDNNL_MSG_PREFIX}ONEDNN will be built from local source at ${ONEDNN_ROOT_DIR}. This version may not be fully compatible with ZenDNN. If unsure, it is recommended to use the standard ZenDNN build.")
      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will use local ONEDNN from ${ONEDNN_ROOT_DIR}")

      set(ONEDNN_DEPS_LINK "${ZENDNNL_DEPS_DIR}/onednn")
      if(NOT "${ONEDNN_ROOT_DIR}" STREQUAL "${ONEDNN_DEPS_LINK}")
        if(EXISTS "${ONEDNN_DEPS_LINK}")
          if(IS_SYMLINK "${ONEDNN_DEPS_LINK}")
            file(REMOVE "${ONEDNN_DEPS_LINK}")
          else()
            message(WARNING "${ZENDNNL_MSG_PREFIX}${ONEDNN_DEPS_LINK} exists and is not a symlink. Skipping symlink creation. Remove it manually if needed.")
          endif()
        endif()
        if(NOT EXISTS "${ONEDNN_DEPS_LINK}")
          execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${ONEDNN_ROOT_DIR}" "${ONEDNN_DEPS_LINK}")
          message(STATUS "${ZENDNNL_MSG_PREFIX}Created symlink ${ONEDNN_DEPS_LINK} -> ${ONEDNN_ROOT_DIR}")
        endif()
      endif()

      ExternalProject_ADD(zendnnl-deps-onednn
        SOURCE_DIR "${ONEDNN_ROOT_DIR}"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn"
        INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/onednn"
        CMAKE_ARGS ${ONEDNN_CMAKE_ARGS}
        BUILD_COMMAND ${ONEDNN_BUILD_COMMAND}
        INSTALL_COMMAND ${ONEDNN_INSTALL_COMMAND})
    else()

      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will download ONEDNN with tag ${ONEDNN_GIT_TAG}")

      ExternalProject_ADD(zendnnl-deps-onednn
        SOURCE_DIR "${ONEDNN_ROOT_DIR}"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/onednn"
        INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/onednn"
        GIT_REPOSITORY ${ONEDNN_GIT_REPO}
        GIT_TAG ${ONEDNN_GIT_TAG}
        GIT_PROGRESS ${ONEDNN_GIT_PROGRESS}
        CMAKE_ARGS ${ONEDNN_CMAKE_ARGS}
        BUILD_COMMAND ${ONEDNN_BUILD_COMMAND}
        INSTALL_COMMAND ${ONEDNN_INSTALL_COMMAND}
        UPDATE_DISCONNECTED TRUE)
    endif()

    list(APPEND ONEDNN_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/onednn")
    list(APPEND ONEDNN_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/onednn")

    set_target_properties(zendnnl-deps-onednn
      PROPERTIES
      ADDITIONAL_CLEAN_FILES "${ONEDNN_CLEAN_FILES}")

    # Windows: oneDNN's static dnnl.lib ships a compiled version resource
    # (version.rc.res). ZenDNN links dnnl.lib with WHOLE_ARCHIVE (see
    # zendnnl/src/CMakeLists.txt), which force-includes that resource; MSVC's
    # manifest-embed re-link then reports it as specified twice and fails with
    # LNK1241 when linking zendnnl.dll / gtests.exe. Strip the (cosmetic)
    # resource from the installed dnnl.lib right after oneDNN installs. Linux
    # (libdnnl.a; no resources, no manifest-embed step) is unaffected, so this
    # step is Windows-only.
    if(WIN32)
      get_filename_component(_zendnnl_msvc_bindir "${CMAKE_CXX_COMPILER}" DIRECTORY)
      ExternalProject_Add_Step(zendnnl-deps-onednn strip_version_res
        COMMAND ${CMAKE_COMMAND}
                "-DLIB_EXE=${_zendnnl_msvc_bindir}/lib.exe"
                "-DTARGET_LIB=${CMAKE_INSTALL_PREFIX}/deps/onednn/lib/dnnl.lib"
                -P "${CMAKE_CURRENT_LIST_DIR}/StripDnnlVersionRes.cmake"
        DEPENDEES install
        COMMENT "Windows: strip oneDNN version.rc.res from dnnl.lib (avoids LNK1241)"
        LOG 1)
    endif()
  else()
    message(WARNING "${ZENDNNL_MSG_PREFIX}ONEDNN will be injected from ${ZENDNNL_ONEDNN_INJECT_DIR}. This version may not be fully compatible with ZenDNN. If unsure, it is recommended to use the standard ZenDNN build.")
    #add a custom target that create a soft link
    set(SYMLNK_DST "${CMAKE_INSTALL_PREFIX}/deps/onednn")
    set(SYMLNK_SRC "${ZENDNNL_ONEDNN_INJECT_DIR}")

    # blocked to test pytorch integration
    # if(EXISTS ${SYMLNK_DST})
    #   file(REMOVE_RECURSE ${SYMLNK_DST})
    # endif()

    message(DEBUG
      "${ZENDNNL_MSG_PREFIX}ONEDNN symlink from ${SYMLNK_SRC} to ${SYMLNK_DST} will be created.")

    # removed if statement to test pytorch build
    # if(EXISTS ${SYMLNK_DST})
    #   ADD_CUSTOM_TARGET(zendnnl-deps-onednn ALL
    #     COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
    #     COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
    # else()
    #   ADD_CUSTOM_TARGET(zendnnl-deps-onednn ALL
    #     COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
    #     COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
    # endif()

    ADD_CUSTOM_TARGET(zendnnl-deps-onednn ALL
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")

  endif()

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-onednn")

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
  #   if(NOT ZENDNNL_ONEDNN_INJECTED)
  #     set(ONEDNN_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/deps/onednn/include)
  #     set(ONEDNN_LIB_DIR ${CMAKE_INSTALL_PREFIX}/deps/onednn/lib)

  #     if(NOT EXISTS ${ONEDNN_INCLUDE_DIR})
  #       file(MAKE_DIRECTORY ${ONEDNN_INCLUDE_DIR})
  #     endif()

  #     add_library(DNNL_dnnl STATIC IMPORTED GLOBAL)
  #     add_dependencies(DNNL_dnnl zendnnl-deps-onednn)
  #     set_target_properties(DNNL_dnnl
  #       PROPERTIES IMPORTED_LOCATION "${ONEDNN_LIB_DIR}/libdnnl.a"
  #                  INCLUDE_DIRECTORIES "${ONEDNN_INCLUDE_DIR}"
  #                  INTERFACE_INCLUDE_DIRECTORIES "${ONEDNN_INCLUDE_DIR}")

  #     add_library(DNNL::dnnl ALIAS DNNL_dnnl)

  #     list(APPEND ZENDNNL_LINK_LIBS "DNNL::dnnl")
  #     list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ONEDNN_INCLUDE_DIR})
  #   endif()
  # endif()
  # !!!
else()
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Building ONEDNN will be skipped.")
endif()
