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

include(ZenDnnlOptions)
if (ZENDNNL_DEPENDS_PARLOOPER)
  # check if zendnnl needs to build this dependency
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Configurig PARLOOPER...")

  if (NOT ZENDNNL_PARLOOPER_INJECTED)

    get_property(PARLOOPER_INSTALL_DIR GLOBAL PROPERTY PARLOOPERROOT)

    list(APPEND PARLOOPER_MAKE_ARGS "AVX=2")
    list(APPEND PARLOOPER_MAKE_ARGS "USE_Cxx_ABI=1")
    list(APPEND PARLOOPER_MAKE_ARGS "PARLOOPER_COMPILER=gcc")

    message(DEBUG "${ZENDNNL_MSG_PREFIX}LIBXSMM_MAKE_ARGS=${PARLOOPER_MAKE_ARGS}")

    set(NPROC ${ZENDNNL_BUILD_SYS_NPROC})

    if (ZENDNNL_LOCAL_PARLOOPER)
      message(DEBUG "Using local ParLooper from ${PARLOOPER_ROOT_DIR}")
      ExternalProject_Add(zendnnl-deps-parlooper
        SOURCE_DIR    "${PARLOOPER_ROOT_DIR}"
        BINARY_DIR    "${CMAKE_CURRENT_BINARY_DIR}/parlooper"
        INSTALL_DIR   "${CMAKE_INSTALL_PREFIX}/deps/parlooper"
        CMAKE_ARGS    "${PARLOOPER_CMAKE_ARGS}"

        # Create build dir and symlinks for sources/headers
        CONFIGURE_COMMAND
          ${CMAKE_COMMAND} -E make_directory <BINARY_DIR> &&
          ${CMAKE_COMMAND} -E rm -rf <BINARY_DIR>/Makefile &&
          ${CMAKE_COMMAND} -E rm -rf <BINARY_DIR>/src &&
          ${CMAKE_COMMAND} -E rm -rf <BINARY_DIR>/include &&
          ${CMAKE_COMMAND} -E create_symlink <SOURCE_DIR>/Makefile <BINARY_DIR>/Makefile &&
          ${CMAKE_COMMAND} -E create_symlink <SOURCE_DIR>/src <BINARY_DIR>/src &&
          ${CMAKE_COMMAND} -E create_symlink <SOURCE_DIR>/include <BINARY_DIR>/include

        # Build inside BINARY_DIR
        BUILD_COMMAND
          ${CMAKE_MAKE_PROGRAM} -C <BINARY_DIR> ${PARLOOPER_MAKE_ARGS} -j${NPROC}

        # Install into deps/parlooper
        INSTALL_COMMAND
          ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/lib &&
          ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/include &&
          ${CMAKE_COMMAND} -E copy <BINARY_DIR>/lib/libparlooper.a <INSTALL_DIR>/lib/ &&
          ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/include <INSTALL_DIR>/include

        BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libparlooper.a
      )
    else()
      ExternalProject_Add(zendnnl-deps-parlooper
        GIT_REPOSITORY  ${PARLOOPER_GIT_REPO}
        GIT_TAG         ${PARLOOPER_GIT_TAG}
        SOURCE_DIR      "${PARLOOPER_ROOT_DIR}"
        BINARY_DIR      "${CMAKE_CURRENT_BINARY_DIR}/parlooper"
        INSTALL_DIR     "${CMAKE_INSTALL_PREFIX}/deps/parlooper"
        CMAKE_ARGS      "${PARLOOPER_CMAKE_ARGS}"

        CONFIGURE_COMMAND
          ${CMAKE_COMMAND} -E make_directory <BINARY_DIR> &&
          ${CMAKE_COMMAND} -E rm -rf <BINARY_DIR>/Makefile &&
          ${CMAKE_COMMAND} -E rm -rf <BINARY_DIR>/src &&
          ${CMAKE_COMMAND} -E rm -rf <BINARY_DIR>/include &&
          ${CMAKE_COMMAND} -E create_symlink <SOURCE_DIR>/Makefile <BINARY_DIR>/Makefile &&
          ${CMAKE_COMMAND} -E create_symlink <SOURCE_DIR>/src <BINARY_DIR>/src &&
          ${CMAKE_COMMAND} -E create_symlink <SOURCE_DIR>/include <BINARY_DIR>/include

        BUILD_COMMAND
          ${CMAKE_MAKE_PROGRAM} -C <BINARY_DIR> ${PARLOOPER_MAKE_ARGS} -j${NPROC}

        INSTALL_COMMAND
          ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/lib &&
          ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/include &&
          ${CMAKE_COMMAND} -E copy <BINARY_DIR>/lib/libparlooper.a <INSTALL_DIR>/lib/ &&
          ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/include <INSTALL_DIR>/include

        BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libparlooper.a
        UPDATE_DISCONNECTED TRUE
      )

      list(APPEND PARLOOPER_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/parlooper")
      list(APPEND PARLOOPER_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/parlooper")
      set_target_properties(zendnnl-deps-parlooper
        PROPERTIES
        ADDITIONAL_CLEAN_FILES "${PARLOOPER_CLEAN_FILES}"
      )
    endif()

  else()
    set(SYMLNK_DST "${CMAKE_INSTALL_PREFIX}/deps/parlooper")
    set(SYMLNK_SRC "${ZENDNNL_PARLOOPER_FWK_DIR}")

    # blocked for consistency with onednn
    # if (EXISTS ${SYMLNK_DST})
    #   file(REMOVE_RECURSE ${SYMLNK_DST})
    # endif()

    message(DEBUG
      "${ZENDNNL_MSG_PREFIX}PARLOOPER symlink from ${SYMLNK_SRC} to ${SYMLNK_DST} will be created."
    )

    # removed if condition for consistency with onednn
    # if (EXISTS ${SYMLNK_DST})
    #   add_custom_target(zendnnl-deps-parlooper ALL
    #     COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
    #     COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
    # else()
    #   add_custom_target(zendnnl-deps-parlooper ALL
    #     COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
    # endif()

    add_custom_target(zendnnl-deps-parlooper ALL
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
  endif()

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-parlooper")
else()
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Building parlooper will be skipped.")
endif()
