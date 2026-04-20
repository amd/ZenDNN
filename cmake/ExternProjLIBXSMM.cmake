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
if (ZENDNNL_DEPENDS_LIBXSMM)

  message(DEBUG "${ZENDNNL_MSG_PREFIX}Configuring LIBXSMM...")

  if (NOT ZENDNNL_LIBXSMM_INJECTED)
    list(APPEND LIBXSMM_MAKE_ARGS "NOFORTRAN=1")
    list(APPEND LIBXSMM_MAKE_ARGS "BLAS=0")

    message(DEBUG "${ZENDNNL_MSG_PREFIX}LIBXSMM_MAKE_ARGS=${LIBXSMM_MAKE_ARGS}")

    get_property(LIBXSMM_INSTALL_DIR GLOBAL PROPERTY LIBXSMMROOT)
    set(NPROC ${ZENDNNL_BUILD_SYS_NPROC})

    if (ZENDNNL_LOCAL_LIBXSMM)
      message(WARNING "${ZENDNNL_MSG_PREFIX}LIBXSMM will be built from local source at ${LIBXSMM_ROOT_DIR}. This version may not be fully compatible with ZenDNN. If unsure, it is recommended to use the standard ZenDNN build.")
      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will use local LIBXSMM from ${LIBXSMM_ROOT_DIR}")

      set(LIBXSMM_DEPS_LINK "${ZENDNNL_DEPS_DIR}/libxsmm")
      if(NOT "${LIBXSMM_ROOT_DIR}" STREQUAL "${LIBXSMM_DEPS_LINK}")
        if(EXISTS "${LIBXSMM_DEPS_LINK}")
          if(IS_SYMLINK "${LIBXSMM_DEPS_LINK}")
            file(REMOVE "${LIBXSMM_DEPS_LINK}")
          else()
            message(WARNING "${ZENDNNL_MSG_PREFIX}${LIBXSMM_DEPS_LINK} exists and is not a symlink. Skipping symlink creation. Remove it manually if needed.")
          endif()
        endif()
        if(NOT EXISTS "${LIBXSMM_DEPS_LINK}")
          execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${LIBXSMM_ROOT_DIR}" "${LIBXSMM_DEPS_LINK}")
          message(STATUS "${ZENDNNL_MSG_PREFIX}Created symlink ${LIBXSMM_DEPS_LINK} -> ${LIBXSMM_ROOT_DIR}")
        endif()
      endif()

      ExternalProject_Add(zendnnl-deps-libxsmm
        SOURCE_DIR   "${LIBXSMM_ROOT_DIR}"
        BINARY_DIR   "${CMAKE_CURRENT_BINARY_DIR}/libxsmm"
        INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/deps/libxsmm"
        BUILD_COMMAND   make -f ${LIBXSMM_ROOT_DIR}/Makefile -j${NPROC} ${LIBXSMM_MAKE_ARGS}
        INSTALL_COMMAND make -f ${LIBXSMM_ROOT_DIR}/Makefile PREFIX=<INSTALL_DIR> ${LIBXSMM_MAKE_ARGS} install
      )
    else()
      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will download LIBXSMM with tag ${LIBXSMM_GIT_TAG}")
      ExternalProject_Add(zendnnl-deps-libxsmm
        SOURCE_DIR       "${LIBXSMM_ROOT_DIR}"
        BINARY_DIR       "${CMAKE_CURRENT_BINARY_DIR}/libxsmm"
        INSTALL_DIR      "${CMAKE_INSTALL_PREFIX}/deps/libxsmm"
        GIT_REPOSITORY   ${LIBXSMM_GIT_REPO}
        GIT_TAG          ${LIBXSMM_GIT_TAG}
        GIT_PROGRESS     ${LIBXSMM_GIT_PROGRESS}
        BUILD_COMMAND   make -f ${LIBXSMM_ROOT_DIR}/Makefile -j${NPROC} ${LIBXSMM_MAKE_ARGS}
        INSTALL_COMMAND make -f ${LIBXSMM_ROOT_DIR}/Makefile PREFIX=<INSTALL_DIR> ${LIBXSMM_MAKE_ARGS} install
        UPDATE_DISCONNECTED TRUE
      )
    endif()

    list(APPEND LIBXSMM_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/libxsmm")
    list(APPEND LIBXSMM_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/libxsmm")
    set_target_properties(zendnnl-deps-libxsmm
      PROPERTIES
      ADDITIONAL_CLEAN_FILES "${LIBXSMM_CLEAN_FILES}"
    )

  else()
    message(WARNING "${ZENDNNL_MSG_PREFIX}LIBXSMM will be injected from ${ZENDNNL_LIBXSMM_INJECT_DIR}. This version may not be fully compatible with ZenDNN. If unsure, it is recommended to use the standard ZenDNN build.")
    set(SYMLNK_DST "${CMAKE_INSTALL_PREFIX}/deps/libxsmm")
    set(SYMLNK_SRC "${ZENDNNL_LIBXSMM_INJECT_DIR}")

    # blocked for consistency with onednn
    # if (EXISTS ${SYMLNK_DST})
    #   file(REMOVE_RECURSE ${SYMLNK_DST})
    # endif()

    message(DEBUG
      "${ZENDNNL_MSG_PREFIX}LIBXSMM symlink from ${SYMLNK_SRC} to ${SYMLNK_DST} will be created."
    )

    # removed if condition for consistency with onednn
    # if (EXISTS ${SYMLNK_DST})
    #   add_custom_target(zendnnl-deps-libxsmm ALL
    #     COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
    #     COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
    # else()
    #   add_custom_target(zendnnl-deps-libxsmm ALL
    #     COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
    # endif()

    add_custom_target(zendnnl-deps-libxsmm ALL
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
  endif()

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-libxsmm")
else()
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Building LIBXSMM will be skipped.")
endif()
