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
      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will use local LIBXSMM from ${LIBXSMM_ROOT_DIR}")
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
    set(SYMLNK_DST "${CMAKE_INSTALL_PREFIX}/deps/libxsmm")
    set(SYMLNK_SRC "${ZENDNNL_LIBXSMM_FWK_DIR}")

    if (EXISTS ${SYMLNK_DST})
      file(REMOVE_RECURSE ${SYMLNK_DST})
    endif()

    message(DEBUG
      "${ZENDNNL_MSG_PREFIX}LIBXSMM symlink from ${SYMLNK_SRC} to ${SYMLNK_DST} will be created."
    )

    if (EXISTS ${SYMLNK_DST})
      add_custom_target(zendnnl-deps-libxsmm ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}"
      )
    else()
      add_custom_target(zendnnl-deps-libxsmm ALL
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}"
      )
    endif()
  endif()

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-libxsmm")
else()
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Building LIBXSMM will be skipped.")
endif()
