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

include(ZenDnnlOptions)

if(ZENDNNL_DEPENDS_AOCLDLP)

  message(DEBUG "${ZENDNNL_MSG_PREFIX}Configurig AOCL-DLP...")

  if (NOT ZENDNNL_AMDBLIS_INJECTED)
    list(APPEND AD_CMAKE_ARGS "-DDLP_THREADING_MODEL=openmp")
    list(APPEND AD_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=Release")
    list(APPEND AD_CMAKE_ARGS "-DCMAKE_VERBOSE_MAKEFILE=OFF")
    list(APPEND AD_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

    # uncoment if openmp root need to be given
    # list(APPEND AD_CMAKE_ARGS "-DDLP_OPENMP_ROOT=/path/to/openmp")

    message(DEBUG "${ZENDNNL_MSG_PREFIX}AOCLDLP_CMAKE_ARGS=${AD_CMAKE_ARGS}")

    set(NPROC ${ZENDNNL_BUILD_SYS_NPROC})
    if (ZENDNNL_LOCAL_AOCLDLP)

      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will use local AOCL-DLP from ${AOCLDLP_ROOT_DIR}")

      ExternalProject_ADD(zendnnl-deps-aocldlp
        SOURCE_DIR "${AOCLDLP_ROOT_DIR}"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/aocldlp"
        INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/aocldlp"
        CMAKE_ARGS ${AD_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --config release --target all -- -j${NPROC}
        INSTALL_COMMAND cmake --build . --config release --target install)
    else()
      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will download AOCL-DLP with tag ${AOCLDLP_GIT_TAG}")
      ExternalProject_ADD(zendnnl-deps-aocldlp
        SOURCE_DIR "${AOCLDLP_ROOT_DIR}"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/aocldlp"
        INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/aocldlp"
        GIT_REPOSITORY ${AOCLDLP_GIT_REPO}
        GIT_TAG ${AOCLDLP_GIT_TAG}
        GIT_PROGRESS ${AOCLDLP_GIT_PROGRESS}
        CMAKE_ARGS ${AD_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --config release --target all -- -j${NPROC}
        INSTALL_COMMAND cmake --build . --config release --target install
        UPDATE_DISCONNECTED TRUE)
    endif()

    list(APPEND AOCLDLP_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/aocldlp")
    list(APPEND AOCLDLP_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/aocldlp")

    set_target_properties(zendnnl-deps-aocldlp
      PROPERTIES
      ADDITIONAL_CLEAN_FILES "${AOCLDLP_CLEAN_FILES}")
  else()
    #add a custom target that create a soft link
    set(SYMLNK_DST "${CMAKE_INSTALL_PREFIX}/deps/aocldlp")
    set(SYMLNK_SRC "${ZENDNNL_AOCLDLP_FWK_DIR}")

    message(DEBUG
      "${ZENDNNL_MSG_PREFIX}AOCL-DLP symlink from ${SYMLNK_SRC} to ${SYMLNK_DST} will be created.")

    if(EXISTS ${SYMLNK_DST})
      ADD_CUSTOM_TARGET(zendnnl-deps-aocldlp ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
    else()
      ADD_CUSTOM_TARGET(zendnnl-deps-aocldlp ALL
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
    endif()
  endif()

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-aocldlp")

  # Manual interface for AOCL-DLP libraries
  # if (NOT ZENDNNL_STANDALONE_BUILD)
  #   if(NOT ZENDNNL_AOCLDLP_INJECTED)
  #     set(ZENDNNL_AOCLDLP_INC_DIR "${CMAKE_INSTALL_PREFIX}/deps/aocldlp/include")
  #     set(ZENDNNL_AOCLDLP_LIB_DIR "${CMAKE_INSTALL_PREFIX}/deps/aocldlp/lib")

  #     if(NOT EXISTS ${ZENDNNL_AMDBLIS_INC_DIR})
  #       file(MAKE_DIRECTORY ${ZENDNNL_AOCLDLP_INC_DIR})
  #     endif()

  #     add_library(zendnnl_aocldlp_static_deps STATIC IMPORTED GLOBAL)
  #     add_dependencies(zendnnl_aocldlp_static_deps zendnnl-deps-aocldlp)
  #     set_target_properties(zendnnl_aocldlp_static_deps
  #       PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AOCLDLP_LIB_DIR}/libaocl_dlp.a"
  #                  INCLUDE_DIRECTORIES "${ZENDNNL_AOCLDLP_INC_DIR}"
  #                  INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AOCLDLP_INC_DIR}")

  #     add_library(aocldlp::aocl_dlp_static ALIAS zendnnl_aocldlp_static_deps)

  #     list(APPEND ZENDNNL_LINK_LIBS "aocldlp::aocl_dlp_static")
  #     list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_AOCLDLP_INC_DIR})
  #   endif()
  # endif()
else()
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Building AOCL-DLP will be skipped.")
endif()
