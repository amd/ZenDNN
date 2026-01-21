# *******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
if (ZENDNNL_DEPENDS_FBGEMM)
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Configuring FBGEMM...")

  if (NOT ZENDNNL_FBGEMM_INJECTED)

    get_property(FBGEMM_INSTALL_DIR GLOBAL PROPERTY FBGEMMROOT)

    list(APPEND FBGEMM_CMAKE_ARGS "-DFBGEMM_BUILD_TESTS=0")
    list(APPEND FBGEMM_CMAKE_ARGS "-DFBGEMM_BUILD_BENCHMARKS=OFF")
    list(APPEND FBGEMM_CMAKE_ARGS "-DFBGEMM_LIBRARY_TYPE=static")
    list(APPEND FBGEMM_CMAKE_ARGS "-DFBGEMM_USE_SANITIZER=OFF")
    list(APPEND FBGEMM_CMAKE_ARGS "-DCMAKE_VERBOSE_MAKEFILE=OFF")
    list(APPEND FBGEMM_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=Release")
    list(APPEND FBGEMM_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")
    list(APPEND FBGEMM_CMAKE_ARGS "-DCMAKE_POSITION_INDEPENDENT_CODE=ON")

    message(DEBUG "${ZENDNNL_MSG_PREFIX}FBGEMM_CMAKE_ARGS=${FBGEMM_CMAKE_ARGS}")

    set(NPROC ${ZENDNNL_BUILD_SYS_NPROC})
    if (ZENDNNL_LOCAL_FBGEMM)

      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will use local FBGEMM from ${FBGEMM_ROOT_DIR}")

      ExternalProject_ADD(zendnnl-deps-fbgemm
        SOURCE_DIR "${FBGEMM_ROOT_DIR}"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fbgemm"
        INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/fbgemm"
        CMAKE_ARGS ${FBGEMM_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --target fbgemm -- -j${NPROC}
        INSTALL_COMMAND cmake --build . --target install)
    else()

      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will download FBGEMM with tag ${FBGEMM_GIT_TAG}")

      ExternalProject_ADD(zendnnl-deps-fbgemm
        SOURCE_DIR "${FBGEMM_ROOT_DIR}"
        BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/fbgemm"
        INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/fbgemm"
        GIT_REPOSITORY ${FBGEMM_GIT_REPO}
        GIT_TAG ${FBGEMM_GIT_TAG}
        GIT_PROGRESS ${FBGEMM_GIT_PROGRESS}
        CMAKE_ARGS ${FBGEMM_CMAKE_ARGS}
        BUILD_COMMAND cmake --build . --target fbgemm -- -j${NPROC}
        INSTALL_COMMAND cmake --build . --target install
        UPDATE_DISCONNECTED TRUE)
    endif()

    # Add custom step to install asmjit library (not installed by default)
    ExternalProject_Add_Step(zendnnl-deps-fbgemm install_asmjit
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CMAKE_CURRENT_BINARY_DIR}/fbgemm/asmjit/libasmjit.a"
              "${CMAKE_INSTALL_PREFIX}/deps/fbgemm/lib/libasmjit.a"
      DEPENDEES install
      COMMENT "Installing asmjit library"
    )

    list(APPEND FBGEMM_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/fbgemm")
    list(APPEND FBGEMM_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/fbgemm")

    set_target_properties(zendnnl-deps-fbgemm
      PROPERTIES
      ADDITIONAL_CLEAN_FILES "${FBGEMM_CLEAN_FILES}")

  else()
      # add a custom target that creates a soft link for framework-injected FBGEMM
    set(SYMLNK_DST "${CMAKE_INSTALL_PREFIX}/deps/fbgemm")
    set(SYMLNK_SRC "${ZENDNNL_FBGEMM_FWK_DIR}")
  
    message(DEBUG
      "${ZENDNNL_MSG_PREFIX}FBGEMM symlink from ${SYMLNK_SRC} to ${SYMLNK_DST} will be created.")
  
    add_custom_target(zendnnl-deps-fbgemm ALL
      COMMAND ${CMAKE_COMMAND} -E rm -rf "${SYMLNK_DST}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink "${SYMLNK_SRC}" "${SYMLNK_DST}")
  
  endif()
  
  list(APPEND ZENDNNL_DEPS "zendnnl-deps-fbgemm")
else()
  message(DEBUG "${ZENDNNL_MSG_PREFIX}Building FBGEMM will be skipped.")
endif()