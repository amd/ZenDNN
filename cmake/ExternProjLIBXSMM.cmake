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
    if(WIN32)
      # LIBXSMM's GNU Makefile is Unix-only (assumes gcc/as/uname/shell), so on
      # Windows build via LIBXSMM's CMake build instead of `make`.
      list(APPEND LIBXSMM_CMAKE_ARGS "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}")
      list(APPEND LIBXSMM_CMAKE_ARGS "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
      list(APPEND LIBXSMM_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=Release")
      list(APPEND LIBXSMM_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")
      list(APPEND LIBXSMM_CMAKE_ARGS "-DBUILD_SHARED_LIBS=OFF")
      # LIBXSMM's fallback GEMM (libxsmm_gemm.c) references the reference Fortran
      # BLAS symbols (sgemm_/dgemm_/sgemv_/dgemv_) unless BLAS is compiled out.
      # The Makefile build (else branch) does this with BLAS=0; LIBXSMM's CMake
      # build has no such option and instead takes the __BLAS=0 compile definition
      # (see the link instructions in LIBXSMM's CMakeLists.txt). The Windows
      # toolchain has no system BLAS to satisfy those symbols, so build BLAS-free
      # to mirror the Linux BLAS=0 build and avoid unresolved externals when
      # linking zendnnl.
      list(APPEND LIBXSMM_CMAKE_ARGS "-DCMAKE_C_FLAGS=-D__BLAS=0")
      # LIBXSMM's CMake build globs every src/*.c into xsmm.lib, including the
      # standalone generator tools that each define their own main(). Upstream
      # drops libxsmm_generator_gemm_driver.c but not
      # libxsmm_binaryexport_generator.c, so the latter's main() lands in
      # xsmm.lib and collides with the examples executable's main() at link time
      # (LNK2005 / LNK1169) once xsmm.lib is whole-archived into zendnnl_archive.
      # The Makefile build never puts these tools in the library; drop the stray
      # tool source so the Windows archive is link-clean as well.
      set(LIBXSMM_WIN_PATCH
        ${CMAKE_COMMAND} -E rm -f <SOURCE_DIR>/src/libxsmm_binaryexport_generator.c)
      # LIBXSMM's CMake build defines no install target, so install the built
      # static library (xsmm.lib) and the public headers manually.
      set(LIBXSMM_WIN_INSTALL
        ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/lib
        COMMAND ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/xsmm.lib <INSTALL_DIR>/lib/xsmm.lib
        COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include <INSTALL_DIR>/include)
    else()
      list(APPEND LIBXSMM_MAKE_ARGS "NOFORTRAN=1")
      list(APPEND LIBXSMM_MAKE_ARGS "BLAS=0")
    endif()

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

      if(WIN32)
        ExternalProject_Add(zendnnl-deps-libxsmm
          SOURCE_DIR   "${LIBXSMM_ROOT_DIR}"
          BINARY_DIR   "${CMAKE_CURRENT_BINARY_DIR}/libxsmm"
          INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/deps/libxsmm"
          PATCH_COMMAND ${LIBXSMM_WIN_PATCH}
          CMAKE_ARGS   ${LIBXSMM_CMAKE_ARGS}
          INSTALL_COMMAND ${LIBXSMM_WIN_INSTALL}
        )
      else()
        ExternalProject_Add(zendnnl-deps-libxsmm
          SOURCE_DIR   "${LIBXSMM_ROOT_DIR}"
          BINARY_DIR   "${CMAKE_CURRENT_BINARY_DIR}/libxsmm"
          INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/deps/libxsmm"
          BUILD_COMMAND   make -f ${LIBXSMM_ROOT_DIR}/Makefile -j${NPROC} ${LIBXSMM_MAKE_ARGS}
          INSTALL_COMMAND make -f ${LIBXSMM_ROOT_DIR}/Makefile PREFIX=<INSTALL_DIR> ${LIBXSMM_MAKE_ARGS} install
        )
      endif()
    else()
      message(DEBUG "${ZENDNNL_MSG_PREFIX}Will download LIBXSMM with tag ${LIBXSMM_GIT_TAG}")
      if(WIN32)
        ExternalProject_Add(zendnnl-deps-libxsmm
          SOURCE_DIR       "${LIBXSMM_ROOT_DIR}"
          BINARY_DIR       "${CMAKE_CURRENT_BINARY_DIR}/libxsmm"
          INSTALL_DIR      "${CMAKE_INSTALL_PREFIX}/deps/libxsmm"
          GIT_REPOSITORY   ${LIBXSMM_GIT_REPO}
          GIT_TAG          ${LIBXSMM_GIT_TAG}
          GIT_PROGRESS     ${LIBXSMM_GIT_PROGRESS}
          PATCH_COMMAND    ${LIBXSMM_WIN_PATCH}
          CMAKE_ARGS       ${LIBXSMM_CMAKE_ARGS}
          INSTALL_COMMAND  ${LIBXSMM_WIN_INSTALL}
          UPDATE_DISCONNECTED TRUE
        )
      else()
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
