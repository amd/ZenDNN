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
  list(APPEND AD_CMAKE_ARGS "-DDLP_THREADING_MODEL=openmp")
  #list(APPEND AD_CMAKE_ARGS "-DDLP_OPENMP_ROOT=/path/to/openmp") //optional, if openmp is not at default place.
  list(APPEND AD_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>")

  message(DEBUG "AD_CMAKE_ARGS=${AD_CMAKE_ARGS}")
  cmake_host_system_information(RESULT NPROC QUERY NUMBER_OF_PHYSICAL_CORES)

  if (ZENDNNL_LOCAL_AOCLDLP)

    message(DEBUG "Using local AOCL-DLP from ${AOCLDLP_ROOT_DIR}")

    ExternalProject_ADD(zendnnl-deps-aocldlp
      SOURCE_DIR "${AOCLDLP_ROOT_DIR}"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/aocldlp"
      INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/aocldlp"
      CMAKE_ARGS ${AD_CMAKE_ARGS}
      BUILD_COMMAND cmake --build . --config release --target all -- -j${NPROC}
      INSTALL_COMMAND cmake --build . --config release --target install
      BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libaocl_dlp.a)
  else()
      message(DEBUG "Public aocl-dlp is not available yet.")
    # ExternalProject_ADD(zendnnl-deps-aocldlp
    #   SOURCE_DIR "${AOCLDLP_ROOT_DIR}"
    #   BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/aocldlp"
    #   INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/aocldlp"
    #   GIT_REPOSITORY ${AOCLDLP_GIT_REPO}
    #   GIT_TAG ${AOCLDLP_GIT_TAG}
    #   GIT_PROGRESS ${AOCLDLP_GIT_PROGRESS}
    #   CMAKE_ARGS ${AD_CMAKE_ARGS}
    #   BUILD_COMMAND cmake --build . --config release --target all -- -j${NPROC}
    #   INSTALL_COMMAND cmake --build . --config release --target install
    #   BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libaocl_dlp.a
    #   UPDATE_DISCONNECTED TRUE)
  endif()

  list(APPEND AOCLDLP_CLEAN_FILES "${CMAKE_CURRENT_BINARY_DIR}/aocldlp")
  list(APPEND AOCLDLP_CLEAN_FILES "${CMAKE_INSTALL_PREFIX}/deps/aocldlp")

  set_target_properties(zendnnl-deps-aocldlp
    PROPERTIES
    ADDITIONAL_CLEAN_FILES "${AOCLDLP_CLEAN_FILES}")

  list(APPEND ZENDNNL_DEPS "zendnnl-deps-aocldlp")

  # Manual interface for AOCL-DLP libraries
  set(ZENDNNL_AOCLDLP_INC_DIR "${CMAKE_INSTALL_PREFIX}/deps/aocldlp/include")
  set(ZENDNNL_AOCLDLP_LIB_DIR "${CMAKE_INSTALL_PREFIX}/deps/aocldlp/lib")

  file(MAKE_DIRECTORY ${ZENDNNL_AOCLDLP_INC_DIR})
  
#   # Shared library target
#   add_library(zendnnl_aocldlp_deps SHARED IMPORTED GLOBAL)
#   add_dependencies(zendnnl_aocldlp_deps zendnnl-deps-aocldlp)
#   set_target_properties(zendnnl_aocldlp_deps
#     PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AOCLDLP_LIB_DIR}/libaocl_dlp.so"
#                INCLUDE_DIRECTORIES "${ZENDNNL_AOCLDLP_INC_DIR}"
#                INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AOCLDLP_INC_DIR}")

#   add_library(aocldlp::aocl_dlp ALIAS zendnnl_aocldlp_deps)
#   list(APPEND ZENDNNL_LINK_LIBS "aocldlp::aocl_dlp")

  # Static library target
  add_library(zendnnl_aocldlp_static_deps STATIC IMPORTED GLOBAL)
  add_dependencies(zendnnl_aocldlp_static_deps zendnnl-deps-aocldlp)
  set_target_properties(zendnnl_aocldlp_static_deps
    PROPERTIES IMPORTED_LOCATION "${ZENDNNL_AOCLDLP_LIB_DIR}/libaocl_dlp.a"
               INCLUDE_DIRECTORIES "${ZENDNNL_AOCLDLP_INC_DIR}"
               INTERFACE_INCLUDE_DIRECTORIES "${ZENDNNL_AOCLDLP_INC_DIR}")

  add_library(aocldlp::aocl_dlp_static ALIAS zendnnl_aocldlp_static_deps)
  list(APPEND ZENDNNL_LINK_LIBS "aocldlp::aocl_dlp_static")
  list(APPEND ZENDNNL_INCLUDE_DIRECTORIES ${ZENDNNL_AOCLDLP_INC_DIR})

else()
  message(DEBUG "skipping building aocl-dlp.")
endif()
