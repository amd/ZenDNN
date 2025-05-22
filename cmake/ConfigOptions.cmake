# *******************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

if(CONFIG_OPTIONS_INCLUDED)
  return()
endif()
set(CONFIG_OTIONS_INCLUDED)

##
# components
##
set(ZENDNNL_BUILD_EXAMPLES ON CACHE BOOL "Build examples")
set(ZENDNNL_BUILD_DOXYGEN_DOCS ON CACHE BOOL "Build doxygen docs")
set(ZENDNNL_BUILD_GTESTS ON CACHE BOOL "Build tests")

##
# special flags
##
set(ZENDNNL_USE_CXX11_ABI_FLAG ON CACHE BOOL "Use CXX11 ABI")
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
add_compile_options(-Wall -Werror)

##
# dependencies
##
set(ZENDNNL_DEPENDS_AMDBLIS ON CACHE BOOL "Add AMD BLIS as a dependency")
set(ZENDNNL_DEPENDS_ONEDNN OFF CACHE BOOL "Add ONEDNN as a dependency")
set(ZENDNNL_DEPENDS_GTEST ON CACHE BOOL "Use google test framework for testing")
set(ZENDNNL_DEPENDS_AOCLUTILS OFF CACHE BOOL "Use aocl utils for hardware identification")

set(ZENDNNL_DEP_DIR ${CMAKE_SOURCE_DIR}/dependencies)
set(ZENDNNL_DEP_BUILD_SUBDIR "zendnnl_build")
set(ZENDNNL_DEP_INSTALL_SUBDIR "zendnnl_build/install")

if(ZENDNNL_DEPENDS_AMDBLIS)
  set(ZENDNNL_AMDBLIS_USE_LOCAL_REPO OFF CACHE BOOL "Use AMDBLIS local repo")
  if(ZENDNNL_AMDBLIS_USE_LOCAL_REPO)
    set(ZENDNNL_AMDBLIS_DIR "/proj/zendnn/sudarshan/Packages/amd-blis-latest/blis"
      CACHE PATH "AMD BLIS root dir")
    set_property(GLOBAL PROPERTY AMDBLISROOT ${ZENDNNL_AMDBLIS_DIR})
  else()
    set(ZENDNNL_AMDBLIS_DIR "${ZENDNNL_DEP_DIR}/blis"
      CACHE PATH "AMD BLIS root dir")
    set_property(GLOBAL
      PROPERTY AMDBLISROOT "${ZENDNNL_AMDBLIS_DIR}/${ZENDNNL_DEP_INSTALL_SUBDIR}")
  endif()

  set(AMDBLIS_GIT_REPO "https://github.com/amd/blis.git")
  # amdblis tag 5.0
  # set(AMDBLIS_GIT_TAG "34d4bbade33a4384cfaff2208c833fc33a311c5d")
  # amdblis tag AOCL-Mar2025-b2
  set(AMDBLIS_GIT_TAG "6d1afeae95b198ea7f19e251a19cba4f0a19813c")
  option(AMDBLIS_GIT_PROGRESS ON)
endif()

if(ZENDNNL_DEPENDS_ONEDNN)
  set(ZENDNNL_ONEDNN_USE_LOCAL_REPO OFF CACHE BOOL "Use ONEDNN local repo")
  if(ZENDNNL_ONEDNN_USE_LOCAL_REPO)
    set(ZENDNNL_ONEDNN_DIR "/proj/zendnn/sudarshan/ZenTorch/Intel/oneDNN"
      CACHE PATH "ONEDNN root dir")
    set_property(GLOBAL PROPERTY ONEDNNROOT ${ZENDNNL_ONEDNN_DIR})
  else()
    set(ZENDNNL_ONEDNN_DIR "${ZENDNNL_DEP_DIR}/oneDNN"
      CACHE PATH "ONEDNN root dir")
    set_property(GLOBAL PROPERTY ONEDNNROOT "${ZENDNNL_ONEDNN_DIR}/${ZENDNNL_DEP_INSTALL_SUBDIR}")
  endif()

  set(ONEDNN_GIT_REPO "https://github.com/oneapi-src/oneDNN.git")
  set(ONEDNN_GIT_TAG "v3.7.1")
  option(ONEDNN_GIT_PROGRESS ON)
endif()

if(ZENDNNL_DEPENDS_GTEST)
  set(ZENDNNL_GTEST_DIR "${ZENDNNL_DEP_DIR}/gtest"
    CACHE PATH "GTEST root dir")
  set_property(GLOBAL
      PROPERTY GTESTROOT "${ZENDNNL_GTEST_DIR}/${ZENDNNL_DEP_INSTALL_SUBDIR}")
  set(GTEST_GIT_REPO "https://github.com/google/googletest.git")
  set(GTEST_GIT_TAG "v1.17.0")
endif()

if(ZENDNNL_DEPENDS_AOCLUTILS)
  set(ZENDNNL_AOCLUTILS_DIR "${ZENDNNL_DEP_DIR}/aocl-utils"
    CACHE PATH "AOCL UTILS root dir")
  set_property(GLOBAL
      PROPERTY AOCLUTILSROOT "${ZENDNNL_AOCLUTILS_DIR}/${ZENDNNL_DEP_INSTALL_SUBDIR}")
  set(AOCLUTILS_GIT_REPO "https://github.com/amd/aocl-utils")
  set(AOCLUTILS_GIT_TAG "5.0")
endif()
