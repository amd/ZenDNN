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

include(ConfigOptions)

get_property(AMDBLIS_INSTALL_DIR GLOBAL PROPERTY AMDBLISROOT)

# set(CONFIGURE_OPTIONS "-a aocl_gemm ")
# set(CONFIGURE_OPTIONS "${CONFIGURE_OPTIONS} --enable-threading=openmp ")
# set(CONFIGURE_OPTIONS "${CONFIGURE_OPTIONS} --enable-cblas ")
# set(CONFIGURE_OPTIONS "${CONFIGURE_OPTIONS} amdzen ")

ExternalProject_ADD(zendnnl_amdblis
  SOURCE_DIR "${ZENDNNL_AMDBLIS_DIR}"
  INSTALL_DIR "${AMDBLIS_INSTALL_DIR}"
  GIT_REPOSITORY ${AMDBLIS_GIT_REPO}
  GIT_TAG ${AMDBLIS_GIT_TAG}
  GIT_PROGRESS ${AMDBLIS_GIT_PROGRESS}
  CMAKE_ARGS -DBLIS_CONFIG_FAMILY=amdzen -DENABLE_ADDON=aocl_gemm -DENABLE_THREADING=openmp -DENABLE_CBLAS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  BUILD_COMMAND cmake --build . --config release --target install -j
  UPDATE_DISCONNECTED TRUE)

#set(AMDBLIS_INCLUDE_DIR ${AMDBLIS_INSTALL_DIR}/include/blis)
set(AMDBLIS_INCLUDE_DIR ${AMDBLIS_INSTALL_DIR}/include)
set(AMDBLIS_LIB ${AMDBLIS_INSTALL_DIR}/lib/libblis-mt.so)
set(AMDBLIS_ARCHIVE_LIB ${AMDBLIS_INSTALL_DIR}/lib/libblis-mt.a)

add_library(amdblis_amdblis INTERFACE)
add_dependencies(amdblis_amdblis zendnnl_amdblis)
target_link_libraries(amdblis_amdblis INTERFACE ${AMDBLIS_LIB})
set_target_properties(amdblis_amdblis
  PROPERTIES
  IMPORTED_LOCATION ${AMDBLIS_LIB}
  INTERFACE_INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR})

add_library(amdblis::amdblis ALIAS amdblis_amdblis)

add_library(amdblis_amdblis_archive INTERFACE)
add_dependencies(amdblis_amdblis_archive zendnnl_amdblis)
target_link_libraries(amdblis_amdblis_archive INTERFACE ${AMDBLIS_ARCHIVE_LIB})
set_target_properties(amdblis_amdblis_archive
  PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${AMDBLIS_INCLUDE_DIR})

add_library(amdblis::amdblis_archive ALIAS amdblis_amdblis_archive)
