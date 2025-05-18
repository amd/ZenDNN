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

get_property(ONEDNN_INSTALL_DIR GLOBAL PROPERTY ONEDNNROOT)

# set(ONEDNN_INCLUDE_DIR ${ONEDNN_INSTALL_DIR}/include)
# set(ONEDNN_LIB ${ONEDNN_INSTALL_DIR}/lib/amdzen/libblis-mt.so)
# set(ONEDNN_ARCHIVE_LIB ${ONEDNN_INSTALL_DIR}/lib/amdzen/libblis-mt.a)

ExternalProject_ADD(zendnnl_onednn
  SOURCE_DIR "${ZENDNNL_ONEDNN_DIR}"
  INSTALL_DIR "${ONEDNN_INSTALL_DIR}"
  GIT_REPOSITORY ${ONEDNN_GIT_REPO}
  GIT_TAG ${ONEDNN_GIT_TAG}
  GIT_PROGRESS ${ONEDNN_GIT_PROGRESS}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${ONEDNN_INSTALL_DIR} -DCMAKE_BUILD_TYPE=Release -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_DOC=OFF -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_GRAPH=OFF
  BUILD_COMMAND make -j8)

set(ONEDNN_INCLUDE_DIR ${ONEDNN_INSTALL_DIR}/include)
set(ONEDNN_LIB ${ONEDNN_INSTALL_DIR}/lib/libdnnl.so)
set(ONEDNN_ARCHIVE_LIB ${ONEDNN_INSTALL_DIR}/lib/libdnnl.a)

add_library(DNNL_dnnl INTERFACE)
add_dependencies(DNNL_dnnl zendnnl_onednn)
target_link_libraries(DNNL_dnnl INTERFACE ${ONEDNN_ARCHIVE_LIB})
set_target_properties(DNNL_dnnl
  PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${ONEDNN_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${ONEDNN_INCLUDE_DIR}
  INTERFACE_LINK_LIBRARIES "dl")

add_library(DNNL::dnnl ALIAS DNNL_dnnl)
