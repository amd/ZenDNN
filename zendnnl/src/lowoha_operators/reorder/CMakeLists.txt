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

set(lowoha_reorder_sources
  lowoha_reorder.cpp
  lowoha_reorder_utils.cpp
  reorder_kernels.cpp)

set(lowoha_reorder_public_headers
  lowoha_reorder_common.hpp
  lowoha_reorder.hpp
  lowoha_reorder_utils.hpp)

set(lowoha_reorder_private_headers
  reorder_kernels.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
    PRIVATE ${lowoha_reorder_sources}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${lowoha_reorder_public_headers}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${lowoha_reorder_private_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
    PRIVATE ${lowoha_reorder_sources}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${lowoha_reorder_public_headers}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${lowoha_reorder_private_headers})
endif()

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

list(APPEND lowoha_reorder_public_headers_list
  ${lowoha_reorder_public_headers})
create_header_file(${zendnnl_inc_file}
  ${inc_subdir}
  "${lowoha_reorder_public_headers_list}")

