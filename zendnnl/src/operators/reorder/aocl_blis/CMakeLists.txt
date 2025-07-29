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

set(aocl_blis_reorder_operator_sources
  reorder_kernel.cpp
  reorder_utils.cpp)

set(aocl_blis_reorder_operator_public_headers
  reorder_utils.hpp)

set(aocl_blis_reorder_operator_private_headers
  reorder_kernel.hpp)

target_sources(zendnnl_archive
  PRIVATE ${aocl_blis_reorder_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${aocl_blis_reorder_operator_private_headers}
  PUBLIC FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${aocl_blis_reorder_operator_public_headers})

target_sources(zendnnl
  PRIVATE ${aocl_blis_reorder_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${aocl_blis_reorder_operator_private_headers}
  PUBLIC FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${aocl_blis_reorder_operator_public_headers})

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# append header files to top level include file
list(APPEND aocl_blis_reorder_operator_public_headers_list
  ${aocl_blis_reorder_operator_public_headers})
create_header_file(${zendnnl_inc_file}
  ${inc_subdir}
  "${aocl_blis_reorder_operator_public_headers_list}")