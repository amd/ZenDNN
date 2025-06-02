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

set(compare_operator_sources
  compare_context.cpp
  compare_operator.cpp
  compare_ref_kernel.cpp)

set(compare_operator_public_headers
  compare_context.hpp
  compare_operator.hpp)

set(compare_operator_private_headers
  compare_kernel_list.hpp
  compare_ref_kernel.hpp)

target_sources(zendnnl_archive
  PRIVATE ${compare_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${compare_operator_private_headers}
  PUBLIC  FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${compare_operator_public_headers})

target_sources(zendnnl
  PRIVATE ${compare_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${compare_operator_private_headers}
  PUBLIC  FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${compare_operator_public_headers})

# target_link_libraries(zendnnl_archive
#   PUBLIC OpenMP::OpenMP_CXX
#   PUBLIC amdblis::amdblis_archive)

# target_link_libraries(zendnnl
#   PUBLIC OpenMP::OpenMP_CXX
#   PUBLIC amdblis::amdblis)

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# append header files to top level include file
list(APPEND compare_operator_public_headers_list
  ${compare_operator_public_headers})
create_header_file(${zendnnl_inc_file}
  ${inc_subdir}
  "${compare_operator_public_headers_list}")

# add_custom_target(append_compare_operator_headers ALL
#   COMMAND ${CMAKE_COMMAND}
#   -DINC_FILE_NAME=${zendnnl_inc_file}
#   -DINC_SUBDIR=${inc_subdir}
#   -DINC_FILE_LIST="${compare_operator_public_headers}"
#   -P${CMAKE_MODULE_PATH}/CreateIncludeFiles.cmake
#   BYPRODUCTS ${include_file})
