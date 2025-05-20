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

set(reorder_operator_sources
  reorder_context.cpp
  reorder_operator.cpp
  reorder_kernel.cpp)

set(reorder_operator_public_headers
  reorder_context.hpp
  reorder_operator.hpp)

set(reorder_operator_private_headers
  reorder_kernel_list.hpp
  reorder_kernel.hpp)

# add_library(reorder_operator_objects OBJECT)

# target_sources(reorder_operator_objects
#   PRIVATE    ${reorder_operator_sources}
#   PRIVATE    ${reorder_operator_private_headers}
#   INTERFACE  ${reorder_operator_public_headers})

# target_compile_options(reorder_operator_objects
#   PUBLIC "-ldl"
#   PUBLIC "-fPIC")

# target_include_directories(reorder_operator_objects
#   PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
#   PRIVATE ${CMAKE_SOURCE_DIR}/common
#   PRIVATE ${CMAKE_SOURCE_DIR}/memory
#   PRIVATE ${CMAKE_SOURCE_DIR}/operators/common)

# target_link_libraries(reorder_operator_objects
#   PUBLIC ops_common_objects
#   PUBLIC common_objects
#   PUBLIC memory_objects)

# target_link_libraries(zendnnl_archive PRIVATE reorder_operator_objects)
# target_link_libraries(zendnnl PRIVATE reorder_operator_objects)

target_sources(zendnnl_archive
  PRIVATE ${reorder_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${reorder_operator_private_headers}
  PUBLIC  FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${reorder_operator_public_headers})

target_sources(zendnnl
  PRIVATE ${reorder_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${reorder_operator_private_headers}
  PUBLIC  FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${reorder_operator_public_headers})

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# append header files to top level include file
add_custom_target(append_reorder_operator_headers ALL
  COMMAND ${CMAKE_COMMAND}
  -DINC_FILE_NAME=${zendnnl_inc_file}
  -DINC_SUBDIR=${inc_subdir}
  -DINC_FILE_LIST="${reorder_operator_public_headers}"
  -P${CMAKE_MODULE_PATH}/CreateIncludeFiles.cmake
  BYPRODUCTS ${zendnnl_inc_file})

# list(APPEND OBJECT_FILES $<TARGET_OBJECTS:reorder_operator_objects>)
# set(OBJECT_FILES ${OBJECT_FILES} PARENT_SCOPE)

# install(FILES ${reorder_operator_public_headers}
#   DESTINATION include/impl)



