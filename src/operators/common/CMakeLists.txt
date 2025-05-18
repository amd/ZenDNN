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
set(ops_common_sources
  post_op.cpp)

set(ops_common_headers
  operator_context.hpp
  operator.hpp
  operator_kernel.hpp
  post_op.hpp)

# add_library(ops_common_objects OBJECT)

# target_sources(ops_common_objects
#   PRIVATE    ${ops_common_sources}
#   INTERFACE  ${ops_common_headers})

# target_compile_options(ops_common_objects
#   PUBLIC "-ldl"
#   PUBLIC "-fPIC")

# target_include_directories(ops_common_objects
#   PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
#   PRIVATE ${CMAKE_SOURCE_DIR}/common
#   PRIVATE ${CMAKE_SOURCE_DIR}/memory)

# target_link_libraries(ops_common_objects
#   PUBLIC common_objects
#   PUBLIC memory_objects)

# target_link_libraries(zendnnl_archive PRIVATE ops_common_objects)
# target_link_libraries(zendnnl PRIVATE ops_common_objects)

target_sources(zendnnl_archive
  PRIVATE ${ops_common_sources}
  PUBLIC  FILE_SET zendnnl_headers TYPE HEADERS FILES ${ops_common_headers})

target_sources(zendnnl
  PRIVATE ${ops_common_sources}
  PUBLIC  FILE_SET zendnnl_headers TYPE HEADERS FILES ${ops_common_headers})

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# append header files to top level include file
add_custom_target(append_ops_common_headers ALL
  COMMAND ${CMAKE_COMMAND}
  -DINC_FILE_NAME=${zendnnl_inc_file}
  -DINC_SUBDIR=${inc_subdir}
  -DINC_FILE_LIST="${ops_common_headers}"
  -P${CMAKE_MODULE_PATH}/CreateIncludeFiles.cmake
  BYPRODUCTS ${zendnnl_inc_file})

# list(APPEND OBJECT_FILES $<TARGET_OBJECTS:ops_common_objects>)
# set(OBJECT_FILES ${OBJECT_FILES} PARENT_SCOPE)

# install(FILES ${ops_common_headers}
#   DESTINATION include/impl)
