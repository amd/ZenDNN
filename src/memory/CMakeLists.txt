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

set(memory_sources
  tensor_storage.cpp
  tensor_options.cpp
  tensor.cpp)

set(memory_headers
  tensor_storage.hpp
  tensor_options.hpp
  tensor.hpp)

# add_library(memory_objects OBJECT)

# target_sources(memory_objects
#   PRIVATE    ${memory_sources}
#   INTERFACE  ${memory_headers})

# target_compile_options(memory_objects
#   PUBLIC "-ldl"
#   PUBLIC "-fPIC")

# target_include_directories(memory_objects
#   PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
#   PRIVATE ${CMAKE_SOURCE_DIR}/common)

# target_link_libraries(memory_objects
#   PUBLIC common_objects)

# target_link_libraries(zendnnl_archive PRIVATE memory_objects)
# target_link_libraries(zendnnl PRIVATE memory_objects)

target_sources(zendnnl_archive
  PRIVATE ${memory_sources}
  PUBLIC  FILE_SET zendnnl_headers TYPE HEADERS FILES ${memory_headers})

target_sources(zendnnl
  PRIVATE ${memory_sources}
  PUBLIC  FILE_SET zendnnl_headers TYPE HEADERS FILES ${memory_headers})

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

add_custom_target(append_memory_headers ALL
  COMMAND ${CMAKE_COMMAND}
  -DINC_FILE_NAME=${zendnnl_inc_file}
  -DINC_SUBDIR=${inc_subdir}
  -DINC_FILE_LIST="${memory_headers}"
  -P${CMAKE_MODULE_PATH}/CreateIncludeFiles.cmake
  BYPRODUCTS ${zendnnl_inc_file})

# install(FILES ${memory_headers}
#   DESTINATION include/impl)

