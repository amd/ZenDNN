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

set(common_sources
  bfloat16.cpp
  data_types.cpp
  dynamic_module.cpp
  zendnnl_exceptions.cpp
  logging.cpp
  hash_object.cpp
  platform_info.cpp
  zendnnl_global_block.cpp)

set(common_headers
  bfloat16.hpp
  data_types.hpp
  dynamic_module.hpp
  zendnnl_exceptions.hpp
  logging.hpp
  hash_object.hpp
  platform_info.hpp
  zendnnl_global_block.hpp
  error_status.hpp
  hash_utils.hpp
  zendnnl_global.hpp)

# build local object library
# add_library(common_objects OBJECT)
# target_include_directories(common_objects
#   PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# target_compile_options(common_objects
#   PUBLIC "-ldl"
#   PUBLIC "-fPIC")
# target_sources(common_objects
#   PRIVATE ${common_sources}
#   PUBLIC  FILE_SET HEADERS FILES ${common_headers})

target_sources(zendnnl_archive
  PRIVATE ${common_sources}
  PUBLIC  FILE_SET zendnnl_headers TYPE HEADERS FILES ${common_headers})
#disbling temporarily for zentorch integration
# target_link_libraries(zendnnl_archive
#   PUBLIC au::aoclutils au::au_cpuid)

target_sources(zendnnl
  PRIVATE ${common_sources}
  PUBLIC FILE_SET zendnnl_headers TYPE HEADERS FILES ${common_headers})

# #disbling temporarily for zentorch integration
# target_link_libraries(zendnnl
#   PUBLIC au::aoclutils au::au_cpuid)

# append local library to top level libraries
# target_link_libraries(zendnnl_archive PUBLIC common_objects)
# target_link_libraries(zendnnl PUBLIC common_objects)

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

add_custom_target(append_common_headers ALL
  COMMAND ${CMAKE_COMMAND}
  -DINC_FILE_NAME="${zendnnl_inc_file}"
  -DINC_SUBDIR="common"
  -DINC_FILE_LIST="${common_headers}"
  -P${CMAKE_MODULE_PATH}/CreateIncludeFiles.cmake
  BYPRODUCTS ${zendnnl_inc_file})

# install(FILES ${common_headers}
#   DESTINATION include/impl)
