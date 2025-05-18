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

set(sample_operator_sources
  sample_context.cpp
  sample_operator.cpp
  sample_fp32_avx512_kernel.cpp
  sample_bf16_avx512_kernel.cpp)

set(sample_operator_public_headers
  sample_context.hpp
  sample_operator.hpp)

set(sample_operator_private_headers
  sample_kernel_list.hpp
  sample_fp32_avx512_kernel.hpp
  sample_bf16_avx512_kernel.hpp)

# add_library(sample_operator_objects OBJECT)

# target_sources(sample_operator_objects
#   PRIVATE    ${sample_operator_sources}
#   PRIVATE    ${sample_operator_private_headers}
#   INTERFACE  ${sample_operator_public_headers})

# target_compile_options(sample_operator_objects
#   PUBLIC "-ldl"
#   PUBLIC "-fPIC")

# target_include_directories(sample_operator_objects
#   PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
#   PRIVATE ${CMAKE_SOURCE_DIR}/common
#   PRIVATE ${CMAKE_SOURCE_DIR}/memory
#   PRIVATE ${CMAKE_SOURCE_DIR}/operators/common)

# target_link_libraries(sample_operator_objects
#   PUBLIC ops_common_objects
#   PUBLIC common_objects
#   PUBLIC memory_objects)

# target_link_libraries(zendnnl_archive PRIVATE sample_operator_objects)
# target_link_libraries(zendnnl PRIVATE sample_operator_objects)

target_sources(zendnnl_archive
  PRIVATE ${sample_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${sample_operator_private_headers}
  PUBLIC  FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${sample_operator_public_headers})

target_sources(zendnnl
  PRIVATE ${sample_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${sample_operator_private_headers}
  PUBLIC  FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${sample_operator_public_headers})

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# append header files to top level include file
add_custom_target(append_sample_operator_headers ALL
  COMMAND ${CMAKE_COMMAND}
  -DINC_FILE_NAME=${zendnnl_inc_file}
  -DINC_SUBDIR=${inc_subdir}
  -DINC_FILE_LIST="${sample_operator_public_headers}"
  -P${CMAKE_MODULE_PATH}/CreateIncludeFiles.cmake
  BYPRODUCTS ${zendnnl_inc_file})

# list(APPEND OBJECT_FILES $<TARGET_OBJECTS:sample_operator_objects>)
# set(OBJECT_FILES ${OBJECT_FILES} PARENT_SCOPE)

# install(FILES ${sample_operator_public_headers}
#   DESTINATION include/impl)



