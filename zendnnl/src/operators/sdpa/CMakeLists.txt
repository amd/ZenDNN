# *******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

set(sdpa_encoder_sources
  sdpa_encoder_context.cpp
  sdpa_encoder_operator_impl.cpp
  sdpa_encoder_fp32_kernel.cpp)

set(sdpa_encoder_public_headers
  sdpa_encoder_context.hpp
  sdpa_encoder_operator_impl.hpp
  sdpa_encoder_operator.hpp)

set(sdpa_encoder_private_headers
  sdpa_encoder_kernel_list.hpp
  sdpa_encoder_fp32_kernel.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
    PRIVATE ${sdpa_encoder_sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${sdpa_encoder_private_headers}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${sdpa_encoder_public_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
    PRIVATE ${sdpa_encoder_sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${sdpa_encoder_private_headers}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${sdpa_encoder_public_headers})
endif()

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# append header files to top level include file
list(APPEND sdpa_encoder_public_headers_list ${sdpa_encoder_public_headers})
create_header_file(${zendnnl_inc_file}
  ${inc_subdir}
  "${sdpa_encoder_public_headers_list}")

# add_custom_target(append_sdpa_encoder_headers ALL
#   COMMAND ${CMAKE_COMMAND}
#   -DINC_FILE_NAME=${zendnnl_inc_file}
#   -DINC_SUBDIR=${inc_subdir}
#   -DINC_FILE_LIST="${sdpa_encoder_public_headers}"
#   -P${CMAKE_MODULE_PATH}/CreateIncludeFiles.cmake
#   BYPRODUCTS ${zendnnl_inc_file})

# list(APPEND OBJECT_FILES $<TARGET_OBJECTS:sdpa_encoder_objects>)
# set(OBJECT_FILES ${OBJECT_FILES} PARENT_SCOPE)

# install(FILES ${sdpa_encoder_public_headers}
#   DESTINATION include/impl)