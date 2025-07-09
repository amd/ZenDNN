# *******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

set(aocl_blis_matmul_operator_sources
  matmul_fp32_avx512_kernel.cpp
  matmul_bf16_avx512_kernel.cpp
  matmul_int8_avx512_kernel.cpp)

if (ZENDNNL_DEPENDS_AOCLDLP)
  list(APPEND aocl_blis_matmul_operator_sources matmul_aocl_dlp_utils.cpp)
  set(aocl_blis_matmul_operator_public_headers
    matmul_aocl_dlp_utils.hpp)
elseif(ZENDNNL_DEPENDS_AMDBLIS)
  list(APPEND aocl_blis_matmul_operator_sources matmul_aocl_blis_utils.cpp)
  set(aocl_blis_matmul_operator_public_headers
    matmul_aocl_blis_utils.hpp)
endif()

set(aocl_blis_matmul_operator_private_headers
  matmul_fp32_avx512_kernel.hpp
  matmul_bf16_avx512_kernel.hpp
  matmul_int8_avx512_kernel.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
    PRIVATE ${aocl_blis_matmul_operator_sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${aocl_blis_matmul_operator_private_headers}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${aocl_blis_matmul_operator_public_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
    PRIVATE ${aocl_blis_matmul_operator_sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${aocl_blis_matmul_operator_private_headers}
    PUBLIC  FILE_SET zendnnl_headers
            TYPE HEADERS FILES ${aocl_blis_matmul_operator_public_headers})
endif()

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# append header files to top level include file
list(APPEND aocl_blis_matmul_operator_public_headers_list
  ${aocl_blis_matmul_operator_public_headers})
create_header_file(${zendnnl_inc_file}
  ${inc_subdir}
  "${aocl_blis_matmul_operator_public_headers_list}")
# add_custom_target(append_matmul_operator_headers ALL
#   COMMAND ${CMAKE_COMMAND}
#   -DINC_FILE_NAME=${zendnnl_inc_file}
#   -DINC_SUBDIR=${inc_subdir}
#   -DINC_FILE_LIST="${aocl_blis_matmul_operator_public_headers}"
#   -P${CMAKE_MODULE_PATH}/CreateIncludeFiles.cmake
#   BYPRODUCTS ${include_file})

# list(APPEND OBJECT_FILES $<TARGET_OBJECTS:matmul_operator_objects>)
# set(OBJECT_FILES ${OBJECT_FILES} PARENT_SCOPE)

# install(FILES ${matmul_operator_public_headers}
#   DESTINATION include/impl)
