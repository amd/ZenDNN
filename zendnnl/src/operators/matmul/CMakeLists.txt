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

# set(matmul_operator_sources
#   matmul_context.cpp
#   matmul_operator.cpp
#   matmul_fp32_avx512_kernel.cpp
#   matmul_bf16_avx512_kernel.cpp
#   matmul_onednn_kernel.cpp)

set(matmul_operator_sources
  matmul_context.cpp
  matmul_operator.cpp
  matmul_fp32_ref_kernel.cpp
  matmul_fp32_avx512_kernel.cpp
  matmul_bf16_avx512_kernel.cpp
  matmul_aocl_utils.cpp)

set(matmul_operator_public_headers
  matmul_context.hpp
  matmul_operator.hpp
  matmul_aocl_utils.hpp)

set(matmul_operator_private_headers
  matmul_kernel_list.hpp
  matmul_fp32_ref_kernel.hpp
  matmul_fp32_avx512_kernel.hpp
  matmul_bf16_avx512_kernel.hpp)

target_sources(zendnnl_archive
  PRIVATE ${matmul_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${matmul_operator_private_headers}
  PUBLIC  FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${matmul_operator_public_headers})
target_link_options(zendnnl_archive
  PUBLIC "-fopenmp")
target_link_libraries(zendnnl_archive
  PUBLIC OpenMP::OpenMP_CXX
  PUBLIC amdblis::amdblis_archive)

target_sources(zendnnl
  PRIVATE ${matmul_operator_sources}
  PRIVATE FILE_SET zendnnl_pvt_headers
          TYPE HEADERS FILES ${matmul_operator_private_headers}
  PUBLIC  FILE_SET zendnnl_headers
          TYPE HEADERS FILES ${matmul_operator_public_headers})
target_link_options(zendnnl
  PUBLIC "-fopenmp")
target_link_libraries(zendnnl
  PUBLIC OpenMP::OpenMP_CXX
  PUBLIC amdblis::amdblis)

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
  BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

# append header files to top level include file
add_custom_target(append_matmul_operator_headers ALL
  COMMAND ${CMAKE_COMMAND}
  -DINC_FILE_NAME=${zendnnl_inc_file}
  -DINC_SUBDIR=${inc_subdir}
  -DINC_FILE_LIST="${matmul_operator_public_headers}"
  -P${CMAKE_MODULE_PATH}/CreateIncludeFiles.cmake
  BYPRODUCTS ${include_file})

# list(APPEND OBJECT_FILES $<TARGET_OBJECTS:matmul_operator_objects>)
# set(OBJECT_FILES ${OBJECT_FILES} PARENT_SCOPE)

# install(FILES ${matmul_operator_public_headers}
#   DESTINATION include/impl)
