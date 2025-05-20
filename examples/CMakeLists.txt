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

set(examples_sources
  examples.cpp
  example_utils.cpp
  tensor_example.cpp
  sample_example.cpp
  matmul_example.cpp
  compare_op_example.cpp
  reorder_example.cpp)

set(example_headers
  example_utils.hpp
  tensor_example.hpp
  sample_example.hpp
  matmul_example.hpp
  compare_op_example.hpp
  reorder_example.hpp)

add_executable(examples ${examples_sources})
target_link_libraries(examples
  zendnnl_archive
  amdblis::amdblis_archive
  OpenMP::OpenMP_CXX)

install(TARGETS examples
  DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)

