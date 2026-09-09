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

add_library(tensor_helper STATIC tensor_factory.cpp)
set_property(TARGET tensor_helper PROPERTY POSITION_INDEPENDENT_CODE ON)

target_sources(tensor_helper
  PUBLIC FILE_SET  tensor_helper_headers
         TYPE      HEADERS
         BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/.."
         FILES     tensor_factory.hpp)

target_include_directories(tensor_helper
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")

target_link_libraries(tensor_helper PUBLIC OpenMP::OpenMP_CXX)
if(TARGET zendnnl::zendnnl_archive)
  set(ZENDNNL_TENSOR_HELPER_CORE_TARGET zendnnl::zendnnl_archive)
elseif(TARGET zendnnl_archive)
  set(ZENDNNL_TENSOR_HELPER_CORE_TARGET zendnnl_archive)
elseif(TARGET zendnnl::zendnnl)
  set(ZENDNNL_TENSOR_HELPER_CORE_TARGET zendnnl::zendnnl)
else()
  set(ZENDNNL_TENSOR_HELPER_CORE_TARGET zendnnl)
endif()

target_include_directories(tensor_helper SYSTEM PUBLIC
  "$<TARGET_PROPERTY:${ZENDNNL_TENSOR_HELPER_CORE_TARGET},INTERFACE_INCLUDE_DIRECTORIES>")
target_compile_definitions(tensor_helper PUBLIC
  "$<TARGET_PROPERTY:${ZENDNNL_TENSOR_HELPER_CORE_TARGET},INTERFACE_COMPILE_DEFINITIONS>")

# Suppress warnings from public umbrella headers while compiling this helper;
# examples and benchdnn consume those same headers as system includes.
if(NOT MSVC)
  target_compile_options(tensor_helper PRIVATE -Wno-unused-function)
endif()
