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

set(zendnnl_src_dir "${CMAKE_CURRENT_SOURCE_DIR}")
set(zendnnl_inc_file "${CMAKE_BINARY_DIR}/zendnnl.hpp")

add_library(zendnnl_archive STATIC)
set_property(TARGET zendnnl_archive
  PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET zendnnl_archive
  PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
target_sources(zendnnl_archive
  INTERFACE FILE_SET zendnnl_top_header TYPE HEADERS BASE_DIRS ${CMAKE_BINARY_DIR} FILES ${zendnnl_inc_file}
  PRIVATE FILE_SET zendnnl_pvt_headers TYPE HEADERS BASE_DIRS ${zendnnl_src_dir}
  PUBLIC FILE_SET zendnnl_headers TYPE HEADERS BASE_DIRS ${zendnnl_src_dir})
target_include_directories(zendnnl_archive
  PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  INTERFACE "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(zendnnl_archive
  PUBLIC ${CMAKE_DL_LIBS})

add_library(zendnnl SHARED)
set_property(TARGET zendnnl
  PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET zendnnl
  PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
target_sources(zendnnl
  INTERFACE FILE_SET zendnnl_top_header TYPE HEADERS BASE_DIRS ${CMAKE_BINARY_DIR} FILES ${zendnnl_inc_file}
  PRIVATE FILE_SET zendnnl_pvt_headers TYPE HEADERS BASE_DIRS ${zendnnl_src_dir}
  PUBLIC FILE_SET zendnnl_headers TYPE HEADERS BASE_DIRS ${zendnnl_src_dir})
target_include_directories(zendnnl
  PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  INTERFACE "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_link_libraries(zendnnl
  PUBLIC ${CMAKE_DL_LIBS})

add_subdirectory(common)
add_subdirectory(memory)
add_subdirectory(operators)

# disabling export as of now
# install(TARGETS zendnnl_archive zendnnl EXPORT zendnnl_targets
#   FILE_SET zendnnl_top_header
#   FILE_SET zendnnl_headers
#   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS zendnnl_archive zendnnl
  FILE_SET zendnnl_top_header
  FILE_SET zendnnl_headers
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#install(FILES ${zendnnl_inc_file} DESTINATION include)

