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

include(CreateIncludeFiles)

# build zendnnl_archive and zendnnl library
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
  INTERFACE "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/zendnnl/include>")
target_link_libraries(zendnnl_archive
  PUBLIC OpenMP::OpenMP_CXX
  PUBLIC ${CMAKE_DL_LIBS}
  PUBLIC nlohmann_json::nlohmann_json)

if (ZENDNNL_DEPENDS_ONEDNN)
  target_link_libraries(zendnnl_archive
    PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::aoclutils>"
    PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,DNNL::dnnl>"
    PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,amdblis::amdblis_archive>")
else()
  target_link_libraries(zendnnl_archive
    PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::aoclutils>"
    PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,amdblis::amdblis_archive>")
endif()

#  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::au_cpuid>"
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
  INTERFACE "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/zendnnl/include>")
target_link_libraries(zendnnl
  PUBLIC OpenMP::OpenMP_CXX
  PUBLIC ${CMAKE_DL_LIBS})

# create top level include file
prolog_header_file(${zendnnl_inc_file})

add_subdirectory(common)
add_subdirectory(memory)
add_subdirectory(operators)

epilog_header_file(${zendnnl_inc_file})

if(ZENDNNL_CODE_COVERAGE)
  target_compile_options(zendnnl_archive PRIVATE --coverage -O0 -g)
  target_link_options(zendnnl_archive PRIVATE --coverage)
endif()

# install zendnnl and create package config files
install(TARGETS zendnnl_archive zendnnl EXPORT zendnnl_targets
  DESTINATION zendnnl/lib
  FILE_SET zendnnl_top_header DESTINATION zendnnl/include
  FILE_SET zendnnl_headers DESTINATION zendnnl/include)

install(FILES "${CMAKE_MODULE_PATH}/ZenDnnlDependenciesOptions.cmake"
  RENAME zendnnl_dependencies_options.cmake
  DESTINATION zendnnl/lib/cmake)

install(FILES "${CMAKE_MODULE_PATH}/ZenDnnlMacros.cmake"
  RENAME zendnnl_macros.cmake
  DESTINATION zendnnl/lib/cmake)

install(FILES "${CMAKE_MODULE_PATH}/FindZLAMDBLIS.cmake"
  DESTINATION zendnnl/lib/cmake)

install(EXPORT zendnnl_targets
  FILE zendnnl_targets.cmake
  NAMESPACE "zendnnl::"
  DESTINATION zendnnl/lib/cmake)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/zendnnl-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/zendnnl-config.cmake
  INSTALL_DESTINATION zendnnl/lib/cmake)

write_basic_package_version_file(
  "zendnnl-version.cmake"
  VERSION ${ZENDNNL_LIBRARY_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/zendnnl-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/zendnnl-version.cmake"
  DESTINATION zendnnl/lib/cmake)
