# *******************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# build archive
if(ZENDNNL_LIB_BUILD_ARCHIVE)
  add_library(zendnnl_archive STATIC)

  set_property(TARGET zendnnl_archive
    PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET zendnnl_archive
    PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)

  target_sources(zendnnl_archive
    INTERFACE FILE_SET  zendnnl_top_header
              TYPE      HEADERS
              BASE_DIRS ${CMAKE_BINARY_DIR}
              FILES     ${zendnnl_inc_file}
    PRIVATE   FILE_SET  zendnnl_pvt_headers
              TYPE      HEADERS
              BASE_DIRS ${zendnnl_src_dir}
    PUBLIC    FILE_SET  zendnnl_headers
              TYPE      HEADERS
              BASE_DIRS ${zendnnl_src_dir})

  target_include_directories(zendnnl_archive
    PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    INTERFACE "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/zendnnl/include>")

  target_link_options(zendnnl_archive
    PUBLIC "-fopenmp")

  target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_JSON=1)
  target_link_libraries(zendnnl_archive
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC ${CMAKE_DL_LIBS}
    PUBLIC nlohmann_json::nlohmann_json)

  target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_AOCLUTILS=1)
  target_link_libraries(zendnnl_archive
    PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::aoclutils>")
  # target_link_libraries(zendnnl_archive
  #   PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::au_cpuid>")

  if(ZENDNNL_DEPENDS_AOCLDLP)
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_AOCLDLP=1)
    if(ZENDNNL_AOCLDLP_INJECTED)
      target_link_libraries(zendnnl_archive
        PRIVATE "$<COMPILE_ONLY:aocldlp::aocl_dlp_static>")
    else()
      target_link_libraries(zendnnl_archive
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,aocldlp::aocl_dlp_static>")
    endif()
  else()
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_AOCLDLP=0)
  endif()

  if(ZENDNNL_DEPENDS_AMDBLIS)
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_AMDBLIS=1)
    if(ZENDNNL_AMDBLIS_INJECTED)
      target_link_libraries(zendnnl_archive
        PRIVATE "$<COMPILE_ONLY:amdblis::amdblis_archive>")
    else()
      target_link_libraries(zendnnl_archive
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,amdblis::amdblis_archive>")
    endif()
  else()
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_AMDBLIS=0)
  endif()

  if(ZENDNNL_DEPENDS_ONEDNN)
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_ONEDNN=1)
    if(ZENDNNL_ONEDNN_INJECTED)
      target_link_libraries(zendnnl_archive
        PRIVATE "$<COMPILE_ONLY:DNNL::dnnl>")
    else()
      target_link_libraries(zendnnl_archive
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,DNNL::dnnl>")
    endif()
  else()
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_ONEDNN=0)
  endif()


  if(ZENDNNL_DEPENDS_LIBXSMM)
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_LIBXSMM=1)
    if(ZENDNNL_LIBXSMM_INJECTED)
      target_link_libraries(zendnnl_archive
        PRIVATE "$<COMPILE_ONLY:libxsmm::libxsmm_archive>")
    else()
      target_link_libraries(zendnnl_archive
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,libxsmm::libxsmm_archive>")
    endif()
  else()
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_LIBXSMM=0)
  endif()

  if(ZENDNNL_DEPENDS_PARLOOPER)
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_PARLOOPER=1)
    if(ZENDNNL_PARLOOPER_INJECTED)
      target_link_libraries(zendnnl_archive
        PRIVATE "$<COMPILE_ONLY:parlooper::parlooper_archive>")
    else()
      target_link_libraries(zendnnl_archive
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,parlooper::parlooper_archive>")
    endif()
  else()
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_PARLOOPER=0)
  endif()

  if(ZENDNNL_DEPENDS_FBGEMM)
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_FBGEMM=1)
    if(ZENDNNL_FBGEMM_INJECTED)
      target_link_libraries(zendnnl_archive
        PRIVATE "$<COMPILE_ONLY:fbgemm::fbgemm_archive>")
    else()
      target_link_libraries(zendnnl_archive
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,fbgemm::fbgemm_archive>")
    endif()
  else()
    target_compile_definitions(zendnnl_archive PUBLIC ZENDNNL_DEPENDS_FBGEMM=0)
  endif()

endif()

# build shared lib
if(ZENDNNL_LIB_BUILD_SHARED)
  add_library(zendnnl SHARED)

  set_property(TARGET zendnnl
    PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET zendnnl
    PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)

  target_sources(zendnnl
    INTERFACE FILE_SET  zendnnl_top_header
              TYPE      HEADERS
              BASE_DIRS ${CMAKE_BINARY_DIR}
              FILES     ${zendnnl_inc_file}
    PRIVATE   FILE_SET  zendnnl_pvt_headers
              TYPE      HEADERS
              BASE_DIRS ${zendnnl_src_dir}
    PUBLIC    FILE_SET  zendnnl_headers
              TYPE      HEADERS
              BASE_DIRS ${zendnnl_src_dir})

  target_include_directories(zendnnl
    PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    INTERFACE "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/zendnnl/include>")

  target_link_options(zendnnl
    PUBLIC "-fopenmp")

  target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_JSON=1)
  target_link_libraries(zendnnl
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC ${CMAKE_DL_LIBS}
    PUBLIC nlohmann_json::nlohmann_json)

  target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_AOCLUTILS=1)
  target_link_libraries(zendnnl
    PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::aoclutils>")
  #  PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,au::au_cpuid>"

  if(ZENDNNL_DEPENDS_AOCLDLP)
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_AOCLDLP=1)
    if(ZENDNNL_AOCLDLP_INJECTED)
      target_link_libraries(zendnnl_archive
        PUBLIC aocldlp::aocl_dlp_static)
    else()
      target_link_libraries(zendnnl
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,aocldlp::aocl_dlp_static>")
    endif()
  else()
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_AOCLDLP=0)
  endif()

  if(ZENDNNL_DEPENDS_AMDBLIS)
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_AMDBLIS=1)
    if(ZENDNNL_AMDBLIS_INJECTED)
      target_link_libraries(zendnnl
        PUBLIC amdblis::amdblis_archive)
    else()
      target_link_libraries(zendnnl
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,amdblis::amdblis_archive>")
    endif()
  else()
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_AMDBLIS=0)
  endif()

  if(ZENDNNL_DEPENDS_ONEDNN)
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_ONEDNN=1)
    if(ZENDNNL_ONEDNN_INJECTED)
      target_link_libraries(zendnnl
        PUBLIC DNNL::dnnl)
    else()
      target_link_libraries(zendnnl
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,DNNL::dnnl")
    endif()
  else()
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_ONEDNN=0)
  endif()

  if(ZENDNNL_DEPENDS_LIBXSMM)
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_LIBXSMM=1)
    if(ZENDNNL_LIBXSMM_INJECTED)
      target_link_libraries(zendnnl
        PUBLIC libxsmm::libxsmm_archive)
    else()
      target_link_libraries(zendnnl
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,libxsmm::libxsmm_archive>")
    endif()
  else()
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_LIBXSMM=0)
  endif()

  if(ZENDNNL_DEPENDS_PARLOOPER)
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_PARLOOPER=1)
    if(ZENDNNL_PARLOOPER_INJECTED)
      target_link_libraries(zendnnl
        PUBLIC parlooper::parlooper_archive)
    else()
      target_link_libraries(zendnnl
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,parlooper::parlooper_archive>")
    endif()
  else()
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_PARLOOPER=0)
  endif()

  if(ZENDNNL_DEPENDS_FBGEMM)
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_FBGEMM=1)
    if(ZENDNNL_FBGEMM_INJECTED)
      target_link_libraries(zendnnl
        PUBLIC fbgemm::fbgemm_archive)
    else()
      target_link_libraries(zendnnl
        PUBLIC "$<LINK_LIBRARY:WHOLE_ARCHIVE,fbgemm::fbgemm_archive>")
    endif()
  else()
    target_compile_definitions(zendnnl PUBLIC ZENDNNL_DEPENDS_FBGEMM=0)
  endif()

endif()

# create top level include file
prolog_header_file(${zendnnl_inc_file})

if(ZENDNNL_DEPENDS_AMDBLIS)
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_AMDBLIS" "1")
else()
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_AMDBLIS" "0")
endif()

if(ZENDNNL_DEPENDS_AOCLDLP)
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_AOCLDLP" "1")
else()
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_AOCLDLP" "0")
endif()

if(ZENDNNL_DEPENDS_ONEDNN)
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_ONEDNN" "1")
else()
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_ONEDNN" "0")
endif()

if(ZENDNNL_DEPENDS_LIBXSMM)
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_LIBXSMM" "1")
else()
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_LIBXSMM" "0")
endif()

if(ZENDNNL_DEPENDS_PARLOOPER)
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_PARLOOPER" "1")
else()
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_PARLOOPER" "0")
endif()

if(ZENDNNL_DEPENDS_FBGEMM)
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_FBGEMM" "1")
else()
  deps_flag_header_file(${zendnnl_inc_file} "ZENDNNL_DEPENDS_FBGEMM" "0")
endif()

add_subdirectory(common)
add_subdirectory(memory)
add_subdirectory(operators)
add_subdirectory(lowoha_operators)

epilog_header_file(${zendnnl_inc_file})

if(ZENDNNL_CODE_COVERAGE)
  message(STATUS "code-coverage enabled for zendnnl source code")
  target_compile_options(zendnnl_archive PRIVATE --coverage -O0 -g)
  target_link_options(zendnnl_archive PRIVATE --coverage)
endif()


if(ZENDNNL_BUILD_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "ASAN enabled for zendnnl source code")
  set(SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer -O1)
  target_compile_options(zendnnl_archive PRIVATE ${SANITIZER_FLAGS})
  target_link_options(zendnnl_archive PRIVATE "-fsanitize=address")
endif()


# install zendnnl and create package config files
if(ZENDNNL_LIB_BUILD_ARCHIVE)
  install(TARGETS zendnnl_archive EXPORT zendnnl_targets
    DESTINATION zendnnl/lib
    FILE_SET zendnnl_top_header DESTINATION zendnnl/include
    FILE_SET zendnnl_headers DESTINATION zendnnl/include)
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  install(TARGETS zendnnl EXPORT zendnnl_targets
    DESTINATION zendnnl/lib
    FILE_SET zendnnl_top_header DESTINATION zendnnl/include
    FILE_SET zendnnl_headers DESTINATION zendnnl/include)
endif()


install(FILES "${CMAKE_MODULE_PATH}/ZenDnnlDependenciesOptions.cmake"
  DESTINATION zendnnl/lib/cmake)

install(FILES "${CMAKE_MODULE_PATH}/ZenDnnlComponentsOptions.cmake"
  DESTINATION zendnnl/lib/cmake)

install(FILES "${CMAKE_MODULE_PATH}/ZenDnnlIntegrationOptions.cmake"
  DESTINATION zendnnl/lib/cmake)

install(FILES "${CMAKE_MODULE_PATH}/ZenDnnlMacros.cmake"
  DESTINATION zendnnl/lib/cmake)

install(FILES "${CMAKE_MODULE_PATH}/FindZLAMDBLIS.cmake"
  DESTINATION zendnnl/lib/cmake)
install(FILES "${CMAKE_MODULE_PATH}/FindAOCLDLP.cmake"
  DESTINATION zendnnl/lib/cmake)
install(FILES "${CMAKE_MODULE_PATH}/FindLIBXSMM.cmake"
  DESTINATION zendnnl/lib/cmake)
install(FILES "${CMAKE_MODULE_PATH}/FindPARLOOPER.cmake"
  DESTINATION zendnnl/lib/cmake)
install(FILES "${CMAKE_MODULE_PATH}/FindFBGEMM.cmake"
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
