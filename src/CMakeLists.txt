#*******************************************************************************
# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
#******************************************************************************/

#===============================================================================
# Copyright 2016-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

file(GLOB HEADERS_ROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/*.hpp
    )
file(GLOB HEADERS_SUBDIR
    ${PROJECT_BINARY_DIR}/inc/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/*.hpp
    )
include_directories_with_host_compiler(${CMAKE_CURRENT_SOURCE_DIR})

if (ZENDNN_LIBRARY_TYPE STREQUAL "SHARED")
    add_definitions_with_host_compiler(-DZENDNN_DLL_EXPORTS)
endif()

# propagate SRC specific flags
append(CMAKE_C_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_SRC_CCXX_FLAGS}")

# propagate no warning flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_SRC_CXX_FLAGS}")
append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_CXX_NOWARN_FLAGS}")

if(NOT ZENDNN_VERBOSE)
    add_definitions_with_host_compiler(-DDISABLE_VERBOSE)
endif()

if(ZENDNN_ENABLE_CONCURRENT_EXEC)
    add_definitions_with_host_compiler(-DZENDNN_ENABLE_CONCURRENT_EXEC)
endif()

if(ZENDNN_ENABLE_PRIMITIVE_CACHE)
    message(STATUS "Primitive cache is enabled")
else()
    add_definitions_with_host_compiler(-DZENDNN_DISABLE_PRIMITIVE_CACHE)
    message(STATUS "Primitive cache is disabled")
endif()

if(ZENDNN_ENABLE_JIT_PROFILING OR ZENDNN_ENABLE_ITT_TASKS)
    if (UNIX AND NOT APPLE)
        # Not every compiler adds -ldl automatically
        list(APPEND EXTRA_SHARED_LIBS "${CMAKE_DL_LIBS}")
    endif()
endif()

if(ZENDNN_EXPERIMENTAL)
    message(STATUS "Experimental features are enabled")
endif()

if(ZENDNN_ENABLE_ITT_TASKS AND NOT ZENDNN_CPU_RUNTIME STREQUAL "NONE")
    # Only supported for certain architectures (see src/common/CMakeLists.txt)
    if(ZENDNN_TARGET_ARCH STREQUAL "AARCH64" OR ZENDNN_TARGET_ARCH STREQUAL "X64")
        add_definitions_with_host_compiler(-DZENDNN_ENABLE_ITT_TASKS)
    endif()
endif()

if(ZENDNN_ENABLE_MAX_CPU_ISA)
    add_definitions_with_host_compiler(-DZENDNN_ENABLE_MAX_CPU_ISA)
endif()

if(ZENDNN_ENABLE_CPU_ISA_HINTS)
    add_definitions_with_host_compiler(-DZENDNN_ENABLE_CPU_ISA_HINTS)
endif()

if(WIN32)
    add_definitions_with_host_compiler(-D_WIN)
    add_definitions_with_host_compiler(-DNOMINMAX)
endif()

# Windows does not support weak/strong symbols and no guarrantees by the linker
# for out_of_memory testing to work. Not tested on macOS
if(UNIX)
    if(ZENDNN_ENABLE_MEM_DEBUG)
        add_definitions_with_host_compiler(-DZENDNN_ENABLE_MEM_DEBUG)
    endif()
endif()

add_subdirectory(common)

if(NOT ZENDNN_CPU_RUNTIME STREQUAL "NONE")
    add_subdirectory(cpu)
endif()

if(NOT ZENDNN_GPU_RUNTIME STREQUAL "NONE")
    add_subdirectory(gpu)
endif()

if(ZENDNN_WITH_SYCL)
    add_subdirectory(sycl)
endif()

get_property(LIB_DEPS GLOBAL PROPERTY ZENDNN_LIB_DEPS)
add_library(${ZENDNN_LIBRARY_NAME} ${ZENDNN_LIBRARY_TYPE}
    ${VERSION_RESOURCE_FILE} ${HEADERS_ROOT} ${HEADERS_SUBDIR} ${LIB_DEPS})
# LINK_PRIVATE for cmake 2.8.11 compatibility
target_link_libraries(${ZENDNN_LIBRARY_NAME} LINK_PRIVATE ${${ZENDNN_LIBRARY_NAME}_INTERFACE})

set_property(TARGET ${ZENDNN_LIBRARY_NAME} PROPERTY VERSION "${ZENDNN_VERSION_MAJOR}.${ZENDNN_VERSION_MINOR}")
set_property(TARGET ${ZENDNN_LIBRARY_NAME} PROPERTY SOVERSION "${ZENDNN_VERSION_MAJOR}")

target_include_directories(${ZENDNN_LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inc>
    # $<INSTALL_PREFIX> is required for compatibility with cmake 2.8
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>
    )

target_link_libraries_build(${ZENDNN_LIBRARY_NAME}
    "${EXTRA_SHARED_LIBS};${EXTRA_STATIC_LIBS}")
target_link_libraries_install(${ZENDNN_LIBRARY_NAME} "${EXTRA_SHARED_LIBS}")
if(ZENDNN_LIBRARY_TYPE STREQUAL "STATIC")
    target_link_libraries_install(${ZENDNN_LIBRARY_NAME} "${EXTRA_STATIC_LIBS}")
endif()

set(LIB_EXPORT_NAME "${ZENDNN_LIBRARY_NAME}-targets")
install(TARGETS ${ZENDNN_LIBRARY_NAME}
    EXPORT "${LIB_EXPORT_NAME}"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# If only Cmake could preserve the directory hierarchy...
foreach(header ${HEADERS_ROOT})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/")
endforeach()
foreach(header ${HEADERS_SUBDIR})
    install(FILES ${header} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/")
endforeach()

# Write version and package config files
set(LIB_CONFIG_GENERATE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(LIB_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${ZENDNN_LIBRARY_NAME}")
set(LIB_VERSION_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${ZENDNN_LIBRARY_NAME}-config-version.cmake")
set(LIB_CONFIG_FILE
    "${LIB_CONFIG_GENERATE_DIR}/${ZENDNN_LIBRARY_NAME}-config.cmake")
write_basic_package_version_file(
    "${LIB_VERSION_FILE}"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    "../cmake/config.cmake.in"
    "${LIB_CONFIG_FILE}"
    INSTALL_DESTINATION ${LIB_CONFIG_INSTALL_DIR})
install(FILES ${LIB_CONFIG_FILE} ${LIB_VERSION_FILE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})
string(TOUPPER "${ZENDNN_LIBRARY_NAME}::" LIB_NAMESPACE)
install(EXPORT ${LIB_EXPORT_NAME}
    NAMESPACE ${LIB_NAMESPACE}
    DESTINATION ${LIB_CONFIG_INSTALL_DIR})

# Install custom find modules for transitive dependencies
if(ZENDNN_CPU_THREADING_RUNTIME STREQUAL "TBB")
    if(WIN32)
        install(FILES "../cmake/win/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(APPLE)
        install(FILES "../cmake/mac/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    elseif(UNIX)
        install(FILES "../cmake/lnx/TBBConfig.cmake" RENAME "FindTBB.cmake"
            DESTINATION ${LIB_CONFIG_INSTALL_DIR})
    endif()
endif()

if(ZENDNN_GPU_RUNTIME STREQUAL "OCL")
    install(FILES
        "../cmake/FindOpenCL.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
endif()

if(ZENDNN_WITH_SYCL)
    install(FILES
        "../cmake/FindLevelZero.cmake"
        DESTINATION ${LIB_CONFIG_INSTALL_DIR})
endif()

# On Windows we need to add zendnn.dll path to CTESTCONFIG_PATH which is later
# passed to ctest and Visual Studio solutions
if(WIN32)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        foreach(BUILD_TYPE Release Debug RelWithDebInfo MinSizeRel)
            append_to_windows_path_list(CTESTCONFIG_PATH
                "${CMAKE_CURRENT_BINARY_DIR}/${BUILD_TYPE}")
        endforeach()
    else()
        append_to_windows_path_list(CTESTCONFIG_PATH
            "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    set(CTESTCONFIG_PATH "${CTESTCONFIG_PATH}" PARENT_SCOPE)
endif()
