#  *******************************************************************************
#  * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  *     http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  *******************************************************************************

include(ConfigOptions)

get_property(AU_INSTALL_DIR GLOBAL PROPERTY AOCLUTILSROOT)

set(AU_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${AU_INSTALL_DIR}")
set(AU_CMAKE_ARGS "${AU_CMAKE_ARGS} -DAU_BUILD_DOCS=OFF")
set(AU_CMAKE_ARGS "${AU_CMAKE_ARGS} -DAU_BUILD_TESTS=OFF")
set(AU_CMAKE_ARGS "${AU_CMAKE_ARGS} -DAU_BUILD_TYPE=Release")
set(AU_CMAKE_ARGS "${AU_CMAKE_ARGS} -DAU_BUILD_EXAMPLES=OFF")
set(AU_CMAKE_ARGS "${AU_CMAKE_ARGS} -DAU_BUILD_STATIC_LIBS=ON")
set(AU_CMAKE_ARGS "${AU_CMAKE_ARGS} -DAU_BUILD_SHARED_LIBS=ON")

ExternalProject_ADD(zendnnl_aoclutils
  SOURCE_DIR "${ZENDNNL_AOCLUTILS_DIR}"
  INSTALL_DIR "${AU_INSTALL_DIR}"
  GIT_REPOSITORY ${AOCLUTILS_GIT_REPO}
  GIT_TAG ${AOCLUTILS_GIT_TAG}
  GIT_PROGRESS ${AOCLUTILS_GIT_PROGRESS}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${AU_INSTALL_DIR} -DAU_BUILD_EXAMPLES=ON
  BUILD_COMMAND make -j8
  UPDATE_DISCONNECTED TRUE)

set(AU_INCLUDE_DIR ${AU_INSTALL_DIR}/include)
set(AU_AOCLUTILS_LIB ${AU_INSTALL_DIR}/lib/libaoclutils.so)
set(AU_AOCLUTILS_ARCHIVE_LIB ${AU_INSTALL_DIR}/lib/libaoclutils.a)
set(AU_AU_CPUID_LIB ${AU_INSTALL_DIR}/lib/libau_cpuid.so)
set(AU_AU_CPUID_ARCHIVE_LIB ${AU_INSTALL_DIR}/lib/libau_cpuid.a)

# au::aoclutils
add_library(au_aoclutils INTERFACE)
add_dependencies(au_aoclutils zendnnl_aoclutils)
target_link_libraries(au_aoclutils INTERFACE ${AU_AOCLUTILS_ARCHIVE_LIB})
set_target_properties(au_aoclutils
  PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${AU_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${AU_INCLUDE_DIR})

add_library(au::aoclutils ALIAS au_aoclutils)

# au_aoclutils_shared
add_library(au_aoclutils_shared INTERFACE)
add_dependencies(au_aoclutils_shared zendnnl_aoclutils)
target_link_libraries(au_aoclutils_shared INTERFACE ${AU_AOCLUTILS_LIB})
set_target_properties(au_aoclutils_shared
  PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${AU_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${AU_INCLUDE_DIR})

add_library(au::aoclutils_shared ALIAS au_aoclutils_shared)

# au::au_cpuid
add_library(au_au_cpuid INTERFACE)
add_dependencies(au_au_cpuid zendnnl_aoclutils)
target_link_libraries(au_au_cpuid INTERFACE ${AU_AU_CPUID_ARCHIVE_LIB})
set_target_properties(au_au_cpuid
  PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${AU_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${AU_INCLUDE_DIR})

add_library(au::au_cpuid ALIAS au_au_cpuid)

# au::au_cpuid_shared
add_library(au_au_cpuid_shared INTERFACE)
add_dependencies(au_au_cpuid_shared zendnnl_aoclutils)
target_link_libraries(au_au_cpuid_shared INTERFACE ${AU_AU_CPUID_LIB})
set_target_properties(au_au_cpuid_shared
  PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${AU_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${AU_INCLUDE_DIR})

add_library(au::au_cpuid_shared ALIAS au_au_cpuid_shared)



