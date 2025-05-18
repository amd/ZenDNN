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

get_property(GTEST_INSTALL_DIR GLOBAL PROPERTY GTESTROOT)

ExternalProject_ADD(zendnnl_gtest
  SOURCE_DIR "${ZENDNNL_GTEST_DIR}"
  INSTALL_DIR "${GTEST_INSTALL_DIR}"
  GIT_REPOSITORY ${GTEST_GIT_REPO}
  GIT_TAG ${GTEST_GIT_TAG}
  GIT_PROGRESS ${GTEST_GIT_PROGRESS}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  BUILD_COMMAND make -j
  UPDATE_DISCONNECTED TRUE)

set(GTEST_INCLUDE_DIR ${GTEST_INSTALL_DIR}/include)
set(GTEST_GTEST_INCLUDE_DIR ${GTEST_INSTALL_DIR}/include/gtest)
set(GTEST_GMOCK_INCLUDE_DIR ${GTEST_INSTALL_DIR}/include/gmock)
set(GTEST_GTEST_LIB ${GTEST_INSTALL_DIR}/lib/libgtest.a)
set(GTEST_GTEST_MAIN_LIB ${GTEST_INSTALL_DIR}/lib/libgtest_main.a)
set(GTEST_GMOCK_LIB ${GTEST_INSTALL_DIR}/lib/libgmock.a)
set(GTEST_GTEST_MAIN_LIB ${GTEST_INSTALL_DIR}/lib/libgmock_main.a)

add_library(GTest_gtest INTERFACE)
add_dependencies(GTest_gtest zendnnl_gtest)
target_link_libraries(GTest_gtest INTERFACE ${GTEST_GTEST_LIB})
set_target_properties(GTest_gtest PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                                "${GTEST_GTEST_INCLUDE_DIR}"
                                "${GTEST_GMOCK_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES "Threads::Threads"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                                       "${GTEST_GTEST_INCLUDE_DIR}"
                                       "${GTEST_GMOCK_INCLUDE_DIR}")

add_library(Gtest::gtest ALIAS GTest_gtest)

# Create imported target GTest::gtest_main
add_library(GTest_gtest_main INTERFACE)
add_dependencies(GTest_gtest_main zendnnl_gtest)
target_link_libraries(GTest_gtest_main INTERFACE ${GTEST_GTEST_MAIN_LIB})
set_target_properties(GTest_gtest_main PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                                "${GTEST_GTEST_INCLUDE_DIR}"
                                "${GTEST_GMOCK_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES "Threads::Threads;GTest::gtest"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                                       "${GTEST_GTEST_INCLUDE_DIR}"
                                       "${GTEST_GMOCK_INCLUDE_DIR}")

add_library(GTest::gtest_main ALIAS GTest_gtest_main)

# Create imported target GTest::gmock
add_library(GTest_gmock INTERFACE)
add_dependencies(GTest_gmock zendnnl_gtest)
target_link_libraries(GTest_gmock INTERFACE ${GTEST_GMOCK_LIB})

set_target_properties(GTest_gmock PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                                "${GTEST_GTEST_INCLUDE_DIR}"
                                "${GTEST_GMOCK_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES "Threads::Threads;GTest::gtest"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                                       "${GTEST_GTEST_INCLUDE_DIR}"
                                       "${GTEST_GMOCK_INCLUDE_DIR}")

add_library(GTest::gmock ALIAS GTest_gmock)

# Create imported target GTest::gmock_main
add_library(GTest_gmock_main INTERFACE)
add_dependencies(GTest_gmock_main zendnnl_gtest)
target_link_libraries(GTest_gmock_main INTERFACE ${GTEST_GMOCK_MAIN_LIB})

set_target_properties(GTest_gmock_main PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                                "${GTEST_GTEST_INCLUDE_DIR}"
                                "${GTEST_GMOCK_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES "Threads::Threads;GTest::gmock"
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                                       "${GTEST_GTEST_INCLUDE_DIR}"
                                       "${GTEST_GMOCK_INCLUDE_DIR}")

add_library(GTest::gmock_main ALIAS GTest_gmock_main)
