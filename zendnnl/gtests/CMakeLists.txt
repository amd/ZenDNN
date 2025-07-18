# ********************************************************************************
# * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

if(ZENDNNL_BUILD_GTEST)
  # find gtest
  set(GTEST_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/deps/gtest")
  set(GTest_ROOT "${GTEST_INSTALL_DIR}")
  set(GTest_DIR "${GTest_ROOT}/lib/cmake/GTest")
  find_package(GTest REQUIRED)
  if(GTest_FOUND)
    message(STATUS "GTest forund at ${GTest_ROOT}")
  endif()

  message(DEBUG "building gtests...")

  # Option to include AI gtests
  option(ZENDNNL_BUILD_AI_GTESTS "Build AI-specific Google Tests" ON)

  set(gtests_sources
    gtest_main.cpp
    gtest_utils.cpp
    test_matmul.cpp
    test_reorder.cpp)

  set(gtests_headers
    gtest_utils.hpp)

  # Add AI gtests if enabled
  if(ZENDNNL_BUILD_AI_GTESTS)
    message(STATUS "Including AI gtests...")
    list(APPEND gtests_sources
      ai_gtests/gtest_utils_ai.cpp
      ai_gtests/test_matmul_ai.cpp)
    list(APPEND gtests_headers
      ai_gtests/gtest_utils_ai.hpp)
  else()
    message(STATUS "AI gtests disabled...")
  endif()

  add_executable(gtests ${gtests_sources})

  target_link_libraries(gtests
    zendnnl_archive
    amdblis::amdblis_archive
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main)

  if(ZENDNNL_CODE_COVERAGE)
    target_compile_options(gtests PRIVATE --coverage -O0 -g)
    target_link_options(gtests PRIVATE --coverage)
  endif()

  install(TARGETS gtests
    DESTINATION gtests)
else()
  message(STATUS "gtests build is disabled...")
endif()
