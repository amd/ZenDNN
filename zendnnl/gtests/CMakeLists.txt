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

  set(gtests_sources
    gtest_main.cpp
    gtest_utils.cpp
    test_matmul.cpp
    test_reorder.cpp
    test_batchmatmul.cpp
    test_embag.cpp
    test_embedding.cpp)

  set(gtests_headers
    gtest_utils.hpp)

  # Add AI gtests if enabled
  if(ZENDNNL_BUILD_AI_GTESTS)
    message(STATUS "Including AI gtests...")
    list(APPEND gtests_sources
      ai_gtests/gtest_utils_ai.cpp
      ai_gtests/test_matmul_ai.cpp
      ai_gtests/gtest_utils_batchmatmul_ai.cpp
      ai_gtests/test_batchmatmul_ai.cpp)
    list(APPEND gtests_headers
      ai_gtests/gtest_utils_ai.hpp
      ai_gtests/gtest_utils_batchmatmul_ai.hpp)
  else()
    message(STATUS "AI gtests disabled...")
  endif()

  add_executable(gtests ${gtests_sources})

  target_link_libraries(gtests
    PRIVATE zendnnl_archive
    PRIVATE OpenMP::OpenMP_CXX
    PRIVATE GTest::gtest
    PRIVATE GTest::gtest_main)

  # Link correct AOCL backend and add matching compile defs
  if(ZENDNNL_DEPENDS_AOCLDLP AND ZENDNNL_AOCLDLP_INJECTED)
      target_link_libraries(gtests
        PRIVATE aocldlp::aocl_dlp_static)
  endif()

  if(ZENDNNL_DEPENDS_AMDBLIS AND ZENDNNL_AMDBLIS_INJECTED)
      target_link_libraries(gtests
        PRIVATE amdblis::amdblis_archive)
  endif()

  # if (ZENDNNL_DEPENDS_AOCLDLP)
  #   target_link_libraries(gtests aocldlp::aocl_dlp_static)
  # elseif(ZENDNNL_DEPENDS_AMDBLIS)
  #   target_link_libraries(gtests amdblis::amdblis_archive)
  # endif()

  if(ZENDNNL_CODE_COVERAGE)
    message(STATUS "code-coverage enabled for zendnnl gtests")
    target_compile_options(gtests PRIVATE --coverage -O0 -g)
    target_link_options(gtests PRIVATE --coverage)
  endif()

  if(ZENDNNL_BUILD_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "ASAN enabled for zendnnl gtests")
    set(SANITIZER_FLAGS -fsanitize=address -fno-omit-frame-pointer -O1)
    target_compile_options(gtests PRIVATE ${SANITIZER_FLAGS})
    target_link_options(gtests PRIVATE "-fsanitize=address")
  endif()

  install(TARGETS gtests
    DESTINATION gtests)
else()
  message(STATUS "gtests build is disabled...")
endif()
