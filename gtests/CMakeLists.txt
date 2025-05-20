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

set(gtests_sources
    gtest_main.cpp
    gtest_utils.cpp
    test_matmul.cpp
    test_reorder.cpp)

set(gtests_headers
    gtest_utils.hpp)

add_executable(gtests ${gtests_sources})

target_link_libraries(gtests
    zendnnl_archive
    amdblis::amdblis_archive
    OpenMP::OpenMP_CXX
    GTest::gtest
    GTest::gtest_main)

install(TARGETS gtests
    DESTINATION ${CMAKE_INSTALL_PREFIX}/gtests)
