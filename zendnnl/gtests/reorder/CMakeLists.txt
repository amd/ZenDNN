# ********************************************************************************
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

# Lightweight CMakeLists invoked from the parent gtests/CMakeLists.txt
# via add_subdirectory(reorder).  Appends sources to the existing `gtests`
# target so the binary stays single-output.
#
# Layout: the reorder gtest surface was split out of the monolithic
# test_reorder.cpp into behavior-focused files (mirroring group_matmul/).
# All bodies share the one `TestReorder` fixture in reorder_test_common.hpp;
# the lone INSTANTIATE_TEST_SUITE_P lives in test_reorder_regular.cpp.
target_sources(gtests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/reorder_test_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reorder_regular.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_static_quant_dequant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dynamic_quant.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_strided_cases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_conversion.cpp
)
