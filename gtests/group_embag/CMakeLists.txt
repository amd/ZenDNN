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
# via add_subdirectory(group_embag).  Appends sources to the existing
# `gtests` target so the binary stays single-output.
#
# Layout: every group_embedding_bag-specific gtest artifact lives under
# this folder.  See group_embag/README.md for the maintainer overview.
target_sources(gtests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/group_embag_test_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_group_embag.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_group_embedding.cpp
)
