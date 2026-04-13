# *******************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Group matmul: parallel expert dispatch, MoE post-op, shared header.
# Paths are relative to lowoha_operators/matmul/ (parent listfile).

set(lowoha_group_matmul_sources
  group_matmul/group_matmul_parallel.cpp
  group_matmul/group_matmul_moe_postop.cpp
  PARENT_SCOPE)

set(lowoha_group_matmul_public_headers
  group_matmul/group_matmul_direct.hpp
  PARENT_SCOPE)
