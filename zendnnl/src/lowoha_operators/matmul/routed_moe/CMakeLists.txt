# *******************************************************************************
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

# Routed MoE direct executor. Paths are relative to lowoha_operators/matmul/
# where the parent source list consumes this variable.
set(lowoha_routed_moe_sources
  routed_moe/routed_moe_validate.cpp
  routed_moe/routed_moe_execute.cpp
  routed_moe/routed_moe_dispatch.cpp
  PARENT_SCOPE)

set(lowoha_routed_moe_public_headers
  routed_moe/routed_moe.hpp
  PARENT_SCOPE)
