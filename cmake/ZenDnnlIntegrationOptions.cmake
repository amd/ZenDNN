# *******************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
include_guard(GLOBAL)
include(CMakeDependentOption)
include(ZenDnnlMacros)

# integration options
option(ZENDNNL_FWK_BUILD "ZenDNNL Framework Build" OFF)
cmake_dependent_option(ZENDNNL_STANDALONE_BUILD "ZenDNNL Standalone Build" ON
  "NOT ZENDNNL_FWK_BUILD" OFF)

# sanity check on integration options
if ((NOT ZENDNNL_FWK_BUILD) AND (NOT ZENDNNL_STANDALONE_BUILD))
  message(FATAL_ERROR "Neither a framework integration nor standalone build is enabled.")
endif()

# informative message
if (ZENDNNL_STANDALONE_BUILD)
  message(STATUS "${ZENDNNL_MSG_PREFIX}Building standalone zendnnl (and components).")
else()
  message(STATUS "${ZENDNNL_MSG_PREFIX}Building fwk integrated zendnnl (and components).")
endif()

# enable dependency injection
enable_dependency_injection(AMDBLIS ZENDNNL_FWK_BUILD)
enable_dependency_injection(AOCLDLP ZENDNNL_FWK_BUILD)
enable_dependency_injection(ONEDNN  ZENDNNL_FWK_BUILD)
enable_dependency_injection(LIBXSMM  ZENDNNL_FWK_BUILD)
enable_dependency_injection(PARLOOPER  ZENDNNL_FWK_BUILD)
enable_dependency_injection(FBGEMM ZENDNNL_FWK_BUILD)