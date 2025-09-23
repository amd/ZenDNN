# *******************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

set(ZENDNNL_DEPENDS_AOCLDLP ON CACHE BOOL "Add AOCL DLP as a dependency")
set(ZENDNNL_DEPENDS_ONEDNN  OFF CACHE BOOL "Add ONEDNN as a dependency")

set(ZENDNNL_DEPENDS_AOCLUTILS ON
  CACHE BOOL "Use aocl utils for hardware identification" FORCE)
set(ZENDNNL_DEPENDS_JSON ON
  CACHE BOOL "Use JSON script for configuration" FORCE)

cmake_dependent_option(ZENDNNL_DEPENDS_AMDBLIS "Add AMDBLIS as a dependency" OFF
  "ZENDNNL_DEPENDS_AOCLDLP" ON)

cmake_dependent_option(ZENDNNL_LOCAL_AMDBLIS "use local AMDBLIS" OFF
  "ZENDNNL_DEPENDS_AMDBLIS" OFF)
cmake_dependent_option(ZENDNNL_LOCAL_AOCLDLP "use local AOCLDLP" OFF
  "ZENDNNL_DEPENDS_AOCLDLP" OFF)
cmake_dependent_option(ZENDNNL_LOCAL_ONEDNN "use local ONEDNN" OFF
  "ZENDNNL_DEPENDS_ONEDNN" OFF)

set(ZENDNNL_LOCAL_AOCLUTILS   OFF CACHE BOOL "use local AOCLUTILS" FORCE)
set(ZENDNNL_LOCAL_JSON        OFF CACHE BOOL "use local JSON" FORCE)

# sanity check on dependencies
if((NOT ZENDNNL_DEPENDS_AMDBLIS) AND (NOT ZENDNNL_DEPENDS_AOCLDLP))
  message(FATAL_ERROR "ZenDNNL has a hard dependency on amd-blis or aocl-dlp.")
endif()

if(ZENDNNL_DEPENDS_AMDBLIS AND ZENDNNL_DEPENDS_AOCLDLP)
  message(FATAL_ERROR "Either aocl-dlp or amd-blis can be enabled but not both.")
endif()

if(NOT ZENDNNL_DEPENDS_AOCLUTILS)
  message(FATAL_ERROR "ZenDNNL has a hard dependency on aocl-utils.")
endif()

if(NOT ZENDNNL_DEPENDS_JSON)
  message(FATAL_ERROR "ZenDNNL has a hard dependency on nlohmann-json.")
endif()

# informative messages
set(ZENDNNL_MSG_DEPS "")
if(ZENDNNL_DEPENDS_AMDBLIS)
  list(APPEND ZENDNNL_MSG_DEPS "amd-blis")
endif()
if(ZENDNNL_DEPENDS_AOCLDLP)
  list(APPEND ZENDNNL_MSG_DEPS "aocl-dlp")
endif()
if(ZENDNNL_DEPENDS_ONEDNN)
  list(APPEND ZENDNNL_MSG_DEPS "onednn")
endif()
if(ZENDNNL_DEPENDS_AOCLUTILS)
  list(APPEND ZENDNNL_MSG_DEPS "aocl-utils")
endif()
if(ZENDNNL_DEPENDS_JSON)
  list(APPEND ZENDNNL_MSG_DEPS "nlohmann-json")
endif()

message(STATUS "${ZENDNNL_MSG_PREFIX}Dependencies : ${ZENDNNL_MSG_DEPS}")

