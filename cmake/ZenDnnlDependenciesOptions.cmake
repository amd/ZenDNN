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

# Consume-only sanity checks and messages for ZENDNNL_DEPENDS_*. Values come
# from Defaults.cmake (build time) or zendnnl-config.cmake @VAR@ baking (consumer).

include_guard(GLOBAL)

# Present at build time, absent in install — OPTIONAL covers both.
include(ZenDnnlDependenciesDefaults OPTIONAL)

# sanity check on dependencies
# AOCL-DLP is now an optional dependency. When disabled, AOCL-DLP backed
# kernels are not built and selecting one at runtime returns an error
# (reference-kernel fallback is planned as a follow-up).
if(NOT ZENDNNL_DEPENDS_AOCLDLP)
  message(STATUS "${ZENDNNL_MSG_PREFIX}AOCL-DLP dependency disabled; "
                 "AOCL-DLP kernels will be unavailable at runtime.")
endif()

if(NOT ZENDNNL_DEPENDS_AOCLUTILS)
  message(FATAL_ERROR "ZenDNNL has a hard dependency on aocl-utils.")
endif()

if(NOT ZENDNNL_DEPENDS_JSON)
  message(FATAL_ERROR "ZenDNNL has a hard dependency on nlohmann-json.")
endif()

# informative messages
set(ZENDNNL_MSG_DEPS "")
if(ZENDNNL_DEPENDS_AOCLDLP)
  list(APPEND ZENDNNL_MSG_DEPS "aocl-dlp")
endif()
if(ZENDNNL_DEPENDS_ONEDNN)
  list(APPEND ZENDNNL_MSG_DEPS "onednn")
endif()
if(ZENDNNL_DEPENDS_LIBXSMM)
  list(APPEND ZENDNNL_MSG_DEPS "libxsmm")
endif()
if(ZENDNNL_DEPENDS_PARLOOPER)
  list(APPEND ZENDNNL_MSG_DEPS "parlooper")
endif()
if(ZENDNNL_DEPENDS_FBGEMM)
  list(APPEND ZENDNNL_MSG_DEPS "fbgemm")
endif()
if(ZENDNNL_DEPENDS_AOCLUTILS)
  list(APPEND ZENDNNL_MSG_DEPS "aocl-utils")
endif()
if(ZENDNNL_DEPENDS_JSON)
  list(APPEND ZENDNNL_MSG_DEPS "nlohmann-json")
endif()

message(STATUS "${ZENDNNL_MSG_PREFIX}Dependencies : ${ZENDNNL_MSG_DEPS}")

