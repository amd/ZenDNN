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

# Build-time defaults for ZENDNNL_DEPENDS_* flags. This file is NOT installed —
# at consumer time (find_package(zendnnl)) the values are baked into
# zendnnl-config.cmake via @VAR@ substitution, so consumers see the actual
# build-time configuration instead of these source defaults.

include_guard(GLOBAL)
include(CMakeDependentOption)

set(ZENDNNL_DEPENDS_AOCLDLP   ON  CACHE BOOL "Add AOCL DLP as a dependency")
set(ZENDNNL_DEPENDS_ONEDNN    ON  CACHE BOOL "Add ONEDNN as a dependency")
set(ZENDNNL_DEPENDS_LIBXSMM   ON  CACHE BOOL "Add LIBXSMM as a dependency")
set(ZENDNNL_DEPENDS_PARLOOPER OFF CACHE BOOL "Add PARLOOPER as a dependency")
set(ZENDNNL_DEPENDS_FBGEMM    ON  CACHE BOOL "Add FBGEMM as a dependency")

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
cmake_dependent_option(ZENDNNL_LOCAL_LIBXSMM "use local LIBXSMM" OFF
  "ZENDNNL_DEPENDS_LIBXSMM" OFF)
cmake_dependent_option(ZENDNNL_LOCAL_PARLOOPER "use local PARLOOPER" OFF
  "ZENDNNL_DEPENDS_PARLOOPER" OFF)
cmake_dependent_option(ZENDNNL_LOCAL_FBGEMM "use local FBGEMM" OFF
  "ZENDNNL_DEPENDS_FBGEMM" OFF)

set(ZENDNNL_LOCAL_AOCLUTILS OFF CACHE BOOL "use local AOCLUTILS")
set(ZENDNNL_LOCAL_JSON      OFF CACHE BOOL "use local JSON")
