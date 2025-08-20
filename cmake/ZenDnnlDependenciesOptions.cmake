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

set(ZENDNNL_DEPENDS_AOCLDLP   OFF CACHE BOOL "Add AOCL DLP as a dependency")
set(ZENDNNL_DEPENDS_AMDBLIS   ON CACHE BOOL "Add AMD BLIS as a dependency")
set(ZENDNNL_DEPENDS_ONEDNN    OFF CACHE BOOL "Add ONEDNN as a dependency")
set(ZENDNNL_DEPENDS_AOCLUTILS ON CACHE BOOL "Use aocl utils for hardware identification")
set(ZENDNNL_DEPENDS_JSON      ON CACHE BOOL "Use JSON script for configuration")

set(ZENDNNL_LOCAL_AOCLDLP     OFF  CACHE BOOL "use local AOCLDLP")
set(ZENDNNL_LOCAL_AMDBLIS     OFF CACHE BOOL "use local AMDBLIS")
set(ZENDNNL_LOCAL_ONEDNN      OFF CACHE BOOL "use local ONEDNN")
set(ZENDNNL_LOCAL_AOCLUTILS   OFF CACHE BOOL "use local AOCLUTILS")
set(ZENDNNL_LOCAL_JSON        OFF CACHE BOOL "use local JSON")
