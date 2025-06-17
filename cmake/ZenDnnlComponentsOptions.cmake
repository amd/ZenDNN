# *******************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

set(ZENDNNL_BUILD_DEPS     ON CACHE BOOL "Build depedencies")
set(ZENDNNL_BUILD_EXAMPLES OFF CACHE BOOL "Build examples")
set(ZENDNNL_BUILD_GTEST    OFF CACHE BOOL "Build gtest")
set(ZENDNNL_BUILD_DOXYGEN  OFF CACHE BOOL "Build doxygen docs")

