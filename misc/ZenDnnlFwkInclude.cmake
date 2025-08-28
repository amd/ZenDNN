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

# integration framework
set(ZENDNNL_ZENTORCH_BUILD ON)

# dependencis and their injections
set(ZENDNNL_DEPENDS_AMDBLIS ON)
#set(ZENDNNL_AMDBLIS_FWK_TGT <framework target>)
#set(ZENDNNL_AMDBLIS_FWK_INSTALL_DIR <framework install dir>)

set(ZENDNNL_DEPENDS_ONEDNN  OFF)
#set(ZENDNNL_ONEDNN_FWK_TGT <framework target>)
#set(ZENDNNL_ONEDNN_FWL_INSTALL_DIR <framework install dir>)

# zendnnl components
set(ZENDNNL_BUILD_EXAMPLES OFF)
set(ZENDNNL_BUILD_GTEST OFF)
set(ZENDNNL_BUILD_DOXYGEN OFF)
set(ZENDNNL_BUILD_BENCHDNN OFF)
set(ZENDNNL_CODE_COVERAGE OFF)

# configuration variables
set(ZENDNNL_VERBOSE_MAKEFILE ON)
set(ZENDNNL_MESSAGE_LOG_LEVEL "DEBUG")
set(ZENDNNL_BUILD_TYPE "Release")
#set(ZENDNNL_INSTALL_PREFIX <zendnnl_install_prefix>)

add_subdirectory(ZenDNN)

