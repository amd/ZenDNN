# /********************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

set(embag_sources
embag_benchdnn.cpp
embag_utils.cpp
embag_tensor_factory.cpp
embag_lowoha.cpp)

add_library(embag_benchdnn STATIC ${embag_sources})

target_link_libraries(embag_benchdnn
  zendnnl::zendnnl_archive)