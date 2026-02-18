# /********************************************************************************
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

set(matmul_sources
matmul_benchdnn.cpp
matmul_utils.cpp
matmul_tensor_factory.cpp
matmul_lowoha.cpp)

add_library(matmul_benchdnn STATIC ${matmul_sources})

# Link to whichever zendnnl library variant was built
if(TARGET zendnnl::zendnnl_archive)
  target_link_libraries(matmul_benchdnn zendnnl::zendnnl_archive)
elseif(TARGET zendnnl::zendnnl)
  target_link_libraries(matmul_benchdnn zendnnl::zendnnl)
else()
  message(FATAL_ERROR "No zendnnl library target found for matmul_benchdnn!")
endif()