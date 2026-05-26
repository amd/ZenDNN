# /********************************************************************************
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

set(sdpa_sources
sdpa_benchdnn.cpp
sdpa_utils.cpp
sdpa_tensor_factory.cpp
sdpa_lowoha.cpp)

add_library(sdpa_benchdnn STATIC ${sdpa_sources})

# Link to whichever zendnnl library variant was built
if(TARGET zendnnl::zendnnl_archive)
  target_link_libraries(sdpa_benchdnn zendnnl::zendnnl_archive)
elseif(TARGET zendnnl::zendnnl)
  target_link_libraries(sdpa_benchdnn zendnnl::zendnnl)
else()
  message(FATAL_ERROR "No zendnnl library target found for sdpa_benchdnn!")
endif()
