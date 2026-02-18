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

set(reorder_sources
reorder_benchdnn.cpp
reorder_parser.cpp)

add_library(reorder_benchdnn STATIC ${reorder_sources})

# Link to whichever zendnnl library variant was built
if(TARGET zendnnl::zendnnl_archive)
  target_link_libraries(reorder_benchdnn zendnnl::zendnnl_archive)
elseif(TARGET zendnnl::zendnnl)
  target_link_libraries(reorder_benchdnn zendnnl::zendnnl)
else()
  message(FATAL_ERROR "No zendnnl library target found for reorder_benchdnn!")
endif()