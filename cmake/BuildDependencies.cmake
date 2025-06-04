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

if(ZENDNNL_BUILD_DEPS)
  include(ExternProjAMDBLIS)
  include(ExternProjAOCLUTILS)
  include(ExternProjGTEST)
else()
  message(STATUS
    "Skipping building dependencies. Build will fail if they are not pre-built.")
endif()

# add a custom target to build only dependencies
add_custom_target(zendnnl-deps
  DEPENDS ${ZENDNNL_DEPS}
)



