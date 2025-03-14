# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(FetchContent)

FetchContent_Declare(fbgemm
    GIT_REPOSITORY https://github.com/pytorch/FBGEMM.git
    GIT_TAG ${FBGEMM_TAG}
    GIT_SUBMODULES_RECURSE TRUE
    SOURCE_DIR ${ZENDNN_DIR}/third_party/fbgemm
    SOURCE_SUBDIR "not-available"
)
FetchContent_MakeAvailable(fbgemm)

message(STATUS "FBGEMM source directory, ${fbgemm_SOURCE_DIR}")

execute_process(COMMAND ${GIT_EXECUTABLE} -c log.showSignature=false log --no-abbrev-commit --oneline -1 --format=%H
  WORKING_DIRECTORY ${fbgemm_SOURCE_DIR}
  RESULT_VARIABLE FBGEMM_RESULT_GITLOG
  OUTPUT_VARIABLE FBGEMM_VERSION_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT FBGEMM_RESULT_GITLOG EQUAL 0)
    message(FATAL_ERROR "FBGEMM, git log, failed")
endif()

set(FBGEMM_VERSION_HASH "${FBGEMM_VERSION_HASH}" CACHE STRING "FBGEMM_VERSION_HASH" FORCE)
message(STATUS "FBGEMM, FBGEMM_VERSION_HASH=${FBGEMM_VERSION_HASH}")
