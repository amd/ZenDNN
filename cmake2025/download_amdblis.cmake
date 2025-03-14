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

FetchContent_Declare(amdblis
    GIT_REPOSITORY https://github.com/amd/blis.git
    GIT_TAG ${AMDBLIS_TAG}
    SOURCE_DIR ${ZENDNN_DIR}/third_party/amdblis
    SOURCE_SUBDIR "not-available"
)
FetchContent_MakeAvailable(amdblis)
message(STATUS "AMDBLIS source directory, ${amdblis_SOURCE_DIR}")

execute_process(COMMAND ${GIT_EXECUTABLE} -c log.showSignature=false log --no-abbrev-commit --oneline -1 --format=%H
    WORKING_DIRECTORY ${amdblis_SOURCE_DIR}
    RESULT_VARIABLE RESULT
    OUTPUT_VARIABLE AMDBLIS_VERSION_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT RESULT EQUAL 0)
    message(FATAL_ERROR "AMDBLIS, git log, failed")
endif()
set(BLIS_VERSION_HASH "${AMDBLIS_VERSION_HASH}" CACHE STRING "BLIS_VERSION_HASH" FORCE)
message(STATUS "AMDBLIS, VERSION_HASH=${AMDBLIS_VERSION_HASH}")
