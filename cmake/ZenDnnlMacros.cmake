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

# required packages
macro(zendnnl_required_packages)
  message(DEBUG "finding required packages...")
  # find openmp
  find_package(OpenMP REQUIRED)
  # pthreads
  find_package(Threads REQUIRED)
endmacro()

# dependencies
macro(find_dependencies  _install_prefix)
  # find amdblis
  if(ZENDNNL_DEPENDS_AMDBLIS)
    set(AMDBLIS_INSTALL_DIR "${_install_prefix}/deps/amdblis")
    find_package(ZLAMDBLIS REQUIRED)
  endif()

  # find aocl utils
  if(ZENDNNL_DEPENDS_AOCLUTILS)
    set(AOCLUTILS_INSTALL_DIR "${_install_prefix}/deps/aoclutils")
    set(aocl-utils_ROOT "${AOCLUTILS_INSTALL_DIR}")
    set(aocl-utils_DIR "${aocl-utils_ROOT}/lib/CMake")
    find_package(aocl-utils REQUIRED)
    if(aocl-utils_FOUND)
      message(STATUS "Found AOCL UTILS at ${aocl-utils_ROOT}")
      include_directories(${aocl-utils_ROOT}/include)
    endif()
  endif()
endmacro()

