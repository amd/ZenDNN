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

add_library(embag_avx512_kernels MODULE
  embag_avx512_kernels.cpp)
target_compile_options(embag_avx512_kernels
  PRIVATE "-mavx512f")
target_link_libraries(embag_avx512_kernels
  PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS embag_avx512_kernels DESTINATION lib)

add_library(embag_avx2_kernels MODULE
  embag_avx2_kernels.cpp)
target_compile_options(embag_avx2_kernels
  PRIVATE "-mavx2")
target_link_libraries(embag_avx2_kernels
  PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS embag_avx2_kernels DESTINATION lib)
