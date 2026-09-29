/********************************************************************************
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

#include <cstdint>
#include "embag_avx2_fp32_bf16_utils.hpp"
#include "embag_avx2_kernels.hpp"

namespace zendnnl {
namespace lowoha {
namespace embag {

// Template instantiations
template void embag_avx2_kernel<float, int64_t, int64_t, float>(const float *,
        const float *, const int64_t *, const int64_t *, float *, int64_t,
        int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx2_kernel<float, int32_t, int32_t, float>(const float *,
        const float *, const int32_t *, const int32_t *, float *, int64_t,
        int64_t, int64_t, int64_t, bool, embag_algo_t, int64_t, bool);

template void embag_avx2_kernel<float, int64_t, int64_t, uint16_t>(
        const float *, const float *, const int64_t *, const int64_t *,
        uint16_t *, int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t,
        int64_t, bool);

template void embag_avx2_kernel<float, int32_t, int32_t, uint16_t>(
        const float *, const float *, const int32_t *, const int32_t *,
        uint16_t *, int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t,
        int64_t, bool);

template void embag_avx2_kernel<uint16_t, int64_t, int64_t, uint16_t>(
        const uint16_t *, const float *, const int64_t *, const int64_t *,
        uint16_t *, int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t,
        int64_t, bool);

template void embag_avx2_kernel<uint16_t, int32_t, int32_t, uint16_t>(
        const uint16_t *, const float *, const int32_t *, const int32_t *,
        uint16_t *, int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t,
        int64_t, bool);

template void embag_avx2_kernel<uint16_t, int64_t, int64_t, float>(
        const uint16_t *, const float *, const int64_t *, const int64_t *,
        float *, int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t,
        int64_t, bool);

template void embag_avx2_kernel<uint16_t, int32_t, int32_t, float>(
        const uint16_t *, const float *, const int32_t *, const int32_t *,
        float *, int64_t, int64_t, int64_t, int64_t, bool, embag_algo_t,
        int64_t, bool);

} //namespace embag
} //namespace lowoha
} //namespace zendnnl
