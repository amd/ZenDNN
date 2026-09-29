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
#ifndef _LOWOHA_EMBAG_AVX2_KERNELS_HPP_
#define _LOWOHA_EMBAG_AVX2_KERNELS_HPP_

#include "common/op_config.hpp"

namespace zendnnl {
namespace lowoha {
namespace embag {

using namespace zendnnl::common;

// Defined in embag_avx2_fp32_bf16_utils.hpp, explicitly instantiated in
// embag_avx2_kernels.cpp.
template <typename InType, typename IndexType, typename OffsetType,
        typename OutType>
void embag_avx2_kernel(const InType *input, const float *weights,
        const IndexType *indices, const OffsetType *offsets, OutType *dst,
        int64_t width, int64_t indsz, int64_t offsz, int64_t padidx,
        bool is_weights, embag_algo_t algo, int64_t dst_stride,
        bool include_last_offset);

} //namespace embag
} //namespace lowoha
} //namespace zendnnl

#endif
