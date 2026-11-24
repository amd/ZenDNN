/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// Weight scales as calls give them: their type and form (per tensor, per
/// channel or per group along K), and the reads and writes of one scale.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_SCALES_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_SCALES_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/ntile_flat_parallel.hpp"
#include "lowoha_operators/matmul/group_matmul/multi_expert/ntile_flat_parallel/pack/nr32_pack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace detail {

inline int64_t element_count(const std::vector<int64_t> &dims) {
    int64_t count = 1;
    for (const int64_t d : dims)
        count *= d;
    return count;
}

/// Scale `i` of `buff`, scales of type `dt` (f32, bf16 or f16), as f32.
inline float scale_at(const void *buff, const data_type_t dt, const size_t i) {
    switch (dt) {
        case data_type_t::bf16:
            return bfloat16_t::bf16_to_f32_val(
                    static_cast<const int16_t *>(buff)[i]);
        case data_type_t::f16:
            return float16_t::f16_to_f32_val(
                    static_cast<const uint16_t *>(buff)[i]);
        default: return static_cast<const float *>(buff)[i];
    }
}

/// Writes `value` as scale `i` of `buff`, scales of type `dt`.
inline void store_scale(
        void *buff, const data_type_t dt, const size_t i, const float value) {
    switch (dt) {
        case data_type_t::bf16:
            static_cast<int16_t *>(buff)[i]
                    = bfloat16_t::f32_to_bf16_val(value);
            break;
        case data_type_t::f16:
            static_cast<uint16_t *>(buff)[i] = float16_t::f32_to_f16_val(value);
            break;
        default: static_cast<float *>(buff)[i] = value;
    }
}

/// Whether `dt` is a type the real scales come in.
inline bool scale_type(const data_type_t dt) {
    return dt == data_type_t::f32 || dt == data_type_t::bf16
            || dt == data_type_t::f16;
}

/// How one projection's experts give their weight scales: each a buffer of
/// `groups` rows of N scales, or of one scale for every column.
struct scale_form_t {
    data_type_t dt = data_type_t::none;
    std::vector<int64_t> dims;
    int64_t groups = 1;
    bool per_tensor = false;
};

/// The form of weight scales `dims` of type `dt` for a `K` x `N` weight of
/// scaled format `f`; false for one a pack cannot take.  Formats whose scales
/// are not real numbers (`nr32::scale_enc`) bring their own forms here.
bool scale_form(const std::vector<int64_t> &dims, const data_type_t dt,
        const nr32::format_desc &f, const int K, const int N,
        scale_form_t &out);

/// The weight's group size for its pack: for a format that always groups its
/// scales (s4), and for another with several groups.
int pack_group_size(
        const nr32::format_desc &f, const int K, const scale_form_t &scales);

} // namespace detail
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_PACK_SCALES_HPP
