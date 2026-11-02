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

#include "lowoha_operators/hstu_attention/reference/lowoha_hstu_ref_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include "common/bfloat16.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/hstu_attention/common/lowoha_hstu_mask.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

namespace {

using zendnnl::common::bfloat16_t;

template <typename in_t, typename out_t>
void run(const in_t *q, const in_t *k, const in_t *v, out_t *out,
        const hstu_attention_params &p) {
    const int64_t B = p.num_seqs, H = p.num_heads;
    const int64_t D = p.qk_dim, Dv = p.v_dim;

    // Flattened (sequence, query row) index: rows of sequence b start at
    // row_begin[b].
    std::vector<int64_t> row_begin(B + 1, 0);
    for (int64_t b = 0; b < B; ++b)
        row_begin[b + 1] = row_begin[b] + seq_rows(p, b).num_q;
    const int64_t total = row_begin[B] * H;

    // The live ICV, not thread_guard::max_threads(): restoring the count cached
    // by the first ZenDNN call would undo a later omp_set_num_threads().
    const int32_t omp_max = omp_get_max_threads();
    const int32_t nt = resolve_num_threads(p.num_threads, omp_max);
    thread_guard guard(nt, omp_max);
    // Allocated here: std::bad_alloc cannot leave an OpenMP region.
    std::vector<float> accs(size_t(nt) * Dv);

#pragma omp parallel num_threads(nt)
    {
        float *acc = accs.data() + size_t(omp_get_thread_num()) * Dv;
#pragma omp for schedule(dynamic, 16)
        for (int64_t g = 0; g < total; ++g) {
            const int64_t h = g % H, row = g / H;
            const int64_t b
                    = std::upper_bound(row_begin.begin(), row_begin.end(), row)
                    - row_begin.begin() - 1;
            const seq_rows_t s = seq_rows(p, b);
            const seq_mask_t mask(p, s.len, s.targets);
            const int64_t r = row - row_begin[b];
            const int64_t i = s.q_pos + r;
            const float scale = row_scale(p, s, i);
            const in_t *qi
                    = q + (s.q_row + r) * p.q_stride_t + h * p.q_stride_h;

            std::fill(acc, acc + Dv, 0.f);
            for (int64_t j = 0; j < s.len; ++j) {
                if (!mask.valid(i, j)) continue;
                const in_t *kj
                        = k + (s.kv_row + j) * p.k_stride_t + h * p.k_stride_h;
                float dot = 0.f;
                for (int64_t d = 0; d < D; ++d)
                    dot += static_cast<float>(qi[d])
                            * static_cast<float>(kj[d]);
                const float x = p.alpha * dot;
                const float w = x / (1.f + std::exp(-x)) * scale;
                const in_t *vj
                        = v + (s.kv_row + j) * p.v_stride_t + h * p.v_stride_h;
                for (int64_t d = 0; d < Dv; ++d)
                    acc[d] += w * static_cast<float>(vj[d]);
            }
            out_t *oi = out + (s.q_row + r) * p.o_stride_t + h * p.o_stride_h;
            for (int64_t d = 0; d < Dv; ++d)
                oi[d] = static_cast<out_t>(acc[d]);
        }
    }
}

template <typename in_t>
status_t run_out(const void *q, const void *k, const void *v, void *out,
        const hstu_attention_params &p) {
    const auto *qt = static_cast<const in_t *>(q);
    const auto *kt = static_cast<const in_t *>(k);
    const auto *vt = static_cast<const in_t *>(v);
    if (p.out_dt == data_type_t::f32)
        run(qt, kt, vt, static_cast<float *>(out), p);
    else if (p.out_dt == data_type_t::bf16)
        run(qt, kt, vt, static_cast<bfloat16_t *>(out), p);
    else
        return status_t::unimplemented;
    return status_t::success;
}

} // namespace

status_t hstu_attention_reference(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params) {
    if (params.qkv_dt == data_type_t::f32)
        return run_out<float>(query, key, value, output, params);
    if (params.qkv_dt == data_type_t::bf16)
        return run_out<bfloat16_t>(query, key, value, output, params);
    return status_t::unimplemented;
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
