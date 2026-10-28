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

#include "lowoha_operators/hstu_attention/common/hstu_attention_plan.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

std::vector<key_range_t> key_ranges(const hstu_attention_params &p) {
    std::vector<key_range_t> need(p.num_seqs, key_range_t {0, 0});
    for (int64_t b = 0; b < p.num_seqs; ++b) {
        const seq_rows_t s = seq_rows(p, b);
        const seq_mask_t mask(p, s.len, s.targets);
        int64_t lo = s.len, hi = 0;
        for (int64_t r = 0; r < s.num_q; ++r) {
            const row_band_t band = mask.band(s.q_pos + r);
            if (band.lo < band.hi) {
                lo = std::min(lo, band.lo);
                hi = std::max(hi, band.hi);
            }
        }
        if (lo < hi) need[b] = key_range_t {lo, hi};
    }
    return need;
}

int64_t choose_block_m(const hstu_attention_params &p, int32_t threads) {
    int64_t bm = kMaxBlockM;
    for (; bm > kMinBlockM; bm /= 2) {
        int64_t tiles = 0;
        for (int64_t b = 0; b < p.num_seqs; ++b)
            tiles += (seq_rows(p, b).num_q + bm - 1) / bm;
        if (tiles * p.num_heads >= kTilesPerThread * threads) break;
    }
    return bm;
}

kv_blocks_t::kv_blocks_t(
        const hstu_attention_params &p, const std::vector<key_range_t> &need)
    : offsets_(p.seq_offsets) {
    first_.resize(p.num_seqs + 1, 0);
    for (int64_t b = 0; b < p.num_seqs; ++b) {
        const int64_t len = p.seq_offsets[b + 1] - p.seq_offsets[b];
        first_[b + 1] = first_[b] + (len + kBlockN - 1) / kBlockN;
        for (int64_t g = first_[b]; g < first_[b + 1]; ++g)
            if (need[b].lo < need[b].hi
                    && (g - first_[b]) * kBlockN < need[b].hi
                    && (g - first_[b] + 1) * kBlockN > need[b].lo)
                jobs_.push_back(g);
    }
}

work_plan_t::work_plan_t(const hstu_attention_params &p, int64_t block_m)
    : p_(p), bm_(block_m) {
    first_.resize(p.num_seqs + 1, 0);
    for (int64_t b = 0; b < p.num_seqs; ++b)
        first_[b + 1] = first_[b]
                + p.num_heads * ((seq_rows(p, b).num_q + bm_ - 1) / bm_);
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
