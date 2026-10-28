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

#ifndef LOWOHA_HSTU_ATTENTION_MASK_HPP
#define LOWOHA_HSTU_ATTENTION_MASK_HPP

#include <algorithm>
#include <cstdint>
#include "lowoha_operators/hstu_attention/lowoha_hstu_attention_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Valid keys of one query row in a causal mask: [lo, hi), plus the row
/// itself when `extra_diag` (the diagonal is always valid but can lie outside
/// the interval, e.g. for target rows).
struct row_band_t {
    int64_t lo;
    int64_t hi;
    bool extra_diag;
};

/**
 * @brief The HSTU valid attention mask of one sequence.
 *
 * `valid` is a literal transcription of `_get_valid_attn_mask` in
 * generative-recommenders. Positions get the id
 *
 *   id(x) = min(c > 0 ? max(0, x - c + 1) : x, max_ids),
 *   max_ids = n - t - (c > 0 ? c - 1 : 0)
 *
 * with c = contextual_seq_len and t = num_targets, so the targets share the
 * id of the first target. With d = id(i) - id(j) (|d| when not causal), key
 * j is valid for query i when i == j or d > 0; a window w additionally needs
 * d <= w unless id(i) >= max_ids - min_full_attn_seq_len; contextual rows
 * (id(i) == 0) see every key with id(j) < max_ids.
 */
class seq_mask_t {
public:
    /// `targets` < 0: no num_targets tensor (ids are not clamped).
    seq_mask_t(const hstu_attention_params &p, int64_t n, int64_t targets)
        : n_(n)
        , ctx_(p.contextual_seq_len)
        , window_(p.max_attn_len)
        , min_full_(p.max_attn_len > 0 ? p.min_full_attn_seq_len : 0)
        , causal_(p.causal)
        , clamp_(targets >= 0) {
        max_ids_
                = n - std::max<int64_t>(targets, 0) - (ctx_ > 0 ? ctx_ - 1 : 0);
    }

    int64_t id(int64_t x) const {
        const int64_t r = ctx_ > 0 ? std::max<int64_t>(0, x - ctx_ + 1) : x;
        return clamp_ ? std::min(r, max_ids_) : r;
    }

    bool valid(int64_t i, int64_t j) const {
        const int64_t ri = id(i), cj = id(j);
        int64_t d = ri - cj;
        if (!causal_ && d < 0) d = -d;
        bool ok = (i == j) || d > 0;
        if (window_ > 0)
            ok = ok
                    && (d <= window_
                            || (min_full_ > 0 && ri >= max_ids_ - min_full_));
        if (ctx_ > 0) ok = ok || (ri == 0 && cj < max_ids_);
        return ok;
    }

    /// Causal masks only: the valid keys of row i as an interval plus the
    /// diagonal. ids are non-decreasing, so {j : id(j) < r} = [0, first(r)).
    row_band_t band(int64_t i) const {
        const int64_t r = id(i);
        int64_t lo = 0, hi = first(r);
        if (ctx_ > 0 && r == 0) {
            hi = first(max_ids_);
        } else if (window_ > 0
                && !(min_full_ > 0 && r >= max_ids_ - min_full_)) {
            lo = first(r - window_);
        }
        if (hi < lo) hi = lo;
        // A history row's band ends right before the row itself.
        if (i == hi) ++hi;
        return {lo, hi, i < lo || i >= hi};
    }

private:
    /// Smallest position whose id is >= r, or n.
    int64_t first(int64_t r) const {
        if (r <= id(0)) return 0;
        if (r > max_ids_) return n_;
        return std::min(n_, r + (ctx_ > 0 ? ctx_ - 1 : 0));
    }

    int64_t n_;
    int64_t ctx_;
    int64_t window_;
    int64_t min_full_;
    bool causal_;
    bool clamp_;
    int64_t max_ids_;
};

/// Rows of sequence b: query/output rows [q_row, q_row + num_q) hold the
/// queries of positions [q_pos, q_pos + num_q); keys start at row kv_row.
struct seq_rows_t {
    int64_t len;
    int64_t targets; ///< -1 without num_targets.
    int64_t kv_row;
    int64_t q_row;
    int64_t q_pos;
    int64_t num_q;
};

inline seq_rows_t seq_rows(const hstu_attention_params &p, int64_t b) {
    seq_rows_t s;
    s.kv_row = p.seq_offsets[b];
    s.len = p.seq_offsets[b + 1] - s.kv_row;
    s.targets = p.num_targets ? p.num_targets[b] : -1;
    if (p.delta_size > 0) {
        s.q_row = b * p.delta_size;
        s.q_pos = s.len - p.delta_size;
        s.num_q = p.delta_size;
    } else {
        s.q_row = s.kv_row;
        s.q_pos = 0;
        s.num_q = s.len;
    }
    return s;
}

inline float row_scale(
        const hstu_attention_params &p, const seq_rows_t &s, int64_t pos) {
    if (p.attn_scale_per_token) return p.attn_scale_per_token[s.kv_row + pos];
    return p.has_attn_scale ? p.attn_scale
                            : 1.f / static_cast<float>(p.max_seq_len);
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_ATTENTION_MASK_HPP
