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

#include "lowoha_operators/hstu_attention/common/lowoha_hstu_pad.hpp"

#include <algorithm>
#include <cstring>
#include <vector>
#include "lowoha_operators/common/omp_thread_control.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

namespace {

size_t dt_bytes(data_type_t dt) {
    return dt == data_type_t::f32 ? sizeof(float) : sizeof(uint16_t);
}

/// Grow-only per-thread buffer `i`.
char *buffer(int i, size_t bytes) {
    thread_local std::vector<char> bufs[4];
    if (bufs[i].size() < bytes) bufs[i].resize(bytes);
    return bufs[i].data();
}

/// A [rows, heads, dim] tensor with token and head strides, in bytes.
struct strided_t {
    const char *data;
    size_t st, sh, row; ///< token stride, head stride, dim * element size
};

/// Heads of token `t` of `a` to the contiguous `dst` (one copy when the
/// heads are adjacent, as in a fused projection).
void gather_heads(const strided_t &a, int64_t t, int64_t H, char *dst) {
    const char *s = a.data + size_t(t) * a.st;
    if (a.sh == a.row) {
        std::memcpy(dst, s, H * a.row);
        return;
    }
    for (int64_t h = 0; h < H; ++h)
        std::memcpy(dst + h * a.row, s + h * a.sh, a.row);
}

/// Q, K and V -> contiguous [num_seqs * max_seq_len, num_heads, dim] in one
/// pass, zero rows after each sequence.
void pad_qkv(const strided_t (&src)[3], const hstu_attention_params &p,
        int32_t nt, char *const (&dst)[3]) {
    const int64_t H = p.num_heads, N = p.max_seq_len;
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t r = 0; r < p.num_seqs * N; ++r) {
        const int64_t b = r / N, t = r % N;
        const bool real = t < p.seq_offsets[b + 1] - p.seq_offsets[b];
        for (int i = 0; i < 3; ++i) {
            char *d = dst[i] + size_t(r) * H * src[i].row;
            if (real)
                gather_heads(src[i], p.seq_offsets[b] + t, H, d);
            else
                std::memset(d, 0, H * src[i].row);
        }
    }
}

/// The real rows of the padded output to `out` (token and head strides in
/// bytes).
void unpad_out(const char *o, size_t row, const hstu_attention_params &p,
        int32_t nt, char *out, size_t st, size_t sh) {
    const int64_t H = p.num_heads, N = p.max_seq_len, B = p.num_seqs;
    const int64_t *off = p.seq_offsets;
#pragma omp parallel for schedule(static) num_threads(nt)
    for (int64_t g = off[0]; g < off[B]; ++g) {
        const int64_t b = std::upper_bound(off, off + B + 1, g) - off - 1;
        const char *s = o + size_t(b * N + g - off[b]) * H * row;
        char *d = out + size_t(g) * st;
        if (sh == row) {
            std::memcpy(d, s, H * row);
            continue;
        }
        for (int64_t h = 0; h < H; ++h)
            std::memcpy(d + h * sh, s + h * row, row);
    }
}

} // namespace

status_t hstu_attention_padded(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &p,
        const hstu_attention_run_t &run) {
    const int64_t B = p.num_seqs, H = p.num_heads, N = p.max_seq_len;
    const int64_t *off = p.seq_offsets;
    const size_t ei = dt_bytes(p.qkv_dt), eo = dt_bytes(p.out_dt);
    const int32_t nt
            = resolve_num_threads(p.num_threads, omp_get_max_threads());

    std::vector<int64_t> pad_off(B + 1), pad_targets(B);
    for (int64_t b = 0; b <= B; ++b)
        pad_off[b] = b * N;
    for (int64_t b = 0; b < B; ++b)
        pad_targets[b] = (p.num_targets ? p.num_targets[b] : 0) + N
                - (off[b + 1] - off[b]);

    const size_t rows = size_t(B * N);
    char *const qkv[3] = {buffer(0, rows * H * p.qk_dim * ei),
            buffer(1, rows * H * p.qk_dim * ei),
            buffer(2, rows * H * p.v_dim * ei)};
    char *o = buffer(3, rows * H * p.v_dim * eo);
    const strided_t src[3]
            = {{static_cast<const char *>(query), p.q_stride_t * ei,
                       p.q_stride_h * ei, p.qk_dim * ei},
                    {static_cast<const char *>(key), p.k_stride_t * ei,
                            p.k_stride_h * ei, p.qk_dim * ei},
                    {static_cast<const char *>(value), p.v_stride_t * ei,
                            p.v_stride_h * ei, p.v_dim * ei}};
    pad_qkv(src, p, nt, qkv);
    std::vector<float> scale;
    if (p.attn_scale_per_token) {
        scale.assign(rows, 0.f);
        for (int64_t b = 0; b < B; ++b)
            std::copy(p.attn_scale_per_token + off[b],
                    p.attn_scale_per_token + off[b + 1], scale.begin() + b * N);
    }

    hstu_attention_params pp = p;
    pp.seq_offsets = pad_off.data();
    pp.num_targets = pad_targets.data();
    pp.attn_scale_per_token = p.attn_scale_per_token ? scale.data() : nullptr;
    pp.q_stride_t = pp.k_stride_t = H * p.qk_dim;
    pp.q_stride_h = pp.k_stride_h = p.qk_dim;
    pp.v_stride_t = pp.o_stride_t = H * p.v_dim;
    pp.v_stride_h = pp.o_stride_h = p.v_dim;
    const status_t st = run(qkv[0], qkv[1], qkv[2], o, pp);
    if (st != status_t::success) return st;

    unpad_out(o, p.v_dim * eo, p, nt, static_cast<char *>(output),
            p.o_stride_t * eo, p.o_stride_h * eo);
    return status_t::success;
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
