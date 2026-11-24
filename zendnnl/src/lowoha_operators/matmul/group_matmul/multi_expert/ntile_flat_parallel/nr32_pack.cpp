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

#include "nr32_pack.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/zendnnl_compat.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

namespace {

/// Bytes one column takes in one block row, and the K elements that row
/// covers.
int row_k(data_type_t wei) {
    switch (wei) {
        case data_type_t::bf16: return 2;
        case data_type_t::f16: return 1;
        case data_type_t::s8: return 4;
        case data_type_t::s4: return 8;
        default: return 0;
    }
}
int row_bytes_per_col(data_type_t wei) {
    return wei == data_type_t::f16 ? 2 : 4;
}

/// Element (k, col) of the caller's weight, as its raw bit pattern.
template <typename T>
T element(const T *w, const weight_desc &d, int k, int col) {
    const size_t ld = static_cast<size_t>(d.ldb);
    return d.trans ? w[static_cast<size_t>(col) * ld + k]
                   : w[static_cast<size_t>(k) * ld + col];
}

/// The raw nibble of s4 element (k, col).
uint8_t nibble(const int8_t *w, const weight_desc &d, int k, int col) {
    const size_t ld = static_cast<size_t>(d.ldb);
    const size_t idx = d.trans ? static_cast<size_t>(col) * ld + k
                               : static_cast<size_t>(k) * ld + col;
    const auto byte = static_cast<uint8_t>(w[idx >> 1]);
    return (idx & 1u) == 0u ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
}

int32_t s4_value(uint8_t raw) {
    return (raw & 0x08u) ? static_cast<int32_t>(raw) - 16
                         : static_cast<int32_t>(raw);
}

float scale_at(const void *scales, data_type_t dt, size_t i) {
    if (dt == data_type_t::bf16)
        return static_cast<float>(static_cast<const bfloat16_t *>(scales)[i]);
    return static_cast<const float *>(scales)[i];
}

/// `out`'s scales: `{G, N}` weight scales `scales` as a pack of `layout`
/// keeps them, `[G][n_pad]` f32 with zeros in the padded columns.
void fill_scales(const pack_layout &layout, const void *scales,
        const data_type_t dt, packed_expert &out) {
    const int N = layout.desc.N;
    out.scale.assign(static_cast<size_t>(layout.groups) * layout.n_pad, 0.0f);
    float max_scale = 0.0f;
    bool nan = false;
    for (int g = 0; g < layout.groups; ++g) {
        for (int col = 0; col < N; ++col) {
            const float v
                    = scale_at(scales, dt, static_cast<size_t>(g) * N + col);
            out.scale[static_cast<size_t>(g) * layout.n_pad + col] = v;
            nan = nan || std::isnan(v);
            max_scale = std::max(max_scale, std::fabs(v));
        }
    }
    out.max_scale = nan ? std::numeric_limits<float>::quiet_NaN() : max_scale;
}

// The 16-bit packs walk one column at a time: reading a column's row along
// K, rather than the block's rows at one K, keeps a row stride of a multiple
// of 4 KiB from mapping every read of a step to the same cache sets.

void pack_bf16(const uint16_t *w, const weight_desc &d, int n_pad,
        uint8_t *blocks, size_t bb) {
    const int K_pair = (d.K + 1) / 2;
    for (int b = 0; b < n_pad / kNr; ++b) {
        auto *dst = reinterpret_cast<uint16_t *>(blocks + b * bb);
        for (int n = 0; n < kNr; ++n) {
            const int col = b * kNr + n;
            for (int kp = 0; kp < K_pair; ++kp) {
                uint16_t *pair = dst + (static_cast<size_t>(kp) * kNr + n) * 2;
                const int k = kp * 2;
                pair[0] = col < d.N ? element(w, d, k, col) : 0;
                pair[1] = col < d.N && k + 1 < d.K ? element(w, d, k + 1, col)
                                                   : 0;
            }
        }
    }
}

void pack_f16(const uint16_t *w, const weight_desc &d, int n_pad,
        uint8_t *blocks, size_t bb) {
    for (int b = 0; b < n_pad / kNr; ++b) {
        auto *dst = reinterpret_cast<uint16_t *>(blocks + b * bb);
        for (int n = 0; n < kNr; ++n) {
            const int col = b * kNr + n;
            for (int k = 0; k < d.K; ++k) {
                dst[static_cast<size_t>(k) * kNr + n]
                        = col < d.N ? element(w, d, k, col) : 0;
            }
        }
    }
}

void pack_s8(const int8_t *w, const weight_desc &d, int n_pad, uint8_t *blocks,
        size_t bb, int32_t *comp) {
    const int K_quad = (d.K + 3) / 4;
    const int full_quads = d.K / 4;
    // A quad never straddles a scale group.
    const int group_quads = d.group_size > 0 ? d.group_size / 4 : K_quad;
    for (int b = 0; b < n_pad / kNr; ++b) {
        auto *dst = reinterpret_cast<int8_t *>(blocks + b * bb);
        for (int n = 0; n < kNr; ++n) {
            const int col = b * kNr + n;
            const auto add = [&](const int kq, const int32_t sum) {
                comp[static_cast<size_t>(kq / group_quads) * n_pad + col]
                        += sum;
            };
            int kq = 0;
            if (d.trans && col < d.N) {
                // A column's weights are one contiguous row.
                const int8_t *row = w + static_cast<size_t>(col) * d.ldb;
                for (; kq < full_quads; ++kq) {
                    int8_t *quad
                            = dst + (static_cast<size_t>(kq) * kNr + n) * 4;
                    std::memcpy(quad, row + kq * 4, 4);
                    add(kq, quad[0] + quad[1] + quad[2] + quad[3]);
                }
            }
            for (; kq < K_quad; ++kq) {
                int8_t *quad = dst + (static_cast<size_t>(kq) * kNr + n) * 4;
                int32_t sum = 0;
                for (int q = 0; q < 4; ++q) {
                    const int k = kq * 4 + q;
                    quad[q] = col < d.N && k < d.K ? element(w, d, k, col) : 0;
                    sum += quad[q];
                }
                add(kq, sum);
            }
        }
    }
}

/// The packed octet of a column whose eight raw nibbles, k = 8o..8o+7, are
/// the four bytes of `raw` in memory order (low nibble first), with the sum
/// of their values.
inline uint32_t s4_octet(uint32_t raw, int32_t &sum) {
    const uint32_t lo = raw & 0x0F0F0F0Fu; // k = 0, 2, 4, 6
    const uint32_t hi = (raw >> 4) & 0x0F0F0F0Fu; // k = 1, 3, 5, 7
    const uint32_t q0 = (lo & 0xFFu) | (((lo >> 16) & 0xFFu) << 4);
    const uint32_t q1 = (hi & 0xFFu) | (((hi >> 16) & 0xFFu) << 4);
    const uint32_t q2 = ((lo >> 8) & 0xFFu) | ((lo >> 24) << 4);
    const uint32_t q3 = ((hi >> 8) & 0xFFu) | ((hi >> 24) << 4);
    const uint32_t out
            = (q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)) ^ 0x88888888u;
    // Each stored nibble is its value plus 8.
    const uint32_t pairs = (out & 0x0F0F0F0Fu) + ((out >> 4) & 0x0F0F0F0Fu);
    sum = static_cast<int32_t>((pairs * 0x01010101u) >> 24) - 64;
    return out;
}

void pack_s4(const int8_t *w, const weight_desc &d, int n_pad, uint8_t *blocks,
        size_t bb, int32_t *comp) {
    const int K_oct = d.K / 8;
    const int oct_per_group = d.group_size / 8;
    // [N, K] rows of an even length start on a byte: a column's octets are
    // four bytes each.
    const bool rows_of_bytes = d.trans && d.ldb % 2 == 0;
    for (int b = 0; b < n_pad / kNr; ++b) {
        uint8_t *dst = blocks + b * bb;
        for (int n = 0; n < kNr; ++n) {
            const int col = b * kNr + n;
            if (rows_of_bytes || col >= d.N) {
                const uint8_t *row = col < d.N
                        ? reinterpret_cast<const uint8_t *>(w)
                                + static_cast<size_t>(col) * (d.ldb / 2)
                        : nullptr;
                for (int o = 0; o < K_oct; ++o) {
                    uint32_t raw = 0;
                    if (row != nullptr) std::memcpy(&raw, row + o * 4, 4);
                    int32_t sum = 0;
                    const uint32_t out = s4_octet(raw, sum);
                    std::memcpy(dst + (static_cast<size_t>(o) * kNr + n) * 4,
                            &out, 4);
                    comp[static_cast<size_t>(o / oct_per_group) * n_pad + col]
                            -= sum * 128;
                }
                continue;
            }
            for (int o = 0; o < K_oct; ++o) {
                uint8_t *oct = dst + (static_cast<size_t>(o) * kNr + n) * 4;
                int32_t sum = 0;
                for (int q = 0; q < 4; ++q) {
                    const uint8_t lo
                            = col < d.N ? nibble(w, d, o * 8 + q, col) : 0;
                    const uint8_t hi
                            = col < d.N ? nibble(w, d, o * 8 + 4 + q, col) : 0;
                    oct[q] = static_cast<uint8_t>(
                            (lo ^ 0x08u) | ((hi ^ 0x08u) << 4));
                    sum += s4_value(lo) + s4_value(hi);
                }
                // An octet never straddles a group.
                comp[static_cast<size_t>(o * 8 / d.group_size) * n_pad + col]
                        -= sum * 128;
            }
        }
    }
}

} // namespace

int padded_cols(const int N) {
    return (N + kNr - 1) / kNr * kNr;
}

size_t block_bytes(const weight_desc &d) {
    const int g = row_k(d.wei);
    if (g == 0) return 0;
    const size_t rows = static_cast<size_t>((d.K + g - 1) / g);
    return rows * kNr * row_bytes_per_col(d.wei);
}

bool packable(const weight_desc &d) {
    if (row_k(d.wei) == 0 || d.K <= 0 || d.N <= 0) return false;
    if (d.ldb < (d.trans ? d.K : d.N)) return false;
    if (d.wei == data_type_t::s4) {
        return d.group_size > 0 && d.K % 8 == 0 && d.group_size % 8 == 0
                && d.K % d.group_size == 0;
    }
    if (d.wei == data_type_t::s8 && d.group_size != 0) {
        return d.group_size > 0 && d.group_size % 4 == 0
                && d.K % d.group_size == 0;
    }
    return d.group_size == 0;
}

pack_layout layout_of(const weight_desc &desc) {
    pack_layout layout;
    layout.desc = desc;
    layout.n_pad = padded_cols(desc.N);
    layout.groups = desc.group_size > 0 ? desc.K / desc.group_size : 1;
    layout.block_bytes = block_bytes(desc);
    return layout;
}

size_t packed_weights::bytes() const {
    size_t total = 0;
    for (const packed_expert &e : experts) {
        if (e.blocks != nullptr) total += block_bytes * (n_pad / kNr);
        total += (e.comp != nullptr ? e.comp->size() * sizeof(int32_t) : 0)
                + e.scale.size() * sizeof(float);
    }
    return total;
}

status_t prepack(const weight_desc &desc,
        const std::vector<const void *> &weights,
        const std::vector<const void *> &scales, const data_type_t scale_dt,
        const int num_threads, packed_weights &out) {
    out = packed_weights {};
    const int E = static_cast<int>(weights.size());
    const bool quantized
            = desc.wei == data_type_t::s8 || desc.wei == data_type_t::s4;
    const bool scaled = quantized && !scales.empty();
    if (!packable(desc) || E == 0
            || (scaled
                    && (static_cast<int>(scales.size()) != E
                            || (scale_dt != data_type_t::f32
                                    && scale_dt != data_type_t::bf16)))) {
        return status_t::unimplemented;
    }
    for (int e = 0; e < E; ++e) {
        if (weights[e] == nullptr || (scaled && scales[e] == nullptr))
            return status_t::unimplemented;
    }

    packed_weights pw;
    static_cast<pack_layout &>(pw) = layout_of(desc);
    pw.experts.resize(static_cast<size_t>(E));
    const size_t expert_bytes = pw.block_bytes * (pw.n_pad / kNr);
    const size_t cols = static_cast<size_t>(pw.groups) * pw.n_pad;

    std::atomic<bool> failed {false};
    // No exception may leave the region: an allocation that fails fails the
    // pack.
#pragma omp parallel for schedule(dynamic) num_threads(std::max(num_threads, 1))
    for (int e = 0; e < E; ++e) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            if (test_api::s_pack_allocation_fails.load(
                        std::memory_order_relaxed)) {
                throw std::bad_alloc();
            }
            packed_expert &px = pw.experts[e];
            void *raw = zendnnl_aligned_alloc(64, expert_bytes);
            if (raw == nullptr) {
                failed.store(true, std::memory_order_relaxed);
                continue;
            }
            px.blocks = std::shared_ptr<uint8_t>(static_cast<uint8_t *>(raw),
                    [](uint8_t *p) { zendnnl_aligned_free(p); });
            uint8_t *blocks = px.blocks.get();
            switch (desc.wei) {
                case data_type_t::bf16:
                    pack_bf16(static_cast<const uint16_t *>(weights[e]), desc,
                            pw.n_pad, blocks, pw.block_bytes);
                    break;
                case data_type_t::f16:
                    pack_f16(static_cast<const uint16_t *>(weights[e]), desc,
                            pw.n_pad, blocks, pw.block_bytes);
                    break;
                default: {
                    auto comp = std::make_shared<std::vector<int32_t>>(cols, 0);
                    if (scaled) fill_scales(pw, scales[e], scale_dt, px);
                    if (desc.wei == data_type_t::s8) {
                        pack_s8(static_cast<const int8_t *>(weights[e]), desc,
                                pw.n_pad, blocks, pw.block_bytes, comp->data());
                    } else {
                        pack_s4(static_cast<const int8_t *>(weights[e]), desc,
                                pw.n_pad, blocks, pw.block_bytes, comp->data());
                    }
                    px.comp = std::move(comp);
                    break;
                }
            }
        } catch (const std::bad_alloc &) {
            failed.store(true, std::memory_order_relaxed);
        }
    }
    if (failed.load()) return status_t::memory_bad_size;
    out = std::move(pw);
    return status_t::success;
}

status_t with_scales(const pack_layout &layout, const packed_expert &bare,
        const void *scales, const data_type_t scale_dt, packed_expert &out) {
    if (scales == nullptr
            || (scale_dt != data_type_t::f32 && scale_dt != data_type_t::bf16)
            || (layout.desc.wei != data_type_t::s8
                    && layout.desc.wei != data_type_t::s4)) {
        return status_t::unimplemented;
    }
    out.blocks = bare.blocks;
    out.comp = bare.comp;
    fill_scales(layout, scales, scale_dt, out);
    return status_t::success;
}

void unpack(const packed_weights &pw, const int e, void *dst, const int ld) {
    const weight_desc &d = pw.desc;
    const uint8_t *blocks = pw.blocks(e);
    const size_t ldz = static_cast<size_t>(ld);
    for (int col = 0; col < d.N; ++col) {
        const uint8_t *blk = blocks + (col / kNr) * pw.block_bytes;
        const int n = col % kNr;
        for (int k = 0; k < d.K; ++k) {
            switch (d.wei) {
                case data_type_t::bf16:
                    static_cast<uint16_t *>(dst)[col * ldz + k]
                            = reinterpret_cast<const uint16_t *>(
                                    blk)[((k / 2) * kNr + n) * 2 + k % 2];
                    break;
                case data_type_t::f16:
                    static_cast<uint16_t *>(dst)[col * ldz + k]
                            = reinterpret_cast<const uint16_t *>(
                                    blk)[static_cast<size_t>(k) * kNr + n];
                    break;
                case data_type_t::s8:
                    static_cast<int8_t *>(dst)[col * ldz + k]
                            = reinterpret_cast<const int8_t *>(
                                    blk)[((k / 4) * kNr + n) * 4 + k % 4];
                    break;
                default: {
                    const int q = k % 8;
                    const uint8_t byte = blk[((k / 8) * kNr + n) * 4 + (q & 3)];
                    const uint8_t raw = static_cast<uint8_t>(
                            ((q < 4 ? byte : byte >> 4) & 0x0Fu) ^ 0x08u);
                    const size_t idx = col * ldz + k;
                    auto *out = static_cast<uint8_t *>(dst) + (idx >> 1);
                    *out = (idx & 1u) == 0u
                            ? static_cast<uint8_t>((*out & 0xF0u) | raw)
                            : static_cast<uint8_t>((*out & 0x0Fu) | (raw << 4));
                    break;
                }
            }
        }
    }
}

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
