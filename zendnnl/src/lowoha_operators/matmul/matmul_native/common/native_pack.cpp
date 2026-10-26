/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <omp.h>

#include "common/zendnnl_global.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::error_handling::log_error;

namespace {

bool has_layout(data_type_t wei_dt, int K, int N, int pack_nr) {
    return (wei_dt == data_type_t::bf16 || wei_dt == data_type_t::s8) && K >= 1
            && (pack_nr == kNativeMinPackNr || pack_nr == kNativeMaxPackNr)
            && N >= pack_nr && N % pack_nr == 0;
}

/// Weight column held at pack column p.
template <bool Interleave>
inline int source_col(int p, int half) {
    if constexpr (Interleave) {
        return (p & 1) ? half + (p >> 1) : (p >> 1);
    } else {
        (void)half;
        return p;
    }
}

template <bool TransB, bool Interleave>
void pack_bf16_block(const uint16_t *w, int K, int N, size_t ldb, int pack_nr,
        int nb, uint16_t *blk) {
    const int half = N / 2;
    const int p0 = nb * pack_nr;
    const int k_pairs = native_k_groups(data_type_t::bf16, K);
    for (int kp = 0; kp < k_pairs; ++kp) {
        const int k = kp * VNNI_PAIR;
        const bool pair = k + 1 < K;
        uint16_t *d = blk + static_cast<size_t>(kp) * pack_nr * VNNI_PAIR;
        for (int j = 0; j < pack_nr; ++j) {
            const size_t c
                    = static_cast<size_t>(source_col<Interleave>(p0 + j, half));
            if constexpr (TransB) {
                const uint16_t *s = w + c * ldb + k;
                d[2 * j] = s[0];
                d[2 * j + 1] = pair ? s[1] : uint16_t {0};
            } else {
                d[2 * j] = w[static_cast<size_t>(k) * ldb + c];
                d[2 * j + 1] = pair ? w[static_cast<size_t>(k + 1) * ldb + c]
                                    : uint16_t {0};
            }
        }
    }
}

template <bool TransB, bool Interleave>
void pack_s8_block(const int8_t *w, int K, int N, size_t ldb, int pack_nr,
        int nb, int8_t *blk) {
    const int half = N / 2;
    const int p0 = nb * pack_nr;
    const int k_quads = native_k_groups(data_type_t::s8, K);
    const size_t quad_stride = static_cast<size_t>(pack_nr) * INT8_VNNI_GRP;
    int32_t sums[kNativeMaxPackNr] = {};
    for (int kq = 0; kq < k_quads; ++kq) {
        const int k = kq * INT8_VNNI_GRP;
        const int rows = std::min(INT8_VNNI_GRP, K - k);
        int8_t *d = blk + kq * quad_stride;
        for (int j = 0; j < pack_nr; ++j) {
            const size_t c
                    = static_cast<size_t>(source_col<Interleave>(p0 + j, half));
            int8_t *q = d + static_cast<size_t>(j) * INT8_VNNI_GRP;
            if constexpr (TransB) {
                const int8_t *s = w + c * ldb + k;
                if (rows == INT8_VNNI_GRP) {
                    std::memcpy(q, s, INT8_VNNI_GRP);
                } else {
                    for (int r = 0; r < INT8_VNNI_GRP; ++r)
                        q[r] = r < rows ? s[r] : int8_t {0};
                }
            } else {
                for (int r = 0; r < INT8_VNNI_GRP; ++r)
                    q[r] = r < rows ? w[static_cast<size_t>(k + r) * ldb + c]
                                    : int8_t {0};
            }
            sums[j] += static_cast<int32_t>(q[0]) + q[1] + q[2] + q[3];
        }
    }
    std::memcpy(blk + k_quads * quad_stride, sums, sizeof(int32_t) * pack_nr);
}

template <bool TransB, bool Interleave>
void unpack_bf16_block(const uint16_t *blk, int K, int N, size_t ldb,
        int pack_nr, int nb, uint16_t *w) {
    const int half = N / 2;
    const int p0 = nb * pack_nr;
    const int k_pairs = native_k_groups(data_type_t::bf16, K);
    for (int kp = 0; kp < k_pairs; ++kp) {
        const int k = kp * VNNI_PAIR;
        const bool pair = k + 1 < K;
        const uint16_t *d = blk + static_cast<size_t>(kp) * pack_nr * VNNI_PAIR;
        for (int j = 0; j < pack_nr; ++j) {
            const size_t c
                    = static_cast<size_t>(source_col<Interleave>(p0 + j, half));
            if constexpr (TransB) {
                uint16_t *s = w + c * ldb + k;
                s[0] = d[2 * j];
                if (pair) s[1] = d[2 * j + 1];
            } else {
                w[static_cast<size_t>(k) * ldb + c] = d[2 * j];
                if (pair)
                    w[static_cast<size_t>(k + 1) * ldb + c] = d[2 * j + 1];
            }
        }
    }
}

template <typename Fn>
void for_each_block(int n_blocks, int num_threads, Fn fn) {
    const int nt = omp_in_parallel()
            ? 1
            : std::max(1, std::min(num_threads, n_blocks));
#pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1)
    for (int nb = 0; nb < n_blocks; ++nb)
        fn(nb);
}

template <bool TransB, bool Interleave>
void pack_typed(data_type_t wei_dt, const void *weight, int K, int N,
        size_t ldb, int pack_nr, void *dst, int num_threads) {
    const size_t block_bytes = native_block_bytes(wei_dt, K, pack_nr);
    uint8_t *out = static_cast<uint8_t *>(dst);
    if (wei_dt == data_type_t::s8) {
        const int8_t *w = static_cast<const int8_t *>(weight);
        for_each_block(N / pack_nr, num_threads, [=](int nb) {
            pack_s8_block<TransB, Interleave>(w, K, N, ldb, pack_nr, nb,
                    reinterpret_cast<int8_t *>(out + nb * block_bytes));
        });
    } else {
        const uint16_t *w = static_cast<const uint16_t *>(weight);
        for_each_block(N / pack_nr, num_threads, [=](int nb) {
            pack_bf16_block<TransB, Interleave>(w, K, N, ldb, pack_nr, nb,
                    reinterpret_cast<uint16_t *>(out + nb * block_bytes));
        });
    }
}

template <bool TransB, bool Interleave>
void unpack_typed(const void *packed, int K, int N, size_t ldb, int pack_nr,
        void *weight, int num_threads) {
    const size_t block_bytes
            = native_block_bytes(data_type_t::bf16, K, pack_nr);
    const uint8_t *in = static_cast<const uint8_t *>(packed);
    uint16_t *w = static_cast<uint16_t *>(weight);
    for_each_block(N / pack_nr, num_threads, [=](int nb) {
        unpack_bf16_block<TransB, Interleave>(
                reinterpret_cast<const uint16_t *>(in + nb * block_bytes), K, N,
                ldb, pack_nr, nb, w);
    });
}

} // namespace

size_t native_packed_bytes(data_type_t wei_dt, int K, int N, int pack_nr) {
    if (!has_layout(wei_dt, K, N, pack_nr)) return 0;
    const size_t bytes = native_block_bytes(wei_dt, K, pack_nr)
            * static_cast<size_t>(N / pack_nr);
    return (bytes + 63) & ~size_t {63};
}

status_t native_pack_weight(data_type_t wei_dt, const void *weight, int K,
        int N, int ldb, int pack_nr, bool transB, bool interleave_split_halves,
        void *dst, int num_threads) {
    if (!has_layout(wei_dt, K, N, pack_nr) || weight == nullptr
            || dst == nullptr || ldb < (transB ? K : N)) {
        log_error("native pack: no native layout for wei=",
                zendnnl::common::dtype_info(wei_dt), " K=", K, " N=", N,
                " ldb=", ldb, " pack_nr=", pack_nr, " transB=", transB,
                " weight=", weight, " dst=", dst);
        return status_t::failure;
    }
    const size_t ld = static_cast<size_t>(ldb);
    if (transB) {
        if (interleave_split_halves)
            pack_typed<true, true>(
                    wei_dt, weight, K, N, ld, pack_nr, dst, num_threads);
        else
            pack_typed<true, false>(
                    wei_dt, weight, K, N, ld, pack_nr, dst, num_threads);
    } else {
        if (interleave_split_halves)
            pack_typed<false, true>(
                    wei_dt, weight, K, N, ld, pack_nr, dst, num_threads);
        else
            pack_typed<false, false>(
                    wei_dt, weight, K, N, ld, pack_nr, dst, num_threads);
    }
    return status_t::success;
}

void native_unpack_bf16(const void *packed, int K, int N, int ldb, int pack_nr,
        bool transB, bool interleave_split_halves, void *weight,
        int num_threads) {
    const size_t ld = static_cast<size_t>(ldb);
    if (transB) {
        if (interleave_split_halves)
            unpack_typed<true, true>(
                    packed, K, N, ld, pack_nr, weight, num_threads);
        else
            unpack_typed<true, false>(
                    packed, K, N, ld, pack_nr, weight, num_threads);
    } else {
        if (interleave_split_halves)
            unpack_typed<false, true>(
                    packed, K, N, ld, pack_nr, weight, num_threads);
        else
            unpack_typed<false, false>(
                    packed, K, N, ld, pack_nr, weight, num_threads);
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
