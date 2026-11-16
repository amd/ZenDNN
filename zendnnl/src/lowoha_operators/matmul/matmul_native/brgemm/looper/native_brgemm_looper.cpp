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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ph_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/f16/native_f16_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/fp32/native_fp32_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/gemv/native_gemv_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_partition.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_epilogue.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::dtype_info;
using zendnnl::common::size_of;
using zendnnl::error_handling::apilog_verbose;
using zendnnl::error_handling::apilog_verbose_enabled;
using zendnnl::error_handling::log_error;
using zendnnl::error_handling::log_info;

namespace {

/// dst[m][k] = src[k * lda + m] for a transposed bf16 or f32 source, in
/// side x side blocks, on threads threads when M * K >= par_min.
template <typename T>
void copy_transposed(const T *src, int lda, int M, int K, T *dst, int side,
        int threads, int64_t par_min) {
    const int m_blocks = ceil_div(M, side);
    const bool parallel = threads > 1 && static_cast<int64_t>(M) * K >= par_min;
#pragma omp parallel for num_threads(threads) schedule(static) if (parallel)
    for (int mbk = 0; mbk < m_blocks; ++mbk) {
        // A block ends at its start plus what is left, at most side: the
        // start plus side may pass the int bound.
        const int m0 = mbk * side;
        const int m_end = m0 + std::min(side, M - m0);
        for (int k0 = 0; k0 < K;) {
            const int k_end = k0 + std::min(side, K - k0);
            for (int m = m0; m < m_end; ++m)
                for (int k = k0; k < k_end; ++k)
                    dst[static_cast<size_t>(m) * K + k]
                            = src[static_cast<size_t>(k) * lda + m];
            k0 = k_end;
        }
    }
}

/// The lanes below lanes (0 .. 16) of a 16-lane mask.
inline __mmask16 f16_lanes(int lanes) {
    return lanes >= 16 ? static_cast<__mmask16>(0xFFFF)
                       : static_cast<__mmask16>((1u << lanes) - 1u);
}

/// Elements below which the calling thread widens an f16 source alone: it
/// takes less time than opening the team.
constexpr int64_t kF16WidenParMin = int64_t {1} << 18;

} // namespace

ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void native_widen_f16_block(const uint16_t *src, int lda, bool trans, int M,
        int K, float *dst, int ld, int rows, int side, int b) {
    // The block, and a transposed source's step of 16 lanes, end at their
    // start plus what is left: the start plus rows or 16 may pass the int
    // bound.
    const int m0 = b * rows;
    const int m_end = m0 + std::min(rows, M - m0);
    if (!trans) {
        for (int m = m0; m < m_end; ++m) {
            const uint16_t *s = src + static_cast<size_t>(m) * lda;
            float *d = dst + static_cast<size_t>(m) * ld;
            for (int k = 0; k < K; k += 16) {
                const __mmask16 lanes = f16_lanes(K - k);
                _mm512_mask_storeu_ps(d + k, lanes,
                        _mm512_cvtph_ps(
                                _mm256_maskz_loadu_epi16(lanes, s + k)));
            }
        }
        return;
    }
    alignas(64) float v[16];
    for (int k0 = 0, k_end = 0; k0 < K; k0 = k_end) {
        k_end = k0 + std::min(side, K - k0);
        for (int k = k0; k < k_end; ++k) {
            const uint16_t *s = src + static_cast<size_t>(k) * lda;
            for (int m = m0, lanes = 0; m < m_end; m += lanes) {
                lanes = std::min(16, m_end - m);
                _mm512_store_ps(v,
                        _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(
                                f16_lanes(lanes), s + m)));
                for (int i = 0; i < lanes; ++i)
                    dst[static_cast<size_t>(m + i) * ld + k] = v[i];
            }
        }
    }
}

namespace {

/// dst[m][k], rows ld apart, = element (m, k) of the f16 source widened to
/// f32 (native_widen_f16_block): in blocks of at most side rows, one block
/// per thread, on threads threads when M * K is at least par_min and
/// kF16WidenParMin.
ZENDNNL_TARGET("avx512f,avx512bw,avx512vl")
void widen_f16_rows(const uint16_t *src, int lda, bool trans, int M, int K,
        float *dst, int ld, int side, int threads, int64_t par_min) {
    const int64_t elems = static_cast<int64_t>(M) * K;
    const bool parallel = threads > 1 && M > 1
            && elems >= std::max(par_min, kF16WidenParMin);
    const int rows = parallel ? std::min(side, ceil_div(M, threads)) : side;
    const int m_blocks = ceil_div(M, rows);
#pragma omp parallel for num_threads(threads) schedule(static) if (parallel)
    for (int mbk = 0; mbk < m_blocks; ++mbk)
        native_widen_f16_block(src, lda, trans, M, K, dst, ld, rows, side, mbk);
}

/// The source rows of a BF16, FP32 or F16_PH call, elements of T, and the
/// kernels of its plan.
template <typename T, typename Kernel>
struct row_operands_t {
    const T *A = nullptr;
    int lda = 0;
    /// Kernels by row count, 1 .. the plan's mr.
    const Kernel *kern = nullptr;
    /// The copy of a transposed source, when A points to it.
    std::unique_ptr<T, native_aligned_free_t> rows;
};

/// The BF16 operands; paired, for a call whose tiles pair gated halves, the
/// kernels of a slab whose second 16 columns lie in the up run.
struct bf16_operands_t : row_operands_t<uint16_t, bf16_brgemm_fn_t> {
    const bf16_brgemm_paired_fn_t *paired = nullptr;
};

/// The FP32 operands; with dst set (an epilogue left with only the store to
/// an f32 dst), the kernels accumulate into the destination itself, rows
/// ldc apart.
struct f32_operands_t : row_operands_t<float, native_fp32_ukernel_fn_t> {
    float *dst = nullptr;
    int ldc = 0;
};

/// Bytes of the buffers of widened F16 sources that no call holds which the
/// process keeps for later calls, over all its threads.
constexpr size_t kF16KeptSourceBytes = size_t {512} << 20;

/// The buffers of widened F16 sources that no call holds, kept so that a
/// call does not map, fault in and unmap the pages of a large source; a
/// release drops them, and a buffer a call held across a release.  They
/// fill the first used slots: a buffer goes back without allocating.
struct f16_source_pool_t {
    std::mutex mutex;
    /// the buffers of the first used slots, and their bytes
    std::array<std::pair<float *, size_t>, kNativeF16KeptSourceSlots> free {};
    int used = 0;
    size_t kept = 0; ///< bytes in free
    uint64_t generation = 0; ///< releases so far
};

f16_source_pool_t &f16_source_pool() {
    // Never destroyed: a thread may still return a buffer after main.
    static f16_source_pool_t *pool = new f16_source_pool_t;
    return *pool;
}

/// The slot of the smallest kept buffer; the pool holds one.  Under the
/// pool's mutex.
int smallest_kept(const f16_source_pool_t &pool) {
    int s = 0;
    for (int i = 1; i < pool.used; ++i)
        if (pool.free[i].second < pool.free[s].second) s = i;
    return s;
}

/// Takes the buffer of slot i out of the pool.  Under the pool's mutex.
std::pair<float *, size_t> take_slot(f16_source_pool_t &pool, int i) {
    const std::pair<float *, size_t> b = pool.free[i];
    pool.kept -= b.second;
    pool.free[i] = pool.free[--pool.used];
    return b;
}

/// The F16 operands: A the call's source rows widened to f32 into rows, and
/// dst as for f32_operands_t.
struct f16_operands_t : row_operands_t<float, native_f16_ukernel_fn_t> {
    float *dst = nullptr;
    int ldc = 0;
    native_f16_source_lease_t widened;
};

/// The F16_PH operands: A the call's f16 source rows, and dst as for
/// f32_operands_t.
struct f16_ph_operands_t
    : row_operands_t<uint16_t, native_f16_ph_ukernel_fn_t> {
    float *dst = nullptr;
    int ldc = 0;
};

/// The INT8 call's operands; with direct, the INT8 kernel stores the
/// destination itself (store).
struct int8_operands_t {
    native_int8_call_t call;
    native_int8_out_t store;
    bool direct = false;
};

/// The kernel calls of tiles column tiles from n0: per K pass, per tile, per
/// row call, the slabs of the tile (one slab, or nr / 32 slabs of 32
/// columns, each inside one block at the weight columns the tile holds), so
/// a row call's source rows serve every slab of the tile in turn and a
/// pass's source rows serve every tile.  A paired tile runs each slab over
/// all its rows instead, as a tile of one slab, and keeps the mb * nr
/// accumulators of its rows until the epilogue pairs them; the slab of a
/// paired tile of 32 columns reads its second 16 columns from the up run.
/// Across K passes each tile keeps its own mb * nr accumulators.  Tail (a
/// weight of padded columns): the epilogue of a tile takes only its columns
/// below N, and a tile past N is skipped.
template <bool Tail>
void bf16_block(const native_plan_t &p, const bf16_operands_t &op,
        const native_weight_view_t &w, const native_epilogue_t &ep, int mb,
        int K, int m0, int n0, int tiles, float *acc) {
    const int b_stride = w.group_stride();
    const int slab = native_bf16_slab_cols(p.nr);
    const int slabs = p.nr / slab;
    assert(slabs <= kNativeBf16MaxTileCols / kNativeBf16SlabCols);
    const bool one_pass = p.kc >= K;
    for (native_k_passes_t pass {K, p.kc}; pass.next();) {
        const int k0 = pass.k0, kl = pass.kl;
        const float beta = k0 == 0 ? 0.0f : 1.0f;
        const size_t k_off = static_cast<size_t>(k0 / VNNI_PAIR) * b_stride;
        for (int t = 0; t < tiles; ++t) {
            const int n = n0 + t * p.nr;
            const int cols = Tail ? std::min(p.nr, w.N - n) : p.nr;
            if (Tail && cols <= 0) continue;
            const uint16_t
                    *slab_b[kNativeBf16MaxTileCols / kNativeBf16SlabCols];
            for (int s = 0; s < slabs; ++s)
                slab_b[s] = reinterpret_cast<const uint16_t *>(
                                    w.column(w.tile_col(n, p.nr, s * slab)))
                        + k_off;
            float *tile_acc
                    = one_pass ? acc : acc + static_cast<size_t>(t) * mb * p.nr;
            if (w.paired) {
                for (int s = 0; s < slabs; ++s) {
                    const size_t second
                            = w.column_distance(w.tile_col(n, p.nr, s * slab),
                                      w.tile_col(n, p.nr, s * slab + 16))
                            / 2;
                    for (int ir = 0; ir < mb; ir += p.mr) {
                        const int mr = std::min(p.mr, mb - ir);
                        const uint16_t *a = op.A
                                + static_cast<size_t>(m0 + ir) * op.lda + k0;
                        float *c = tile_acc + static_cast<size_t>(ir) * p.nr
                                + s * slab;
                        if (second == 16 * VNNI_PAIR)
                            op.kern[mr](a, op.lda, slab_b[s], b_stride, c, p.nr,
                                    kl, kl, beta, nullptr, fused_postop_t::none,
                                    nullptr, 0);
                        else
                            op.paired[mr](a, op.lda, slab_b[s], b_stride,
                                    second, c, p.nr, kl, kl, beta, nullptr,
                                    fused_postop_t::none, nullptr, 0);
                    }
                }
                if (k0 + kl == K)
                    for (int ir = 0; ir < mb; ir += p.mr)
                        apply_epilogue_tile(ep,
                                tile_acc + static_cast<size_t>(ir) * p.nr, p.nr,
                                m0 + ir, std::min(p.mr, mb - ir), n, p.nr);
                continue;
            }
            for (int ir = 0; ir < mb; ir += p.mr) {
                const int mr = std::min(p.mr, mb - ir);
                float *c = one_pass ? tile_acc
                                    : tile_acc + static_cast<size_t>(ir) * p.nr;
                const uint16_t *a
                        = op.A + static_cast<size_t>(m0 + ir) * op.lda + k0;
                for (int s = 0; s < slabs; ++s)
                    op.kern[mr](a, op.lda, slab_b[s], b_stride, c + s * slab,
                            p.nr, kl, kl, beta, nullptr, fused_postop_t::none,
                            nullptr, 0);
                if (k0 + kl == K)
                    apply_epilogue_tile(
                            ep, c, p.nr, m0 + ir, mr, n, Tail ? cols : p.nr);
            }
        }
    }
}

/// The kernel calls of tiles column tiles from n0: per K pass, per tile, per
/// row call, one call over the tile's nr columns, so a pass's source rows
/// serve every tile.  Across K passes each tile keeps its own mb * nr
/// accumulators, in acc or, with op.dst, in the destination.  Tail as
/// bf16_block; a tile of fewer columns than nr accumulates in acc, whatever
/// op.dst.
template <bool Tail>
void f32_block(const native_plan_t &p, const f32_operands_t &op,
        const native_weight_view_t &w, const native_epilogue_t &ep, int mb,
        int K, int m0, int n0, int tiles, float *acc) {
    const int b_stride = w.group_stride();
    const bool one_pass = p.kc >= K;
    for (native_k_passes_t pass {K, p.kc}; pass.next();) {
        const int k0 = pass.k0, kl = pass.kl;
        const float beta = k0 == 0 ? 0.0f : 1.0f;
        const size_t k_off = static_cast<size_t>(k0) * b_stride;
        for (int t = 0; t < tiles; ++t) {
            const int n = n0 + t * p.nr;
            const int cols = Tail ? std::min(p.nr, w.N - n) : p.nr;
            if (Tail && cols <= 0) continue;
            const float *b
                    = reinterpret_cast<const float *>(w.block(n / w.pack_nr))
                    + n % w.pack_nr + k_off;
            float *tile_acc
                    = one_pass ? acc : acc + static_cast<size_t>(t) * mb * p.nr;
            for (int ir = 0; ir < mb; ir += p.mr) {
                const int mr = std::min(p.mr, mb - ir);
                const float *a
                        = op.A + static_cast<size_t>(m0 + ir) * op.lda + k0;
                if (op.dst != nullptr && (!Tail || cols == p.nr)) {
                    op.kern[mr](a, op.lda, b, b_stride,
                            op.dst + static_cast<size_t>(m0 + ir) * op.ldc + n,
                            op.ldc, kl, kl, beta);
                    continue;
                }
                float *c = one_pass ? tile_acc
                                    : tile_acc + static_cast<size_t>(ir) * p.nr;
                op.kern[mr](a, op.lda, b, b_stride, c, p.nr, kl, kl, beta);
                if (k0 + kl == K)
                    apply_epilogue_tile(
                            ep, c, p.nr, m0 + ir, mr, n, Tail ? cols : p.nr);
            }
        }
    }
}

/// f32_block over an f16 weight, from the widened source rows; its K
/// passes step without passing K.
template <bool Tail>
void f16_block(const native_plan_t &p, const f16_operands_t &op,
        const native_weight_view_t &w, const native_epilogue_t &ep, int mb,
        int K, int m0, int n0, int tiles, float *acc) {
    const int b_stride = w.group_stride();
    const bool one_pass = p.kc >= K;
    for (int k0 = 0, kl = 0; k0 < K; k0 += kl) {
        kl = std::min(p.kc, K - k0);
        const float beta = k0 == 0 ? 0.0f : 1.0f;
        const size_t k_off = static_cast<size_t>(k0) * b_stride;
        for (int t = 0; t < tiles; ++t) {
            const int n = n0 + t * p.nr;
            const int cols = Tail ? std::min(p.nr, w.N - n) : p.nr;
            if (Tail && cols <= 0) continue;
            const uint16_t *b
                    = reinterpret_cast<const uint16_t *>(w.block(n / w.pack_nr))
                    + n % w.pack_nr + k_off;
            float *tile_acc
                    = one_pass ? acc : acc + static_cast<size_t>(t) * mb * p.nr;
            for (int ir = 0; ir < mb; ir += p.mr) {
                const int mr = std::min(p.mr, mb - ir);
                const float *a
                        = op.A + static_cast<size_t>(m0 + ir) * op.lda + k0;
                if (op.dst != nullptr && (!Tail || cols == p.nr)) {
                    op.kern[mr](a, op.lda, b, b_stride,
                            op.dst + static_cast<size_t>(m0 + ir) * op.ldc + n,
                            op.ldc, kl, kl, beta);
                    continue;
                }
                float *c = one_pass ? tile_acc
                                    : tile_acc + static_cast<size_t>(ir) * p.nr;
                op.kern[mr](a, op.lda, b, b_stride, c, p.nr, kl, kl, beta);
                if (k0 + kl == K)
                    apply_epilogue_tile(
                            ep, c, p.nr, m0 + ir, mr, n, Tail ? cols : p.nr);
            }
        }
    }
}

/// f16_block on the F16_PH kernels, from the call's f16 source rows: the
/// plan's K passes are whole blocks of the kernels' f16 accumulation, so
/// each block covers the same K whatever the plan.  A tile of 64 columns
/// reads its second 32 columns from the next pack block.
template <bool Tail>
void f16_ph_block(const native_plan_t &p, const f16_ph_operands_t &op,
        const native_weight_view_t &w, const native_epilogue_t &ep, int mb,
        int K, int m0, int n0, int tiles, float *acc) {
    const int b_stride = w.group_stride();
    const size_t b_next = w.block_bytes / sizeof(uint16_t);
    const bool one_pass = p.kc >= K;
    for (int k0 = 0, kl = 0; k0 < K; k0 += kl) {
        kl = std::min(p.kc, K - k0);
        const float beta = k0 == 0 ? 0.0f : 1.0f;
        const size_t k_off = static_cast<size_t>(k0) * b_stride;
        for (int t = 0; t < tiles; ++t) {
            const int n = n0 + t * p.nr;
            const int cols = Tail ? std::min(p.nr, w.N - n) : p.nr;
            if (Tail && cols <= 0) continue;
            const uint16_t *b
                    = reinterpret_cast<const uint16_t *>(w.block(n / w.pack_nr))
                    + n % w.pack_nr + k_off;
            float *tile_acc
                    = one_pass ? acc : acc + static_cast<size_t>(t) * mb * p.nr;
            for (int ir = 0; ir < mb; ir += p.mr) {
                const int mr = std::min(p.mr, mb - ir);
                const uint16_t *a
                        = op.A + static_cast<size_t>(m0 + ir) * op.lda + k0;
                if (op.dst != nullptr && (!Tail || cols == p.nr)) {
                    op.kern[mr](a, op.lda, b, b_stride, b_next,
                            op.dst + static_cast<size_t>(m0 + ir) * op.ldc + n,
                            op.ldc, kl, beta);
                    continue;
                }
                float *c = one_pass ? tile_acc
                                    : tile_acc + static_cast<size_t>(ir) * p.nr;
                op.kern[mr](a, op.lda, b, b_stride, b_next, c, p.nr, kl, beta);
                if (k0 + kl == K)
                    apply_epilogue_tile(
                            ep, c, p.nr, m0 + ir, mr, n, Tail ? cols : p.nr);
            }
        }
    }
}

/// The INT8 tiles of tiles column tiles from n0, each over all of K, stored
/// by the epilogue unless direct stores them.  Tail as bf16_block; a tile of
/// fewer columns than nr computes them into acc (compute_int8_tail_tile) and
/// the epilogue stores them, whatever direct.
template <bool Tail>
void int8_block(const native_plan_t &p, const native_int8_call_t &call,
        const native_weight_view_t &w, const native_epilogue_t &ep,
        const native_int8_out_t *direct, int mb, int m0, int n0, int tiles,
        float *acc) {
    for (int t = 0; t < tiles; ++t) {
        const int n = n0 + t * p.nr;
        if constexpr (Tail) {
            const int cols = w.N - n;
            if (cols <= 0) continue;
            if (cols < p.nr) {
                compute_int8_tail_tile(p, call, w, m0, mb, n, cols, acc, p.nr);
                apply_epilogue_tile(ep, acc, p.nr, m0, mb, n, cols);
                continue;
            }
        }
        compute_int8_tile(p, call, w, m0, mb, n, p.nr, acc, p.nr, direct);
        if (direct == nullptr)
            apply_epilogue_tile(ep, acc, p.nr, m0, mb, n, p.nr);
    }
}

/// The widths of an INT8 width mask, widest first.
const char *int8_cw_text(uint8_t mask) {
    static const char *const kText[] = {"-", "16", "32", "32/16"};
    return kText[mask & kNativeInt8CwAll];
}

/// A call of native_brgemm_execute and what its stages share.
struct execute_call_t {
    const GemmDescriptor &desc;
    const native_weight_ref_t &weight;
    const void *src;
    void *dst;
    const void *bias;
    const matmul_params &params;
    bool bf16;
    bool int8;
    bool f32;
    bool f16;
};

ZENDNNL_ALWAYS_INLINE inline execute_call_t make_call(
        const GemmDescriptor &desc, const native_weight_ref_t &weight,
        const void *src, void *dst, const void *bias,
        const matmul_params &params) {
    const bool bf16 = desc.src_dt == data_type_t::bf16
            && desc.wei_dt == data_type_t::bf16;
    const bool dynamic_src = params.dynamic_quant
            && (desc.src_dt == data_type_t::bf16
                    || desc.src_dt == data_type_t::f32);
    const bool int8
            = (desc.wei_dt == data_type_t::s8 || desc.wei_dt == data_type_t::s4)
            && (desc.src_dt == data_type_t::u8 || desc.src_dt == data_type_t::s8
                    || dynamic_src);
    const bool f32 = desc.src_dt == data_type_t::f32
            && desc.wei_dt == data_type_t::f32;
    const bool f16 = desc.src_dt == data_type_t::f16
            && desc.wei_dt == data_type_t::f16;
    return {desc, weight, src, dst, bias, params, bf16, int8, f32, f16};
}

/// The shape, the leading dimensions and the buffers of the call.
ZENDNNL_ALWAYS_INLINE inline status_t check_call_operands(
        const execute_call_t &c, const char **why) {
    const GemmDescriptor &desc = c.desc;
    const int M = desc.M, N = desc.N, K = desc.K;
    const int n_out = desc.gated_act != matmul_gated_act_t::none ? N / 2 : N;
    const int min_lda = desc.transA ? M : K;
    if (M < 0 || N < 1 || K < 1 || desc.ldc < n_out
            || ((c.bf16 || c.f32 || c.f16) && desc.lda < min_lda))
        return turned_away(
                status_t::failure, "a bad shape or leading dimension", why);
    if (M > 0
            && (c.src == nullptr || c.weight.weight == nullptr
                    || c.dst == nullptr))
        return turned_away(status_t::failure, "a null src, weight or dst", why);
    // The bf16 kernels offset the source rows of their calls, up to 12, in
    // an int; a transposed source is copied into rows K apart.
    if (c.bf16 && !desc.transA && M > 1
            && static_cast<int64_t>(std::min(M, 12) - 1) * desc.lda + K
                    > std::numeric_limits<int32_t>::max())
        return turned_away(status_t::unimplemented,
                "a bf16 source of rows past the bf16 kernels' int offsets",
                why);
    return status_t::success;
}

/// The dtypes, the CPU's ISA, the shape and the buffers of the call.
ZENDNNL_ALWAYS_INLINE inline status_t check_call(
        const execute_call_t &c, const char **why) {
    const GemmDescriptor &desc = c.desc;
    if (!c.bf16 && !c.int8 && !c.f32 && !c.f16)
        return turned_away(status_t::unimplemented,
                "data types other than bf16 x bf16, u8 / s8 x s8, s8 x s4, "
                "f32 x f32 and f16 x f16",
                why);
    const UarchParams &uarch = detect_uarch();
    const bool kernel_isa
            = c.f32 || c.f16 || (c.bf16 ? uarch.avx512bf16 : uarch.avx512vnni);
    if (!kernel_isa || !native_base_isa())
        return turned_away(status_t::isa_unsupported,
                c.f32 || c.f16 ? "no AVX-512 F / BW / VL / DQ or FMA"
                               : (c.bf16 ? "no AVX512-BF16 or AVX-512 F / BW / "
                                           "VL / DQ or FMA"
                                         : "no AVX512-VNNI or AVX-512 F / BW / "
                                           "VL / DQ or FMA"),
                why);
    if (c.f32 && desc.dst_dt != data_type_t::f32
            && desc.dst_dt != data_type_t::bf16)
        return turned_away(status_t::unimplemented,
                "an f32 call into a dst other than f32 or bf16", why);
    if (c.f16 && desc.dst_dt != data_type_t::f16
            && desc.dst_dt != data_type_t::f32)
        return turned_away(status_t::unimplemented,
                "an f16 call into a dst other than f16 or f32", why);
    return check_call_operands(c, why);
}

/// The INT8 decision of the call, and the check of the call against it.
status_t decide_int8(const execute_call_t &c, native_int8_decision_t &decision,
        const char **why) {
    decision = decide_int8_call(c.desc, c.params, c.bias);
    return check_int8_call(c.desc, c.params, decision, why);
}

/// The epilogue of the call over paired tiles or not; with bias_in_dequant
/// the INT8 dequantization takes the bias instead; buffers as
/// prepare_epilogue.
status_t prepare_call_epilogue(const execute_call_t &c,
        const native_int8_decision_t &decision, bool bias_in_dequant,
        bool paired, bool buffers, native_epilogue_t &ep, const char **why) {
    GemmDescriptor ep_desc = c.desc;
    if (c.int8
            && ((decision.scheme == native_int8_scheme_t::per_channel
                        && decision.order == native_int8_order_t::reference)
                    || decision.scales_after_groups)) {
        // The per-channel INT8 tile applies alpha and beta to the integer
        // sum, the group tile before the weight scales that apply after the
        // groups; the epilogue applies those of another per-group call.
        ep_desc.alpha = 1.0f;
        ep_desc.beta = 0.0f;
    }
    return prepare_epilogue(ep_desc, c.params,
            bias_in_dequant ? nullptr : c.bias, c.dst, paired, ep, why,
            buffers);
}

/// The INT8 call of the plan and, when the epilogue is left with only the
/// store, the direct store.
status_t prepare_int8_operands(const execute_call_t &c,
        const native_int8_decision_t &decision, const native_plan_t &p,
        bool bias_in_dequant, const native_epilogue_t &ep, int8_operands_t &op,
        const char **why) {
    const GemmDescriptor &desc = c.desc;
    const status_t st = prepare_int8_call(desc, c.params, decision, c.src,
            p.copy_block, p.copy_par_min, op.call, why);
    if (st != status_t::success) return st;
    if (desc.beta != 0.0f) {
        op.call.dst = c.dst;
        op.call.dst_dt = desc.dst_dt;
        op.call.ldc = desc.ldc;
    }
    if (bias_in_dequant) {
        op.call.bias = c.bias;
        op.call.bias_dt = c.params.dtypes.bias;
    }
    // An epilogue left with only the store: the INT8 kernel stores the
    // bf16 / f32 destination itself.
    if (p.int8_direct_store && decision.direct_store && ep.store_only) {
        op.store.dst = c.dst;
        op.store.bf16 = desc.dst_dt == data_type_t::bf16;
        op.store.ldc = desc.ldc;
        op.direct = true;
    }
    return status_t::success;
}

/// The source rows of a BF16, FP32 or F16_PH call, a transposed source
/// copied once into rows K apart on threads threads.
template <typename T, typename Kernel>
status_t prepare_source_rows(const execute_call_t &c, const native_plan_t &p,
        int threads, row_operands_t<T, Kernel> &op) {
    const GemmDescriptor &desc = c.desc;
    const int M = desc.M, K = desc.K;
    op.A = static_cast<const T *>(c.src);
    op.lda = desc.lda;
    if (desc.transA) {
        const size_t bytes = static_cast<size_t>(M) * K * sizeof(T);
        op.rows.reset(static_cast<T *>(
                zendnnl_aligned_alloc(64, (bytes + 63) & ~size_t {63})));
        if (!op.rows) {
            log_error(
                    "native BRGEMM: no buffer for the transposed "
                    "source (",
                    bytes, " bytes)");
            return status_t::failure;
        }
        copy_transposed(op.A, desc.lda, M, K, op.rows.get(), p.copy_block,
                threads, p.copy_par_min);
        op.A = op.rows.get();
        op.lda = K;
    }
    return status_t::success;
}

/// The BF16 source rows (prepare_source_rows) and the kernels of the plan's
/// rows per call.
status_t prepare_bf16_operands(const execute_call_t &c, const native_plan_t &p,
        int threads, bf16_operands_t &op) {
    const status_t st = prepare_source_rows(c, p, threads, op);
    if (st != status_t::success) return st;
    const native_kernel_registry_t &kernels = native_kernel_registry();
    if (!kernels.covers(native_kernel_kind_t::bf16, p.mr, p.nr)) {
        log_error("native BRGEMM: no bf16 kernels for 1 .. ", p.mr,
                " rows at nr ", p.nr);
        return status_t::failure;
    }
    op.kern = kernels.bf16_table(p.nr);
    return status_t::success;
}

/// The FP32 source rows (prepare_source_rows), the kernels of the plan's
/// rows per call and, when the epilogue is left with only the store to an
/// f32 dst, the destination the kernels accumulate into.
status_t prepare_f32_operands(const execute_call_t &c, const native_plan_t &p,
        int threads, const native_epilogue_t &ep, f32_operands_t &op) {
    const status_t st = prepare_source_rows(c, p, threads, op);
    if (st != status_t::success) return st;
    const native_kernel_registry_t &kernels = native_kernel_registry();
    if (!kernels.covers(native_kernel_kind_t::f32, p.mr, p.nr)) {
        log_error("native BRGEMM: no f32 kernels for 1 .. ", p.mr,
                " rows at nr ", p.nr);
        return status_t::failure;
    }
    op.kern = kernels.f32_table(p.nr);
    if (ep.store_only && c.desc.dst_dt == data_type_t::f32) {
        op.dst = static_cast<float *>(c.dst);
        op.ldc = c.desc.ldc;
    }
    return status_t::success;
}

/// The F16 source rows widened to f32 once for the call (transposed for a
/// transposed source) into a buffer of the call (native_f16_source_lease_t) on
/// threads threads, rows K apart, or K + 16 when K floats are a multiple of
/// 4 KiB (rows that far apart map to one L1 set); then the kernels and the
/// destination as prepare_f32_operands.
status_t prepare_f16_operands(const execute_call_t &c, const native_plan_t &p,
        int threads, const native_epilogue_t &ep, f16_operands_t &op) {
    const GemmDescriptor &desc = c.desc;
    const int M = desc.M, K = desc.K;
    const native_kernel_registry_t &kernels = native_kernel_registry();
    if (!kernels.covers(native_kernel_kind_t::f16, p.mr, p.nr)) {
        log_error("native BRGEMM: no f16 kernels for 1 .. ", p.mr,
                " rows at nr ", p.nr);
        return status_t::failure;
    }
    op.kern = kernels.f16_table(p.nr);
    const int ld = K % 1024 == 0 ? K + 16 : K;
    const size_t bytes = static_cast<size_t>(M) * ld * sizeof(float);
    float *rows = op.widened.take(bytes);
    if (rows == nullptr) {
        log_error("native BRGEMM: no buffer for the widened f16 source (",
                bytes, " bytes)");
        return status_t::failure;
    }
    widen_f16_rows(static_cast<const uint16_t *>(c.src), desc.lda, desc.transA,
            M, K, rows, ld, p.copy_block, threads, p.copy_par_min);
    op.A = rows;
    op.lda = ld;
    if (ep.store_only && desc.dst_dt == data_type_t::f32) {
        op.dst = static_cast<float *>(c.dst);
        op.ldc = desc.ldc;
    }
    return status_t::success;
}

/// The F16 source rows as the call holds them (prepare_source_rows), then
/// the F16_PH kernels and the destination as prepare_f32_operands.
status_t prepare_f16_ph_operands(const execute_call_t &c,
        const native_plan_t &p, int threads, const native_epilogue_t &ep,
        f16_ph_operands_t &op) {
    const status_t st = prepare_source_rows(c, p, threads, op);
    if (st != status_t::success) return st;
    const native_kernel_registry_t &kernels = native_kernel_registry();
    if (!kernels.covers(native_kernel_kind_t::f16_ph, p.mr, p.nr)) {
        log_error("native BRGEMM: no f16_ph kernels for 1 .. ", p.mr,
                " rows at nr ", p.nr);
        return status_t::failure;
    }
    op.kern = kernels.f16_ph_table(p.nr);
    if (ep.store_only && c.desc.dst_dt == data_type_t::f32) {
        op.dst = static_cast<float *>(c.dst);
        op.ldc = c.desc.ldc;
    }
    return status_t::success;
}

/// copy = the N pre-activation columns of the call's dst, rows N apart, in
/// blocks of the plan's copy_block rows, on threads threads when M * N >=
/// its copy_par_min.
status_t copy_pre_activation(const execute_call_t &c, const native_plan_t &p,
        int threads, std::unique_ptr<uint8_t[]> &copy) {
    const GemmDescriptor &desc = c.desc;
    const size_t esz = size_of(desc.dst_dt);
    const size_t row_bytes = static_cast<size_t>(desc.N) * esz;
    const size_t ld_bytes = static_cast<size_t>(desc.ldc) * esz;
    copy.reset(new (std::nothrow)
                    uint8_t[static_cast<size_t>(desc.M) * row_bytes]);
    if (!copy) {
        log_error("native BRGEMM: no buffer for the pre-activation of beta (",
                static_cast<size_t>(desc.M) * row_bytes, " bytes)");
        return status_t::failure;
    }
    const uint8_t *src = static_cast<const uint8_t *>(c.dst);
    uint8_t *out = copy.get();
    const int blocks = ceil_div(desc.M, p.copy_block);
    const bool parallel = threads > 1
            && static_cast<int64_t>(desc.M) * desc.N >= p.copy_par_min;
#pragma omp parallel for num_threads(threads) schedule(static) if (parallel)
    for (int b = 0; b < blocks; ++b) {
        const int m0 = b * p.copy_block;
        const int m_end = m0 + std::min(p.copy_block, desc.M - m0);
        for (int m = m0; m < m_end; ++m)
            std::memcpy(out + static_cast<size_t>(m) * row_bytes,
                    src + static_cast<size_t>(m) * ld_bytes, row_bytes);
    }
    return status_t::success;
}

/// Runs the plan's work units on a team of up to threads threads; team
/// receives the threads of the region.  Tail: a weight of padded columns.
/// int8_op is the INT8 call's operands, null for any other call.  Staged
/// gated rows pair on the same team once every unit is done, or, too few of
/// them for the barrier that takes to pay, on the calling thread after it.
template <bool Tail>
status_t run_units(const execute_call_t &c, const native_plan_t &p, int threads,
        const native_weight_view_t &view, const native_epilogue_t &ep,
        const bf16_operands_t &bf16_op, const int8_operands_t *int8_op,
        const f32_operands_t &f32_op, const f16_operands_t &f16_op,
        const f16_ph_operands_t &f16_ph_op, int &team) {
    const int K = c.desc.K;
    const bool bf16 = c.bf16;
    const bool f32 = c.f32;
    const bool f16_ph = c.f16 && p.kind == native_kernel_kind_t::f16_ph;
    const bool f16 = c.f16 && !f16_ph;
    const native_int8_out_t *direct
            = int8_op != nullptr && int8_op->direct ? &int8_op->store : nullptr;
    // Every unit covers all of K and tiles column tiles of nr columns.
    auto tile = [&](const native_work_unit_t &u, float *acc) {
        if (bf16)
            bf16_block<Tail>(
                    p, bf16_op, view, ep, u.rows, K, u.m0, u.n0, u.tiles, acc);
        else if (f32)
            f32_block<Tail>(
                    p, f32_op, view, ep, u.rows, K, u.m0, u.n0, u.tiles, acc);
        else if (f16_ph)
            f16_ph_block<Tail>(p, f16_ph_op, view, ep, u.rows, K, u.m0, u.n0,
                    u.tiles, acc);
        else if (f16)
            f16_block<Tail>(
                    p, f16_op, view, ep, u.rows, K, u.m0, u.n0, u.tiles, acc);
        else
            int8_block<Tail>(p, int8_op->call, view, ep, direct, u.rows, u.m0,
                    u.n0, u.tiles, acc);
    };
    // Workers past the units have nothing to do.
    const native_partition_t part(p, c.desc.M, 0, p.n_tiles);
    struct pairing_t {
        const native_epilogue_t *ep;
        int M;
    };
    const pairing_t pairing {&ep, c.desc.M};
    const native_team_step_t pair {[](const void *ctx, int i, int n) {
        const pairing_t &q = *static_cast<const pairing_t *>(ctx);
        pair_staged_halves(*q.ep, q.M, i, n);
    }, &pairing};
    const bool pair_in_team = ep.staged_halves
            && staged_pair_units(ep, c.desc.M)
                    >= 4 * static_cast<int64_t>(threads);
    // An M = 1 call on a layout of at most a quarter of the L2 runs its
    // units on the calling thread: one core reads that layout in less time
    // than a region of the call's team takes to open, and a region of fewer
    // threads would make the runtime end and start the others again.
    const bool one_thread = c.desc.M == 1
            && native_packed_bytes(view.wei_dt, view.K, view.N, view.pack_nr)
                    <= static_cast<size_t>(detect_uarch().l2_bytes) / 4;
    const status_t st = native_run_cpu_team(part,
            one_thread ? 1
                       : static_cast<int>(
                                 std::min<int64_t>(p.threads, part.units())),
            threads, p.scratch_floats, tile, &team,
            pair_in_team ? &pair : nullptr);
    // After work no status may let the call fall back to another route.
    if (st != status_t::success) {
        log_error("native BRGEMM: no accumulator scratch of ", p.scratch_floats,
                " floats");
        return status_t::failure;
    }
    if (ep.staged_halves && !pair_in_team)
        pair_staged_halves(ep, c.desc.M, 0, 1);
    return status_t::success;
}

/// run_units of the call: the tiles of a weight of N not a multiple of its
/// blocks run on its padded width and store only the columns below N.
status_t run_team(const execute_call_t &c, const native_plan_t &p, int threads,
        const native_weight_view_t &view, const native_epilogue_t &ep,
        const bf16_operands_t &bf16_op, const int8_operands_t *int8_op,
        const f32_operands_t &f32_op, const f16_operands_t &f16_op,
        const f16_ph_operands_t &f16_ph_op, int &team) {
    if (view.N % kNativeLayoutPackNr != 0)
        return run_units<true>(c, p, threads, view, ep, bf16_op, int8_op,
                f32_op, f16_op, f16_ph_op, team);
    return run_units<false>(c, p, threads, view, ep, bf16_op, int8_op, f32_op,
            f16_op, f16_ph_op, team);
}

/// The GEMV kernels of an M = 1 call, by its type: none on the tiles.
enum class gemv_kind_t : uint8_t {
    none,
    bf16,
    f32,
    f16,
    f16_ph,
    s8,
    s8_group,
    s4
};

const char *gemv_name(gemv_kind_t kind) {
    switch (kind) {
        case gemv_kind_t::bf16: return "bf16";
        case gemv_kind_t::f32: return "f32";
        case gemv_kind_t::f16: return "f16";
        case gemv_kind_t::f16_ph: return "f16_ph";
        case gemv_kind_t::s8: return "s8";
        case gemv_kind_t::s8_group: return "s8_group";
        case gemv_kind_t::s4: return "s4";
        default: return "none";
    }
}

/// The GEMV kernels of the call: every M = 1 call runs on them.
gemv_kind_t gemv_kind_of(const execute_call_t &c, const native_plan_t &p,
        const int8_operands_t *int8_op) {
    if (c.desc.M != 1) return gemv_kind_t::none;
    if (c.bf16) return gemv_kind_t::bf16;
    if (c.f32) return gemv_kind_t::f32;
    if (c.f16) {
        if (p.kind != native_kernel_kind_t::f16_ph) return gemv_kind_t::f16;
        // A plan of the f16_ph tiles stays on them without the f16_ph GEMV
        // kernels (ZENDNNL_NATIVE_F16_PH_KERNELS builds both or neither).
        return select_native_f16_ph_gemv(1) != nullptr ? gemv_kind_t::f16_ph
                                                       : gemv_kind_t::none;
    }
    if (c.desc.wei_dt == data_type_t::s4) return gemv_kind_t::s4;
    return int8_op->call.scheme == native_int8_scheme_t::per_group
            ? gemv_kind_t::s8_group
            : gemv_kind_t::s8;
}

/// An M = 1 call on the GEMV kernels and what its workers share.  Its
/// columns come in units a worker takes whole: blocks of 32 columns; for a
/// gated call whose tiles pair the halves (the view is paired), a gate block
/// and the up block paired with it, or where the halves are not whole
/// blocks, a run of 16 gate columns and its up run.
struct gemv_call_t {
    const execute_call_t &c;
    const native_plan_t &p;
    const native_weight_view_t &view;
    const native_epilogue_t &ep;
    gemv_kind_t kind;
    const bf16_operands_t &bf16_op;
    const int8_operands_t *int8_op;
    const f32_operands_t &f32_op;
    const f16_operands_t &f16_op;
    const f16_ph_operands_t &f16_ph_op;
    /// Units: 32 columns each, plain; paired: 64 (whole blocks) or 32 (runs).
    int units = 0;
    bool runs = false;
};

/// The units of the call's columns.
void gemv_units(gemv_call_t &g) {
    const int N = g.view.N;
    if (!g.view.paired) {
        g.units = ceil_div(N, kNativeLayoutPackNr);
    } else if ((N / 2) % kNativeLayoutPackNr == 0) {
        g.units = N / (2 * kNativeLayoutPackNr);
    } else {
        g.units = N / kNativeLayoutPackNr;
        g.runs = true;
    }
}

/// The tile (native_weight_view_t) unit u of the call forms: pack columns
/// n0 .. n0 + nr, of which cols lie below N.
struct gemv_tile_t {
    int n0 = 0;
    int nr = 0;
    int cols = 0;
};

gemv_tile_t gemv_tile(const gemv_call_t &g, int u) {
    gemv_tile_t t;
    t.nr = g.view.paired && !g.runs ? 2 * kNativeLayoutPackNr
                                    : kNativeLayoutPackNr;
    t.n0 = u * t.nr;
    t.cols = std::min(t.nr, g.view.N - t.n0);
    return t;
}

/// Units units[0 .. nu) of the call (at most kNativeGemvMaxBlocks blocks)
/// into acc, which holds 32 * kNativeGemvMaxBlocks floats, unit i's tile at
/// the sum of the earlier tiles' widths, then the epilogue of each tile: the
/// bits the tiles give the same columns.
void gemv_units_call(
        const gemv_call_t &g, const int *units, int nu, float *acc) {
    const native_weight_view_t &w = g.view;
    const int N = w.N;
    const bool whole_pairs = w.paired && !g.runs;
    // The kernel's blocks: a unit's block; a gate block then its up block;
    // or a gate run of 16 columns and its up run.
    native_gemv_cols_t cols;
    cols.runs = g.runs;
    gemv_tile_t tiles[kNativeGemvMaxBlocks];
    for (int i = 0; i < nu; ++i) {
        const int u = units[i];
        tiles[i] = gemv_tile(g, u);
        if (whole_pairs) {
            const int gate = u * kNativeLayoutPackNr;
            cols.lo[cols.nb++] = w.column(gate);
            cols.lo[cols.nb++] = w.column(N / 2 + gate);
        } else if (g.runs) {
            cols.lo[cols.nb] = w.column(u * 16);
            cols.hi[cols.nb++] = w.column(N / 2 + u * 16);
        } else {
            cols.lo[cols.nb++] = w.column(u * kNativeLayoutPackNr);
        }
    }
    // The epilogue of each tile.
    const auto epilogue = [&]() {
        float *c = acc;
        for (int i = 0; i < nu; c += tiles[i++].nr)
            apply_epilogue_tile(
                    g.ep, c, tiles[i].nr, 0, 1, tiles[i].n0, tiles[i].cols);
    };
    const int K = w.K;
    switch (g.kind) {
        case gemv_kind_t::bf16:
            select_native_bf16_gemv(cols.nb, cols.runs)(
                    g.bf16_op.A, K, cols, acc);
            epilogue();
            return;
        case gemv_kind_t::f32:
            select_native_f32_gemv(cols.nb)(g.f32_op.A, K, cols, acc);
            epilogue();
            return;
        case gemv_kind_t::f16:
            select_native_f16_gemv(cols.nb)(g.f16_op.A, K, cols, acc);
            epilogue();
            return;
        case gemv_kind_t::f16_ph:
            select_native_f16_ph_gemv(cols.nb)(g.f16_ph_op.A, K, cols, acc);
            epilogue();
            return;
        default: break;
    }
    const native_int8_call_t &call = g.int8_op->call;
    if (g.kind == gemv_kind_t::s8) {
        // The fused order finishes in the kernel; the reference order, and
        // the last tile of a weight of padded columns, through the tiles'
        // finish over the kernel's sums of all of K.
        const bool tail = tiles[nu - 1].cols < tiles[nu - 1].nr;
        if (call.order == native_int8_order_t::fused && !tail) {
            native_s8_gemv_args_t a;
            a.a = call.A;
            a.k_quads = w.k_groups;
            a.cols = cols;
            for (int i = 0, b = 0; i < nu; ++i) {
                const gemv_tile_t &t = tiles[i];
                for (int h = 0; h < t.nr; h += kNativeLayoutPackNr, ++b) {
                    a.col_lo[b] = w.tile_col(t.n0, t.nr, h);
                    a.col_hi[b] = w.tile_col(t.n0, t.nr, h + 16);
                    a.cs_lo[b] = w.col_sum(a.col_lo[b]);
                    a.cs_hi[b] = w.col_sum(a.col_hi[b]);
                }
            }
            a.zp = call.src_zp != nullptr ? call.src_zp[0] : 0;
            a.wide_zp = call.wide_zp;
            a.src_scale = call.src_scale[0];
            a.wei_scale = call.wei_scale;
            a.wei_scale_dt = call.wei_scale_dt;
            a.wei_scale_per_column = call.wei_scale_stride != 0;
            a.bias = call.bias;
            a.bias_dt = call.bias_dt;
            a.c = acc;
            select_native_s8_fused_gemv(cols.nb, cols.runs, call.src_signed)(a);
            epilogue();
            return;
        }
        select_native_s8_gemv(cols.nb, cols.runs, call.src_signed)(
                call.A, w.k_groups, cols, reinterpret_cast<int32_t *>(acc));
        float *c = acc;
        for (int i = 0; i < nu; c += tiles[i++].nr) {
            const gemv_tile_t &t = tiles[i];
            finish_int8_channel_tile(
                    g.p, call, w, 0, 1, t.n0, t.nr, t.cols, c, t.nr);
            apply_epilogue_tile(g.ep, c, t.nr, 0, 1, t.n0, t.cols);
        }
        return;
    }
    // Per-group weight scales, or an s4 weight: the last block of a weight of
    // padded columns runs on the tiles' tail tile, which reads no scale past
    // N.
    const bool s4 = g.kind == gemv_kind_t::s4;
    const int full
            = !w.paired && tiles[nu - 1].cols < tiles[nu - 1].nr ? nu - 1 : nu;
    if (full > 0) {
        native_gemv_group_args_t a;
        a.cols = cols;
        a.cols.nb = whole_pairs ? 2 * full : full;
        a.a = call.A;
        a.a_step = static_cast<size_t>(call.group_quads) * 4;
        a.src_u8 = !call.src_signed;
        a.shift = call.group_shift;
        a.src_zp = !call.src_signed && call.src_zp != nullptr ? call.src_zp[0]
                                                              : 0;
        a.wide_zp = call.wide_zp;
        a.k_quads = call.group_quads;
        a.b_step = s4
                ? 0
                : static_cast<size_t>(call.group_quads) * w.group_stride();
        a.groups = call.groups;
        a.src_scale = call.src_scale;
        a.src_scale_step = call.src_scale_group_stride;
        a.unit_scales = call.scales_after_groups;
        a.wei_scale = call.wei_scale;
        a.wei_scale_dt = call.wei_scale_dt;
        a.wei_scale_ld = static_cast<size_t>(N);
        // The weight columns of each kernel block's two runs of 16, as its
        // tile holds them.
        for (int i = 0, b = 0; i < full; ++i) {
            const gemv_tile_t &t = tiles[i];
            for (int h = 0; h < t.nr; h += kNativeLayoutPackNr, ++b) {
                a.col_lo[b] = w.tile_col(t.n0, t.nr, h);
                a.col_hi[b] = w.tile_col(t.n0, t.nr, h + 16);
            }
        }
        a.order = call.order;
        a.src_groups = call.src_scale_group_stride != 0;
        a.finish = !a.src_groups;
        a.c = acc;
        select_native_group_gemv(a.cols.nb, cols.runs, s4, a.src_u8)(a);
        if (a.unit_scales) {
            float *c = acc;
            for (int i = 0; i < full; c += tiles[i++].nr)
                finish_int8_tile_after_groups(g.p, call, w, 0, 1, tiles[i].n0,
                        tiles[i].nr, c, tiles[i].nr);
        }
    }
    if (full < nu) {
        const gemv_tile_t &t = tiles[nu - 1];
        compute_int8_tail_tile(g.p, call, w, 0, 1, t.n0, t.cols,
                acc + full * kNativeLayoutPackNr, t.nr);
    }
    epilogue();
}

/// Blocks one GEMV kernel call reads side by side, each a stream of its own:
/// two, the streams per core DRAM and the L3 serve best, unless a worker's
/// share of a bf16, f32 or f16 layout fits half its L2, where four blocks
/// give the multiply-adds the accumulators that hide their latency.  The
/// INT8 and s4 kernels split their exact sums over more accumulators of
/// their own.
int gemv_call_blocks(
        gemv_kind_t kind, const native_weight_view_t &view, int workers) {
    switch (kind) {
        case gemv_kind_t::bf16:
        case gemv_kind_t::f32:
        case gemv_kind_t::f16:
        case gemv_kind_t::f16_ph: {
            const size_t share = native_packed_bytes(view.wei_dt, view.K,
                                         view.N, view.pack_nr)
                    / static_cast<size_t>(std::max(1, workers));
            return share <= static_cast<size_t>(detect_uarch().l2_bytes) / 2
                    ? 4
                    : 2;
        }
        default: return kNativeGemvMaxIntBlocks;
    }
}

/// Runs the call's units on the GEMV kernels: each worker its share of whole
/// units, all of K each, in kernel calls of at most gemv_call_blocks blocks;
/// one worker inline, else a region of the caller's team as
/// native_run_cpu_team opens it, the workers spread over its threads.
/// Staged gated rows pair as run_units pairs them.
status_t run_gemv(gemv_call_t &g, int workers, int threads, int &team) {
    gemv_units(g);
    const int units = g.units;
    workers = std::max(1, std::min(workers, units));
    // Blocks a unit reads.
    const int unit_blocks = g.view.paired && !g.runs ? 2 : 1;
    const int per_call = std::max(
            1, gemv_call_blocks(g.kind, g.view, workers) / unit_blocks);
    // A worker splits its units into fronts, as many as a kernel call
    // takes, and each call takes the next unit of every front: a front
    // walks its units in memory order, so each of a call's streams goes on
    // from the block it read last.
    const auto run = [&](int w, int n) {
        alignas(64) float acc[kNativeGemvMaxBlocks * kNativeLayoutPackNr];
        if (g.c.int8 && g.int8_op->call.group_shift != nullptr)
            ready_group_shifts(g.int8_op->call, 0, 1);
        const int u0 = static_cast<int>(static_cast<int64_t>(units) * w / n);
        const int u1
                = static_cast<int>(static_cast<int64_t>(units) * (w + 1) / n);
        const int count = u1 - u0;
        if (count <= 0) return;
        const int fronts = std::min(per_call, count);
        int first[kNativeGemvMaxBlocks], last[kNativeGemvMaxBlocks];
        for (int f = 0; f < fronts; ++f) {
            first[f] = u0 + count * f / fronts;
            last[f] = u0 + count * (f + 1) / fronts;
        }
        for (int step = 0; step < ceil_div(count, fronts); ++step) {
            int list[kNativeGemvMaxBlocks];
            int nu = 0;
            for (int f = 0; f < fronts; ++f)
                if (first[f] + step < last[f]) list[nu++] = first[f] + step;
            gemv_units_call(g, list, nu, acc);
        }
    };
    const native_epilogue_t &ep = g.ep;
    const int M = g.c.desc.M;
    const bool pair_in_team = ep.staged_halves
            && staged_pair_units(ep, M) >= 4 * static_cast<int64_t>(threads);
    if (workers <= 1 || threads <= 1) {
        run(0, 1);
        team = 1;
    } else {
        const int region = std::max(workers, threads);
#pragma omp parallel num_threads(region)
        {
            const int given = omp_get_num_threads();
            const int n = std::min(workers, given);
            const int tid = omp_get_thread_num();
            // Fewer workers than threads spread evenly over the team, and so
            // over the L3 domains of its cores, each domain's memory
            // bandwidth serving its share of the columns.
            const int64_t at = static_cast<int64_t>(tid) * n;
            if (at % given < n) run(static_cast<int>(at / given), n);
            if (tid == 0) team = given;
            if (pair_in_team) {
#pragma omp barrier
                pair_staged_halves(ep, M, tid, given);
            }
        }
    }
    if (ep.staged_halves && !(pair_in_team && workers > 1 && threads > 1))
        pair_staged_halves(ep, M, 0, 1);
    return status_t::success;
}

/// The plan the paired tiles of a gated call over [gate | up] halves run
/// (native_weight_view_t) at width (native_paired_width): planned's, with
/// tiles width wide when that is wider, and for BF16 the accumulators of a
/// tile's rows.  Where the wider tiles leave the static deal fewer units
/// than threads, row blocks of fewer rows give back the units the plan was
/// balanced for.
native_plan_t paired_plan(const native_plan_t &planned,
        const native_problem_t &problem, int width) {
    native_plan_t p = planned;
    if (width > planned.nr) {
        p.nr = width;
        p.n_tiles = problem.N / p.nr;
        if (p.unit_tiles > 0)
            p.unit_tiles = std::max(1, p.unit_tiles * planned.nr / width);
        const int64_t units
                = static_cast<int64_t>(planned.n_tiles) * planned.m_blocks;
        if (p.unit_tiles == 0 && p.col_groups == 0
                && static_cast<int64_t>(p.n_tiles) * p.m_blocks < p.threads
                && units > static_cast<int64_t>(p.n_tiles) * p.m_blocks) {
            const int want
                    = static_cast<int>(std::min<int64_t>(units, p.threads));
            const int m_blocks = std::min(problem.M, ceil_div(want, p.n_tiles));
            p.mb = ceil_div(problem.M, m_blocks);
            p.m_blocks = ceil_div(problem.M, p.mb);
        }
        p.scratch_floats = native_plan_scratch_floats(p, problem);
    }
    // A paired BF16 tile keeps the accumulators of all its rows.
    if (problem.bf16())
        p.scratch_floats
                = std::max(p.scratch_floats, static_cast<size_t>(p.mb) * p.nr);
    return p;
}

/// The [MATMUL.NATIVE_BRGEMM] line of a call that ran; paired: its tiles
/// paired the halves of a gated call.
void log_plan_line(const execute_call_t &c, const native_plan_t &p,
        const native_int8_decision_t &decision, bool bias_in_dequant,
        bool direct, int pack_nr, int team, bool paired, gemv_kind_t gemv) {
    const GemmDescriptor &desc = c.desc;
    // A BF16, FP32 or F16 call has a K pass; an INT8 call has call shapes
    // and, unless its weight scales are per group, a K pass, a direct store
    // and a bias placement; a per-group call has its scheme and its direct
    // store, and over an s4 weight its group size and order.
    const std::string kc_field
            = c.bf16 || c.f32 || c.f16 ? " kc=" + std::to_string(p.kc) : "";
    // An F16 call names its kernels: the widening F16 ones or F16_PH.
    const std::string kernels_field = c.f16 ? std::string(" kernels=")
                    + (p.kind == native_kernel_kind_t::f16_ph ? "f16_ph"
                                                              : "f16")
                                            : std::string();
    // The blocks schedule names the two fields its deal reads: the unit
    // width and the threads of a domain.
    const std::string blocks_field = p.unit_tiles > 0
            ? " unit_tiles=" + std::to_string(p.unit_tiles)
                    + " domain_threads=" + std::to_string(p.domain_threads)
            : "";
    std::string int8_fields;
    if (c.int8) {
        int8_fields = " int8_rows=" + std::to_string(p.int8_rows_nv1) + "/"
                + std::to_string(p.int8_rows_nv2)
                + " int8_cw=" + int8_cw_text(p.int8_cw_mask);
        if (decision.scheme != native_int8_scheme_t::per_channel)
            int8_fields += std::string(" int8_scheme=")
                    + native_int8_scheme_name(decision.scheme)
                    + " direct=" + std::to_string(direct ? 1 : 0);
        if (desc.wei_dt == data_type_t::s4)
            int8_fields += " wei=s4 gs="
                    + std::to_string(desc.K / std::max(1, decision.groups))
                    + " order="
                    + (decision.order == native_int8_order_t::fused
                                    ? "fused"
                                    : "reference")
                    + " pass_budget=" + std::to_string(p.int8_pass_budget);
        else if (decision.scheme == native_int8_scheme_t::per_channel)
            int8_fields += " int8_pass="
                    + (p.int8_pass_quads > 0 ? std::to_string(p.int8_pass_quads)
                                             : std::string("rule"))
                    + " direct=" + std::to_string(direct ? 1 : 0)
                    + " dequant_bias="
                    + std::to_string(bias_in_dequant ? 1 : 0);
    }
    // A call of padded columns names the width its plan, the tuning file
    // and ZENDNNL_MATMUL_NATIVE_CONFIG match (make_native_problem).
    const std::string padded_field = desc.N % kNativeLayoutPackNr != 0
            ? " padded_n=" + std::to_string(native_padded_n(desc.N, pack_nr))
            : std::string();
    // A gated call over [gate | up] halves names the width at which its
    // tiles paired them, or that it staged its rows.
    const std::string pairing_field = native_gated_halves(desc.gated_act)
            ? " pairing="
                    + (paired ? std::to_string(p.nr) : std::string("staged"))
            : std::string();
    apilog_verbose("[MATMUL.NATIVE_BRGEMM] M=", desc.M, " N=", desc.N,
            " K=", desc.K, " pack_nr=", pack_nr, " nr=", p.nr, " mb=", p.mb,
            kc_field, kernels_field, " tiles=", p.tiles(),
            " threads=", p.threads, " col_groups=", p.col_groups, blocks_field,
            " transA=", desc.transA ? 1 : 0, " gemv=", gemv_name(gemv),
            " mr=", p.mr, " row_unit=", p.row_unit, " traversal=",
            p.traversal == native_traversal_t::m_inner ? "m_inner" : "n_inner",
            int8_fields, " copy_block=", p.copy_block,
            " source=", native_plan_source_names(p),
            " cfg=", native_plan_config_text(p), " team=", team, padded_field,
            pairing_field);
}

/// The native tiles of the call: every step up to the acquisition of the
/// weight's layout, which is the first that may change the weight, then
/// (without CheckOnly) the run, an M = 1 call on the GEMV kernels unless
/// tiles_only.
template <bool CheckOnly>
status_t run_tiles(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params, const char **why,
        bool tiles_only = false) {
    static const bool s_log = apilog_verbose_enabled();
    const execute_call_t c = make_call(desc, weight, src, dst, bias, params);
    status_t st = check_call(c, why);
    if (st != status_t::success) return st;
    if (desc.M == 0) return status_t::success;
    // The layout rule checked the shape; the planner and the tile grid take
    // a weight of N not a multiple of its blocks at its padded width.
    assert(native_layout_fits(desc.wei_dt, desc.K, desc.N, weight.pack_nr));

    // Every check that can turn the call away runs before any work: a call
    // turned away leaves the source and the weight as they were.
    native_int8_decision_t decision;
    if (c.int8) {
        st = decide_int8(c, decision, why);
        if (st != status_t::success) return st;
    }
    const native_problem_t problem
            = make_native_problem(desc, params, bias, weight.pack_nr, decision);
    native_plan_t p = native_planner().plan(problem);
    // The gate and up halves of a gated weight pair up in the tiles of the
    // BF16 and INT8 kernels, from the weight's plain layout, in whole blocks
    // or in runs of 16 columns.
    const int paired_width
            = native_gated_halves(desc.gated_act) && !c.f32 && !c.f16
            ? native_paired_width(desc.N, p.nr, p.m_blocks, p.threads)
            : 0;
    const bool paired = paired_width > 0;
    if (paired) p = paired_plan(p, problem, paired_width);
    const bool bias_in_dequant
            = c.int8 && p.int8_bias_in_dequant && decision.bias_in_dequant;
    native_epilogue_t ep;
    st = prepare_call_epilogue(
            c, decision, bias_in_dequant, paired, !CheckOnly, ep, why);
    if (st != status_t::success) return st;
    std::optional<int8_operands_t> int8_op;
    if (c.int8) {
        st = prepare_int8_operands(
                c, decision, p, bias_in_dequant, ep, int8_op.emplace(), why);
        if (st != status_t::success) return st;
    }
    if (CheckOnly) return status_t::success;
    bf16_operands_t bf16_op;
    if (c.bf16) {
        st = prepare_bf16_operands(c, p, problem.threads, bf16_op);
        if (st != status_t::success) return st;
        if (paired) bf16_op.paired = bf16_brgemm_paired_table();
    }
    f32_operands_t f32_op;
    if (c.f32) {
        st = prepare_f32_operands(c, p, problem.threads, ep, f32_op);
        if (st != status_t::success) return st;
    }
    f16_operands_t f16_op;
    f16_ph_operands_t f16_ph_op;
    if (c.f16 && p.kind == native_kernel_kind_t::f16_ph) {
        st = prepare_f16_ph_operands(c, p, problem.threads, ep, f16_ph_op);
        if (st != status_t::success) return st;
    } else if (c.f16) {
        st = prepare_f16_operands(c, p, problem.threads, ep, f16_op);
        if (st != status_t::success) return st;
    }
    // A swiglu output column i takes the place of pre-activation column i,
    // which a tile of other columns may still read for beta: beta reads a
    // copy of the pre-activation columns instead.
    std::unique_ptr<uint8_t[]> beta_copy;
    if (desc.gated_act == matmul_gated_act_t::swiglu_oai_mul
            && desc.beta != 0.0f) {
        st = copy_pre_activation(c, p, problem.threads, beta_copy);
        if (st != status_t::success) return st;
        ep.beta_src = beta_copy.get();
        ep.beta_ld = desc.N;
        if (int8_op) {
            int8_op->call.dst = beta_copy.get();
            int8_op->call.ldc = desc.N;
        }
    }

    // Last: a first call at weight cache mode 2 may rewrite the weight in
    // place, so nothing after it may turn the call away.
    native_weight_view_t view;
    native_weight_lease_t lease;
    st = acquire_native_weight(weight, desc.wei_dt, desc.K, desc.N, desc.ldb,
            desc.transB, view, lease, problem.threads);
    if (st != status_t::success) return st;
    view.paired = paired;

    int team = 1;
    const int8_operands_t *int8_ops = int8_op ? &*int8_op : nullptr;
    const gemv_kind_t gemv
            = tiles_only ? gemv_kind_t::none : gemv_kind_of(c, p, int8_ops);
    if (gemv != gemv_kind_t::none) {
        gemv_call_t g {c, p, view, ep, gemv, bf16_op, int8_ops, f32_op, f16_op,
                f16_ph_op};
        // An M = 1 call on a layout of at most a quarter of the L2 runs on
        // the calling thread (see run_units).
        const bool one_thread
                = native_packed_bytes(view.wei_dt, view.K, view.N, view.pack_nr)
                <= static_cast<size_t>(detect_uarch().l2_bytes) / 4;
        st = run_gemv(g, one_thread ? 1 : p.threads, problem.threads, team);
    } else {
        st = run_team(c, p, problem.threads, view, ep, bf16_op, int8_ops,
                f32_op, f16_op, f16_ph_op, team);
    }
    if (st != status_t::success) return st;

    if (s_log)
        log_plan_line(c, p, decision, bias_in_dequant,
                int8_op && int8_op->direct, view.pack_nr, team, paired, gemv);
    return status_t::success;
}

} // namespace

native_f16_source_lease_t::~native_f16_source_lease_t() {
    if (buf_ == nullptr) return;
    // At most every kept buffer, and this one.
    std::array<float *, kNativeF16KeptSourceSlots + 1> drop;
    int dropped = 0;
    f16_source_pool_t &pool = f16_source_pool();
    {
        const std::lock_guard<std::mutex> lock(pool.mutex);
        if (generation_ == pool.generation && bytes_ <= kF16KeptSourceBytes) {
            if (pool.used == kNativeF16KeptSourceSlots) {
                const int s = smallest_kept(pool);
                if (pool.free[s].second < bytes_)
                    drop[dropped++] = take_slot(pool, s).first;
            }
            if (pool.used < kNativeF16KeptSourceSlots) {
                pool.free[pool.used++] = {buf_, bytes_};
                pool.kept += bytes_;
                buf_ = nullptr;
            }
            while (pool.kept > kF16KeptSourceBytes)
                drop[dropped++] = take_slot(pool, smallest_kept(pool)).first;
        }
    }
    if (buf_ != nullptr) drop[dropped++] = buf_;
    for (int i = 0; i < dropped; ++i)
        zendnnl_aligned_free(drop[i]);
}

float *native_f16_source_lease_t::take(size_t bytes) {
    bytes = (bytes + 63) & ~size_t {63};
    f16_source_pool_t &pool = f16_source_pool();
    {
        const std::lock_guard<std::mutex> lock(pool.mutex);
        generation_ = pool.generation;
        int best = -1;
        for (int i = 0; i < pool.used; ++i)
            if (pool.free[i].second >= bytes
                    && (best < 0
                            || pool.free[i].second < pool.free[best].second))
                best = i;
        if (best >= 0) {
            const std::pair<float *, size_t> b = take_slot(pool, best);
            buf_ = b.first;
            bytes_ = b.second;
            return buf_;
        }
    }
    buf_ = static_cast<float *>(zendnnl_aligned_alloc(64, bytes));
    bytes_ = buf_ != nullptr ? bytes : 0;
    return buf_;
}

void release_native_f16_sources() {
    std::array<std::pair<float *, size_t>, kNativeF16KeptSourceSlots> drop;
    int dropped = 0;
    f16_source_pool_t &pool = f16_source_pool();
    {
        const std::lock_guard<std::mutex> lock(pool.mutex);
        drop = pool.free;
        dropped = pool.used;
        pool.used = 0;
        pool.kept = 0;
        ++pool.generation;
    }
    for (int i = 0; i < dropped; ++i)
        zendnnl_aligned_free(drop[i].first);
}

size_t native_f16_kept_source_bytes() {
    f16_source_pool_t &pool = f16_source_pool();
    const std::lock_guard<std::mutex> lock(pool.mutex);
    return pool.kept;
}

status_t native_brgemm_execute(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params, const char **why) {
    return run_tiles<false>(desc, weight, src, dst, bias, params, why);
}

status_t native_brgemm_check(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params, const char **why) {
    return run_tiles<true>(desc, weight, src, dst, bias, params, why);
}

status_t native_brgemm_execute_on_tiles(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params, const char **why) {
    return run_tiles<false>(desc, weight, src, dst, bias, params, why, true);
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
