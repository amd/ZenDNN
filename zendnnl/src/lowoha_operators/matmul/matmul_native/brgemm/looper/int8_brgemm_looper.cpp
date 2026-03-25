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

#include "lowoha_operators/matmul/matmul_native/brgemm/looper/int8_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/brgemm_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/int8/int8_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/common/postop.hpp"
#include "operators/matmul/matmul_config.hpp"
#include "common/zendnnl_global.hpp"
#include "common/bfloat16.hpp"

#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using namespace zendnnl::error_handling;
using zendnnl::ops::matmul_config_t;

// ── INT8 VNNI B packing (panel-based, same layout as KC but per N-panel) ──
// Packs s8 weights into NR_PACK-wide panels with 4-byte VNNI groups.
// Also computes col_sum[n] for zero-point compensation.
__attribute__((target("avx512f,avx512bw,avx512vl")))
static void pack_b_int8_vnni_panel(
    const int8_t *B, int ldb, int K, int N, bool transB,
    int8_t *packed, int32_t *col_sum) {

    const int K_padded = (K + 3) & ~3;
    const int k_quads  = K_padded / 4;
    const int np = (N + NR_PACK - 1) / NR_PACK;
    const int vnni_stride = NR_PACK * 4;

    std::memset(col_sum, 0, static_cast<size_t>(N) * sizeof(int32_t));

    for (int jp = 0; jp < np; ++jp) {
        const int j0 = jp * NR_PACK;
        const int nr_act = std::min(NR_PACK, N - j0);
        int8_t *dst_panel = packed
            + static_cast<size_t>(jp) * k_quads * vnni_stride;

        for (int kq = 0; kq < k_quads; ++kq) {
            int8_t *d = dst_panel + kq * vnni_stride;
            const int k_base = kq * 4;

            for (int n = 0; n < nr_act; ++n) {
                int32_t sum = 0;
                for (int i = 0; i < 4; ++i) {
                    const int k = k_base + i;
                    int8_t val = 0;
                    if (k < K) {
                        val = transB ? B[(j0 + n) * ldb + k]
                                     : B[k * ldb + (j0 + n)];
                    }
                    d[n * 4 + i] = val;
                    sum += val;
                }
                col_sum[j0 + n] += sum;
            }
            for (int n = nr_act; n < NR_PACK; ++n) {
                d[n * 4 + 0] = 0;
                d[n * 4 + 1] = 0;
                d[n * 4 + 2] = 0;
                d[n * 4 + 3] = 0;
            }
        }
    }
}

// Alpha scaling for fp32 tile (applied after dequant when alpha != 1).
__attribute__((target("avx512f")))
static void scale_tile_fp32(float *C, int ldc, int mr, int nr, float alpha) {
    __m512 av = _mm512_set1_ps(alpha);
    for (int m = 0; m < mr; ++m) {
        float *row = C + m * ldc;
        int n = 0;
        for (; n + 15 < nr; n += 16)
            _mm512_storeu_ps(row + n, _mm512_mul_ps(av, _mm512_loadu_ps(row + n)));
        for (; n < nr; ++n)
            row[n] *= alpha;
    }
}

// Beta accumulation: C_dst = alpha * C_fp32 + beta * C_old_fp32.
__attribute__((target("avx512f")))
static void beta_accumulate(float *C_new, int ldc_new,
                            const float *C_old, int ldc_old,
                            int mr, int nr, float beta) {
    __m512 bv = _mm512_set1_ps(beta);
    for (int m = 0; m < mr; ++m) {
        float *dst = C_new + m * ldc_new;
        const float *old_row = C_old + m * ldc_old;
        int n = 0;
        for (; n + 15 < nr; n += 16)
            _mm512_storeu_ps(dst + n, _mm512_add_ps(
                _mm512_loadu_ps(dst + n),
                _mm512_mul_ps(bv, _mm512_loadu_ps(old_row + n))));
        for (; n < nr; ++n)
            dst[n] += beta * old_row[n];
    }
}

// Requantize fp32 → s8 or u8: out = clamp(round(val / dst_scale) + dst_zp)
__attribute__((target("avx512f,avx512bw,avx512vl")))
static void requantize_tile(const float *C_fp32, int ldc_fp32,
                            void *dst, int ldc_dst,
                            int mr, int nr,
                            data_type_t dst_dt,
                            float dst_scale, int32_t dst_zp) {
    const __m512 inv_scale = _mm512_set1_ps(1.0f / dst_scale);
    const __m512i vzp = _mm512_set1_epi32(dst_zp);

    for (int m = 0; m < mr; ++m) {
        const float *src_row = C_fp32 + m * ldc_fp32;
        int n = 0;
        for (; n + 15 < nr; n += 16) {
            __m512 val = _mm512_mul_ps(_mm512_loadu_ps(src_row + n), inv_scale);
            __m512i i32 = _mm512_add_epi32(_mm512_cvtps_epi32(val), vzp);
            if (dst_dt == data_type_t::u8) {
                __m512i clamped = _mm512_max_epi32(_mm512_min_epi32(i32,
                    _mm512_set1_epi32(255)), _mm512_setzero_si512());
                __m128i packed = _mm512_cvtusepi32_epi8(clamped);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(
                    static_cast<uint8_t *>(dst) + m * ldc_dst + n), packed);
            } else {
                __m512i clamped = _mm512_max_epi32(_mm512_min_epi32(i32,
                    _mm512_set1_epi32(127)), _mm512_set1_epi32(-128));
                __m128i packed = _mm512_cvtepi32_epi8(clamped);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(
                    static_cast<int8_t *>(dst) + m * ldc_dst + n), packed);
            }
        }
        for (; n < nr; ++n) {
            float val = src_row[n] / dst_scale;
            int32_t i = static_cast<int32_t>(std::nearbyintf(val)) + dst_zp;
            if (dst_dt == data_type_t::u8) {
                i = std::max(0, std::min(255, i));
                static_cast<uint8_t *>(dst)[m * ldc_dst + n] = static_cast<uint8_t>(i);
            } else {
                i = std::max(-128, std::min(127, i));
                static_cast<int8_t *>(dst)[m * ldc_dst + n] = static_cast<int8_t>(i);
            }
        }
    }
}

// Convert fp32 tile → bf16 output.
__attribute__((target("avx512f,avx512bf16")))
static void convert_fp32_to_bf16(const float *C_fp32, int ldc_fp32,
                                  uint16_t *C_bf16, int ldc_bf16,
                                  int mr, int nr) {
    for (int m = 0; m < mr; ++m) {
        const float *src_row = C_fp32 + m * ldc_fp32;
        uint16_t *dst_row = C_bf16 + m * ldc_bf16;
        int n = 0;
        for (; n + 15 < nr; n += 16) {
            __m256bh bf = _mm512_cvtneps_pbh(_mm512_loadu_ps(src_row + n));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst_row + n), (__m256i)bf);
        }
        for (; n < nr; ++n) {
            uint32_t bits;
            std::memcpy(&bits, &src_row[n], sizeof(bits));
            dst_row[n] = static_cast<uint16_t>(
                (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
        }
    }
}

__attribute__((target("avx512f,avx512bf16,avx512bw,avx512vl,avx512vnni,fma")))
void int8_brgemm_execute(
    const GemmDescriptor &desc,
    const UarchParams &uarch,
    const void *src, const void *weight, void *dst,
    const void *bias, matmul_params &params) {

    const int M = desc.M, N = desc.N, K = desc.K;
    const int lda = desc.lda, ldb = desc.ldb, ldc = desc.ldc;
    const bool transA = desc.transA;
    const bool transB = desc.transB;
    const float alpha = desc.alpha;
    const float beta  = desc.beta;
    const bool src_is_u8  = (desc.src_dt == data_type_t::u8);
    const bool dst_is_bf16 = (desc.dst_dt == data_type_t::bf16);
    const bool dst_is_fp32 = (desc.dst_dt == data_type_t::f32);
    const bool dst_is_int8 = (desc.dst_dt == data_type_t::s8 ||
                              desc.dst_dt == data_type_t::u8);
    const int K_padded = (K + 3) & ~3;

    // ── Extract quantization parameters ──
    auto qp = extract_int8_quant(params);
    const float src_scale = qp.src_scale;
    int32_t src_zp = qp.src_zp;
    float wei_scale_default = 1.0f;
    const float *wei_scale_ptr = qp.wei_scale ? qp.wei_scale : &wei_scale_default;
    const int wei_scale_count = qp.wei_scale_count;
    if (wei_scale_count != 0 && wei_scale_count != 1 && wei_scale_count != N)
        return;

    // dst scale/zp for requantization
    float dst_scale = 1.0f;
    int32_t dst_zp = 0;
    if (dst_is_int8) {
        if (params.quant_params.dst_scale.buff)
            dst_scale = *static_cast<const float *>(params.quant_params.dst_scale.buff);
        if (params.quant_params.dst_zp.buff) {
            if (params.quant_params.dst_zp.dt == data_type_t::s32)
                dst_zp = *static_cast<const int32_t *>(params.quant_params.dst_zp.buff);
        }
    }

    // ── Convert s8 source to u8 ──
    const uint8_t *A_u8 = nullptr;
    std::unique_ptr<uint8_t[]> a_buf;
    int32_t effective_zp = src_zp;

    if (src_is_u8) {
        A_u8 = static_cast<const uint8_t *>(src);
    } else {
        const size_t a_elems = static_cast<size_t>(M) * lda;
        a_buf.reset(new uint8_t[a_elems]);
        const int8_t *a_s8 = static_cast<const int8_t *>(src);
        for (size_t i = 0; i < a_elems; ++i)
            a_buf[i] = static_cast<uint8_t>(static_cast<int>(a_s8[i]) + 128);
        A_u8 = a_buf.get();
        effective_zp = src_zp + 128;
    }

    // For transA: the microkernel reads A[m * lda + k]. With transA,
    // the logical A[m][k] = src[k * lda_orig + m]. The caller already
    // set lda to reflect the transposed stride, so no extra handling needed.
    // The u8 conversion above covers the full buffer regardless of layout.

    // ── Bias → fp32 ──
    const bool has_bias = (desc.bias != nullptr);
    const float *bias_f = nullptr;
    std::unique_ptr<float[]> bias_buf;
    if (has_bias) {
        if (desc.bias_dt == data_type_t::bf16) {
            bias_buf.reset(new float[N]);
            const uint16_t *bb = static_cast<const uint16_t *>(bias);
            for (int n = 0; n < N; ++n) {
                uint32_t bits = static_cast<uint32_t>(bb[n]) << 16;
                std::memcpy(&bias_buf[n], &bits, sizeof(float));
            }
            bias_f = bias_buf.get();
        } else {
            bias_f = static_cast<const float *>(bias);
        }
    }

    // ── Plan ──
    BrgemmPlan plan = plan_int8_brgemm(desc, uarch);
    const int MR = plan.MR, NR = plan.NR, BK = plan.BK;
    const int MB = plan.MB, NB = plan.NB;

    // Separate fused (activation) and remaining (binary) post-ops.
    fused_postop_t fused_op = fused_postop_t::none;
    std::vector<matmul_post_op> remaining_postops;
    for (auto &po : params.postop_) {
        auto pt = po.po_type;
        bool is_activation = (pt == post_op_type_t::relu && po.alpha == 0.0f)
                          || pt == post_op_type_t::gelu_tanh
                          || pt == post_op_type_t::gelu_erf
                          || pt == post_op_type_t::sigmoid
                          || pt == post_op_type_t::tanh
                          || pt == post_op_type_t::swish;
        if (is_activation && fused_op == fused_postop_t::none)
            fused_op = detect_fused_postop(params);
        else if (pt != post_op_type_t::none)
            remaining_postops.push_back(po);
    }
    const bool has_remaining_postops = !remaining_postops.empty();

    // When alpha != 1, beta != 0, dst is int8, or there are remaining
    // post-ops, the microkernel writes to fp32 scratch. The epilogue
    // applies: bias → alpha → beta → activation → remaining postops → store.
    const bool need_fp32_buf = (alpha != 1.0f || beta != 0.0f
                                || dst_is_int8 || has_remaining_postops);

    // ── Pack B + compute col_sum ──
    const int8_t *B_raw = static_cast<const int8_t *>(weight);
    const int np = (N + NR_PACK - 1) / NR_PACK;
    const int k_quads = K_padded / 4;
    const int vnni_stride = NR_PACK * 4;
    const size_t pack_total = static_cast<size_t>(np) * k_quads * vnni_stride;

    static int32_t s_weight_cache =
        matmul_config_t::instance().get_weight_cache();
    const bool is_weights_const = desc.is_weights_const;
    const bool can_cache = is_weights_const && (s_weight_cache != 0);

    const int8_t *packed_ptr = nullptr;
    const int32_t *col_sum_ptr = nullptr;
    const INT8PrepackedWeight *prepacked_b = nullptr;
    std::unique_ptr<int8_t[]> owned_pack;
    std::unique_ptr<int32_t[]> owned_cs;
    [[maybe_unused]] const char *pack_source = "prepack";

    if (can_cache) {
        PrepackedWeightKey bk{weight, K, N, ldb, transB};
        prepacked_b = INT8PrepackedWeightCache::instance().get_or_prepack(
            bk, B_raw);
        if (prepacked_b) {
            packed_ptr = prepacked_b->data;
            col_sum_ptr = prepacked_b->col_sum;
            pack_source = "global_cache";
        }
    }

    if (!prepacked_b) {
        owned_pack.reset(new (std::nothrow) int8_t[pack_total]);
        owned_cs.reset(new (std::nothrow) int32_t[N]);
        if (!owned_pack || !owned_cs) return;

        pack_b_int8_vnni_panel(B_raw, ldb, K, N, transB,
                               owned_pack.get(), owned_cs.get());
        packed_ptr = owned_pack.get();
        col_sum_ptr = owned_cs.get();
        pack_source = "thread_local_prepack";
    }

    auto hot_kernel = select_int8_brgemm_kernel(MR, NR);
    if (!hot_kernel) return;

    const int num_threads = plan.num_threads;

    // For microkernel: when need_fp32_buf, pass nullptr for C_bf16 so
    // kernel writes to fp32 only. Bias is folded into kernel when alpha=1.
    // When need_fp32_buf, kernel produces dequantized fp32 without bias/activation.
    // The epilogue applies: + bias → × alpha → + beta*C_old → activation → store.

    #pragma omp parallel num_threads(num_threads)
    {
        // Per-thread fp32 scratch for epilogue (reused across tiles)
        float *tile_fp32 = nullptr;
        std::unique_ptr<float[]> tile_buf;
        if (need_fp32_buf) {
            tile_buf.reset(new float[MR * NR]);
            tile_fp32 = tile_buf.get();
        }

        #pragma omp for collapse(2) schedule(static)
        for (int ic = 0; ic < M; ic += MB) {
            for (int jc = 0; jc < N; jc += NB) {
                const int mb_act = std::min(MB, M - ic);
                const int nb_act = std::min(NB, N - jc);
                const int n_strips = (nb_act + NR - 1) / NR;

                for (int js = 0; js < n_strips; ++js) {
                    const int col = jc + js * NR;
                    const int nr_act = std::min(NR, N - col);
                    const int panel_idx = col / NR_PACK;
                    const int in_panel_off = col % NR_PACK;
                    const int8_t *pb = packed_ptr
                        + static_cast<size_t>(panel_idx) * k_quads * vnni_stride
                        + in_panel_off * 4;

                    const int32_t *cs = col_sum_ptr + col;
                    const float *ws = (wei_scale_count > 1)
                        ? wei_scale_ptr + col : wei_scale_ptr;
                    int ws_cnt = (wei_scale_count > 1) ? nr_act : 1;
                    // When need_fp32_buf: kernel does dequant only (no bias,
                    // no activation). Epilogue applies bias → alpha → beta → activation → store.
                    const float *tb = (has_bias && !need_fp32_buf)
                        ? bias_f + col : nullptr;
                    fused_postop_t kernel_op = need_fp32_buf
                        ? fused_postop_t::none : fused_op;

                    const int m_panels = (mb_act + MR - 1) / MR;
                    for (int mp = 0; mp < m_panels; ++mp) {
                        const int row = ic + mp * MR;
                        const int mr_act = std::min(MR, M - row);
                        const uint8_t *a_tile = A_u8 + row * lda;

                        float *c_fp32_dst;
                        int ldc_fp32;
                        if (need_fp32_buf) {
                            c_fp32_dst = tile_fp32;
                            ldc_fp32 = NR;
                        } else if (dst_is_fp32) {
                            c_fp32_dst = static_cast<float *>(dst) + row * ldc + col;
                            ldc_fp32 = ldc;
                        } else {
                            c_fp32_dst = tile_fp32 ? tile_fp32
                                : static_cast<float *>(dst) + row * ldc + col;
                            ldc_fp32 = tile_fp32 ? NR : ldc;
                        }

                        uint16_t *c_bf16_dst = (!need_fp32_buf && dst_is_bf16)
                            ? static_cast<uint16_t *>(dst) + row * ldc + col
                            : nullptr;

                        if (mr_act == MR && nr_act == NR) {
                            hot_kernel(
                                a_tile, lda, pb, vnni_stride,
                                c_fp32_dst, ldc_fp32,
                                K, BK, cs, effective_zp, src_scale,
                                ws, ws_cnt, tb, kernel_op,
                                c_bf16_dst, c_bf16_dst ? ldc : 0);
                        } else {
                            int8_brgemm_tail_kernel(
                                a_tile, lda, pb, vnni_stride,
                                c_fp32_dst, ldc_fp32,
                                K, BK, mr_act, nr_act,
                                cs, effective_zp, src_scale,
                                ws, ws_cnt, tb, kernel_op,
                                c_bf16_dst, c_bf16_dst ? ldc : 0);
                        }

                        // ── Epilogue (need_fp32_buf path): ──
                        // C = PostOps(α · dequant(A·B) + Bias + β · C_old)
                        if (need_fp32_buf) {
                            // Step 3: × alpha
                            if (alpha != 1.0f)
                                scale_tile_fp32(tile_fp32, NR, mr_act, nr_act, alpha);

                            // Step 4: + bias
                            if (has_bias) {
                                for (int m = 0; m < mr_act; ++m)
                                    for (int n = 0; n < nr_act; ++n)
                                        tile_fp32[m * NR + n] += bias_f[col + n];
                            }

                            if (beta != 0.0f) {
                                if (dst_is_fp32) {
                                    float *c_old = static_cast<float *>(dst) + row * ldc + col;
                                    beta_accumulate(tile_fp32, NR, c_old, ldc, mr_act, nr_act, beta);
                                } else if (dst_is_bf16) {
                                    // Read old bf16 C, convert to fp32, accumulate
                                    const uint16_t *old_bf = static_cast<const uint16_t *>(dst) + row * ldc + col;
                                    for (int m = 0; m < mr_act; ++m)
                                        for (int n = 0; n < nr_act; ++n) {
                                            uint32_t bits = static_cast<uint32_t>(old_bf[m * ldc + n]) << 16;
                                            float old_val;
                                            std::memcpy(&old_val, &bits, sizeof(float));
                                            tile_fp32[m * NR + n] += beta * old_val;
                                        }
                                } else if (dst_is_int8) {
                                    // Read old s8/u8 C, convert to fp32, accumulate
                                    for (int m = 0; m < mr_act; ++m)
                                        for (int n = 0; n < nr_act; ++n) {
                                            float old_val;
                                            if (desc.dst_dt == data_type_t::u8)
                                                old_val = static_cast<float>(
                                                    static_cast<const uint8_t *>(dst)[row * ldc + col + m * ldc + n]);
                                            else
                                                old_val = static_cast<float>(
                                                    static_cast<const int8_t *>(dst)[row * ldc + col + m * ldc + n]);
                                            tile_fp32[m * NR + n] += beta * old_val;
                                        }
                                }
                            }

                            if (fused_op != fused_postop_t::none) {
                                for (int m = 0; m < mr_act; ++m) {
                                    float *r = tile_fp32 + m * NR;
                                    int n = 0;
                                    for (; n + 15 < nr_act; n += 16) {
                                        __m512 v = _mm512_loadu_ps(r + n);
                                        _mm512_storeu_ps(r + n, apply_fused_postop(v, fused_op));
                                    }
                                    for (; n < nr_act; ++n) {
                                        __m512 v = _mm512_set1_ps(r[n]);
                                        r[n] = _mm_cvtss_f32(_mm512_castps512_ps128(
                                            apply_fused_postop(v, fused_op)));
                                    }
                                }
                            }

                            // Apply remaining (non-fused) post-ops (binary_add/mul)
                            if (has_remaining_postops) {
                                apply_postops_tile(tile_fp32, NR,
                                    mr_act, nr_act, row, col,
                                    nullptr, remaining_postops);
                            }

                            // Store to final destination
                            if (dst_is_int8) {
                                void *dst_row = (desc.dst_dt == data_type_t::u8)
                                    ? static_cast<void *>(static_cast<uint8_t *>(dst) + row * ldc + col)
                                    : static_cast<void *>(static_cast<int8_t *>(dst) + row * ldc + col);
                                requantize_tile(tile_fp32, NR, dst_row, ldc,
                                    mr_act, nr_act, desc.dst_dt, dst_scale, dst_zp);
                            } else if (dst_is_bf16) {
                                uint16_t *bf_dst = static_cast<uint16_t *>(dst) + row * ldc + col;
                                convert_fp32_to_bf16(tile_fp32, NR, bf_dst, ldc, mr_act, nr_act);
                            } else {
                                float *fp_dst = static_cast<float *>(dst) + row * ldc + col;
                                for (int m = 0; m < mr_act; ++m)
                                    std::memcpy(fp_dst + m * ldc, tile_fp32 + m * NR,
                                                nr_act * sizeof(float));
                            }
                        }
                    }
                }
            }
        }
    }

    static bool s_log = apilog_info_enabled();
    if (s_log) {
        const char *dst_str = dst_is_fp32 ? "fp32" : dst_is_bf16 ? "bf16"
                            : desc.dst_dt == data_type_t::u8 ? "u8" : "s8";
        apilog_info("Native INT8 BRGEMM looper: M=", M, " N=", N, " K=", K,
                    " MB=", MB, " NB=", NB, " BK=", BK,
                    " MR=", MR, " NR=", NR,
                    " alpha=", alpha, " beta=", beta,
                    " transA=", transA ? "true" : "false",
                    " threads=", num_threads,
                    " src=", src_is_u8 ? "u8" : "s8",
                    " dst=", dst_str,
                    " pack=", pack_source);
    }
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
