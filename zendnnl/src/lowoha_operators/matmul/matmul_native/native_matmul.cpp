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

#include "lowoha_operators/matmul/matmul_native/native_matmul.hpp"

#include <algorithm>
#include <exception>
#include <sstream>
#include <string>

#include "common/data_types.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/common/omp_thread_control.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_gemv_narrow.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/bf16_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/bf16_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/fp32_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/int8_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/int8_gemv_direct.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_brgemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/gemm_descriptor.hpp"
#include "lowoha_operators/matmul/matmul_native/common/kernel_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_operator_bridge.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_weight_cache.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/bf16_gemm_looper.hpp"
#include "lowoha_operators/matmul/matmul_native/gemm/looper/fp32_gemm_looper.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::dtype_info;
using zendnnl::common::size_of;
using namespace zendnnl::error_handling;

// ════════════════════════════════════════════════════════════════════════
// Best-algo selector for BF16 M≤8 (GEMV/decode) shapes.
// Called by ALGO 0 (dynamic_dispatch) to route M≤8 shapes to
// native_brgemm or aocl_dlp_blocked.
//
// M=5-8: always native_brgemm (the BRGEMM looper has different cache
// access patterns; the per-L3 rule below covers only M=1-4).
//
// M=1-4 multi-thread: native_brgemm unless BKC packed data overflows
// the L3 of one L3 domain while raw B fits in aggregate L3. In that case DLP's
// unpacked streaming stays L3-resident. Exception: K > 5120 shapes
// with low NR-pad waste (≤25%) benefit from BKC K-contiguous format
// even with L3 overflow.
// Validated against 72 LLM shapes × {M=1,2,4,8} × {32,64,128}t.
// Catches 14 losing shapes (up to -195%) with 1 minor regression (+5%).
//
// M=1 single-thread: shape-based DLP gates for small-N high-K where
// BKC dispatch overhead exceeds DLP's thin-dispatcher advantage.
// Decision tree for L1d=48KB, L2=1MB and 32MB of L3 per L3 domain:
//
//   Rules are evaluated in order (first match wins):
//
//   ┌──────────────────────────────────┬──────────┬────────────────────────┐
//   │ Shape characteristic             │ Best     │ Reason                 │
//   ├──────────────────────────────────┼──────────┼────────────────────────┤
//   │ M≤4 MT: BKC per L3 ≥ its L3,   │ DLP      │ BKC overflows its L3; │
//   │   raw B < agg L3,               │          │ DLP stays L3-resident  │
//   │   K≤5120 or NR-waste>25%        │          │                        │
//   │ N ≤ 4                            │ BRGEMM   │ Narrow-GEMV xmm kernel │
//   │                                  │          │ inside bf16_gemv_direct│
//   │                                  │          │ is pack-free, 0–25%    │
//   │                                  │          │ waste.  Beats DLP by   │
//   │                                  │          │ a wide margin on       │
//   │                                  │          │ narrow-N decode shapes.│
//   │ N≤32, K≥256                     │ DLP      │ 50-75% NR-pad waste    │
//   │ N=33-48, K≥384                 │ DLP      │ BKC dispatch overhead  │
//   │ N∈(49,63), ≥25%pad, K≥8N,384  │ DLP      │ High zero-pad BW waste │
//   │ N∈(65,127), ≥25%pad, K≥5N,384 │ DLP      │ Double dispatch + waste │
//   │ Packed B ≤ L2                    │ BRGEMM   │ BKC+flat GEMV kernel   │
//   │ Everything else                  │ BRGEMM   │ General BRGEMM looper  │
//   └──────────────────────────────────┴──────────┴────────────────────────┘
// ════════════════════════════════════════════════════════════════════════
static matmul_algo_t bf16_gemv_best_algo_impl(
        int M, int N, int K, int num_threads, bool transB) {
    // M=5-8: always native BRGEMM. The per-L3 overflow rule below covers the
    // M=1-4 paths (BKC GEMV + decode flat); M=5-8 uses the general BRGEMM
    // looper, which has different cache access patterns.
    if (M > 4) return matmul_algo_t::native_brgemm;

    // ── M=1-4 multi-thread: per-L3 overflow gate ─────────────────────
    // BKC packing pads each thread's N-slice to a multiple of BKC_NR_PAD (64).
    // When the packed footprint per L3 domain exceeds its L3 but raw B fits in
    // aggregate L3, DLP streams unpacked B from L3 while BKC spills to DRAM.
    // High-K shapes (K > 5120) with low NR-pad waste (≤25%) still benefit
    // from BKC's K-contiguous streaming pattern even with L3 overflow.
    if (num_threads > 1) {
        static const UarchParams &uarch = detect_uarch();
        const int nt = m1_gemv_cap_threads(N, num_threads);
        const int K_padded = (K + 1) & ~1;
        const int chunk = (N + nt - 1) / nt;
        const int n_pad = ((chunk + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
        const int ccx_w = std::max(1, uarch.ccx_cores);
        const size_t per_ccd = static_cast<size_t>(K_padded) * n_pad
                * sizeof(uint16_t) * ccx_w;
        const size_t l3_ccd = static_cast<size_t>(uarch.l3_bytes_per_ccd);

        if (per_ccd >= l3_ccd) {
            const size_t raw_B = static_cast<size_t>(K) * N * sizeof(uint16_t);
            const int num_ccds = std::max(1, nt / ccx_w);
            const size_t agg_l3 = l3_ccd * num_ccds;
            const int nr_waste_pct
                    = n_pad > 0 ? (n_pad - chunk) * 100 / n_pad : 0;

            if (raw_B < agg_l3 && (K <= 5120 || nr_waste_pct > 25))
                return matmul_algo_t::aocl_dlp_blocked;
        }
        return matmul_algo_t::native_brgemm;
    }

    // ── M=1 single-thread heuristics below ────────────────────────────
    if (num_threads != 1) return matmul_algo_t::native_brgemm;

    // ── Narrow-GEMV zone (M == 1, N ≤ 4, !transB) ──────────────────────
    // native_brgemm's M=1 entry (`bf16_gemv_direct`) routes N ≤ 4 to
    // the pack-free `bf16_gemv_narrow` xmm kernel.  That kernel beats
    // DLP by a wide margin across all K on narrow-N decode shapes
    // (e.g., N=1 at K=234 is ~2.8× DLP, N=3 at K~6000 is ~1.3× DLP).
    // Must short-circuit BEFORE the N ≤ 32
    // DLP zone below, otherwise high-K narrow shapes hit the legacy
    // DLP rule and miss the narrow kernel entirely.  Applies at any
    // K (narrow kernel has no K-based dispatch overhead).
    //
    // Gated on M == 1 because `bf16_gemv_direct` itself only engages
    // for M=1 (returns false otherwise).  For M=2..4 with N ≤ 4, the
    // dispatch falls through to `bf16_brgemm_execute` — standard
    // BRGEMM, not the narrow kernel — which was NOT re-validated at
    // narrow-N after the narrow-kernel addition.  Keeping those
    // shapes on the legacy DLP rule avoids a silent behaviour change
    // on M=2..4 narrow-N paths that haven't been benchmarked.
    //
    // Gated on `!transB` because `bf16_gemv_narrow` only engages for
    // row-major (non-transposed) B (see bf16_gemv_direct.cpp:347
    // `if (N <= kBf16GemvNarrowMaxN && !desc.transB)`).  When transB
    // is true the narrow path is bypassed and BKC packing fires
    // instead — for N ≤ 4 BKC pads to NR=64, wasting 75–94% of every
    // VDPBF16PS load on zero-padded lanes — measurably worse than
    // DLP.  Letting transB=true narrow shapes fall through to the
    // DLP rules below routes them onto the better backend.
    if (M == 1 && !transB && N <= kBf16GemvNarrowMaxN)
        return matmul_algo_t::native_brgemm;

    const int packed_N = ((N + BKC_NR_PAD - 1) / BKC_NR_PAD) * BKC_NR_PAD;
    const int pad_waste = packed_N - N;
    const size_t packed_K = static_cast<size_t>((K + 1) & ~1);
    const size_t b_packed_bytes = packed_K * packed_N * sizeof(uint16_t);

    // ── DLP zone 1: small-N with large K ────────────────────────────
    // N≤48: BKC dispatch overhead (pack + block iteration + tail-only
    // kernel) dominates for K-dominant shapes. DLP's thin dispatcher
    // and row-major streaming avoid this entirely.
    //   N∈[5,32], K≥256: 50-75% NR-pad waste on every ZMM load.
    //     (N ≤ 4 is handled above by the narrow-GEMV zone.)
    //   N=33-48, K≥384: no pad waste but BKC dispatch cost is high
    //     relative to the small per-row compute.
    if (N <= 32 && K >= 256) return matmul_algo_t::aocl_dlp_blocked;
    if (N > 32 && N <= 48 && K >= 384) return matmul_algo_t::aocl_dlp_blocked;

    // ── DLP zone 2: NR-padding waste with K-dominant shapes ─────────
    // When packed_N has ≥25% zero columns AND K is large relative to N,
    // the BKC kernel wastes bandwidth on zero-padded loads.
    //   N ∈ (48,64) with padding: DLP wins for K-dominant (K ≥ 8*N).
    //   N ∈ (64,128) with tail: double dispatch overhead + padding.
    //     DLP wins only for very K-dominant shapes (K ≥ 5*N, K ≥ 384).
    if (pad_waste * 4 >= packed_N && N > 48) {
        if (N < 64 && K >= 8 * N && K >= 384)
            return matmul_algo_t::aocl_dlp_blocked;
        if (N > 64 && N < 128 && K >= 5 * N && K >= 384)
            return matmul_algo_t::aocl_dlp_blocked;
    }

    // ── BKC-GEMV zone: packed B fits in L2 ──────────────────────────
    // Block-aware packing eliminates stride gaps for any N.
    // Wide blocks (384 cols, NP=5/6) used for N ∈ (256, 512].
    {
        static const size_t l2 = static_cast<size_t>(detect_uarch().l2_bytes);
        if (b_packed_bytes <= l2) return matmul_algo_t::native_brgemm;
    }

    // ── BRGEMM (default for L3-resident shapes) ────────────────────
    return matmul_algo_t::native_brgemm;
}

// Wrapper that logs the dispatch decision.
matmul_algo_t bf16_gemv_best_algo(
        int M, int N, int K, int num_threads, bool transB) {
    matmul_algo_t result
            = bf16_gemv_best_algo_impl(M, N, K, num_threads, transB);
    static bool s_log = apilog_info_enabled();
    if (s_log) {
        const char *name = (result == matmul_algo_t::native_brgemm) ? "BRGEMM"
                : (result == matmul_algo_t::aocl_dlp_blocked)       ? "DLP"
                                                                    : "???";
        apilog_info("ALGO0 decode dispatch: M=", M, " K=", K, " N=", N,
                " threads=", num_threads, " transB=", transB,
                " B=", static_cast<size_t>(K) * N * 2 / 1024,
                "KB"
                " → ",
                name);
    }
    return result;
}

// Common descriptor setup — shared by both ALGO 10 and ALGO 11.
static GemmDescriptor make_desc(bool transA, bool transB, int M, int N, int K,
        float alpha, float beta, int lda, int ldb, int ldc,
        bool is_weights_const, int num_threads, const matmul_params &params) {
    GemmDescriptor desc;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.lda = lda;
    desc.ldb = ldb;
    desc.ldc = ldc;
    desc.transA = transA;
    desc.transB = transB;
    desc.alpha = alpha;
    desc.beta = beta;
    desc.src_dt = params.dtypes.src;
    desc.wei_dt = params.dtypes.wei;
    desc.dst_dt = params.dtypes.dst;
    desc.bias_dt = params.dtypes.bias;
    desc.src_elem_size = size_of(desc.src_dt);
    desc.wei_elem_size = size_of(desc.wei_dt);
    desc.dst_elem_size = size_of(desc.dst_dt);
    desc.bias = nullptr;
    desc.is_weights_const = is_weights_const;
    desc.num_threads = num_threads;
    return desc;
}

bool native_matmul_execute(matmul_algo_t kernel, char layout, bool transA,
        bool transB, int M, int N, int K, float alpha, const void *src, int lda,
        const void *weight, int ldb, const void *bias, float beta, void *dst,
        int ldc, bool is_weights_const, int num_threads,
        matmul_params &params) {

    (void)layout;
    if (M <= 0 || N <= 0 || K <= 0) return true;

    // Native fused Swish implements SiLU (alpha == 1) only. Parameterized
    // Swish must fall back to DLP, which consumes the post-op alpha.
    for (const auto &po : params.postop_) {
        if (po.po_type == post_op_type_t::swish && po.alpha != 1.0f) {
            log_info(
                    "Native matmul: non-unit Swish alpha is not supported, "
                    "falling back to aocl_dlp");
            return false;
        }
    }

    const bool is_bf16 = (params.dtypes.src == data_type_t::bf16
            && params.dtypes.wei == data_type_t::bf16);

    // ════════════════════════════════════════════════════════════════════
    // native_brgemm as dynamic dispatch or the auto tuner selects it:
    // BRGEMM-based paths with GEMM fallback.  ZENDNNL_MATMUL_ALGO=11 runs
    // matmul_direct on native_layout_matmul instead.
    //   BF16 + AVX512BF16 → bf16_brgemm_execute
    //   FP32 + AVX512F → brgemm_execute (FP32 BRGEMM)
    //   No AVX512 → gemm_execute (scalar/AVX2 fallback)
    // ════════════════════════════════════════════════════════════════════
    if (kernel == matmul_algo_t::native_brgemm) {
        // AIGtest Failed for aplha = 0 and beta = -0.5
        // fall back to DLP which handles scalars.
        // TODO: Fix this in the future
        if (alpha == 0.0f) {
            log_info(
                    "Native BRGEMM: alpha=0 not supported, falling back to "
                    "aocl_dlp");
            return false;
        }

        const UarchParams &uarch = detect_uarch();
        int nt_exec = num_threads;
        if (M == 1 && is_bf16 && num_threads > 1)
            nt_exec = m1_gemv_cap_threads(N, num_threads);
        GemmDescriptor desc = make_desc(transA, transB, M, N, K, alpha, beta,
                lda, ldb, ldc, is_weights_const, nt_exec, params);
        desc.bias = bias;

        // INT8 paths: u8/s8 source × s8 weights → bf16/fp32.
        // M=1 single-thread: try BKC GEMV fast path first (L2-resident B).
        // All other shapes: general INT8 BRGEMM looper.
        const bool is_int8 = ((params.dtypes.src == data_type_t::u8
                                      || params.dtypes.src == data_type_t::s8)
                && params.dtypes.wei == data_type_t::s8);
        if (is_int8) {
            if (!uarch.avx512vnni) return false;

            // Native INT8 supports:
            //   src: per-tensor scale + per-tensor zero point only
            //   wei: per-tensor (wei_scale_count==1) or per-channel (==N)
            // Per-group, per-token, and other granularities (e.g. src_scale with
            // dims > 1, or wei_scale_count not in {0,1,N}) fall back to DLP.
            {
                auto qp = extract_int8_quant(params);
                if (qp.wei_scale_count != 0 && qp.wei_scale_count != 1
                        && qp.wei_scale_count != N)
                    return false;
                const auto &src_dims = params.quant_params.src_scale.dims;
                if (!src_dims.empty() && src_dims.back() > 1) return false;
                const auto &zp_dims = params.quant_params.src_zp.dims;
                if (!zp_dims.empty() && zp_dims.back() > 1) return false;
            }

            // INT8 BKC GEMV is single-thread only today; BF16 M=1 can use bf16_gemv_direct
            // with nt>1. Parallel INT8 M=1 goes straight to BRGEMM.
            if (M == 1 && num_threads == 1) {
                if (int8_gemv_direct(
                            desc, uarch, src, weight, dst, bias, params))
                    return true;
            }
            int8_brgemm_execute(desc, uarch, src, weight, dst, bias, params);
            return true;
        }

        if (is_bf16 && uarch.avx512bf16) {
            // M=1: try BKC GEMV first (single-thread: global/thread-local pack;
            // multi-thread: per-thread N-slices, each packs its columns). On false,
            // fall through to BRGEMM looper.
            if (M == 1) {
                if (bf16_gemv_direct(
                            desc, uarch, src, weight, dst, bias, params))
                    return true;
            }
            bf16_brgemm_execute(desc, uarch, src, weight, dst, bias, params);
        } else if (uarch.avx512f) {
            // FP32 BRGEMM
            brgemm_execute(desc, uarch, src, weight, dst, bias, params);
        } else {
            gemm_execute(desc, uarch, src, weight, dst, bias, params);
        }
        return true;
    }

    // ════════════════════════════════════════════════════════════════════
    // ALGO 10 (Native GEMM): GEMM-based paths only.
    //   All shapes → bf16_gemm_execute (Planner → Looper → GEMM kernel)
    //   No BRGEMM microkernel, no GEMV special path.
    //   Uses inner-product style, adaptive NR, simpler looper.
    // ════════════════════════════════════════════════════════════════════

    const UarchParams &uarch = detect_uarch();
    GemmDescriptor desc = make_desc(transA, transB, M, N, K, alpha, beta, lda,
            ldb, ldc, is_weights_const, num_threads, params);
    desc.bias = bias;

    if (is_bf16 && uarch.avx512bf16) {
        bf16_gemm_execute(desc, uarch, src, weight, dst, bias, params);
    } else {
        // FP32 GEMM (avx512f or scalar fallback — gemm_execute handles both)
        gemm_execute(desc, uarch, src, weight, dst, bias, params);
    }
    return true;
}

native_layout_query_t native_layout_query_for(
        const matmul_params &params, int K, int N, int ldb, bool transB) {
    native_layout_query_t q;
    q.wei_dt = params.dtypes.wei;
    q.src_dt = params.dtypes.src;
    const data_type_t compute = params.dtypes.compute;
    if (params.dynamic_quant
            && (q.src_dt == data_type_t::bf16 || q.src_dt == data_type_t::f32)
            && q.wei_dt == data_type_t::s8
            && (compute == data_type_t::s8 || compute == data_type_t::u8)) {
        q.src_dt = compute;
    }
    q.K = K;
    q.N = N;
    q.ldb = ldb;
    q.transB = transB;
    if (has_per_group_wei_scale(params.quant_params, N)) {
        const int64_t groups = params.quant_params.wei_scale.dims[0];
        q.wei_group_size = K % groups == 0 ? static_cast<int>(K / groups) : -1;
    }
    return q;
}

namespace {

/// Why the kernel did not serve a call it was given.
const char *kernel_status_text(status_t st) {
    switch (st) {
        case status_t::unimplemented: return "the kernel does not implement it";
        case status_t::isa_unsupported:
            return "the CPU lacks an instruction set the kernel needs";
        default: return "the kernel failed";
    }
}

const char *gated_act_name(matmul_gated_act_t act) {
    switch (act) {
        case matmul_gated_act_t::silu_and_mul: return "silu_and_mul";
        case matmul_gated_act_t::gelu_and_mul: return "gelu_and_mul";
        case matmul_gated_act_t::swiglu_oai_mul: return "swiglu_oai_mul";
        default: return "none";
    }
}

/// Where the weight of a call the native BRGEMM takes comes from.
enum class native_source_t : uint8_t {
    none, ///< not for the native BRGEMM: the other routes serve the call
    raw, ///< a raw weight the layout rule places on the layout
    /// An AOCL prepack whose raw weight the layout rule places on the layout:
    /// unreordered, then packed per the weight cache mode.  AOCL still reads
    /// it until a weight cache mode 2 pack rewrites it in place.
    aocl_prepacked,
    in_place, ///< a buffer an earlier native call rewrote in place
};

/// The bytes at weight a call vouches for as its weight buffer: what it
/// reads from a bf16 / s8 weight (an AOCL prepack this route unreorders, or
/// the span of a plain weight), or the capacity it states when larger; 0
/// when neither is known.
size_t weight_buffer_bytes(
        const matmul_params &params, int K, int N, int ldb, bool transB) {
    size_t reads = 0;
    const data_type_t wdt = params.dtypes.wei;
    if ((wdt == data_type_t::bf16 || wdt == data_type_t::s8) && K > 0 && N > 0
            && ldb > 0 && params.packing.pack_format_b == 0) {
        if (params.mem_format_b == 'r') {
            if (aocl_prepacked_weight(params)
                    && aocl_prepack_unreorderable(params))
                reads = aocl_prepack_bytes(
                        K, N, ldb, transB, wdt == data_type_t::s8);
        } else if (params.mem_format_b == 'n') {
            reads = native_raw_weight_bytes(wdt, K, N, ldb, transB);
        }
    }
    return std::max(reads, params.wei_buffer_capacity_bytes);
}

/// reason is set when the weight is an in-place image the call cannot read.
/// owned is weight_buffer_bytes of the call; image is what its check for an
/// in-place image found.
native_source_t classify_source(const matmul_params &params, const void *weight,
        int K, int N, int ldb, bool transB, int batch_count,
        bool process_native, size_t owned, int &pack_nr, const char *&reason,
        native_image_check_t &image) {
    // A native in-place image has this route as its only reader, whatever
    // algorithm the call selects.
    if (native_weight_find_image(weight, owned, image)) {
        const native_layout_decision_t rule = decide_native_layout(
                native_layout_query_for(params, K, N, ldb, transB));
        pack_nr = kNativeLayoutPackNr;
        if (!rule.native)
            reason = rule.reason;
        else if (batch_count != 1)
            reason = "batched call";
        return native_source_t::in_place;
    }
    bool aocl = false;
    if (params.mem_format_b == 'r') {
        if (!process_native || !aocl_prepacked_weight(params))
            return native_source_t::none;
        aocl = true;
    } else if (params.mem_format_b != 'n' || !native_brgemm_requested(params)) {
        return native_source_t::none;
    }

    const native_layout_decision_t rule = decide_native_layout(
            native_layout_query_for(params, K, N, ldb, transB));
    if (!rule.native || batch_count != 1 || params.packing.pack_format_b != 0)
        return native_source_t::none;
    pack_nr = rule.pack_nr;
    if (aocl)
        return aocl_prepack_unreorderable(params)
                ? native_source_t::aocl_prepacked
                : native_source_t::none;
    // A buffer a custom-kernel pack rewrote in place has its reader on that
    // route.
    if (custom_kernel_packed_in_place(weight)) return native_source_t::none;
    return native_source_t::raw;
}

} // namespace

bool native_layout_matmul(char layout, bool transA, bool transB, int M, int N,
        int K, float alpha, const void *src, int lda, const void *weight,
        int ldb, const void *bias, float beta, void *dst, int ldc,
        bool is_weights_const, int batch_count, int num_threads,
        matmul_gated_act_t gated_act, matmul_params &params,
        zendnnl::profile::profiler_t *profiler, status_t &status,
        const matmul_batch_params_t *batch) {
    const bool process_native = matmul_config_t::is_native_brgemm_algo();
    int pack_nr = 0;
    const char *reason = nullptr;
    const size_t owned = weight_buffer_bytes(params, K, N, ldb, transB);
    native_weight_ref_t ref;
    const native_source_t source
            = classify_source(params, weight, K, N, ldb, transB, batch_count,
                    process_native, owned, pack_nr, reason, ref.image);
    if (source == native_source_t::none) return false;
    bool in_place = source == native_source_t::in_place;
    if (reason == nullptr && layout != 'r' && layout != 'R')
        reason = "column-major call";
    if (reason == nullptr && params.mem_format_a != 'n')
        reason = "reordered source";
    // The one-op group route defines a gated call with beta or post-ops, and
    // takes only an f32 / bf16 / f16 gated dst.
    if (reason == nullptr && gated_act != matmul_gated_act_t::none
            && (beta != 0.0f || !params.postop_.empty()
                    || (params.dtypes.dst != data_type_t::f32
                            && params.dtypes.dst != data_type_t::bf16)))
        reason = "gated call with beta, post-ops or a dst other than f32 / "
                 "bf16";

    ref.weight = weight;
    ref.pack_nr = pack_nr;
    ref.aocl_blocked = source == native_source_t::aocl_prepacked
            || (in_place && aocl_prepacked_weight(params));
    ref.interleave_split_halves = gated_act == matmul_gated_act_t::silu_and_mul
            || gated_act == matmul_gated_act_t::gelu_and_mul;
    ref.weight_cache_mode = is_weights_const
            ? effective_weight_cache_type(params.weight_cache_type)
            : 0;
    // A buffer rewritten in place must have no other reader, which only the
    // process-wide ALGO 11 guarantees (every call then reaches the image
    // check of classify_source); a call that selects the native BRGEMM by
    // itself caches the layout out of place instead.
    if (ref.weight_cache_mode == 2 && !process_native)
        ref.weight_cache_mode = 1;
    ref.capacity_bytes = params.wei_buffer_capacity_bytes;

    if (reason != nullptr) {
        // A concurrent first call may have rewritten the buffer since it was
        // classified; then no other route may read it either.
        if (!in_place) in_place = native_weight_packed_in_place(weight, owned);
        if (!in_place) {
            log_info("native BRGEMM: a ", reason, " is not served");
            return false;
        }
        status = status_t::unimplemented;
    } else {
        GemmDescriptor desc = make_desc(transA, transB, M, N, K, alpha, beta,
                lda, ldb, ldc, is_weights_const, num_threads, params);
        desc.bias = bias;
        desc.gated_act = gated_act;
        const thread_guard tg(num_threads, thread_guard::max_threads());
        // Nothing may throw out of matmul_direct (allocations of the call's
        // buffers, an executor).
        try {
            status = native_brgemm_execute(desc, ref, src, dst, bias, params);
        } catch (const std::exception &e) {
            apilog_error("matmul_direct: native BRGEMM: ", e.what());
            status = status_t::failure;
        }
        if ((status == status_t::unimplemented
                    || status == status_t::isa_unsupported)
                && !in_place && !native_weight_packed_in_place(weight, owned)) {
            log_info(
                    "native BRGEMM does not serve this call; the other "
                    "routes do");
            return false;
        }
    }

    // The line matmul_direct logs for every other route, with the fields of
    // this one.
    std::string line;
    if (apilog_info_enabled() || profiler != nullptr) {
        const std::string po_dtypes = post_op_dtypes_text(params);
        std::ostringstream ss;
        ss << "LOWOHA matmul_direct: M=" << M << ", N=" << N << ", K=" << K
           << ", alpha=" << alpha << ", beta=" << beta << ", lda=" << lda
           << ", ldb=" << ldb << ", ldc=" << ldc
           << ", transA=" << (transA ? "true" : "false")
           << ", transB=" << (transB ? "true" : "false")
           << ", input_dtype=" << dtype_info(params.dtypes.src)
           << ", weight_dtype=" << dtype_info(params.dtypes.wei)
           << ", output_dtype=" << dtype_info(params.dtypes.dst)
           << ", bias_dtype=" << dtype_info(params.dtypes.bias)
           << ", bias=" << (bias != nullptr ? "true" : "false")
           << ", is_weights_const=" << (is_weights_const ? "true" : "false")
           << ", post_op=[" << post_op_names_text(params) << "]"
           << ", post_op_dtype=[" << (po_dtypes.empty() ? "none" : po_dtypes)
           << "], Batch_A=" << (batch ? batch->Batch_A : batch_count)
           << ", Batch_B=" << (batch ? batch->Batch_B : batch_count)
           << ", plugin_op=" << params.plugin_op
           << ", dynamic_quant=" << (params.dynamic_quant ? "true" : "false")
           << ", kernel=native_brgemm, layout=native_nr" << pack_nr
           << ", weight_cache=";
        if (in_place)
            ss << "inplace";
        else
            ss << ref.weight_cache_mode;
        ss << ", gated_act=" << gated_act_name(gated_act);
        line = ss.str();
    }
    if (apilog_info_enabled()) apilog_info(line);
    if (profiler != nullptr) {
        profiler->tbp_stop();
        profilelog_verbose(line, ", weight_address=", weight,
                ", time=", profiler->tbp_elapsedtime(),
                profiler->get_res_str());
    }
    if (status != status_t::success) {
        apilog_error("matmul_direct: native BRGEMM did not serve the call (",
                reason != nullptr ? reason : kernel_status_text(status), ")",
                in_place ? "; no other route reads a buffer it rewrote" : "");
    }
    return true;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
