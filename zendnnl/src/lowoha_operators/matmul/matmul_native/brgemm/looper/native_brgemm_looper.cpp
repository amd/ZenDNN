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
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/bf16/bf16_brgemm_ukernel.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/kernel/native_kernel_registry.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_int8_tile.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/looper/native_partition.hpp"
#include "lowoha_operators/matmul/matmul_native/brgemm/planner/native_planner.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_epilogue.hpp"
#include "lowoha_operators/matmul/matmul_native/common/native_pack.hpp"
#include "lowoha_operators/matmul/matmul_native/native_layout_policy.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::dtype_info;
using zendnnl::error_handling::apilog_verbose;
using zendnnl::error_handling::apilog_verbose_enabled;
using zendnnl::error_handling::log_error;

namespace {

struct aligned_free_t {
    void operator()(void *p) const { zendnnl_aligned_free(p); }
};

int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

/// dst[m][k] = src[k * lda + m] for a transposed bf16 source, in side x side
/// blocks, on threads threads when M * K >= par_min.
void copy_transposed(const uint16_t *src, int lda, int M, int K, uint16_t *dst,
        int side, int threads, int64_t par_min) {
    const int m_blocks = ceil_div(M, side);
    const bool parallel = threads > 1 && static_cast<int64_t>(M) * K >= par_min;
#pragma omp parallel for num_threads(threads) schedule(static) if (parallel)
    for (int mbk = 0; mbk < m_blocks; ++mbk) {
        const int m_end = std::min(M, (mbk + 1) * side);
        for (int k0 = 0; k0 < K; k0 += side) {
            const int k_end = std::min(K, k0 + side);
            for (int m = mbk * side; m < m_end; ++m)
                for (int k = k0; k < k_end; ++k)
                    dst[static_cast<size_t>(m) * K + k]
                            = src[static_cast<size_t>(k) * lda + m];
        }
    }
}

struct bf16_operands_t {
    const uint16_t *A = nullptr;
    int lda = 0;
    /// Kernels by row count, 1 .. the plan's mr.
    const bf16_brgemm_fn_t *kern = nullptr;
};

void bf16_block(const native_plan_t &p, const bf16_operands_t &op,
        const native_weight_view_t &w, const native_epilogue_t &ep, int mb,
        int K, int m0, int n0, float *acc) {
    const int b_stride = w.group_stride();
    const uint16_t *b
            = reinterpret_cast<const uint16_t *>(w.block(n0 / w.pack_nr))
            + (n0 % w.pack_nr) * VNNI_PAIR;
    const bool one_pass = p.kc >= K;
    for (int k0 = 0; k0 < K; k0 += p.kc) {
        const int kl = std::min(p.kc, K - k0);
        const float beta = k0 == 0 ? 0.0f : 1.0f;
        const uint16_t *bk = b + static_cast<size_t>(k0 / VNNI_PAIR) * b_stride;
        for (int ir = 0; ir < mb; ir += p.mr) {
            const int mr = std::min(p.mr, mb - ir);
            float *c = one_pass ? acc : acc + static_cast<size_t>(ir) * p.nr;
            op.kern[mr](op.A + static_cast<size_t>(m0 + ir) * op.lda + k0,
                    op.lda, bk, b_stride, c, p.nr, kl, kl, beta, nullptr,
                    fused_postop_t::none, nullptr, 0);
            if (k0 + kl == K)
                apply_epilogue_tile(ep, c, p.nr, m0 + ir, mr, n0, p.nr);
        }
    }
}

void int8_block(const native_plan_t &p, const native_int8_call_t &call,
        const native_weight_view_t &w, const native_epilogue_t &ep,
        const native_int8_out_t *direct, int mb, int m0, int n0, float *acc) {
    compute_int8_tile(p, call, w, m0, mb, n0, p.nr, acc, p.nr, direct);
    if (direct == nullptr) apply_epilogue_tile(ep, acc, p.nr, m0, mb, n0, p.nr);
}

/// The widths of an INT8 width mask, widest first.
const char *int8_cw_text(uint8_t mask) {
    static const char *const kText[] = {"-", "16", "32", "32/16"};
    return kText[mask & kNativeInt8CwAll];
}

} // namespace

status_t native_brgemm_execute(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, matmul_params &params) {
    const int M = desc.M, N = desc.N, K = desc.K;
    const bool bf16 = desc.src_dt == data_type_t::bf16
            && desc.wei_dt == data_type_t::bf16;
    const bool dynamic_src = params.dynamic_quant
            && (desc.src_dt == data_type_t::bf16
                    || desc.src_dt == data_type_t::f32);
    const bool int8 = desc.wei_dt == data_type_t::s8
            && (desc.src_dt == data_type_t::u8 || desc.src_dt == data_type_t::s8
                    || dynamic_src);
    if (!bf16 && !int8) {
        log_error("native BRGEMM: src=", dtype_info(desc.src_dt),
                " wei=", dtype_info(desc.wei_dt), " is not served");
        return status_t::unimplemented;
    }
    const UarchParams &uarch = detect_uarch();
    if ((bf16 ? !uarch.avx512bf16 : !uarch.avx512vnni) || !native_base_isa()) {
        log_error("native BRGEMM: the CPU lacks ",
                bf16 ? "AVX512-BF16" : "AVX512-VNNI",
                " or AVX-512 F / BW / VL / DQ or FMA");
        return status_t::isa_unsupported;
    }
    const int n_out = desc.gated_act != matmul_gated_act_t::none ? N / 2 : N;
    const int min_lda = desc.transA ? M : K;
    if (M < 0 || N < 1 || K < 1 || desc.ldc < n_out
            || (bf16 && desc.lda < min_lda)) {
        log_error("native BRGEMM: bad shape M=", M, " N=", N, " K=", K,
                " lda=", desc.lda, " ldc=", desc.ldc);
        return status_t::failure;
    }
    if (M > 0 && (src == nullptr || dst == nullptr)) {
        log_error("native BRGEMM: null src or dst");
        return status_t::failure;
    }
    if (M == 0) return status_t::success;
    // The planner and the tile grid need N to be whole pack blocks.
    if ((weight.pack_nr != kNativeMinPackNr
                && weight.pack_nr != kNativeMaxPackNr)
            || N < weight.pack_nr || N % weight.pack_nr != 0) {
        log_error("native BRGEMM: N=", N,
                " is not a positive multiple of pack_nr=", weight.pack_nr);
        return status_t::failure;
    }

    // Every check that can turn the call away runs before any work: a call
    // turned away leaves the source and the weight as they were.
    status_t st = status_t::success;
    native_int8_decision_t decision;
    if (int8) {
        decision = decide_int8_call(desc, params, bias);
        st = check_int8_call(desc, params, decision);
        if (st != status_t::success) return st;
    }
    const native_problem_t problem
            = make_native_problem(desc, params, bias, weight.pack_nr);
    const native_plan_t p = native_planner().plan(problem);
    const bool bias_in_dequant
            = int8 && p.int8_bias_in_dequant && decision.bias_in_dequant;
    GemmDescriptor ep_desc = desc;
    if (int8 && decision.scheme == native_int8_scheme_t::per_channel
            && decision.order == native_int8_order_t::reference) {
        // The per-channel INT8 tile applies alpha and beta to the integer
        // sum; the epilogue applies those of a per-group call.
        ep_desc.alpha = 1.0f;
        ep_desc.beta = 0.0f;
    }
    native_epilogue_t ep;
    st = prepare_epilogue(ep_desc, params, bias_in_dequant ? nullptr : bias,
            dst, weight.interleave_split_halves, ep);
    if (st != status_t::success) return st;

    native_int8_call_t call;
    native_int8_out_t store;
    const native_int8_out_t *direct = nullptr;
    if (int8) {
        st = prepare_int8_call(desc, params, decision, src, p.copy_block,
                p.copy_par_min, call);
        if (st != status_t::success) return st;
        if (desc.beta != 0.0f) {
            call.dst = dst;
            call.dst_dt = desc.dst_dt;
            call.ldc = desc.ldc;
        }
        if (bias_in_dequant) {
            call.bias = bias;
            call.bias_dt = params.dtypes.bias;
        }
        // An epilogue left with only the store: the INT8 kernel stores the
        // bf16 / f32 destination itself.
        if (p.int8_direct_store && decision.direct_store
                && ep.gated == matmul_gated_act_t::none && ep.ops.empty()
                && !ep.requant && ep.bias == nullptr
                && ep.bias_raw == nullptr) {
            store.dst = dst;
            store.bf16 = desc.dst_dt == data_type_t::bf16;
            store.ldc = desc.ldc;
            direct = &store;
        }
    }

    bf16_operands_t op;
    std::unique_ptr<uint16_t, aligned_free_t> a_rows;
    if (bf16) {
        op.A = static_cast<const uint16_t *>(src);
        op.lda = desc.lda;
        if (desc.transA) {
            const size_t bytes = static_cast<size_t>(M) * K * sizeof(uint16_t);
            a_rows.reset(static_cast<uint16_t *>(
                    zendnnl_aligned_alloc(64, (bytes + 63) & ~size_t {63})));
            if (!a_rows) {
                log_error(
                        "native BRGEMM: no buffer for the transposed "
                        "source (",
                        bytes, " bytes)");
                return status_t::failure;
            }
            copy_transposed(op.A, desc.lda, M, K, a_rows.get(), p.copy_block,
                    problem.threads, p.copy_par_min);
            op.A = a_rows.get();
            op.lda = K;
        }
        const native_kernel_registry_t &kernels = native_kernel_registry();
        if (!kernels.covers(native_kernel_kind_t::bf16, p.mr, p.nr)) {
            log_error("native BRGEMM: no bf16 kernels for 1 .. ", p.mr,
                    " rows at nr ", p.nr);
            return status_t::failure;
        }
        op.kern = kernels.bf16_table(p.nr);
    }

    // Last: a first call at weight cache mode 2 may rewrite the weight in
    // place, so nothing after it may turn the call away.
    native_weight_view_t view;
    native_weight_lease_t lease;
    st = acquire_native_weight(weight, desc.wei_dt, K, N, desc.ldb, desc.transB,
            view, lease, problem.threads);
    if (st != status_t::success) return st;

    // Every unit covers all of K and nr columns.
    auto tile = [&](const native_work_unit_t &u, float *acc) {
        if (bf16)
            bf16_block(p, op, view, ep, u.rows, K, u.m0, u.n0, acc);
        else
            int8_block(p, call, view, ep, direct, u.rows, u.m0, u.n0, acc);
    };
    // Workers past the tiles have nothing to do.
    const int64_t units = static_cast<int64_t>(p.m_blocks) * p.n_tiles;
    int team = 1;
    st = native_run_cpu_team(native_partition_t(p, M, 0, p.n_tiles),
            static_cast<int>(std::min<int64_t>(p.threads, units)),
            problem.threads, p.scratch_floats, tile, &team);
    // After work no status may let the call fall back to another route.
    if (st != status_t::success) {
        log_error("native BRGEMM: no accumulator scratch of ", p.scratch_floats,
                " floats");
        return status_t::failure;
    }

    static const bool s_log = apilog_verbose_enabled();
    if (s_log) {
        // A BF16 call has a K pass; an INT8 call has call shapes and, unless
        // its weight scales are per group, a K pass, a direct store and a
        // bias placement.
        const std::string kc_field = bf16 ? " kc=" + std::to_string(p.kc) : "";
        std::string int8_fields;
        if (int8) {
            int8_fields = " int8_rows=" + std::to_string(p.int8_rows_nv1) + "/"
                    + std::to_string(p.int8_rows_nv2)
                    + " int8_cw=" + int8_cw_text(p.int8_cw_mask);
            if (decision.scheme != native_int8_scheme_t::per_channel)
                int8_fields += std::string(" int8_scheme=")
                        + native_int8_scheme_name(decision.scheme);
            else
                int8_fields += " int8_pass="
                        + (p.int8_pass_quads > 0
                                        ? std::to_string(p.int8_pass_quads)
                                        : std::string("rule"))
                        + " direct=" + std::to_string(direct != nullptr ? 1 : 0)
                        + " dequant_bias="
                        + std::to_string(bias_in_dequant ? 1 : 0);
        }
        apilog_verbose("[MATMUL.NATIVE_BRGEMM] M=", M, " N=", N, " K=", K,
                " pack_nr=", view.pack_nr, " nr=", p.nr, " mb=", p.mb, kc_field,
                " tiles=", p.tiles(), " threads=", p.threads,
                " col_groups=", p.col_groups, " transA=", desc.transA ? 1 : 0,
                " mr=", p.mr, " row_unit=", p.row_unit, " traversal=",
                p.traversal == native_traversal_t::m_inner ? "m_inner"
                                                           : "n_inner",
                int8_fields, " copy_block=", p.copy_block,
                " source=", native_plan_source_names(p),
                " cfg=", native_plan_config_text(p), " team=", team);
    }
    return status_t::success;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
