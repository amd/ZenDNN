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
#include <cassert>
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
#include "lowoha_operators/matmul/matmul_native/common/native_utils.hpp"
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
    /// The copy of a transposed source, when A points to it.
    std::unique_ptr<uint16_t, native_aligned_free_t> rows;
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
/// columns), so a row call's source rows serve every slab of the tile in
/// turn and a pass's source rows serve every tile.  Across K passes each
/// tile keeps its own mb * nr accumulators.
void bf16_block(const native_plan_t &p, const bf16_operands_t &op,
        const native_weight_view_t &w, const native_epilogue_t &ep, int mb,
        int K, int m0, int n0, int tiles, float *acc) {
    const int b_stride = w.group_stride();
    const int slab = native_bf16_slab_cols(p.nr);
    const int slabs = p.nr / slab;
    assert(slabs <= kNativeBf16MaxTileCols / kNativeBf16SlabCols);
    const bool one_pass = p.kc >= K;
    for (int k0 = 0; k0 < K; k0 += p.kc) {
        const int kl = std::min(p.kc, K - k0);
        const float beta = k0 == 0 ? 0.0f : 1.0f;
        const size_t k_off = static_cast<size_t>(k0 / VNNI_PAIR) * b_stride;
        for (int t = 0; t < tiles; ++t) {
            const int n = n0 + t * p.nr;
            const uint16_t
                    *slab_b[kNativeBf16MaxTileCols / kNativeBf16SlabCols];
            for (int s = 0; s < slabs; ++s) {
                const int col = n + s * slab;
                slab_b[s] = reinterpret_cast<const uint16_t *>(
                                    w.block(col / w.pack_nr))
                        + (col % w.pack_nr) * VNNI_PAIR + k_off;
            }
            float *tile_acc
                    = one_pass ? acc : acc + static_cast<size_t>(t) * mb * p.nr;
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
                    apply_epilogue_tile(ep, c, p.nr, m0 + ir, mr, n, p.nr);
            }
        }
    }
}

void int8_block(const native_plan_t &p, const native_int8_call_t &call,
        const native_weight_view_t &w, const native_epilogue_t &ep,
        const native_int8_out_t *direct, int mb, int m0, int n0, int tiles,
        float *acc) {
    for (int t = 0; t < tiles; ++t) {
        const int n = n0 + t * p.nr;
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
};

execute_call_t make_call(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params) {
    const bool bf16 = desc.src_dt == data_type_t::bf16
            && desc.wei_dt == data_type_t::bf16;
    const bool dynamic_src = params.dynamic_quant
            && (desc.src_dt == data_type_t::bf16
                    || desc.src_dt == data_type_t::f32);
    const bool int8 = desc.wei_dt == data_type_t::s8
            && (desc.src_dt == data_type_t::u8 || desc.src_dt == data_type_t::s8
                    || dynamic_src);
    return {desc, weight, src, dst, bias, params, bf16, int8};
}

/// The dtypes, the CPU's ISA, the shape and the buffers of the call.
status_t check_call(const execute_call_t &c) {
    const GemmDescriptor &desc = c.desc;
    const int M = desc.M, N = desc.N, K = desc.K;
    if (!c.bf16 && !c.int8) {
        log_error("native BRGEMM: src=", dtype_info(desc.src_dt),
                " wei=", dtype_info(desc.wei_dt), " is not served");
        return status_t::unimplemented;
    }
    const UarchParams &uarch = detect_uarch();
    if ((c.bf16 ? !uarch.avx512bf16 : !uarch.avx512vnni)
            || !native_base_isa()) {
        log_error("native BRGEMM: the CPU lacks ",
                c.bf16 ? "AVX512-BF16" : "AVX512-VNNI",
                " or AVX-512 F / BW / VL / DQ or FMA");
        return status_t::isa_unsupported;
    }
    const int n_out = desc.gated_act != matmul_gated_act_t::none ? N / 2 : N;
    const int min_lda = desc.transA ? M : K;
    if (M < 0 || N < 1 || K < 1 || desc.ldc < n_out
            || (c.bf16 && desc.lda < min_lda)) {
        log_error("native BRGEMM: bad shape M=", M, " N=", N, " K=", K,
                " lda=", desc.lda, " ldc=", desc.ldc);
        return status_t::failure;
    }
    if (M > 0 && (c.src == nullptr || c.dst == nullptr)) {
        log_error("native BRGEMM: null src or dst");
        return status_t::failure;
    }
    return status_t::success;
}

/// The INT8 decision of the call, and the check of the call against it.
status_t decide_int8(
        const execute_call_t &c, native_int8_decision_t &decision) {
    decision = decide_int8_call(c.desc, c.params, c.bias);
    return check_int8_call(c.desc, c.params, decision);
}

/// The epilogue of the call; with bias_in_dequant the INT8 dequantization
/// takes the bias instead.
status_t prepare_call_epilogue(const execute_call_t &c,
        const native_int8_decision_t &decision, bool bias_in_dequant,
        native_epilogue_t &ep) {
    GemmDescriptor ep_desc = c.desc;
    if (c.int8 && decision.scheme == native_int8_scheme_t::per_channel
            && decision.order == native_int8_order_t::reference) {
        // The per-channel INT8 tile applies alpha and beta to the integer
        // sum; the epilogue applies those of a per-group call.
        ep_desc.alpha = 1.0f;
        ep_desc.beta = 0.0f;
    }
    return prepare_epilogue(ep_desc, c.params,
            bias_in_dequant ? nullptr : c.bias, c.dst,
            c.weight.interleave_split_halves, ep);
}

/// The INT8 call of the plan and, when the epilogue is left with only the
/// store, the direct store.
status_t prepare_int8_operands(const execute_call_t &c,
        const native_int8_decision_t &decision, const native_plan_t &p,
        bool bias_in_dequant, const native_epilogue_t &ep,
        int8_operands_t &op) {
    const GemmDescriptor &desc = c.desc;
    const status_t st = prepare_int8_call(desc, c.params, decision, c.src,
            p.copy_block, p.copy_par_min, op.call);
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

/// The BF16 source rows, a transposed source copied once into rows K apart
/// on threads threads, and the kernels of the plan's rows per call.
status_t prepare_bf16_operands(const execute_call_t &c, const native_plan_t &p,
        int threads, bf16_operands_t &op) {
    const GemmDescriptor &desc = c.desc;
    const int M = desc.M, K = desc.K;
    op.A = static_cast<const uint16_t *>(c.src);
    op.lda = desc.lda;
    if (desc.transA) {
        const size_t bytes = static_cast<size_t>(M) * K * sizeof(uint16_t);
        op.rows.reset(static_cast<uint16_t *>(
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
    const native_kernel_registry_t &kernels = native_kernel_registry();
    if (!kernels.covers(native_kernel_kind_t::bf16, p.mr, p.nr)) {
        log_error("native BRGEMM: no bf16 kernels for 1 .. ", p.mr,
                " rows at nr ", p.nr);
        return status_t::failure;
    }
    op.kern = kernels.bf16_table(p.nr);
    return status_t::success;
}

/// Runs the plan's work units on a team of up to threads threads; team
/// receives the threads of the region.
status_t run_team(const execute_call_t &c, const native_plan_t &p, int threads,
        const native_weight_view_t &view, const native_epilogue_t &ep,
        const bf16_operands_t &bf16_op, const int8_operands_t &int8_op,
        int &team) {
    const int K = c.desc.K;
    const bool bf16 = c.bf16;
    const native_int8_out_t *direct = int8_op.direct ? &int8_op.store : nullptr;
    // Every unit covers all of K and tiles column tiles of nr columns.
    auto tile = [&](const native_work_unit_t &u, float *acc) {
        if (bf16)
            bf16_block(
                    p, bf16_op, view, ep, u.rows, K, u.m0, u.n0, u.tiles, acc);
        else
            int8_block(p, int8_op.call, view, ep, direct, u.rows, u.m0, u.n0,
                    u.tiles, acc);
    };
    // Workers past the units have nothing to do.
    const native_partition_t part(p, c.desc.M, 0, p.n_tiles);
    const status_t st = native_run_cpu_team(part,
            static_cast<int>(std::min<int64_t>(p.threads, part.units())),
            threads, p.scratch_floats, tile, &team);
    // After work no status may let the call fall back to another route.
    if (st != status_t::success) {
        log_error("native BRGEMM: no accumulator scratch of ", p.scratch_floats,
                " floats");
        return status_t::failure;
    }
    return status_t::success;
}

/// The [MATMUL.NATIVE_BRGEMM] line of a call that ran.
void log_plan_line(const execute_call_t &c, const native_plan_t &p,
        const native_int8_decision_t &decision, bool bias_in_dequant,
        bool direct, int pack_nr, int team) {
    const GemmDescriptor &desc = c.desc;
    // A BF16 call has a K pass; an INT8 call has call shapes and, unless
    // its weight scales are per group, a K pass, a direct store and a
    // bias placement; a per-group call has its scheme and its direct store.
    const std::string kc_field = c.bf16 ? " kc=" + std::to_string(p.kc) : "";
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
        else
            int8_fields += " int8_pass="
                    + (p.int8_pass_quads > 0 ? std::to_string(p.int8_pass_quads)
                                             : std::string("rule"))
                    + " direct=" + std::to_string(direct ? 1 : 0)
                    + " dequant_bias="
                    + std::to_string(bias_in_dequant ? 1 : 0);
    }
    apilog_verbose("[MATMUL.NATIVE_BRGEMM] M=", desc.M, " N=", desc.N,
            " K=", desc.K, " pack_nr=", pack_nr, " nr=", p.nr, " mb=", p.mb,
            kc_field, " tiles=", p.tiles(), " threads=", p.threads,
            " col_groups=", p.col_groups, blocks_field,
            " transA=", desc.transA ? 1 : 0, " mr=", p.mr,
            " row_unit=", p.row_unit, " traversal=",
            p.traversal == native_traversal_t::m_inner ? "m_inner" : "n_inner",
            int8_fields, " copy_block=", p.copy_block,
            " source=", native_plan_source_names(p),
            " cfg=", native_plan_config_text(p), " team=", team);
}

} // namespace

status_t native_brgemm_execute(const GemmDescriptor &desc,
        const native_weight_ref_t &weight, const void *src, void *dst,
        const void *bias, const matmul_params &params) {
    const execute_call_t c = make_call(desc, weight, src, dst, bias, params);
    status_t st = check_call(c);
    if (st != status_t::success) return st;
    if (desc.M == 0) return status_t::success;
    // The planner and the tile grid need N to be whole pack blocks, which the
    // layout rule checked.
    assert(native_layout_fits(desc.wei_dt, desc.K, desc.N, weight.pack_nr));

    // Every check that can turn the call away runs before any work: a call
    // turned away leaves the source and the weight as they were.
    native_int8_decision_t decision;
    if (c.int8) {
        st = decide_int8(c, decision);
        if (st != status_t::success) return st;
    }
    const native_problem_t problem
            = make_native_problem(desc, params, bias, weight.pack_nr, decision);
    const native_plan_t p = native_planner().plan(problem);
    const bool bias_in_dequant
            = c.int8 && p.int8_bias_in_dequant && decision.bias_in_dequant;
    native_epilogue_t ep;
    st = prepare_call_epilogue(c, decision, bias_in_dequant, ep);
    if (st != status_t::success) return st;
    int8_operands_t int8_op;
    if (c.int8) {
        st = prepare_int8_operands(
                c, decision, p, bias_in_dequant, ep, int8_op);
        if (st != status_t::success) return st;
    }
    bf16_operands_t bf16_op;
    if (c.bf16) {
        st = prepare_bf16_operands(c, p, problem.threads, bf16_op);
        if (st != status_t::success) return st;
    }

    // Last: a first call at weight cache mode 2 may rewrite the weight in
    // place, so nothing after it may turn the call away.
    native_weight_view_t view;
    native_weight_lease_t lease;
    st = acquire_native_weight(weight, desc.wei_dt, desc.K, desc.N, desc.ldb,
            desc.transB, view, lease, problem.threads);
    if (st != status_t::success) return st;

    int team = 1;
    st = run_team(c, p, problem.threads, view, ep, bf16_op, int8_op, team);
    if (st != status_t::success) return st;

    static const bool s_log = apilog_verbose_enabled();
    if (s_log)
        log_plan_line(c, p, decision, bias_in_dequant, int8_op.direct,
                view.pack_nr, team);
    return status_t::success;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
