/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "dispatch.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <immintrin.h>

#include "../single_expert/single_expert_hooks.hpp"
#include "../single_expert/single_expert_policy.hpp"
#include "common/zendnnl_compat.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/aocl_unreorder_cache.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_machine.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"
#include "pack.hpp"
#include "qwen_gemma_llama/bf16_decode_microkernel.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace custom_kernel {

using zendnnl::error_handling::apilog_info;
using zendnnl::error_handling::apilog_info_enabled;
using zendnnl::error_handling::apilog_verbose;
using zendnnl::error_handling::apilog_verbose_enabled;

bool bf16_isa_supported() {
    return avx512bf16_available();
}

// ─────────────────────────────────────────────────────────────────────
// Convenience predicate — variant is in the DQ-INT8 family:
//   {kS8_S8_BF16_SYM, kU8_S8_BF16_ASYM,   // bf16 dst
//    kS8_S8_F32_SYM,  kU8_S8_F32_ASYM}.   // f32 dst (act=none only)
// Hoisted to one place so dispatch.cpp, the warmer, and the
// debug-assert in `dispatch_tile()` agree on the int8 family
// membership.  Keeping it `inline` (no namespace qualifier) lets the
// optimiser fold the call into a single `cmp + or` at every callsite.
// `is_int8_variant` now lives in the public header (`dispatch.hpp`)
// so call sites in the N-tile executor can branch on it without
// touching dispatch internals; this TU keeps using the header-level
// definition.
// ─────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// CallContext lifecycle — `release_owned_buffers()` + `reset()`.
//
// The destructor + the in-place reset are the only places the
// caller-owned packed buffers can be freed (the `disable_cache=true`
// branch of `get_or_pack_weight_bf16()` skips the LRU singleton, so
// no other code path knows these pointers exist).  Implementing both
// in dispatch.cpp keeps the `<array>` member's destructor trivial
// while routing the actual free through `free_owned_packed_weight()`
// (pack.cpp's matched companion to `aligned_alloc` from the
// disable-cache branch).
// ─────────────────────────────────────────────────────────────────────
void CallContext::release_owned_buffers() {
    // Iterate every slot — both `owned_packed_ptrs` and `packed_ptrs`
    // are zero-initialised on construction, so the bulk of slots cost
    // one branch each on a typical (active_count << kMaxExperts) call.
    // The `free_owned_packed_weight` helper itself is null-safe so the
    // inner check is only an optimisation to skip the call entirely
    // when the slot is empty.
    for (auto &ptr : owned_packed_ptrs) {
        if (ptr != nullptr) {
            free_owned_packed_weight(ptr);
            ptr = nullptr;
        }
    }
    // DQ-INT8 sibling — symmetric semantics with a different free
    // sink (the int8 cache and the bf16 cache are disjoint singletons
    // and their disable-cache buffers come from different
    // aligned_alloc'ed regions tagged by the per-pack family in
    // `pack.cpp`).
    for (auto &ptr : owned_packed_ptrs_int8) {
        if (ptr != nullptr) {
            free_owned_packed_weight_int8(ptr);
            ptr = nullptr;
        }
    }
    // FP16 sibling — disjoint free sink (the f16 cache and its
    // disable-cache buffers come from a separate aligned_alloc region
    // tagged by the per-pack family in `pack.cpp`).
    for (auto &ptr : owned_packed_ptrs_f16) {
        if (ptr != nullptr) {
            free_owned_packed_weight_f16(ptr);
            ptr = nullptr;
        }
    }
    for (auto &ptr : owned_packed_ptrs_w4) {
        if (ptr != nullptr) {
            free_owned_packed_weight_w4(ptr);
            ptr = nullptr;
        }
    }
}

void CallContext::reset() {
    // Free caller-owned packs FIRST so subsequent field zeroing does
    // not strand the pointers.  `release_owned_buffers()` clears
    // `owned_packed_ptrs` itself, so the loops below only need to
    // touch the remaining state.
    release_owned_buffers();
    enabled = false;
    variant = KernelVariant::kUnsupported;
    pack_nr = 0;
    pack_in_place = false;
    NV = 0;
    max_mr = 0;
    base_max_mr = 0;
    kblock_auto = false;
    prompt_kc = 0;
    prompt_kc_eligible = false;
    single_expert = false;
    f16_prompt = false;
    subtile_cols = 0;
    act_kind = ActivationKind::none;
    bias_kind = BiasKind::none;
    compute_int = IntCompute::kS8_Sym;
    // Restore the same safe default as the member initializer: f32 scales.
    // `dispatch_tile_int8()` derives src/wei scale byte strides from
    // `scale_kind`, so a stale value across a CallContext reuse would read
    // scales at the wrong element width if the next caller forgot to set it.
    scale_kind = ScaleKind::kF32;
    src_u8 = false;
    for (auto &fn : kernel_table)
        fn = nullptr;
    for (auto &fn : kernel_table_int8)
        fn = nullptr;
    for (auto &fn : kernel_table_f16)
        fn = nullptr;
    for (auto &fn : kernel_table_f16_kblock)
        fn = nullptr;
    for (auto &row : kernel_table_f16_prompt)
        for (auto &fn : row)
            fn = nullptr;
    for (auto &row : kernel_table_f16_prompt_f32)
        for (auto &fn : row)
            fn = nullptr;
    for (auto &fn : kernel_table_s4)
        fn = nullptr;
    for (auto &fn : kernel_table_s4_offset)
        fn = nullptr;
    for (auto &fn : kernel_table_mxfp4)
        fn = nullptr;
    // A stale `group_size` across a CallContext reuse would make the s4
    // kernel walk the wrong number of compensation rows.
    group_size = 0;
    wei_scale_grp_stride = 0;
    src_scale_groups = 1;
    packed_ptrs.fill(nullptr);
    packed_ptrs_int8.fill(nullptr);
    packed_ptrs_f16.fill(nullptr);
    packed_ptrs_w4.fill(nullptr);
    for (auto &lease : aocl_custom_kernel_weight_leases)
        lease.reset();
    subtile_cols_per_expert.fill(0);
}

// ────────────────────────────────────────────────────────────────────
// Variant resolution — pure switch over the (src, wei, dst) dtype
// tuple.  No I/O, no allocation, no logging — `prepare_for_call()`
// owns the logging (this helper is also called from gtests that
// want to assert the routing without any side effect).
//
// The wired-up rows are: the BF16 family (BF16-BF16-BF16 /
// BF16-BF16-F32), the DQ-INT8 family (s8 wei × {bf16,f32} dst, sym /
// asym), and the FP16 family (F16-F16-F16 / F16-F16-F32).  Any other
// tuple falls through to `kUnsupported`, which the caller routes to
// DLP.  See the full truth table on `resolve_variant` in dispatch.hpp.
// ────────────────────────────────────────────────────────────────────
KernelVariant resolve_variant(data_type_t src, data_type_t wei, data_type_t dst,
        bool dynamic_quant, data_type_t compute_dtype) noexcept {
    // BF16 family — `dynamic_quant` MUST be false (any
    // dynamic_quant=true call must go through the DQ-INT8 branch
    // below; we keep these mutually exclusive to avoid silent
    // mis-routing).
    if (!dynamic_quant && src == data_type_t::bf16
            && wei == data_type_t::bf16) {
        if (dst == data_type_t::bf16) return KernelVariant::kBF16_BF16_BF16;
        if (dst == data_type_t::f32) return KernelVariant::kBF16_BF16_F32;
    }
    // DQ-INT8 family — two ways the per-token-quantized s8/u8 src reaches
    // the CK microkernel, both with `wei == s8` and the same dequant math:
    //
    //   1. Runtime hoist (legacy): caller passes bf16/f32 src with
    //      `dynamic_quant = true`; the N-tile executor hoists it to s8/u8
    //      (per-token src_scale) before `dispatch_tile`.
    //   2. Grouped pre-quant (ZENDNNL_ENABLE_GROUP_DQ, default on): the
    //      `group_dynamic_quant` pre-pass already quantized the src to s8,
    //      set `dtypes.src = s8`, cleared `dynamic_quant`, and left a
    //      per-token `src_scale` buffer in params.  The src arrives here
    //      ALREADY s8, so accept `src == s8` directly — `n_tile_supported`
    //      has already validated the grouped per-token src_scale + per-
    //      channel wei_scale shape, so this only fires on the DQ-INT8
    //      grouped form (a static s8s8 GEMM would have dst = s8, not
    //      bf16/f32, and is rejected by the dst checks below).
    //
    // Discriminator on the compute dtype (identical for both):
    //   * `compute_dtype = s8` → kS8_S8_BF16_SYM  (sym, no src_zp).
    //   * `compute_dtype = u8` → kU8_S8_BF16_ASYM (asym, with src_zp).
    const bool dq_int8_src
            = (dynamic_quant
                      && (src == data_type_t::bf16 || src == data_type_t::f32))
            || (src
                    == data_type_t::
                            s8); // (1) runtime hoist, (2) grouped pre-quant
    if (dq_int8_src && wei == data_type_t::s8) {
        if (dst == data_type_t::bf16) {
            if (compute_dtype == data_type_t::s8)
                return KernelVariant::kS8_S8_BF16_SYM;
            if (compute_dtype == data_type_t::u8)
                return KernelVariant::kU8_S8_BF16_ASYM;
        }
        // FP32 dst — direct store of the dequantised accumulator (Act=none
        // only; gated kinds stay BF16 and are refused in fill_kernel_table_int8
        // / the microkernel selector).  Mirror of the bf16 family's
        // bf16/f32 dst split.
        if (dst == data_type_t::f32) {
            if (compute_dtype == data_type_t::s8)
                return KernelVariant::kS8_S8_F32_SYM;
            if (compute_dtype == data_type_t::u8)
                return KernelVariant::kU8_S8_F32_ASYM;
        }
    }
    // Static U8xS8: the caller has already quantized both operands.
    // Source and weight metadata validation (per-tensor scales, per-tensor
    // source zp, no weight zp) is owned by the routing/N-tile gates because
    // this pure dtype resolver cannot inspect quant buffers. `compute=none`
    // is the normal public-API form; accept an explicit u8 discriminator too.
    if (!dynamic_quant && src == data_type_t::u8 && wei == data_type_t::s8
            && (compute_dtype == data_type_t::none
                    || compute_dtype == data_type_t::u8)) {
        if (dst == data_type_t::bf16) return KernelVariant::kU8_S8_BF16_ASYM;
        if (dst == data_type_t::f32) return KernelVariant::kU8_S8_F32_ASYM;
        if (dst == data_type_t::u8) return KernelVariant::kU8_S8_U8_ASYM;
    }
    // FP16 family — native AVX-512-FP16.  `dynamic_quant` MUST be false
    // (the FP16 path is non-quant, like the bf16 family); a quantised
    // f16 tuple has no CK route and falls through to kUnsupported.
    //   (f16, f16, f16) → kF16_F16_F16
    //   (f16, f16, f32) → kF16_F16_F32  (act=none only; gated kinds are
    //                                    f16-dst, refused by select_*)
    if (!dynamic_quant && src == data_type_t::f16 && wei == data_type_t::f16) {
        if (dst == data_type_t::f16) return KernelVariant::kF16_F16_F16;
        if (dst == data_type_t::f32) return KernelVariant::kF16_F16_F32;
    }
    // Four-bit weights — the weight type picks the format (s4 → W4A8,
    // f4_e2m1 → MXFP4).  W4A8 also has an f32 output, valid for act=none;
    // the kernel-table gate below rejects f32 gated activation stores.
    // Matching here does not engage the kernel; `prepare_for_call` still
    // applies the shape gates.
    const bool w4_src
            = (dynamic_quant
                      && (src == data_type_t::bf16 || src == data_type_t::f32))
            || (src == data_type_t::s8);
    if (w4_src && compute_dtype == data_type_t::s8
            && (dst == data_type_t::bf16 || dst == data_type_t::f32)) {
        const bool f32_dst = dst == data_type_t::f32;
        if (wei == data_type_t::s4)
            return f32_dst ? KernelVariant::kS8_S4_F32_SYM
                           : KernelVariant::kS8_S4_BF16_SYM;
        if (wei == data_type_t::f4_e2m1)
            return f32_dst ? KernelVariant::kS8_MXFP4_F32_SYM
                           : KernelVariant::kS8_MXFP4_BF16_SYM;
    }
    return KernelVariant::kUnsupported;
}

// Pick the pack/microkernel NR for one (K, N) shape.  Env override
// (from `get_grp_matmul_custom_kernel_nr()` in parallel_common.hpp)
// wins when it divides N; otherwise default to NR=32 — it gives a
// clean register budget against the 32-zmm budget on AVX-512 and
// keeps M chunking small on small-M decode shapes.  NR=64 stays
// available via env override for shapes that benefit.
//
// Now a public symbol (declared in dispatch.hpp) so the warm-pack
// helpers in `group_matmul/prepack/` can compute the same NR the
// dispatcher will pack under, keeping cache keys aligned.
int plan_pack_nr(int /*K*/, int N) {
    if (N <= 0) return 0;
    const int env_nr = get_grp_matmul_custom_kernel_nr();
    if (env_nr != 0) return ((N % env_nr) == 0) ? env_nr : 0;
    if ((N % kNRMin) == 0) return kNRMin;
    if ((N % kNRMax) == 0) return kNRMax;
    return 0;
}

int64_t ck_l2_budget_bytes() {
    return grp_machine::machine().l2_budget_bytes;
}

// ─────────────────────────────────────────────────────────────────────
// Internal helpers — owned entirely by this file.  The callers never call
// these directly; the only public entries are `prepare_for_call()`
// and `dispatch_tile()` below.
// ─────────────────────────────────────────────────────────────────────
namespace {

// L2-friendly sub-tile width (in cols) for (M_max, K, pack_nr).
// Sized so each microkernel call's working set — input + the B
// strip this thread streams — fits in this CPU's per-core L2 with
// some headroom (`l2_budget` uses a 4/5 fraction of detected L2 to
// leave room for caller-side spills).  Accumulators live in zmm
// registers, so they don't enter the budget.  Returned value is a
// multiple of pack_nr, minimum one o-block.
//
// `src_elem_bytes` and `wei_elem_bytes` describe the per-element
// byte volume the kernel consumes:
//   * BF16 family       → src=2, wei=2, comp_bytes_per_col=0.
//   * DQ-INT8 family    → src=1 (the hoisted s8/u8 stripe the int8
//     microkernel reads), wei=1, comp_bytes_per_col=4 (the int32
//     compensation row appended after each o-block — adds a
//     constant per-column overhead independent of K).
// Without dtype-awareness the bf16 byte model (2,2,0) was used for
// the int8 family too, over-budgeting the working set by 2x and
// shrinking subtiles below their byte-budget optimum.  See R2 in
// `analysis_scratch/int8_ntile_audit.md`.
int pick_l2_subtile_cols(int M_max, int K, int pack_nr, int src_elem_bytes,
        int wei_elem_bytes, int comp_bytes_per_col) {
    if (M_max <= 0 || K <= 0 || pack_nr <= 0 || src_elem_bytes <= 0
            || wei_elem_bytes <= 0)
        return pack_nr;
    const int64_t l2_budget = ck_l2_budget_bytes();

    const int64_t input_bytes
            = static_cast<int64_t>(M_max) * K * src_elem_bytes;
    const int64_t weight_bytes_per_col
            = static_cast<int64_t>(K) * wei_elem_bytes
            + static_cast<int64_t>(comp_bytes_per_col);
    if (l2_budget <= input_bytes || weight_bytes_per_col <= 0) {
        return pack_nr;
    }
    const int64_t cols_raw = (l2_budget - input_bytes) / weight_bytes_per_col;
    const int cols_aligned = static_cast<int>((cols_raw / pack_nr) * pack_nr);
    return std::max(cols_aligned, pack_nr);
}

// Per-variant byte model used by `pick_l2_subtile_cols`.  The
// switch is constexpr-friendly so the subtile selection turns
// into two branchless multiplies + one constant load on the
// hot prepare-for-call path; the runtime overhead vs the previous
// hard-coded `sizeof(uint16_t)` is microscopic.
struct SubtileBytes {
    int src_bytes_per_elem;
    int wei_bytes_per_elem;
    int comp_bytes_per_col;
};
inline SubtileBytes subtile_bytes_for_variant(KernelVariant v) {
    if (is_int8_variant(v)) {
        // The hoisted src is s8/u8 (1 byte) — the int8 microkernel
        // reads it directly; the bf16 caller src is replaced by the
        // pre-OMP hoist before the kernel runs.
        // Comp row is int32 per-column appended after each o-block.
        return {1, 1, 4};
    }
    if (is_w4a8_variant(v)) {
        // The true W4A8 footprint is `K/2 + (K/group_size)*4` bytes per
        // column, which this bytes-per-element model cannot express and
        // `group_size` is not visible here anyway.  Reusing the int8
        // model over-estimates the nibble slab 2x, so the subtile comes
        // out narrower than optimal — an L2 blocking heuristic only.
        return {1, 1, 4};
    }
    if (is_f16_variant(v)) {
        // FP16 family — src/wei are native f16 (2 bytes each), no
        // compensation row (non-quant, like bf16).  Numerically identical
        // to the bf16 fallthrough below, but stated explicitly so a future
        // narrower family (e.g. FP8 / INT4) cannot silently inherit the
        // 2-byte bf16 model via the fallthrough.
        return {2, 2, 0};
    }
    // BF16 family (default) — src/wei are bf16 (2 bytes each), no
    // compensation row.
    return {2, 2, 0};
}

// Resolve the per-MR microkernel function-pointer table for one
// (NV, act_kind, dst_dt) tuple.  Called once by `prepare_for_call()`.
// Per-tile lookup is then a direct `kernel_table[mr_now]` — no switch.
status_t fill_kernel_table(int NV, int max_mr, ActivationKind act_kind,
        OutputType dst_dt, bf16_ukernel_fn_t (&kernel_table)[kMaxMR + 1]) {
    for (int mr = 1; mr <= max_mr; ++mr) {
        kernel_table[mr] = select_bf16_ukernel(mr, NV, act_kind, dst_dt);
        if (kernel_table[mr] == nullptr) return status_t::failure;
    }
    return status_t::success;
}

// DQ-INT8 sibling of `fill_kernel_table` — same loop shape, different
// selector (`select_int8_ukernel`).  Carries both the `compute`
// (kS8_Sym / kU8_Asym) axis and the `dst_dt` (kBf16 / kF32) axis;
// the selector returns nullptr for any (gated_act, kF32) tuple so a
// mis-resolved f32 gated variant refuses here rather than at runtime.
status_t fill_kernel_table_int8(int NV, IntCompute compute,
        ActivationKind act_kind, OutputType dst_dt,
        int8_ukernel_fn_t (&kernel_table)[kMaxMR + 1], int group_k = 0,
        bool src_u8 = false) {
    const bool q8_group = group_k == 32 && compute == IntCompute::kS8_Sym
            && (NV == 2 || NV == 4);
    const int max_mr = q8_group ? std::min(max_mr_for_nv_int8(NV, compute), 4)
                                : max_mr_for_nv_int8(NV, compute);
    for (int mr = 1; mr <= max_mr; ++mr) {
        kernel_table[mr] = q8_group
                ? select_int8_q8_group_ukernel(mr, NV, act_kind, dst_dt)
                : select_int8_ukernel(
                          mr, NV, compute, act_kind, dst_dt, src_u8);
        if (kernel_table[mr] == nullptr) return status_t::failure;
    }
    return status_t::success;
}

// FP16 sibling of `fill_kernel_table` — same loop shape, different
// selector (`select_f16_ukernel`).  On a toolchain without AVX-512-FP16
// intrinsics `select_f16_ukernel` returns nullptr for every MR, so
// this fails and `prepare_for_call` refuses the call (caller → AOCL).
status_t fill_kernel_table_f16(int NV, ActivationKind act_kind,
        OutputType dst_dt, f16_ukernel_fn_t (&kernel_table)[kMaxMR + 1]) {
    const int max_mr = max_mr_for_nv(NV);
    for (int mr = 1; mr <= max_mr; ++mr) {
        kernel_table[mr] = select_f16_ukernel(mr, NV, act_kind, dst_dt);
        if (kernel_table[mr] == nullptr) return status_t::failure;
    }
    return status_t::success;
}

status_t fill_kernel_table_f16_kblock(
        int NV, f16_kblock_ukernel_fn_t (&kernel_table)[kMaxMR + 1]) {
    const int max_mr = max_mr_for_nv(NV);
    for (int mr = 1; mr <= max_mr; ++mr) {
        kernel_table[mr] = select_f16_kblock_ukernel(mr, NV);
        if (kernel_table[mr] == nullptr) return status_t::failure;
    }
    return status_t::success;
}

// W4A8 sibling of `fill_kernel_table`: no Compute / OutputType axis, and a
// tighter MR bound.  Slots above it stay null, so `dispatch_tile` must
// clamp its MR chunking to `ctx.max_mr`.
status_t fill_kernel_table_s4(int NV, ActivationKind act_kind,
        int src_scale_groups, OutputType dst_dt,
        w4a8_ukernel_fn_t (&kernel_table)[kMaxMR + 1],
        w4a8_ukernel_fn_t (&kernel_table_offset)[kMaxMR + 1],
        S4SourceKind source = S4SourceKind::kS8) {
    const int max_mr = max_mr_for_nv_s4(NV);
    const S4SrcScaleGranularity granularity = src_scale_groups > 1
            ? S4SrcScaleGranularity::kPerGroup
            : S4SrcScaleGranularity::kPerToken;
    const S4OutputKind output = dst_dt == OutputType::kF32
            ? S4OutputKind::kF32
            : S4OutputKind::kBf16;
    for (int mr = 1; mr <= max_mr; ++mr) {
        kernel_table[mr] = select_w4a8_ukernel(mr, NV, act_kind, source, output,
                granularity,
                /*has_offset=*/false);
        kernel_table_offset[mr] = select_w4a8_ukernel(mr, NV, act_kind, source,
                output, granularity,
                /*has_offset=*/true);
        if (kernel_table[mr] == nullptr || kernel_table_offset[mr] == nullptr) {
            return status_t::failure;
        }
    }
    return status_t::success;
}

// The MXFP4 variants' table, from their own kernel family.  As for s4, an
// f32 dst takes act = none only.
status_t fill_kernel_table_mxfp4(int NV, ActivationKind act_kind,
        int src_scale_groups, OutputType dst_dt,
        mxfp4_ukernel_fn_t (&kernel_table)[kMaxMR + 1]) {
    if (dst_dt == OutputType::kF32 && act_kind != ActivationKind::none)
        return status_t::failure;
    const int max_mr = max_mr_for_nv_mxfp4(NV);
    for (int mr = 1; mr <= max_mr; ++mr) {
        kernel_table[mr] = select_mxfp4_ukernel(
                mr, NV, act_kind, /*per_block_src_scale=*/src_scale_groups > 1);
        if (kernel_table[mr] == nullptr) return status_t::failure;
    }
    return status_t::success;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────
// Public entries
// ─────────────────────────────────────────────────────────────────────

status_t prepare_for_call(grp_matmul_gated_act_t act, data_type_t src_dtype,
        data_type_t wei_dtype, data_type_t dst_dtype, data_type_t act_dtype,
        data_type_t bias_dtype, const std::vector<bool> &transA,
        const std::vector<bool> &transB, const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &K,
        const std::vector<int> &ldb, const std::vector<float> &alpha,
        const std::vector<float> &beta, const std::vector<const void *> &weight,
        const std::vector<bool> &is_weights_const, CallContext &out,
        bool dynamic_quant, data_type_t compute_dtype,
        const std::vector<bool> &weights_prepacked, int group_size,
        int wei_scale_grp_stride,
        const std::vector<bool> &weights_aocl_prepacked, int32_t num_threads,
        int32_t weight_cache_mode, bool allow_in_place,
        const std::vector<size_t> &wei_capacities, int src_scale_groups,
        const std::vector<bool> &weights_aocl_sym_prepacked,
        int aocl_unreorder_group_size,
        const std::vector<const void *> &pack_scales,
        const std::vector<int> &pack_scale_ld) {

    // Reset the full CallContext to defaults on entry.  Callers may
    // reuse a single context across calls (e.g. an OMP region that
    // dispatches multiple group_matmul_direct calls back-to-back), so
    // we cannot rely on the destructor to clear stale state from a
    // previous successful prepare.  If we only cleared `enabled` and
    // a refusal exited before assigning `variant` / `pack_nr` /
    // `kernel_table[...]` / `packed_ptrs[...]` / `subtile_cols_per_expert[...]`,
    // those fields would carry over from the prior call — making any
    // future read that bypasses the `enabled` gate (a debug assert,
    // an APILOG line, or a refactor that misses the gate) silently
    // observe a stale-but-plausible config.  `reset()` value-init's
    // every field via the in-class defaults declared in
    // `CallContext` (dispatch.hpp): `enabled = false`, `variant =
    // kUnsupported`, scalar fields = 0 / sentinel, fn-pointer table
    // and the std::array members zero-filled.  Critically `reset()`
    // also calls `release_owned_buffers()` BEFORE zeroing
    // `owned_packed_ptrs` so a previous call's caller-owned packed
    // weights (disable-cache mode) are freed rather than stranded.
    // Cost: ~2 KB of zeroing once per call on the cold prep path,
    // negligible vs the surrounding kernel work.
    out.reset();

    // ── Refusal logging helper ───────────────────────────────────────
    // Every early return below represents a concrete dispatch-contract
    // violation that forces the caller to fall back to the standard
    // AOCL / BRGEMM path.  Log the reason so a model-level read makes
    // it immediately obvious why the custom kernel was skipped — a
    // silent refusal just produced `kernel=standard` with no clue
    // whether the env was off, a dtype mismatched, or the pack-NR
    // check failed.
    //
    // Gated on the verbose level (`ZENDNNL_API_LOG_LEVEL=4`): these
    // lines fire per-expert per-call from inside the ALGO 3 OMP region,
    // so they belong on the per-thread / per-event channel.  Info
    // level (3) stays clean and shows only the consolidated planner
    // and prepack summary lines.  Single cached `apilog_verbose_enabled()`
    // check.
    //
    // Lambda overhead note: `dt_name` is captureless (`[]`) — a
    // stateless functor with zero runtime construction cost.  `refuse`
    // is `[&]` cosmetically (see capture-note paragraph below for why):
    // `s_refuse_log` has STATIC storage duration (function-local
    // `static const`), so a captureless `[]` for `refuse` would also
    // compile and reference it correctly per
    // [expr.prim.lambda.capture] — captures are only required for
    // variables with AUTOMATIC storage duration.  `[&]` produces the
    // same closure body in this case (no actual captures are emitted
    // because `s_refuse_log` is still static-duration), so the
    // construction cost is still zero.  The `[&]` form is used purely
    // to silence static-analysis tooling that mis-reads the
    // captureless variant as a capture omission.
    //
    // Net impact on a successful `prepare_for_call` with logging
    // disabled: zero — neither lambda object is ever invoked on the
    // success path, and `dt_name` only gets called from the failure
    // paths that also read `s_refuse_log`.
    static const bool s_refuse_log = apilog_verbose_enabled();
    auto refuse = [&](const char *reason_tag,
                          const char *detail = nullptr) -> status_t {
        if (s_refuse_log) {
            if (detail != nullptr) {
                apilog_verbose("[GRP_MATMUL.CK REFUSED] reason=", reason_tag,
                        " (", detail, ")");
            } else {
                apilog_verbose("[GRP_MATMUL.CK REFUSED] reason=", reason_tag);
            }
        }
        return status_t::failure;
    };
    // W4A8 refusals log at info, not verbose: for the other families a
    // refusal is a performance footnote, here it silently removes the
    // whole feature.
    static const bool s_w4a8_log = apilog_info_enabled();
    // Only four-bit calls reach `refuse_w4a8`, so the entry is set there.
    const W4FormatEntry *w4_entry = w4_format_entry(wei_dtype);
    auto refuse_w4a8
            = [&](const char *reason_tag, const char *detail) -> status_t {
        if (s_w4a8_log && w4_entry != nullptr) {
            apilog_info("[GRP_MATMUL.CK.W4A8 REFUSED] fmt=", w4_entry->name,
                    " ", w4_entry->label,
                    " CK was not called; reason=", reason_tag, " (", detail,
                    ")",
                    (w4_entry->ck_only ? " — the call fails; "
                                       : " — falling back to AOCL DLP"),
                    (w4_entry->ck_only ? w4_entry->label : ""),
                    (w4_entry->ck_only ? " has no other backend" : ""));
        }
        return status_t::failure;
    };
    auto dt_name = [](data_type_t dt) -> const char * {
        switch (dt) {
            case data_type_t::none: return "none";
            case data_type_t::f32: return "f32";
            case data_type_t::bf16: return "bf16";
            case data_type_t::f16: return "f16";
            case data_type_t::s8: return "s8";
            case data_type_t::u8: return "u8";
            default: return "?";
        }
    };

    // ── Run-once invariants (CPU + dtypes + activation) ──────────────
    // The dispatcher serves two ISA families: bf16 variants need
    // AVX-512 BF16 (VDPBF16PS); DQ-INT8 variants need AVX-512 VNNI
    // (VPDPBUSD).  A CPU with AVX-512 BF16 also has AVX-512 VNNI, but
    // some x86 CPUs have VNNI WITHOUT BF16.  Refuse early ONLY when
    // NEITHER family's ISA is present; the precise per-variant ISA gate
    // runs after `resolve_variant()` below (bf16 -> BF16, int8 -> VNNI)
    // so a VNNI-only host can still serve the DQ-INT8 fast path.
    if (!avx512bf16_available() && !avx512vnni_available()
            && !avx512f16_available())
        return refuse("no_avx512_bf16_vnni_or_fp16");

    // ── Library-wide weight-cache toggle ─────────────────────────────
    // `ZENDNNL_MATMUL_WEIGHT_CACHE` (read via
    // `matmul_config_t::get_weight_cache()`) is the process-wide
    // contract for "the caller is allowed to cache weight-derived
    // packed buffers across calls".  The CK pack arena
    // (`custom_kernel/pack.cpp::pack_cache_singleton`) is keyed on the
    // raw user weight pointer and has eviction disabled
    // (UINT32_MAX capacity), so when the framework mutates weight
    // addresses between calls — exactly the scenario the env knob is
    // designed for — a stale pointer key would collide with a fresh
    // upload and the dispatcher would read freed memory in
    // `dispatch_tile`.
    //
    // Resolution: when wc_mode == 0 we DO NOT refuse CK.
    // Instead each per-expert pack below is routed through
    // `get_or_pack_weight_bf16(..., disable_cache=true)`, which
    // allocates a fresh aligned buffer, packs into it, and returns
    // the raw pointer without touching the LRU singleton.
    // (wc_mode 1 and 2 both use the LRU; only 0 disables it.)  The raw
    // pointers land in `out.owned_packed_ptrs[i]` AND
    // `out.packed_ptrs[i]` (alias), so `dispatch_tile()` reads them
    // transparently and the `CallContext` destructor /
    // `release_owned_buffers()` frees them once the OMP region
    // unwinds.  Cost: per-call pack work, no inter-call amortisation,
    // no stale pointer hits — exactly the semantic the env knob
    // promises.
    //
    // The matching prepack-side gate lives in
    // `prepack/prepack_custom_kernel.cpp::warm_pack_all_custom_kernel_experts`
    // (skip warming when the toggle is non-1 since the warmer can
    // only populate the LRU cache, which the runtime will not
    // consult); the prepack fingerprint
    // (`prepack/prepack.cpp::fingerprint`) folds the toggle into its
    // hash so flipping the env mid-process re-arms both code paths
    // on the next call.
    // Weight-cache mode drives how the CK pack is stored:
    //   0 → no cache: pack into per-call caller-owned buffers (LRU bypass).
    //   1 → out-of-place cache: pack into an LRU-owned buffer.
    //   2 → in-place (default): bf16/even-K and contiguous f16 packs are
    //       written back into the caller's weight buffer (saves a packed
    //       buffer); int8 (compensation row makes the pack larger than the
    //       raw weight) and odd-K bf16 fall back to the out-of-place cache
    //       inside get_or_pack_weight_*.
    // NOTE: reaching here with WC==2 means one of three things:
    //   * pinned ALGO 3 + custom-kernel — each weight has a single consumer,
    //     so the CK pack mutates it in place (`ck_in_place` below is true);
    //   * an AUTO call holding the custom-kernel in-place licence — a
    //     BF16/FP16 MoE call whose phases both run ALGO 3 and whose AOCL
    //     reorders are capped out-of-place, so CK is the single mutator
    //     (`ck_in_place` is true); or
    //   * any other AUTO call in mixed in-place mode — the grouped dispatcher
    //     kept WC=2 but also set `grp_auto_mixed_inplace`, reserving the one
    //     in-place mutation for the AOCL full-weight (prompt) reorder.  CK
    //     decode must then stay OUT-OF-PLACE (`ck_in_place` is false),
    //     otherwise CK would be a second mutator of the shared buffer and
    //     corrupt it.
    // (Any other AUTO run has already downgraded WC 2→1 upstream, so it never
    // reaches here with WC==2.)
    auto &matmul_config = zendnnl::common::matmul_config_t::instance();
    const int wc_mode = weight_cache_mode >= 0
            ? weight_cache_mode
            : matmul_config.get_weight_cache();
    const bool cache_off = (wc_mode == 0);
    // The caller decides whether this pack may mutate the weight
    // (`allow_in_place`): `n_tile_execute` refuses it when its fallback needs
    // the raw weight, and in AUTO mixed in-place mode (grp ALGO 0 + WC=2 +
    // prepack/cross_warm), where exactly ONE layout may mutate the shared
    // buffer and that owner is the AOCL full-weight (prompt) reorder, unless
    // the call holds the custom-kernel in-place licence.  Pinned ALGO 3 + CK
    // (env_algo==3, mixed flag false) keeps true in-place as before.
    const bool ck_in_place = (wc_mode == 2) && allow_in_place;
    out.pack_in_place = ck_in_place;
    if (cache_off && s_refuse_log) {
        apilog_verbose(
                "[GRP_MATMUL.CK NOCACHE] "
                "weight_cache_type == 0 — packing into per-call "
                "caller-owned buffers (LRU singleton bypassed)");
    }

    // A / B / C dtype gate: route through `resolve_variant()` which is
    // the single source of truth for "which (src, wei, dst, dynamic_quant,
    // compute_dtype) tuples the custom kernel can serve".  `kUnsupported`
    // here means the caller falls back to DLP.  Routing through the
    // variant keeps the dispatcher's gate aligned with the gtest matrix
    // in `gtests/group_matmul/custom_kernel/`, which calls
    // `resolve_variant()` directly to assert the table.
    //
    // DQ-INT8 family additionally requires the AVX-512 VNNI feature
    // (VPDPBUSD), which is a strict superset of AVX-512 BF16 on the
    // AVX-512-VNNI CPUs, but on broader x86 silicon
    // is an independent feature flag.  Refuse cleanly with a specific
    // tag so the user can distinguish "no AVX-512 VNNI" from "wrong
    // dtype" in apilog.
    const KernelVariant variant = resolve_variant(
            src_dtype, wei_dtype, dst_dtype, dynamic_quant, compute_dtype);
    if (variant == KernelVariant::kUnsupported) {
        if (s_refuse_log) {
            apilog_verbose(
                    "[GRP_MATMUL.CK REFUSED] reason="
                    "unsupported_dtype (src=",
                    dt_name(src_dtype), " wei=", dt_name(wei_dtype),
                    " dst=", dt_name(dst_dtype),
                    " dynamic_quant=", (dynamic_quant ? 1 : 0),
                    " compute=", dt_name(compute_dtype),
                    " — see resolve_variant() in custom_kernel/dispatch.cpp"
                    " for the supported table)");
        }
        return status_t::failure;
    }
    // Per-variant ISA gate (the early run-once check above only confirms
    // at least ONE family's ISA exists).  bf16 variants require AVX-512
    // BF16; int8 variants require AVX-512 VNNI; fp16 variants require
    // AVX-512-FP16 (checked by its own gate below).  Splitting the gate
    // here lets a VNNI-only host (no BF16) still serve DQ-INT8 and W4A8
    // (whose BF16 store is an integer RNE sequence, not VCVTNEPS2BF16),
    // and a FP16-only host (no BF16) still serve the FP16 variants — the
    // bf16 ISA requirement must therefore only apply to the bf16 variants.
    if (!is_int8_variant(variant) && !is_f16_variant(variant)
            && !is_w4a8_variant(variant) && !avx512bf16_available()) {
        return refuse("avx512bf16_not_available",
                "BF16 custom kernel requires AVX-512 BF16 (VDPBF16PS)"
                " — runtime CPU detection failed");
    }
    if (is_int8_variant(variant) && !avx512vnni_available()) {
        return refuse("avx512vnni_not_available",
                "DQ-INT8 custom kernel requires AVX-512 VNNI "
                "(VPDPBUSD) — runtime CPU detection failed");
    }
    if (is_w4a8_variant(variant) && !avx512vnni_available()) {
        return refuse_w4a8("avx512vnni_not_available",
                "W4A8 custom kernel requires AVX-512 VNNI "
                "(VPDPBUSD) — runtime CPU detection failed");
    }
    // group_size == 0 means the caller was never ported to the s4 path;
    // a non-multiple of the octet would leave the kernel a partial group.
    if (is_w4a8_variant(variant)
            && (group_size <= 0 || (group_size % kW4Octet) != 0)) {
        return refuse_w4a8("w4_group_size_unsupported",
                "W4A8 custom kernel requires a positive group_size that "
                "is a multiple of 8; falling back to AOCL DLP");
    }
    if (is_w4a8_variant(variant) && wei_scale_grp_stride <= 0) {
        return refuse_w4a8("w4_missing_wei_scale_stride",
                "W4A8 custom kernel requires the caller's {G, N} weight "
                "scale group stride (the expert's full N)");
    }
    if (is_w4a8_variant(variant) && src_scale_groups <= 0) {
        return refuse_w4a8("w4_missing_src_scale_groups",
                "W4A8 custom kernel requires a positive source-scale group "
                "count");
    }
    // Split-halves gated acts: the pack re-permutes weight columns, so
    // the `{G, N}` scale must be permuted on all G rows to match —
    // see `materialise_f32_wei_scale_cached`'s `n_groups` in the N-tile
    // hoist.  `swiglu_oai_mul` arrives already interleaved.
    // FP16 variants require native AVX-512-FP16 (VFMADD*PH).  On CPUs without that ISA
    // and other AVX-512 parts without FP16, `avx512f16_available()`
    // returns false (and on a toolchain without the intrinsics the
    // microkernels weren't compiled), so refuse cleanly and fall back
    // to AOCL DLP.
    if (is_f16_variant(variant) && !avx512f16_available()) {
        return refuse("avx512fp16_not_available",
                "FP16 custom kernel requires native AVX-512-FP16 "
                "(VFMADD*PH) — runtime CPU / toolchain detection "
                "failed");
    }
    // Cache the variant on the per-call context so `dispatch_tile()`
    // can route to the right kernel instantiation without re-running
    // the dtype switch per tile.
    out.variant = variant;
    // Asymmetric (u8 src + per-row src_zp) for BOTH the bf16-dst and
    // f32-dst asym variants; symmetric otherwise.  Keying on the
    // resolved variant (not just the bf16 one) ensures the f32-dst asym
    // path selects the kU8_Asym microkernels and honours src_zp.
    out.compute_int = (variant == KernelVariant::kU8_S8_BF16_ASYM
                              || variant == KernelVariant::kU8_S8_F32_ASYM
                              || variant == KernelVariant::kU8_S8_U8_ASYM)
            ? IntCompute::kU8_Asym
            : IntCompute::kS8_Sym;
    // Activation gate.
    //
    // The kernel handles four activation states inline:
    //
    //   * `none` — plain matmul; the epilogue stores the wide output.
    //   * `swiglu_oai_mul` — fused-in-kernel activation.  Caller
    //     provides W13 already interleaved as `[g0, u0, g1, u1, ...]`;
    //     the pair-pack store helper deinterleaves and applies the
    //     activation in registers (one fused matmul + activation pass,
    //     halved-width output).
    //   * `silu_and_mul` — fused-in-kernel activation.  Caller's W13
    //     is in canonical split-halves `[gate_cols | up_cols]`; the
    //     prepack permutes source columns so the CK arena physically
    //     matches the swiglu_oai_mul layout.  Same in-register
    //     epilogue contract.
    //   * `gelu_and_mul` — fused-in-kernel activation.  Caller-side
    //     contract is identical to `silu_and_mul`; the prepack uses
    //     the same column-interleave permutation, and the in-register
    //     epilogue uses a `gelu_tanh` polynomial approximation
    //     (max delta ≤ 1.5e-3 vs the reference's `gelu_erf`,
    //     comfortably inside the BF16 tolerance band).
    //
    // Bias is NOT supported on `silu_and_mul` / `gelu_and_mul` yet —
    // the bias-into-init epilogue would need to read
    // `[gate_bias | up_bias]` in permuted order; planned follow-up.
    if (act != grp_matmul_gated_act_t::swiglu_oai_mul
            && act != grp_matmul_gated_act_t::silu_and_mul
            && act != grp_matmul_gated_act_t::gelu_and_mul
            && act != grp_matmul_gated_act_t::none) {
        return refuse("unsupported_activation",
                "custom kernel admits only none / swiglu_oai_mul / "
                "silu_and_mul / gelu_and_mul (fused).  Other "
                "activations require the caller to translate to "
                "act=none and apply the activation as a separate "
                "post-pass.");
    }
    // silu_and_mul / gelu_and_mul + bias is not yet supported by the
    // fused epilogue (split-halves bias would need to be read in
    // permuted order to match the interleaved layout).  Refuse the
    // combination cleanly so the caller falls back to the
    // separate-pass path with canonical bias-into-init.  Typical MoE
    // layers are bias-free on W13, so this refusal does not affect
    // common decode shapes.
    if ((act == grp_matmul_gated_act_t::silu_and_mul
                || act == grp_matmul_gated_act_t::gelu_and_mul)
            && bias_dtype != data_type_t::none) {
        return refuse("split_halves_act_with_bias_not_fused",
                "silu_and_mul / gelu_and_mul fused-CK path is "
                "bias-free in this release; fall back to "
                "separate-pass for biased calls (planned "
                "follow-up).");
    }
    // `act_dtype` only matters when an activation is actually applied —
    // for `act = none` (plain GEMM) we accept any act_dtype (including
    // `none`) so callers that have no activation at all don't have to
    // fabricate a dummy value.
    // Fused activation dtype must match the family's store dtype: bf16
    // for the BF16 / DQ-INT8 families (their gated store helpers write
    // BF16), f16 for the FP16 family (its gated store helpers write
    // F16).  `variant` is already resolved above, so gate on it.
    if (act != grp_matmul_gated_act_t::none) {
        const data_type_t want_act_dt = is_f16_variant(variant)
                ? data_type_t::f16
                : data_type_t::bf16;
        if (act_dtype != want_act_dt) {
            return refuse("unsupported_act_dtype",
                    is_f16_variant(variant)
                            ? "fused activation requires act_dtype=f16 "
                              "(FP16 family)"
                            : "fused activation requires act_dtype=bf16");
        }
    }
    // Bias dtype resolution — handled by all three microkernel families
    // (bf16, DQ-INT8, FP16):
    //   * no bias buffer at all (caller passes nullptr per-expert).
    //   * bf16 bias — load 16 bf16 lanes, convert to fp32 in-register.
    //   * fp32 bias — load 16 fp32 lanes directly.
    //   * f16  bias — load 16 f16 lanes.  The FP16 kernel
    //     seeds its __m512h accumulator directly; the bf16 / DQ-INT8
    //     kernels widen to fp32 via `_mm512_cvtph_ps` (VCVTPH2PS, part of
    //     AVX-512F — NOT the native AVX-512-FP16 ISA), so f16 bias works on
    //     any bf16/VNNI-capable host regardless of FP16 ISA / toolchain.
    // Anything else (e.g. s8) falls back to the standard path.
    BiasKind bias_kind = BiasKind::none;
    if (bias_dtype == data_type_t::none) {
        bias_kind = BiasKind::none;
    } else if (bias_dtype == data_type_t::bf16) {
        bias_kind = BiasKind::bf16;
    } else if (bias_dtype == data_type_t::f32) {
        bias_kind = BiasKind::fp32;
    } else if (bias_dtype == data_type_t::f16) {
        bias_kind = BiasKind::f16;
    } else {
        return refuse("unsupported_bias_dtype",
                "custom kernel supports none/bf16/fp32/f16 bias only");
    }

    const int num_ops = static_cast<int>(M.size());
    if (num_ops <= 0 || num_ops > CallContext::kMaxExperts) {
        return refuse(
                "num_ops_out_of_range", "num_ops must be in [1, kMaxExperts]");
    }

    // ── Pack-NR planner — uniform across experts in any call we
    // accept (the caller has already asserted N uniformity diagnostically). ──
    int pack_nr = 0;
    for (int i = 0; i < num_ops; ++i) {
        if (M[i] <= 0) continue;
        pack_nr = plan_pack_nr(K[i], N[i]);
        break; // first active expert is representative
    }
    if (pack_nr != kNRMin && pack_nr != kNRMax) {
        return refuse("N_not_multiple_of_pack_nr",
                "pack_nr must be 32 or 64 and divide N");
    }

    // ── Per-expert contract gate ─────────────────────────────────────
    // Any failing expert disables the custom kernel for the whole
    // call — we don't want a mixed dispatch where some tiles take the
    // custom path and others fall back inside the OMP loop.
    int m_max = 0;
    int K_for_subtile = 0;
    // Per-expert ldb sanity (also catches a missized `ldb` vector
    // before we index it inside the pack).  Relaxed from strict
    // equality so callers that pass a longer `ldb[]` (e.g.
    // `group_matmul_direct` keeps ldb at the framework's prepack-
    // extras size while M is sliced to the active count) still pass
    // — only the first `num_ops` entries are read by the dispatch
    // loop below.  Undersized vectors still fail.
    if (ldb.size() < static_cast<size_t>(num_ops))
        return refuse("ldb_size_mismatch", "ldb vector size below num_ops");
    for (int i = 0; i < num_ops; ++i) {
        if (M[i] <= 0) continue;
        if (transA[i])
            return refuse("transA_not_supported",
                    "custom kernel reads src in row-major only");
        // Note: transB[i]=true is now SUPPORTED (PyTorch [N,K] layout).
        // The pack reads with the appropriate addressing and the cache
        // key folds transB into the discriminator so the two layouts
        // never alias.
        if (weight[i] == nullptr) return refuse("null_weight_in_active_expert");
        if ((N[i] % pack_nr) != 0)
            return refuse("N_not_multiple_of_pack_nr",
                    "an active expert's N is not a multiple of pack_nr");
        // The int8 microkernel reads the per-row src in 4-byte (K-quad)
        // broadcasts; the packed weight is zero-padded to a K-quad but
        // the hoisted src buffer is exactly K bytes per row, so a
        // K % 4 != 0 tail would over-read the src row.  Refuse the int8
        // CK path for unaligned K (the AOCL DLP sym-quant fallback
        // handles any K).  bf16 (K-pair) is unaffected.
        if (is_int8_variant(variant) && (K[i] % kVNNIInt8Quad) != 0)
            return refuse("int8_K_not_multiple_of_4",
                    "DQ-INT8 custom kernel requires K divisible by 4 "
                    "(VNNI K-quad); falling back to AOCL DLP");
        if (is_int8_variant(variant) && K[i] > kMaxSafeInt8AccumK)
            return refuse("int8_K_exceeds_s32_accumulator",
                    "INT8 custom kernel K can overflow its signed int32 "
                    "accumulator; falling back to AOCL DLP");
        // With the `group_size % 8` check above, this also guarantees
        // `K % 8 == 0`: no partial trailing octet in the pack, and no
        // src over-read in the kernel's 4-byte broadcasts.
        if (is_w4a8_variant(variant) && (K[i] % group_size) != 0)
            return refuse_w4a8("w4_K_not_multiple_of_group_size",
                    "W4A8 custom kernel requires K divisible by "
                    "group_size; falling back to AOCL DLP");
        if (is_w4a8_variant(variant) && src_scale_groups > 1
                && src_scale_groups != K[i] / group_size) {
            return refuse_w4a8("w4_src_wei_group_count_mismatch",
                    "W4A8 custom kernel requires activation and weight "
                    "scales to use the same K-group count");
        }
        if (alpha[i] != 1.0f || beta[i] != 0.0f)
            return refuse("alpha_beta_not_supported",
                    "custom kernel requires alpha=1, beta=0 per expert");
        // ldb must accommodate the inner row of the chosen layout.
        const int min_ldb = transB[i] ? K[i] : N[i];
        if (ldb[i] < min_ldb)
            return refuse("ldb_below_min_row_stride",
                    "an active expert's ldb is smaller than min row stride");
        // is_weights_const = false means the caller may mutate this
        // expert's weight buffer in-place between calls.  The CK pack
        // cache is keyed on the source weight pointer and has no per-
        // expert "skip cache" branch (unlike AOCL DLP's `run_dlp(...)`
        // at aocl_kernel.cpp:1700-1702), so a cached packed copy from a
        // previous call could silently shadow a mutated weight and
        // produce wrong results.  Refuse the whole call so the caller
        // falls back to the standard AOCL DLP path, which honours the
        // flag at runtime.  Empty `is_weights_const` is treated as "all
        // const" (legacy callers that don't pass the field).
        if (!is_weights_const.empty()
                && static_cast<size_t>(i) < is_weights_const.size()
                && !is_weights_const[i])
            return refuse("non_const_weight_in_active_expert",
                    "custom kernel pack cache cannot honour an active "
                    "expert's is_weights_const=false");
        if (M[i] > m_max) m_max = M[i];
        if (K_for_subtile == 0) K_for_subtile = K[i];
    }
    if (m_max == 0) {
        // No active experts — mark disabled but treat as a no-op so the
        // caller can still complete its own bookkeeping.  Not strictly a
        // "refusal" (nothing to dispatch), but we log so all_zero_M
        // problems are visible too.
        return refuse("all_experts_have_M_zero");
    }

    // ── Build the run context ─────────────────────────────────────────
    out.pack_nr = pack_nr;
    out.NV = pack_nr / 16;
    // W4A8 caps MR lower: its per-group f32 accumulator lives alongside
    // the s32 one.  The general bound would index null table slots.
    out.max_mr = is_mxfp4_variant(variant) ? max_mr_for_nv_mxfp4(out.NV)
            : is_w4a8_variant(variant)     ? max_mr_for_nv_s4(out.NV)
            : is_int8_variant(variant)
            ? max_mr_for_nv_int8(out.NV, out.compute_int)
            : max_mr_for_nv(out.NV);
    if (is_w4a8_variant(variant)
            || (is_int8_variant(variant) && group_size == 32)) {
        out.group_size = group_size;
        out.wei_scale_grp_stride = wei_scale_grp_stride;
        out.src_scale_groups = src_scale_groups;
    }
    out.kblock_auto = se_kblock_auto(num_ops, m_max);
    out.single_expert = (num_ops == 1);
    const bool bf16_prompt
            = se_bf16_prompt(out.single_expert, out.NV, m_max, variant);
    out.f16_prompt = se_f16_prompt(out.single_expert, out.NV, m_max, variant);
    out.base_max_mr = (bf16_prompt || out.f16_prompt) ? out.max_mr : 0;
    out.max_mr = se_prompt_max_mr(out.max_mr, bf16_prompt, out.f16_prompt);
    // Q8 per-group keeps an f32 sum beside the int32 dpbusd accumulators.
    // MR 4 fits both in zmm; a wider row tile spills the K loop.
    if (out.group_size == 32 && is_int8_variant(out.variant))
        out.max_mr = std::min(out.max_mr, 4);
    // Map framework activation → CK ActivationKind.  Order matters: refusal
    // gate above guarantees `act` is one of {none, swiglu_oai_mul,
    // silu_and_mul, gelu_and_mul} at this point.
    out.act_kind = (act == grp_matmul_gated_act_t::swiglu_oai_mul)
            ? ActivationKind::swiglu_oai_mul
            : (act == grp_matmul_gated_act_t::silu_and_mul)
            ? ActivationKind::silu_and_mul
            : (act == grp_matmul_gated_act_t::gelu_and_mul)
            ? ActivationKind::gelu_and_mul
            : ActivationKind::none;
    out.bias_kind = bias_kind;
    out.prompt_kc_eligible = se_prompt_kc_eligible(bf16_prompt, out.f16_prompt,
            K_for_subtile, out.act_kind != ActivationKind::none,
            bias_kind != BiasKind::none);
    out.prompt_kc = se_prompt_kc(out.prompt_kc_eligible, out.f16_prompt,
            pack_nr, K_for_subtile, ck_l2_budget_bytes());
    // Representative subtile_cols (worst-case m_max) — used by APILOG /
    // debug.  Dispatch reads per-expert values below.  Byte model is
    // dtype-aware: bf16 family uses (2,2,0); int8 family uses (1,1,4)
    // — see `subtile_bytes_for_variant` for the rationale.
    const SubtileBytes sb = subtile_bytes_for_variant(out.variant);
    out.subtile_cols = pick_l2_subtile_cols(m_max, K_for_subtile, pack_nr,
            sb.src_bytes_per_elem, sb.wei_bytes_per_elem,
            sb.comp_bytes_per_col);

    // Fill the per-MR kernel-pointer table from the resolved variant:
    // BF16 family → `kernel_table` (template axis is `OutputType`); DQ-INT8
    // family → `kernel_table_int8` (template axes are `IntCompute` and
    // `OutputType` — bf16 or f32 dst).  Only the populated table is read
    // by `dispatch_tile`; the other stays zero-initialised.
    if (is_int8_variant(out.variant)) {
        const OutputType dst_dt = dst_dt_for_variant(out.variant);
        if (fill_kernel_table_int8(out.NV, out.compute_int, out.act_kind,
                    dst_dt, out.kernel_table_int8, out.group_size)
                != status_t::success) {
            return refuse("kernel_table_int8_fill_failed",
                    "no DQ-INT8 microkernel for this (NV, compute, "
                    "act_kind, dst_dt) tuple — note gated act + FP32-dst "
                    "is intentionally rejected (BF16 dst only)");
        }
    } else if (is_mxfp4_variant(out.variant)) {
        if (fill_kernel_table_mxfp4(out.NV, out.act_kind, out.src_scale_groups,
                    dst_dt_for_variant(out.variant), out.kernel_table_mxfp4)
                != status_t::success) {
            return refuse_w4a8("kernel_table_mxfp4_fill_failed",
                    "no MXFP4 microkernel for this (NV, act_kind, dst_dt) "
                    "tuple; a gated act stores bf16 only");
        }
    } else if (is_w4a8_variant(out.variant)) {
        if (fill_kernel_table_s4(out.NV, out.act_kind, out.src_scale_groups,
                    dst_dt_for_variant(out.variant), out.kernel_table_s4,
                    out.kernel_table_s4_offset)
                != status_t::success) {
            return refuse_w4a8("kernel_table_s4_fill_failed",
                    "no W4A8 microkernel for this (NV, act_kind) tuple");
        }
    } else if (is_f16_variant(out.variant)) {
        const OutputType dst_dt = dst_dt_for_variant(out.variant);
        if (fill_kernel_table_f16(
                    out.NV, out.act_kind, dst_dt, out.kernel_table_f16)
                != status_t::success) {
            return refuse("kernel_table_f16_fill_failed",
                    "no FP16 microkernel for this (NV, act_kind, dst_dt) "
                    "tuple — note gated act + FP32-dst is intentionally "
                    "rejected (F16 dst only), and on a toolchain without "
                    "AVX-512-FP16 intrinsics every FP16 selector slot is "
                    "nullptr so the call falls back to AOCL DLP");
        }
        if (fill_kernel_table_f16_kblock(out.NV, out.kernel_table_f16_kblock)
                != status_t::success) {
            return refuse("kernel_table_f16_kblock_fill_failed",
                    "no FP16 partial-K microkernel for this NV/MR tuple");
        }
        const bool f32_chunks = out.f16_prompt && out.prompt_kc_eligible
                && out.variant == KernelVariant::kF16_F16_F16;
        for (int nb = 1; f32_chunks && nb <= kF16PromptNb; ++nb) {
            for (int mr = 1; mr <= out.max_mr; ++mr) {
                out.kernel_table_f16_prompt_f32[nb - 1][mr]
                        = select_f16_prompt_ukernel(
                                mr, nb, ActivationKind::none, OutputType::kF32);
                if (out.kernel_table_f16_prompt_f32[nb - 1][mr] == nullptr)
                    return refuse("kernel_table_f16_prompt_fill_failed",
                            "no FP32-destination FP16 prompt microkernel for "
                            "the K chunks");
            }
        }
        for (int nb = 1; out.f16_prompt && nb <= kF16PromptNb; ++nb) {
            for (int mr = 1; mr <= out.max_mr; ++mr) {
                out.kernel_table_f16_prompt[nb - 1][mr]
                        = select_f16_prompt_ukernel(
                                mr, nb, out.act_kind, dst_dt);
                if (out.kernel_table_f16_prompt[nb - 1][mr] == nullptr)
                    return refuse("kernel_table_f16_prompt_fill_failed",
                            "no FP16 prompt microkernel for this (MR, NB, "
                            "act_kind, dst_dt) tuple");
            }
        }
    } else {
        const OutputType dst_dt = dst_dt_for_variant(out.variant);
        if (fill_kernel_table(
                    out.NV, out.max_mr, out.act_kind, dst_dt, out.kernel_table)
                != status_t::success) {
            return refuse("kernel_table_fill_failed",
                    "no microkernel for this (NV, act_kind, dst_dt) tuple "
                    "— note swiglu_oai_mul + FP32-dst is intentionally "
                    "rejected (BF16 dst only)");
        }
    }

    // ── Pre-pack every active expert's weight (single-threaded; the
    // LRU mutex never contends in the per-tile path).  When the env
    // knob `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_SUBTILE_PER_EXPERT=1`
    // is set, also size each expert's L2-friendly subtile individually:
    // small-M experts (low A panel footprint) get a wider subtile for
    // better B reuse.  Default (knob=0): leave `subtile_cols_per_expert`
    // zero-filled so `dispatch_tile` falls back to the global m_max-
    // sized `ctx.subtile_cols` — the production baseline.
    const bool per_expert_subtile
            = get_grp_matmul_custom_kernel_subtile_per_expert();
    // For `silu_and_mul` and `gelu_and_mul` the caller's W13 is in
    // canonical split-halves layout; the prepack permutes source
    // columns into the interleaved layout the in-register fused
    // epilogue expects.  Both split-halves variants share the SAME
    // permutation — they differ only in the kernel-side activation
    // math, not in the pack layout, so the cache key bit
    // (`kInterleaveSplitMarker` in pack.cpp) is shared and a packed
    // arena warmed for silu can be reused for gelu on the same
    // weight pointer.
    // `swiglu_oai_mul` callers already provide interleaved weight at
    // the API boundary, so the flag stays false for them.
    const bool interleave_split_halves
            = (act == grp_matmul_gated_act_t::silu_and_mul)
            || (act == grp_matmul_gated_act_t::gelu_and_mul);
    out.packed_ptrs.fill(nullptr);
    out.packed_ptrs_int8.fill(nullptr);
    out.packed_ptrs_f16.fill(nullptr);
    out.subtile_cols_per_expert.fill(0);
    // `out.owned_packed_ptrs` / `owned_packed_ptrs_int8` /
    // `owned_packed_ptrs_f16` were already zeroed by `out.reset()` at
    // entry (via `release_owned_buffers()`); only the `cache_off` branch
    // below stores into them.
    const bool variant_is_int8 = is_int8_variant(out.variant);
    const bool variant_is_f16 = is_f16_variant(out.variant);
    const bool variant_is_w4 = is_w4a8_variant(out.variant);
    for (int i = 0; i < num_ops; ++i) {
        if (M[i] <= 0) continue;
        bool was_hit_unused = false;
        // ── Caller-prepacked weight (mem_format_b == 'r') ──────────────
        // The weight is ALREADY in the CK VNNI layout (produced by the
        // moe_custom_kernel weight-prepack).  Alias it directly and skip
        // the pack + LRU entirely — the caller owns the buffer lifetime,
        // so it must NOT enter the cache (clear() would free caller
        // memory) and must NOT be recorded in owned_packed_ptrs.
        const bool prepacked_i = !weights_prepacked.empty()
                && static_cast<size_t>(i) < weights_prepacked.size()
                && weights_prepacked[i];
        const bool aocl_prepacked_i = !weights_aocl_prepacked.empty()
                && static_cast<size_t>(i) < weights_aocl_prepacked.size()
                && weights_aocl_prepacked[i];
        const bool aocl_sym_prepacked_i = !weights_aocl_sym_prepacked.empty()
                && static_cast<size_t>(i) < weights_aocl_sym_prepacked.size()
                && weights_aocl_sym_prepacked[i];
        if (prepacked_i && aocl_prepacked_i) {
            return refuse("conflicting_prepacked_weight_provenance",
                    "one expert cannot be both CK-VNNI-packed and "
                    "AOCL-DLP-blocked");
        }
        if (prepacked_i) {
            if (variant_is_w4) {
                // The prepack API has no s4 family (`prepack_params_t`
                // carries no `group_size`), so an 'r' s4 weight cannot
                // have come from this library.  Aliasing it would feed
                // the kernel raw nibbles in place of the octet layout.
                return refuse_w4a8("w4_caller_prepack_unsupported",
                        "W4A8 custom kernel has no caller-prepacked "
                        "('r') weight format; pass raw packed s4 with "
                        "mem_format_b = 'n'");
            }
            if (variant_is_int8) {
                out.packed_ptrs_int8[i]
                        = static_cast<const int8_t *>(weight[i]);
            } else if (variant_is_f16) {
                out.packed_ptrs_f16[i]
                        = static_cast<const float16_t *>(weight[i]);
            } else {
                out.packed_ptrs[i] = static_cast<const bfloat16_t *>(weight[i]);
            }
            if (per_expert_subtile) {
                out.subtile_cols_per_expert[i] = pick_l2_subtile_cols(M[i],
                        K[i], pack_nr, sb.src_bytes_per_elem,
                        sb.wei_bytes_per_elem, sb.comp_bytes_per_col);
            }
            continue;
        }

        // AOCL blocked -> CK conversion belongs to CK preparation, not
        // scalar/group AUTO routing. Reaching this point means every CK
        // structural/ISA/shape contract and function-table check above has
        // succeeded. AOCL fallbacks retain the original blocked pointer.
        //
        // The derived cache owns one right-sized final CK arena. On a miss,
        // AOCL unreorders into temporary raw storage and CK packs directly
        // into the arena before publication.
        const void *ck_weight = weight[i];
        int ck_ldb = ldb[i];
        if (aocl_prepacked_i) {
            // AOCL's one four-bit inverse is INT4's.
            const bool aocl_s4 = variant_is_w4 && wei_dtype == data_type_t::s4;
            const bool bf16_variant
                    = out.variant == KernelVariant::kBF16_BF16_BF16
                    || out.variant == KernelVariant::kBF16_BF16_F32;
            const bool static_u8s8_variant = variant_is_int8
                    && src_dtype == data_type_t::u8
                    && wei_dtype == data_type_t::s8 && !dynamic_quant;
            const bool sym_int8_variant = variant_is_int8
                    && !static_u8s8_variant
                    && out.compute_int == IntCompute::kS8_Sym
                    && aocl_sym_prepacked_i;
            if (!bf16_variant && !variant_is_f16 && !aocl_s4
                    && !static_u8s8_variant && !sym_int8_variant) {
                return refuse("aocl_unreorder_unsupported_variant",
                        "AOCL unreorder-to-CK does not support this variant; "
                        "dynamic asymmetric INT8 keeps its AOCL fallback");
            }
            bool derived_cache_hit = false;
            std::shared_ptr<void> derived_ck_weight;
            status_t conversion_status = status_t::failure;
            const char *family_name = "bf16";
            if (aocl_s4) {
                family_name = "s8s4_sym";
                conversion_status = get_or_cache_aocl_ck_weight_s4(weight[i],
                        K[i], N[i], ldb[i], transB[i], num_threads, pack_nr,
                        interleave_split_halves, group_size, derived_ck_weight,
                        &derived_cache_hit, wc_mode);
            } else if (sym_int8_variant) {
                family_name = "s8s8_sym";
                conversion_status = get_or_cache_aocl_ck_weight_int8_sym(
                        weight[i], K[i], N[i], ldb[i], transB[i], num_threads,
                        pack_nr, interleave_split_halves,
                        aocl_unreorder_group_size, derived_ck_weight,
                        &derived_cache_hit, wc_mode);
            } else if (static_u8s8_variant) {
                family_name = "u8s8";
                conversion_status = get_or_cache_aocl_ck_weight_int8(weight[i],
                        K[i], N[i], ldb[i], transB[i], num_threads, pack_nr,
                        interleave_split_halves, derived_ck_weight,
                        &derived_cache_hit, wc_mode);
            } else if (variant_is_f16) {
                family_name = "f16";
                conversion_status = get_or_cache_aocl_ck_weight_f16(weight[i],
                        K[i], N[i], ldb[i], transB[i], num_threads, pack_nr,
                        interleave_split_halves, derived_ck_weight,
                        &derived_cache_hit, wc_mode);
            } else {
                conversion_status = get_or_cache_aocl_ck_weight(weight[i], K[i],
                        N[i], ldb[i], transB[i], num_threads, pack_nr,
                        interleave_split_halves, derived_ck_weight,
                        &derived_cache_hit, wc_mode);
            }
            if (conversion_status != status_t::success || !derived_ck_weight) {
                return refuse("aocl_to_ck_conversion_failed",
                        "CK declined; AOCL fallback keeps the original "
                        "blocked weight");
            }
            out.aocl_custom_kernel_weight_leases[i] = derived_ck_weight;
            if (aocl_s4) {
                out.packed_ptrs_w4[i]
                        = static_cast<const int8_t *>(derived_ck_weight.get());
            } else if (variant_is_int8) {
                out.packed_ptrs_int8[i]
                        = static_cast<const int8_t *>(derived_ck_weight.get());
            } else if (variant_is_f16) {
                out.packed_ptrs_f16[i] = static_cast<const float16_t *>(
                        derived_ck_weight.get());
            } else {
                out.packed_ptrs[i] = static_cast<const bfloat16_t *>(
                        derived_ck_weight.get());
            }
            apilog_verbose("[GRP_MATMUL.CK AOCL->CK] expert=", i,
                    " family=", family_name,
                    " cache=", (derived_cache_hit ? "hit" : "miss"),
                    " K=", K[i], " N=", N[i], " transB=", (transB[i] ? 1 : 0));
            if (per_expert_subtile) {
                out.subtile_cols_per_expert[i] = pick_l2_subtile_cols(M[i],
                        K[i], pack_nr, sb.src_bytes_per_elem,
                        sb.wei_bytes_per_elem, sb.comp_bytes_per_col);
            }
            continue;
        }

        if (variant_is_f16) {
            // FP16 path — caller's weight is f16; the plain (non-K-
            // interleaved) FP16 pack mirrors the bf16 contract for
            // is_weights_const / pack_nr / ldb / transB.
            status_t pst = get_or_pack_weight_f16(
                    static_cast<const float16_t *>(ck_weight), K[i], N[i],
                    ck_ldb, pack_nr,
                    /*transB=*/transB[i],
                    /*interleave_split_halves=*/interleave_split_halves,
                    &out.packed_ptrs_f16[i],
                    /*was_hit_out=*/&was_hit_unused,
                    /*disable_cache=*/cache_off,
                    /*in_place=*/ck_in_place,
                    /*wei_buffer_capacity_bytes=*/0);
            if (pst != status_t::success) {
                return refuse("weight_pack_failed",
                        "get_or_pack_weight_f16 returned failure — "
                        "see preceding log_error for OOM/arg detail");
            }
            if (cache_off) {
                out.owned_packed_ptrs_f16[i] = out.packed_ptrs_f16[i];
            }
        } else if (variant_is_int8) {
            // DQ-INT8 path — caller's weight is signed s8 (`is_weights_const`
            // / pack_nr / ldb / transB contracts are the same as bf16).
            // The int8 pack writes both the VNNI-quad weight slab and the
            // per-column compensation row; the microkernel reads both
            // through a single `int8_t *`.
            const size_t expert_cap
                    = (i < static_cast<int>(wei_capacities.size()))
                    ? wei_capacities[static_cast<size_t>(i)]
                    : 0;
            const int pack_group = (group_size == 32) ? 32 : 0;
            status_t pst = get_or_pack_weight_int8(
                    static_cast<const int8_t *>(ck_weight), K[i], N[i], ck_ldb,
                    pack_nr,
                    /*transB=*/transB[i],
                    /*interleave_split_halves=*/interleave_split_halves,
                    &out.packed_ptrs_int8[i],
                    /*was_hit_out=*/&was_hit_unused,
                    /*disable_cache=*/cache_off,
                    /*in_place=*/ck_in_place && expert_cap > 0, expert_cap,
                    pack_group);
            if (pst != status_t::success) {
                return refuse("weight_pack_failed",
                        "get_or_pack_weight_int8 returned failure — "
                        "see preceding log_error for OOM/arg detail");
            }
            if (cache_off) {
                out.owned_packed_ptrs_int8[i] = out.packed_ptrs_int8[i];
            }
        } else if (variant_is_w4) {
            const W4Format fmt = w4_format_for_variant(out.variant);
            const uint8_t *scale_i = nullptr;
            int scale_ld_i = wei_scale_grp_stride;
            if (w4_scales_in_pack(fmt)) {
                const size_t e = static_cast<size_t>(i);
                scale_i = e < pack_scales.size()
                        ? static_cast<const uint8_t *>(pack_scales[e])
                        : nullptr;
                scale_ld_i = e < pack_scale_ld.size() ? pack_scale_ld[e] : 0;
                if (scale_i == nullptr || scale_ld_i <= 0) {
                    return refuse_w4a8("mxfp4_scales_missing",
                            "MXFP4 packs its scales and needs each expert's "
                            "f8_e8m0 {G, N} wei_scale");
                }
            }
            // Caller's weight is the nibble-packed four-bit buffer, the
            // same bytes the AOCL s4 path receives for W4A8.
            status_t pst = get_or_pack_weight_w4(
                    static_cast<const int8_t *>(ck_weight), K[i], N[i], ck_ldb,
                    pack_nr,
                    /*transB=*/transB[i],
                    /*interleave_split_halves=*/interleave_split_halves,
                    /*group_size=*/group_size, &out.packed_ptrs_w4[i],
                    /*was_hit_out=*/&was_hit_unused,
                    /*disable_cache=*/cache_off, fmt, scale_i, scale_ld_i);
            if (pst != status_t::success) {
                return refuse_w4a8("weight_pack_failed",
                        "get_or_pack_weight_w4 returned failure — "
                        "see preceding log_error for OOM/arg detail");
            }
            if (cache_off) {
                out.owned_packed_ptrs_w4[i] = out.packed_ptrs_w4[i];
            }
        } else {
            // BF16 path — unchanged.
            status_t pst = get_or_pack_weight_bf16(
                    static_cast<const bfloat16_t *>(ck_weight), K[i], N[i],
                    ck_ldb, pack_nr,
                    /*transB=*/transB[i],
                    /*interleave_split_halves=*/interleave_split_halves,
                    &out.packed_ptrs[i],
                    /*was_hit_out=*/&was_hit_unused,
                    /*disable_cache=*/cache_off,
                    /*in_place=*/ck_in_place,
                    /*wei_buffer_capacity_bytes=*/0);
            if (pst != status_t::success) {
                return refuse("weight_pack_failed",
                        "get_or_pack_weight_bf16 returned failure — "
                        "see preceding log_error for OOM/arg detail");
            }
            // Only the no-cache (WC=0) path hands back a caller-owned buffer to
            // free.  The in-place (WC=2) path returns the caller's own weight
            // pointer (cached as an `in_place` entry) — NOT owned here.
            if (cache_off) { out.owned_packed_ptrs[i] = out.packed_ptrs[i]; }
        }
        if (per_expert_subtile) {
            out.subtile_cols_per_expert[i] = pick_l2_subtile_cols(M[i], K[i],
                    pack_nr, sb.src_bytes_per_elem, sb.wei_bytes_per_elem,
                    sb.comp_bytes_per_col);
        }
    }

    out.enabled = true;

    // Counterpart to the REFUSED lines above; once per call, so it
    // cannot flood a hot loop.
    // A format whose pack holds the scales reads each expert's from its own
    // pack, so the call-wide stride does not describe it.
    const W4Format out_fmt = w4_format_for_variant(out.variant);
    const bool scales_in_pack
            = is_w4a8_variant(out.variant) && w4_scales_in_pack(out_fmt);
    const auto wei_scale_stride_label = [&]() {
        return scales_in_pack ? std::string("in-pack")
                              : std::to_string(out.wei_scale_grp_stride);
    };
    if (is_w4a8_variant(out.variant) && s_w4a8_log) {
        const int rep_K = (K.empty() ? 0 : K[0]);
        const int groups = (out.group_size > 0) ? (rep_K / out.group_size) : 0;
        apilog_info("[GRP_MATMUL.CK.W4A8 ENGAGED] fmt=",
                w4_format_name(out_fmt), " Calling ", w4_format_label(out_fmt),
                " custom kernel: group_size=", out.group_size, " G=", groups,
                " K=", rep_K,
                " wei_scale_grp_stride=", wei_scale_stride_label(),
                " pack_nr=", out.pack_nr, " NV=", out.NV,
                " max_mr=", out.max_mr, " act=",
                (out.act_kind == ActivationKind::swiglu_oai_mul
                                ? "swiglu_oai_mul"
                                : out.act_kind == ActivationKind::silu_and_mul
                                ? "silu_and_mul"
                                : out.act_kind == ActivationKind::gelu_and_mul
                                ? "gelu_and_mul"
                                : "none"),
                " scale_kind=",
                (out.scale_kind == ScaleKind::kBf16 ? "bf16" : "f32"), " bias=",
                (out.bias_kind == BiasKind::none                  ? "none"
                                : out.bias_kind == BiasKind::bf16 ? "bf16"
                                : out.bias_kind == BiasKind::f16  ? "f16"
                                                                  : "f32"),
                " num_ops=", num_ops);
    }

    if (s_refuse_log) {
        const char *variant_name
                = (out.variant == KernelVariant::kBF16_BF16_BF16)
                ? "bf16_bf16_bf16"
                : (out.variant == KernelVariant::kBF16_BF16_F32)
                ? "bf16_bf16_f32"
                : (out.variant == KernelVariant::kS8_S8_BF16_SYM)
                ? "s8_s8_bf16_sym"
                : (out.variant == KernelVariant::kU8_S8_BF16_ASYM)
                ? "u8_s8_bf16_asym"
                : (out.variant == KernelVariant::kS8_S8_F32_SYM)
                ? "s8_s8_f32_sym"
                : (out.variant == KernelVariant::kU8_S8_F32_ASYM)
                ? "u8_s8_f32_asym"
                : (out.variant == KernelVariant::kU8_S8_U8_ASYM)
                ? "u8_s8_u8_asym"
                : (out.variant == KernelVariant::kF16_F16_F16) ? "f16_f16_f16"
                : (out.variant == KernelVariant::kF16_F16_F32) ? "f16_f16_f32"
                : (out.variant == KernelVariant::kS8_S4_BF16_SYM)
                ? "s8_s4_bf16_sym"
                : (out.variant == KernelVariant::kS8_S4_F32_SYM)
                ? "s8_s4_f32_sym"
                : (out.variant == KernelVariant::kS8_MXFP4_BF16_SYM)
                ? "s8_mxfp4_bf16_sym"
                : (out.variant == KernelVariant::kS8_MXFP4_F32_SYM)
                ? "s8_mxfp4_f32_sym"
                : "unsupported";
        apilog_verbose("[GRP_MATMUL.CK ENGAGED] variant=", variant_name,
                " group_size=", out.group_size, // 0 for non-W4A8 families
                " wei_scale_grp_stride=", wei_scale_stride_label(),
                " pack_nr=", out.pack_nr, " NV=", out.NV,
                " max_mr=", out.max_mr, " subtile_cols=", out.subtile_cols,
                " act_kind=",
                (out.act_kind == ActivationKind::swiglu_oai_mul
                                ? "swiglu_oai_mul"
                                : out.act_kind == ActivationKind::silu_and_mul
                                ? "silu_and_mul"
                                : out.act_kind == ActivationKind::gelu_and_mul
                                ? "gelu_and_mul"
                                : "none"),
                " bias_kind=",
                (out.bias_kind == BiasKind::none                  ? "none"
                                : out.bias_kind == BiasKind::bf16 ? "bf16"
                                : out.bias_kind == BiasKind::f16  ? "f16"
                                                                  : "fp32"),
                " num_ops=", num_ops,
                " per_expert_subtile=", (per_expert_subtile ? 1 : 0));
    }
    return status_t::success;
}

// (The ahead-of-time warm-pack — `warm_pack_all_custom_kernel_experts`
// + `PackProbeStats` — moved to
// `group_matmul/prepack/prepack_custom_kernel.{hpp,cpp}`.  This file
// keeps the per-call dispatcher path; `plan_pack_nr` was promoted to
// the public dispatch.hpp surface so the prepack module can compute
// the same NR the dispatcher will pack under.)

// The o-block-outer threshold (`kOblockOuterMinCalls`) and its single-expert
// gate are `se_oblock_outer` in `single_expert/single_expert_hooks.hpp`.

// Visits every (MR call, o-block) pair of one sub-tile, o-block outer when
// `oblk_outer` is set.  Each pair writes a disjoint output block.
template <typename CallOne>
inline void for_each_call_block(
        bool oblk_outer, int n_calls, int n_blocks, const CallOne &call_one) {
    if (oblk_outer) {
        for (int b = 0; b < n_blocks; ++b)
            for (int c = 0; c < n_calls; ++c)
                call_one(c, b);
    } else {
        for (int c = 0; c < n_calls; ++c)
            for (int b = 0; b < n_blocks; ++b)
                call_one(c, b);
    }
}

// DQ-INT8 dispatch tile — owns the per-tile sub-tile + per-MR loop
// for the `kS8_S8_BF16_SYM` / `kU8_S8_BF16_ASYM` variants.  Splits
// out of the public `dispatch_tile` so the BF16 loop body stays
// straight-line and any future Phase-2 changes to the int8 hot
// path land here without disturbing the bf16 sibling.
//
// Pointer convention:
//   * `A`         — caller's `src` cast to `uint8_t *`.  For
//     `kS8_S8_BF16_SYM` the caller passes the s8 buffer
//     (`int8_t *`) but the microkernel re-interprets it as
//     `uint8_t *` and XORs each broadcast with `0x80808080` to
//     run VPDPBUSD; the compensation row precomputed at pack time
//     undoes the resulting `+128 × sum_wei` bias.  Casting once
//     here keeps the signature uniform across compute flavours.
//   * `Bpacked`   — `int8_t *` to this expert's packed weight slab
//     (one o-block of weight bytes followed by `pack_nr` int32
//     compensation lanes; the microkernel walks both regions via
//     byte arithmetic, see int8_microkernel.cpp).
//   * `src_scale_full` / `src_zp_full` / `wei_scale_full` — full
//     M / M / N vectors as the caller passed to the public
//     `dispatch_tile`.  The dispatcher slices to the per-(m_off,
//     mr_now) and per-(sub_col_base, n_blocks) windows.  `src_zp`
//     is passed straight through (nullptr for sym).
//
// Tight / wide destination handling is identical to the bf16
// sibling — gated activation epilogues store to the half-width tight
// arena; act=none stores to the wide [M, N] dst.  The dst element
// width is variant-dependent (`dst_elem_bytes_int8` below): bf16 for
// the *_BF16_* variants, f32 for the *_F32_* (act=none) variants.
namespace {
inline void dispatch_tile_int8(const CallContext &ctx, int expert_idx, int M,
        int K, int n_tile, int col_start, const void *src, int lda,
        const void *bias, void *tight_dst, int tight_ldc,
        const void *src_scale_full, const int32_t *src_zp_full,
        const void *wei_scale_full, const Int8EpilogueConfig &epilogue) {
    // Per-tile contract: src_scale + wei_scale are required; src_zp
    // is required iff the variant is asym.  The caller (N-tile
    // executor) populates both from the hoisted dynamic-quant output;
    // mis-routing here would silently dequant to zero, which the
    // microkernel's epilogue would propagate.  Assert in debug.
    assert(src_scale_full != nullptr
         && "dispatch_tile_int8: src_scale is null — N-tile hoist "
            "must populate this for DQ-INT8");
    assert(wei_scale_full != nullptr
         && "dispatch_tile_int8: wei_scale is null — caller must "
            "thread the per-expert wei_scale");
    // Asym requires src_zp — true for both the bf16-dst and f32-dst
    // asym variants.  Key off the resolved compute family rather than a
    // single variant so the f32-dst asym path is covered too.
    if (ctx.compute_int == IntCompute::kU8_Asym) {
        assert(src_zp_full != nullptr
           && "dispatch_tile_int8: src_zp is null on the asym path "
              "— hoist must populate it for compute=u8");
    }

    const int8_t *Bpacked_full = ctx.packed_ptrs_int8[expert_idx];
    const auto *A_bytes = static_cast<const uint8_t *>(src);
    std::byte *Tight_bytes = static_cast<std::byte *>(tight_dst);
    // Dst element width tracks the resolved variant: BF16 (2 bytes) for
    // the kS8/U8_*_BF16_* variants, FP32 (4 bytes) for the kS8/U8_*_F32_*
    // variants.  The kernel itself reinterprets the `void *` dst to the
    // right type via its `OutputType` template; the dispatcher only needs the
    // element stride here for pointer arithmetic.  FP32 dst is act=none
    // only (gated kinds are BF16-dst), so the gated path below always
    // sees the BF16 width.
    const OutputType dst_dt = dst_dt_for_variant(ctx.variant);
    const size_t dst_elem_bytes_int8 = (dst_dt == OutputType::kF32)
            ? sizeof(float)
            : (dst_dt == OutputType::kU8) ? sizeof(uint8_t)
                                          : sizeof(bfloat16_t);

    const int K_quad = (K + kVNNIInt8Quad - 1) / kVNNIInt8Quad;
    // Per-o-block stride in BYTES: weight slab + per-column int32
    // compensation row.  Mirror of the pack-time
    // `bytes_per_oblock` expression in pack.cpp; the two MUST stay
    // in lockstep.
    const int comp_rows = (ctx.group_size > 0 && (K % ctx.group_size) == 0)
            ? (K / ctx.group_size)
            : 1;
    const size_t o_blk_stride_bytes
            = static_cast<size_t>(K_quad) * ctx.pack_nr * kVNNIInt8Quad
            + static_cast<size_t>(comp_rows) * ctx.pack_nr * sizeof(int32_t);

    const int subtile_cols = (ctx.subtile_cols_per_expert[expert_idx] > 0)
            ? ctx.subtile_cols_per_expert[expert_idx]
            : ctx.subtile_cols;

    // Balanced MR partition — identical scheme to the bf16 sibling.
    // See dispatch_tile (bf16) for the rationale.
    const int n_calls = (M + ctx.max_mr - 1) / ctx.max_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;

    const size_t bias_elem_bytes = (ctx.bias_kind == BiasKind::fp32)
            ? sizeof(float)
            : sizeof(bfloat16_t); // bf16 or f16 (both 2B)

    // Scale element width — the microkernel reads src/wei scales as bf16
    // (converting on load) or f32.  The dispatcher slices both scale
    // buffers by byte offset using this stride; the kernel re-casts the
    // `const void *` per `ctx.scale_kind`.
    const size_t scale_elem_bytes = (ctx.scale_kind == ScaleKind::kBf16)
            ? sizeof(bfloat16_t)
            : sizeof(float);
    const bool oblk_outer = se_oblock_outer(ctx.single_expert, n_calls);
    const auto mr_of = [mr_base, n_big](int c) {
        return (c < n_big) ? mr_base + 1 : mr_base;
    };
    const auto m_off_of = [mr_base, n_big](int c) {
        return c * mr_base + std::min(c, n_big);
    };

    const bool is_gated_act = (ctx.act_kind == ActivationKind::swiglu_oai_mul)
            || (ctx.act_kind == ActivationKind::silu_and_mul)
            || (ctx.act_kind == ActivationKind::gelu_and_mul);

    if (is_gated_act) {
        for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
            const int sub_n = std::min(subtile_cols, n_tile - sub_off);
            const int sub_col_base = col_start + sub_off;
            const int n_blocks = sub_n / ctx.pack_nr;

            const int8_t *Bpacked_blk_base = Bpacked_full
                    + static_cast<size_t>(sub_col_base / ctx.pack_nr)
                            * o_blk_stride_bytes;
            const char *bias_blk_base = (bias != nullptr)
                    ? static_cast<const char *>(bias)
                            + static_cast<size_t>(sub_col_base)
                                    * bias_elem_bytes
                    : nullptr;
            const char *wei_scale_blk_base
                    = static_cast<const char *>(wei_scale_full)
                    + (epilogue.wei_scale_per_tensor
                                    ? 0
                                    : static_cast<size_t>(sub_col_base)
                                            * scale_elem_bytes);

            const auto call_one = [&](int c, int b) {
                const int mr_now = mr_of(c);
                const int m_off = m_off_of(c);
                const int8_ukernel_fn_t kfn = ctx.kernel_table_int8[mr_now];

                const uint8_t *A_chunk
                        = A_bytes + static_cast<size_t>(m_off) * lda;
                const int src_row
                        = epilogue.group_k > 0 ? epilogue.src_row_groups : 1;
                const void *src_scale_chunk
                        = static_cast<const char *>(src_scale_full)
                        + (epilogue.src_quant_per_tensor
                                        ? 0
                                        : static_cast<size_t>(m_off)
                                                * static_cast<size_t>(src_row)
                                                * scale_elem_bytes);
                const int32_t *src_zp_chunk = (src_zp_full != nullptr)
                        ? src_zp_full
                                + (epilogue.src_quant_per_tensor ? 0 : m_off)
                        : nullptr;

                std::byte *Tight_row_base = Tight_bytes
                        + (static_cast<size_t>(m_off) * tight_ldc
                                  + (sub_col_base / 2))
                                * dst_elem_bytes_int8;

                const int8_t *Bpacked_blk = Bpacked_blk_base
                        + static_cast<size_t>(b) * o_blk_stride_bytes;
                const void *bias_blk = (bias_blk_base != nullptr)
                        ? static_cast<const void *>(bias_blk_base
                                  + static_cast<size_t>(b) * ctx.pack_nr
                                          * bias_elem_bytes)
                        : nullptr;
                const void *wei_scale_blk = static_cast<const void *>(
                        wei_scale_blk_base
                        + (epilogue.wei_scale_per_tensor
                                        ? 0
                                        : static_cast<size_t>(b) * ctx.pack_nr
                                                * scale_elem_bytes));
                // 16 cols per (g, u) pair × NV/2 pairs per kernel call —
                // mirror of the bf16 sibling's tight stride.
                std::byte *Tight_row = Tight_row_base
                        + static_cast<size_t>(b) * (ctx.pack_nr / 2)
                                * dst_elem_bytes_int8;

                kfn(A_chunk, lda, Bpacked_blk, src_scale_chunk, src_zp_chunk,
                        wei_scale_blk, ctx.scale_kind, epilogue, bias_blk,
                        ctx.bias_kind,
                        /*Cout=*/nullptr, /*ldc=*/0, Tight_row, tight_ldc, K);
            };

            for_each_call_block(oblk_outer, n_calls, n_blocks, call_one);
        }
        return;
    }

    // Act = none — wide output to the [M, N] dst arena.
    for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
        const int sub_n = std::min(subtile_cols, n_tile - sub_off);
        const int sub_col_base = col_start + sub_off;
        const int n_blocks = sub_n / ctx.pack_nr;

        const int8_t *Bpacked_blk_base = Bpacked_full
                + static_cast<size_t>(sub_col_base / ctx.pack_nr)
                        * o_blk_stride_bytes;
        const char *bias_blk_base = (bias != nullptr)
                ? static_cast<const char *>(bias)
                        + static_cast<size_t>(sub_col_base) * bias_elem_bytes
                : nullptr;
        const char *wei_scale_blk_base
                = static_cast<const char *>(wei_scale_full)
                + (epilogue.wei_scale_per_tensor
                                ? 0
                                : static_cast<size_t>(sub_col_base)
                                        * scale_elem_bytes);

        const auto call_one = [&](int c, int b) {
            const int mr_now = mr_of(c);
            const int m_off = m_off_of(c);
            const int8_ukernel_fn_t kfn = ctx.kernel_table_int8[mr_now];

            const uint8_t *A_chunk = A_bytes + static_cast<size_t>(m_off) * lda;
            const int src_row
                    = epilogue.group_k > 0 ? epilogue.src_row_groups : 1;
            const void *src_scale_chunk
                    = static_cast<const char *>(src_scale_full)
                    + (epilogue.src_quant_per_tensor
                                    ? 0
                                    : static_cast<size_t>(m_off)
                                            * static_cast<size_t>(src_row)
                                            * scale_elem_bytes);
            const int32_t *src_zp_chunk = (src_zp_full != nullptr)
                    ? src_zp_full + (epilogue.src_quant_per_tensor ? 0 : m_off)
                    : nullptr;

            std::byte *Wide_row_base = Tight_bytes
                    + (static_cast<size_t>(m_off) * tight_ldc + sub_col_base)
                            * dst_elem_bytes_int8;

            const int8_t *Bpacked_blk = Bpacked_blk_base
                    + static_cast<size_t>(b) * o_blk_stride_bytes;
            const void *bias_blk = (bias_blk_base != nullptr)
                    ? static_cast<const void *>(bias_blk_base
                              + static_cast<size_t>(b) * ctx.pack_nr
                                      * bias_elem_bytes)
                    : nullptr;
            const void *wei_scale_blk = static_cast<const void *>(
                    wei_scale_blk_base
                    + (epilogue.wei_scale_per_tensor ? 0
                                                     : static_cast<size_t>(b)
                                            * ctx.pack_nr * scale_elem_bytes));
            std::byte *Wide_row = Wide_row_base
                    + static_cast<size_t>(b) * ctx.pack_nr
                            * dst_elem_bytes_int8;

            Int8EpilogueConfig call_epilogue = epilogue;
            const int block_col = sub_col_base + b * ctx.pack_nr;
            if (call_epilogue.binary_mul != nullptr) {
                call_epilogue.binary_mul += static_cast<size_t>(m_off)
                                * call_epilogue.binary_mul_ld
                        + block_col;
            }
            if (call_epilogue.binary_add != nullptr) {
                call_epilogue.binary_add += static_cast<size_t>(m_off)
                                * call_epilogue.binary_add_ld
                        + block_col;
            }
            kfn(A_chunk, lda, Bpacked_blk, src_scale_chunk, src_zp_chunk,
                    wei_scale_blk, ctx.scale_kind, call_epilogue, bias_blk,
                    ctx.bias_kind, Wide_row, tight_ldc,
                    /*Cout_tight=*/nullptr, /*ldc_tight=*/0, K);
        };

        for_each_call_block(oblk_outer, n_calls, n_blocks, call_one);
    }
}

// Deep-K blocked DQ-INT8 path for the dense symmetric W2 shape.  The normal
// NR=32 kernel completes compensation/dequant after every MR call, forcing a
// >L2 B block to be re-streamed once per MR call.  This path accumulates
// L2-sized K chunks into s32 scratch, reuses each B chunk across all MR calls,
// then applies the full-K compensation and dequant exactly once. O-block stays
// outermost so each thread walks the full packed-weight block monotonically;
// a K-major traversal would interleave distant slices from several 1 MiB
// blocks and defeat the cold-weight streaming pattern.
inline void dispatch_tile_int8_kblocked(const CallContext &ctx, int expert_idx,
        int M, int K, int n_tile, int col_start, const void *src, int lda,
        void *dst, int ldc, const void *src_scale, const void *wei_scale) {
    constexpr int pack_nr = 32;
    assert(ctx.pack_nr == pack_nr && (K % kVNNIInt8Quad) == 0);
    assert(src != nullptr && dst != nullptr && src_scale != nullptr
            && wei_scale != nullptr);
    const auto *A = static_cast<const uint8_t *>(src);
    const int8_t *Bpacked_full = ctx.packed_ptrs_int8[expert_idx];
    const int K_quad_total = K / kVNNIInt8Quad;
    const size_t kq_stride_bytes = pack_nr * kVNNIInt8Quad;
    const size_t weight_bytes_per_oblock
            = static_cast<size_t>(K_quad_total) * kq_stride_bytes;
    const size_t o_blk_stride_bytes
            = weight_bytes_per_oblock + pack_nr * sizeof(int32_t);
    const int n_blocks = n_tile / pack_nr;

    thread_local std::vector<int32_t> c_accum;
    c_accum.resize(static_cast<size_t>(M) * pack_nr);

    const int n_calls = (M + ctx.max_mr - 1) / ctx.max_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;
    const int8_accum_ukernel_fn_t fn_small
            = select_int8_accum_ukernel(mr_base, ctx.src_u8);
    const int8_accum_ukernel_fn_t fn_big = (n_big > 0)
            ? select_int8_accum_ukernel(mr_base + 1, ctx.src_u8)
            : fn_small;
    assert(fn_small != nullptr && fn_big != nullptr);

    const int64_t l2_budget = ck_l2_budget_bytes();
    int kc = static_cast<int>(
            l2_budget / static_cast<int64_t>(pack_nr + ctx.max_mr));
    kc &= ~(kVNNIInt8Quad - 1);
    if (kc < kVNNIInt8Quad) kc = kVNNIInt8Quad;

    const size_t scale_elem_bytes = (ctx.scale_kind == ScaleKind::kBf16)
            ? sizeof(bfloat16_t)
            : sizeof(float);
    for (int b = 0; b < n_blocks; ++b) {
        const int global_block = col_start / pack_nr + b;
        const int8_t *Bblock = Bpacked_full
                + static_cast<size_t>(global_block) * o_blk_stride_bytes;

        for (int kc_off = 0; kc_off < K; kc_off += kc) {
            const int kc_now = std::min(kc, K - kc_off);
            const size_t kq_off = static_cast<size_t>(kc_off / kVNNIInt8Quad);
            const int8_t *Bpk = Bblock + kq_off * kq_stride_bytes;
            const bool add = (kc_off != 0);
            int m_off = 0;
            for (int c = 0; c < n_calls; ++c) {
                const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
                const int8_accum_ukernel_fn_t kfn
                        = (c < n_big) ? fn_big : fn_small;
                const uint8_t *A_chunk = A + static_cast<size_t>(m_off) * lda
                        + static_cast<size_t>(kc_off);
                int32_t *C
                        = c_accum.data() + static_cast<size_t>(m_off) * pack_nr;
                kfn(A_chunk, lda, Bpk, kc_now, C, pack_nr, add);
                m_off += mr_now;
            }
        }

        const int32_t *comp = reinterpret_cast<const int32_t *>(
                Bblock + weight_bytes_per_oblock);
        const void *wei_scale_block
                = static_cast<const void *>(static_cast<const char *>(wei_scale)
                        + static_cast<size_t>(col_start + b * pack_nr)
                                * scale_elem_bytes);
        auto *dst_block = static_cast<bfloat16_t *>(dst) + col_start
                + static_cast<size_t>(b) * pack_nr;
        finish_int8_accum_nv2_sym_bf16(c_accum.data(), pack_nr, M, comp,
                src_scale, wei_scale_block, ctx.scale_kind, dst_block, ldc);
    }
}

// Per worker thread: a K chunk of one unpacked NR=32 W4A8 block, the f32 sums
// of a tile's rows and the `[12][32]` int32 group sums of
// `dispatch_tile_s4_unpacked`.
struct S4UnpackScratch {
    using buf_t = std::unique_ptr<void, decltype(&zendnnl_aligned_free)>;
    buf_t b {nullptr, &zendnnl_aligned_free};
    size_t b_cap = 0;
    buf_t f {nullptr, &zendnnl_aligned_free};
    size_t f_cap = 0;
    alignas(64) int32_t c[12 * 32];

    static void *grow(buf_t &p, size_t &cap, size_t bytes) {
        if (bytes > cap) {
            p.reset(zendnnl_aligned_alloc(64, bytes));
            cap = p ? bytes : 0;
        }
        return p.get();
    }
};

// `dispatch_tile_w4` for the NR=32 tiles of a `src_u8` call: each ~512-deep K
// chunk of a 32-column block is unpacked to s8 once, into L1, and every MR
// call of the tile sums its groups on the XOR-free DQ-INT8 accumulator over
// up to 12 rows, rather than every MR call unpacking in registers.  With s8
// source bytes the accumulator's per-broadcast XOR costs more than the
// unpack it saves, so those calls keep the in-register kernels.  Same bits
// as the in-register kernels.  Returns false, having written nothing, when it
// does not apply.
bool dispatch_tile_s4_unpacked(const CallContext &ctx, int M, int K, int n_tile,
        int col_start, const uint8_t *A_bytes, int lda, const void *bias,
        std::byte *Tight_bytes, int tight_ldc, const void *src_scale_full,
        const void *wei_scale_full, const int8_t *Bpacked_full,
        size_t o_blk_stride_bytes, int subtile_cols, bool is_gated_act) {
    constexpr int kNr = 32;
    constexpr int kMaxRows = 12;
    constexpr int kChunkK = 512;
    if (ctx.pack_nr != kNr || !ctx.src_u8) return false;
    const S4SrcScaleGranularity gran = ctx.src_scale_groups > 1
            ? S4SrcScaleGranularity::kPerGroup
            : S4SrcScaleGranularity::kPerToken;
    const w4a8_unpacked_store_fn_t sfn
            = select_w4a8_unpacked_store(ctx.act_kind,
                    dst_dt_for_variant(ctx.variant) == OutputType::kF32
                            ? S4OutputKind::kF32
                            : S4OutputKind::kBf16,
                    gran);
    if (sfn == nullptr) return false;
    const w4a8_unpacked_groups_fn_t gfn = select_w4a8_unpacked_groups(gran);
    const int gs = ctx.group_size;
    const int G = K / gs;
    const int chunk_groups = std::max(1, kChunkK / gs);
    static thread_local S4UnpackScratch scratch;
    auto *unpacked = static_cast<int8_t *>(S4UnpackScratch::grow(scratch.b,
            scratch.b_cap, static_cast<size_t>(chunk_groups) * gs * kNr));
    auto *facc = static_cast<float *>(S4UnpackScratch::grow(scratch.f,
            scratch.f_cap, static_cast<size_t>(M) * kNr * sizeof(float)));
    if (unpacked == nullptr || facc == nullptr) return false;

    const size_t dst_elem = dst_elem_bytes(ctx.variant);
    const size_t bias_elem = (ctx.bias_kind == BiasKind::fp32)
            ? sizeof(float)
            : sizeof(bfloat16_t);
    const size_t scale_elem = (ctx.scale_kind == ScaleKind::kBf16)
            ? sizeof(bfloat16_t)
            : sizeof(float);
    const size_t group_packed_bytes
            = static_cast<size_t>(gs / kW4Octet) * kNr * kW4BytesPerOctetCol;
    const int n_calls = (M + kMaxRows - 1) / kMaxRows;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;
    const int out_cols = is_gated_act ? kNr / 2 : kNr;
    const auto src_scale_at = [&](int m_off) {
        return static_cast<const char *>(src_scale_full)
                + static_cast<size_t>(m_off) * ctx.src_scale_groups
                * scale_elem;
    };

    for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
        const int sub_n = std::min(subtile_cols, n_tile - sub_off);
        for (int b = 0; b < sub_n / kNr; ++b) {
            const int col = col_start + sub_off + b * kNr;
            const int8_t *blk = Bpacked_full
                    + static_cast<size_t>(col / kNr) * o_blk_stride_bytes;
            const auto *comp = reinterpret_cast<const int32_t *>(
                    blk + static_cast<size_t>(G) * group_packed_bytes);
            const void *wei_scale_blk
                    = static_cast<const char *>(wei_scale_full)
                    + col * scale_elem;
            std::memset(facc, 0, static_cast<size_t>(M) * kNr * sizeof(float));
            for (int g0 = 0; g0 < G; g0 += chunk_groups) {
                const int ng = std::min(chunk_groups, G - g0);
                unpack_s4_block_nr32(
                        blk + g0 * group_packed_bytes, ng * gs, unpacked);
                int m_off = 0;
                for (int c = 0; c < n_calls; ++c) {
                    const int mr = (c < n_big) ? mr_base + 1 : mr_base;
                    gfn(mr, A_bytes + static_cast<size_t>(m_off) * lda, lda,
                            unpacked, comp,
                            select_int8_accum_ukernel(mr, ctx.src_u8),
                            src_scale_at(m_off), wei_scale_blk, ctx.scale_kind,
                            g0, ng, G, gs, ctx.wei_scale_grp_stride,
                            facc + static_cast<size_t>(m_off) * kNr, scratch.c);
                    m_off += mr;
                }
            }
            const void *bias_blk = (bias != nullptr)
                    ? static_cast<const char *>(bias) + col * bias_elem
                    : nullptr;
            int m_off = 0;
            for (int c = 0; c < n_calls; ++c) {
                const int mr = (c < n_big) ? mr_base + 1 : mr_base;
                std::byte *row = Tight_bytes
                        + (static_cast<size_t>(m_off) * tight_ldc
                                  + static_cast<size_t>(col / kNr) * out_cols)
                                * dst_elem;
                sfn(mr, facc + static_cast<size_t>(m_off) * kNr,
                        src_scale_at(m_off), ctx.scale_kind, bias_blk,
                        ctx.bias_kind, is_gated_act ? nullptr : row,
                        is_gated_act ? 0 : tight_ldc,
                        is_gated_act ? row : nullptr,
                        is_gated_act ? tight_ldc : 0);
                m_off += mr;
            }
        }
    }
    return true;
}

// Sub-tile + per-MR loop for the four-bit variants.  Shaped like
// `dispatch_tile_int8`, minus `src_zp` (W4A8 is symmetric), with a bf16 or
// f32 dst.  The weight scale is sliced by COLUMN only; the kernel adds
// `g * wei_scale_grp_stride` itself to walk the `{G, N}` rows, so that
// stride is the expert's full N, not the tile width.  The source scale has
// `src_scale_groups` values per row.  The MXFP4 variants walk the same tiles
// with their own kernel family, whose scales are in the pack.
inline void dispatch_tile_w4(const CallContext &ctx, int expert_idx, int M,
        int K, int n_tile, int col_start, const void *src, int lda,
        const void *bias, void *tight_dst, int tight_ldc,
        const void *src_scale_full, const void *wei_scale_full,
        const int32_t *src_group_sums_full, const void *wei_zp_full) {

    const W4Format fmt = w4_format_for_variant(ctx.variant);
    // A format whose pack holds its scales never reads the caller's.
    if (w4_scales_in_pack(fmt)) wei_scale_full = nullptr;
    // A format that seeds each group from its row's source sum takes them
    // from the N-tile hoist for the whole call; without them, each row chunk
    // computes its own, once per sub-tile.
    const bool needs_sums = w4_needs_src_group_sums(fmt);
    const bool mxfp4_kernel = is_mxfp4_variant(ctx.variant);
    const OutputType dst_dt = dst_dt_for_variant(ctx.variant);
    const int G = (ctx.group_size > 0) ? K / ctx.group_size : 0;
    thread_local std::vector<int32_t> chunk_sums;
    if (needs_sums && src_group_sums_full == nullptr)
        chunk_sums.resize(static_cast<size_t>(ctx.max_mr) * G);
    assert(src_scale_full != nullptr
            && "dispatch_tile_w4: N-tile hoist must populate the source scale");
    assert((wei_scale_full != nullptr || w4_scales_in_pack(fmt))
            && "dispatch_tile_w4: caller must thread the {G, N} wei scale");
    assert(ctx.group_size > 0
            && "dispatch_tile_w4: prepare_for_call must record group_size");

    const int8_t *Bpacked_full = ctx.packed_ptrs_w4[expert_idx];
    const auto *A_bytes = static_cast<const uint8_t *>(src);
    std::byte *Tight_bytes = static_cast<std::byte *>(tight_dst);
    const size_t dst_elem_bytes_w4 = dst_elem_bytes(ctx.variant);

    // `prepare_for_call` guarantees `K % 8 == 0` and `K % group_size == 0`.
    const size_t o_blk_stride_bytes
            = w4_oblock_bytes(K, ctx.pack_nr, ctx.group_size, fmt);

    const int subtile_cols = (ctx.subtile_cols_per_expert[expert_idx] > 0)
            ? ctx.subtile_cols_per_expert[expert_idx]
            : ctx.subtile_cols;

    const int n_calls = (M + ctx.max_mr - 1) / ctx.max_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;

    const size_t bias_elem_bytes = (ctx.bias_kind == BiasKind::fp32)
            ? sizeof(float)
            : sizeof(bfloat16_t); // bf16 or f16 (both 2B)
    const size_t scale_elem_bytes = (ctx.scale_kind == ScaleKind::kBf16)
            ? sizeof(bfloat16_t)
            : sizeof(float);

    const bool is_gated_act = (ctx.act_kind == ActivationKind::swiglu_oai_mul)
            || (ctx.act_kind == ActivationKind::silu_and_mul)
            || (ctx.act_kind == ActivationKind::gelu_and_mul);

    // The unpacked s4 path has no per-group offset. Q4_K stays on the
    // kernel below, which subtracts wei_zp.
    if (wei_zp_full == nullptr
            && dispatch_tile_s4_unpacked(ctx, M, K, n_tile, col_start, A_bytes,
                    lda, bias, Tight_bytes, tight_ldc, src_scale_full,
                    wei_scale_full, Bpacked_full, o_blk_stride_bytes,
                    subtile_cols, is_gated_act))
        return;

    for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
        const int sub_n = std::min(subtile_cols, n_tile - sub_off);
        const int sub_col_base = col_start + sub_off;
        const int n_blocks = sub_n / ctx.pack_nr;

        const int8_t *Bpacked_blk_base = Bpacked_full
                + static_cast<size_t>(sub_col_base / ctx.pack_nr)
                        * o_blk_stride_bytes;
        const char *bias_blk_base = (bias != nullptr)
                ? static_cast<const char *>(bias)
                        + static_cast<size_t>(sub_col_base) * bias_elem_bytes
                : nullptr;
        const char *wei_scale_blk_base = (wei_scale_full != nullptr)
                ? static_cast<const char *>(wei_scale_full)
                        + static_cast<size_t>(sub_col_base) * scale_elem_bytes
                : nullptr;
        const char *wei_zp_blk_base = wei_zp_full != nullptr
                ? static_cast<const char *>(wei_zp_full)
                        + static_cast<size_t>(sub_col_base) * scale_elem_bytes
                : nullptr;

        int m_off = 0;
        // One {MR, G} activation sum per row chunk. Every column block of
        // this tile multiplies the same source, so the Q4_K offset reuses it.
        static thread_local std::vector<float> act_group_sums;
        for (int c = 0; c < n_calls; ++c) {
            const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
            const w4a8_ukernel_fn_t *table = wei_zp_full != nullptr
                    ? ctx.kernel_table_s4_offset
                    : ctx.kernel_table_s4;
            const w4a8_ukernel_fn_t kfn = table[mr_now];
            const mxfp4_ukernel_fn_t mxfp4_kfn = ctx.kernel_table_mxfp4[mr_now];

            const uint8_t *A_chunk = A_bytes + static_cast<size_t>(m_off) * lda;
            const void *src_scale_chunk
                    = static_cast<const char *>(src_scale_full)
                    + static_cast<size_t>(m_off) * ctx.src_scale_groups
                            * scale_elem_bytes;
            const int32_t *sums_chunk = nullptr;
            if (needs_sums && src_group_sums_full != nullptr) {
                sums_chunk
                        = src_group_sums_full + static_cast<size_t>(m_off) * G;
            } else if (needs_sums) {
                s8_group_sums(A_chunk, lda, mr_now, K, ctx.group_size,
                        chunk_sums.data());
                sums_chunk = chunk_sums.data();
            }

            // Gated acts write the halved [M, N/2] tight arena, so the
            // column origin halves; act=none writes the wide [M, N].
            std::byte *Row_base = Tight_bytes
                    + (static_cast<size_t>(m_off) * tight_ldc
                              + (is_gated_act ? (sub_col_base / 2)
                                              : sub_col_base))
                            * dst_elem_bytes_w4;

            const float *act_group_sum = nullptr;
            if (wei_zp_full != nullptr) {
                const size_t nsum
                        = static_cast<size_t>(mr_now) * static_cast<size_t>(G);
                if (act_group_sums.size() < nsum) act_group_sums.resize(nsum);
                for (int m = 0; m < mr_now; ++m) {
                    const uint8_t *row = A_chunk + static_cast<size_t>(m) * lda;
                    for (int g = 0; g < G; ++g) {
                        const uint8_t *p
                                = row + static_cast<size_t>(g) * ctx.group_size;
                        int sum = 0;
                        for (int kk = 0; kk < ctx.group_size; ++kk) {
                            sum += static_cast<int8_t>(p[kk]);
                        }
                        act_group_sums[static_cast<size_t>(m) * G + g]
                                = static_cast<float>(sum);
                    }
                }
                act_group_sum = act_group_sums.data();
            }

            for (int b = 0; b < n_blocks; ++b) {
                const int8_t *Bpacked_blk = Bpacked_blk_base
                        + static_cast<size_t>(b) * o_blk_stride_bytes;
                const void *bias_blk = (bias_blk_base != nullptr)
                        ? static_cast<const void *>(bias_blk_base
                                  + static_cast<size_t>(b) * ctx.pack_nr
                                          * bias_elem_bytes)
                        : nullptr;
                const void *wei_scale_blk = (wei_scale_blk_base != nullptr)
                        ? static_cast<const void *>(wei_scale_blk_base
                                  + static_cast<size_t>(b) * ctx.pack_nr
                                          * scale_elem_bytes)
                        : nullptr;
                const void *wei_zp_blk = wei_zp_blk_base != nullptr
                        ? static_cast<const void *>(wei_zp_blk_base
                                  + static_cast<size_t>(b) * ctx.pack_nr
                                          * scale_elem_bytes)
                        : nullptr;
                std::byte *Row = Row_base
                        + static_cast<size_t>(b)
                                * (is_gated_act ? (ctx.pack_nr / 2)
                                                : ctx.pack_nr)
                                * dst_elem_bytes_w4;

                if (mxfp4_kernel) {
                    if (is_gated_act) {
                        mxfp4_kfn(A_chunk, lda, Bpacked_blk, src_scale_chunk,
                                ctx.scale_kind, bias_blk, ctx.bias_kind,
                                /*Cout=*/nullptr, dst_dt,
                                /*ldc=*/0, Row, tight_ldc, K, ctx.group_size,
                                sums_chunk);
                    } else {
                        mxfp4_kfn(A_chunk, lda, Bpacked_blk, src_scale_chunk,
                                ctx.scale_kind, bias_blk, ctx.bias_kind, Row,
                                dst_dt, tight_ldc,
                                /*Cout_tight=*/nullptr, /*ldc_tight=*/0, K,
                                ctx.group_size, sums_chunk);
                    }
                } else if (is_gated_act) {
                    kfn(A_chunk, lda, Bpacked_blk, src_scale_chunk,
                            wei_scale_blk, ctx.scale_kind, bias_blk,
                            ctx.bias_kind, /*Cout=*/nullptr, /*ldc=*/0, Row,
                            tight_ldc, K, ctx.group_size,
                            ctx.wei_scale_grp_stride, wei_zp_blk,
                            act_group_sum);
                } else {
                    kfn(A_chunk, lda, Bpacked_blk, src_scale_chunk,
                            wei_scale_blk, ctx.scale_kind, bias_blk,
                            ctx.bias_kind, Row, tight_ldc,
                            /*Cout_tight=*/nullptr, /*ldc_tight=*/0, K,
                            ctx.group_size, ctx.wei_scale_grp_stride,
                            wei_zp_blk, act_group_sum);
                }
            }
            m_off += mr_now;
        }
    }
}
} // namespace

// FP16 dispatch tile — owns the per-tile sub-tile + per-MR loop for the
// `kF16_F16_F16` / `kF16_F16_F32` variants.  Structurally identical to
// the BF16 `dispatch_tile` body (no scales / zp / compensation), with
// two differences isolated here:
//   * `o_blk_stride` is the plain FP16 pack stride `K * pack_nr`
//     (the native-FMA pack carries no K-pair interleave), vs the bf16
//     `K_pair * pack_nr * 2`.
//   * the act=none dst element width tracks the resolved variant
//     (F16 = 2B for kF16_F16_F16, FP32 = 4B for kF16_F16_F32); gated
//     epilogues are F16-dst-only (2B) like the bf16 sibling.
namespace {
ZENDNNL_TARGET("avx512f")
inline void narrow_f32_rows_to_f16(
        float16_t *dst, int ldd, const float *src, int lds, int rows, int cols);

inline bool dispatch_tile_f16_kblocked(const CallContext &ctx, int expert_idx,
        int M, int K, int n_tile, int col_start, const void *src, int lda,
        void *dst, int ldc) {
    if (ctx.variant != KernelVariant::kF16_F16_F16
            || ctx.act_kind != ActivationKind::none || !ctx.single_expert
            || ctx.f16_prompt || M != 32)
        return false;

    const int64_t l2_budget = grp_machine::machine().l2_budget_bytes;
    const int64_t bytes_per_k
            = static_cast<int64_t>(ctx.pack_nr + M) * sizeof(float16_t);
    if (static_cast<int64_t>(K) * bytes_per_k <= l2_budget) return false;

    int k_block = static_cast<int>(l2_budget / bytes_per_k);
    k_block = (k_block / 64) * 64;
    if (k_block < 256 || k_block >= K) return false;
    if (test_api::s_capture_kblock.load(std::memory_order_relaxed))
        test_api::s_last_kblock_used.store(true, std::memory_order_relaxed);

    const auto *A = static_cast<const float16_t *>(src);
    const float16_t *Bpacked_full = ctx.packed_ptrs_f16[expert_idx];
    auto *C = static_cast<float16_t *>(dst);
    const size_t o_blk_stride = static_cast<size_t>(K) * ctx.pack_nr;
    const int n_blocks = n_tile / ctx.pack_nr;
    const int n_calls = (M + ctx.max_mr - 1) / ctx.max_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;
    thread_local std::vector<float> c32;
    c32.resize(static_cast<size_t>(M) * ctx.pack_nr);

    for (int b = 0; b < n_blocks; ++b) {
        const int block_col = col_start + b * ctx.pack_nr;
        const float16_t *Bpacked_block = Bpacked_full
                + static_cast<size_t>(block_col / ctx.pack_nr) * o_blk_stride;
        for (int k_begin = 0; k_begin < K; k_begin += k_block) {
            const int k_len = std::min(k_block, K - k_begin);
            const bool add = k_begin != 0;
            int m_off = 0;
            for (int c = 0; c < n_calls; ++c) {
                const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
                const f16_kblock_ukernel_fn_t kfn
                        = ctx.kernel_table_f16_kblock[mr_now];
                kfn(A + static_cast<size_t>(m_off) * lda + k_begin, lda,
                        Bpacked_block
                                + static_cast<size_t>(k_begin) * ctx.pack_nr,
                        c32.data() + static_cast<size_t>(m_off) * ctx.pack_nr,
                        ctx.pack_nr, k_len, add);
                m_off += mr_now;
            }
        }
        narrow_f32_rows_to_f16(
                C + block_col, ldc, c32.data(), ctx.pack_nr, M, ctx.pack_nr);
    }
    return true;
}

// Single-expert FP16 prompt tile (`ctx.f16_prompt`): groups of
// `kF16PromptNb` o-blocks outer, MR calls inner, so a group's B strip
// is read once and reused by every MR call.  No L2 sub-tiles: they would
// cut a tall tile to one o-block per group.  An act=none tile taller than
// one MR call with `ctx.prompt_kc` set walks K in chunks of that depth (at
// least `kPromptKcMin`, rounded up to whole `kF16AccumBlockK` blocks)
// outermost, each chunk continuing from the FP32 sums of the previous ones.
// dst[r][c] = F16(src[r][c]) over `rows` x `cols` (cols a multiple of 16).
ZENDNNL_TARGET("avx512f")
inline void narrow_f32_rows_to_f16(float16_t *dst, int ldd, const float *src,
        int lds, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float16_t *d = dst + static_cast<size_t>(r) * ldd;
        const float *s = src + static_cast<size_t>(r) * lds;
        for (int c = 0; c < cols; c += 16)
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(d + c),
                    _mm512_cvtps_ph(_mm512_loadu_ps(s + c),
                            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
}

inline void dispatch_tile_f16_prompt(const CallContext &ctx, int expert_idx,
        int M, int K, int n_tile, int col_start, const void *src, int lda,
        const void *bias, void *tight_dst, int tight_ldc) {
    const float16_t *Bpacked_full = ctx.packed_ptrs_f16[expert_idx];
    const auto *A = static_cast<const float16_t *>(src);
    const int pack_nr = ctx.pack_nr;
    const size_t o_blk_stride = static_cast<size_t>(K) * pack_nr;
    const size_t dst_elem_bytes = (ctx.variant == KernelVariant::kF16_F16_F32)
            ? sizeof(float)
            : sizeof(float16_t);
    const size_t bias_elem_bytes = (ctx.bias_kind == BiasKind::fp32)
            ? sizeof(float)
            : sizeof(float16_t);
    const bool gated = ctx.act_kind != ActivationKind::none;
    const int kc = (!gated && ctx.prompt_kc > 0 && ctx.prompt_kc < K
                           && M > ctx.max_mr)
            ? (std::max(ctx.prompt_kc, kPromptKcMin) + kF16AccumBlockK - 1)
                    / kF16AccumBlockK * kF16AccumBlockK
            : K;
    if (kc < K && test_api::s_capture_kblock.load(std::memory_order_relaxed))
        test_api::s_last_kblock_used.store(true, std::memory_order_relaxed);
    // K chunks of an F16 destination accumulate in an FP32 scratch and are
    // narrowed once, so the result matches the full K bit for bit.
    const bool f32_scratch
            = kc < K && ctx.variant == KernelVariant::kF16_F16_F16;
    thread_local std::vector<float> c32;
    if (f32_scratch) c32.resize(static_cast<size_t>(M) * n_tile);
    const int n_calls = (M + ctx.max_mr - 1) / ctx.max_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;
    const int n_blocks = n_tile / pack_nr;
    const float16_t *B_tile = Bpacked_full
            + static_cast<size_t>(col_start / pack_nr) * o_blk_stride;

    for (int k0 = 0; k0 < K; k0 += kc) {
        const int kk = std::min(kc, K - k0);
        for (int b = 0; b < n_blocks; b += kF16PromptNb) {
            const int nb = std::min(kF16PromptNb, n_blocks - b);
            const int col = col_start + b * pack_nr;
            const float16_t *B_blk = B_tile
                    + static_cast<size_t>(b) * o_blk_stride
                    + static_cast<size_t>(k0) * pack_nr;
            const void *bias_blk = (bias != nullptr && k0 == 0)
                    ? static_cast<const void *>(static_cast<const char *>(bias)
                              + static_cast<size_t>(col) * bias_elem_bytes)
                    : nullptr;
            int m_off = 0;
            for (int c = 0; c < n_calls; ++c) {
                const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
                const f16_prompt_ukernel_fn_t kfn
                        = ctx.kernel_table_f16_prompt[nb - 1][mr_now];
                const float16_t *A_chunk
                        = A + static_cast<size_t>(m_off) * lda + k0;
                if (gated) {
                    float16_t *dst = static_cast<float16_t *>(tight_dst)
                            + static_cast<size_t>(m_off) * tight_ldc + col / 2;
                    kfn(A_chunk, lda, B_blk, o_blk_stride, bias_blk,
                            ctx.bias_kind, nullptr, 0, dst, tight_ldc, kk,
                            false);
                } else if (f32_scratch) {
                    float *dst = c32.data()
                            + static_cast<size_t>(m_off) * n_tile
                            + (col - col_start);
                    ctx.kernel_table_f16_prompt_f32[nb - 1][mr_now](A_chunk,
                            lda, B_blk, o_blk_stride, bias_blk, ctx.bias_kind,
                            dst, n_tile, nullptr, 0, kk, k0 > 0);
                } else {
                    std::byte *dst = static_cast<std::byte *>(tight_dst)
                            + (static_cast<size_t>(m_off) * tight_ldc + col)
                                    * dst_elem_bytes;
                    kfn(A_chunk, lda, B_blk, o_blk_stride, bias_blk,
                            ctx.bias_kind, dst, tight_ldc, nullptr, 0, kk,
                            k0 > 0);
                }
                m_off += mr_now;
            }
        }
    }
    if (f32_scratch)
        narrow_f32_rows_to_f16(static_cast<float16_t *>(tight_dst) + col_start,
                tight_ldc, c32.data(), n_tile, M, n_tile);
}

inline void dispatch_tile_f16(const CallContext &ctx, int expert_idx, int M,
        int K, int n_tile, int col_start, const void *src, int lda,
        const void *bias, void *tight_dst, int tight_ldc) {
    if (ctx.f16_prompt) {
        dispatch_tile_f16_prompt(ctx, expert_idx, M, K, n_tile, col_start, src,
                lda, bias, tight_dst, tight_ldc);
        return;
    }

    const float16_t *Bpacked_full = ctx.packed_ptrs_f16[expert_idx];
    const auto *A = static_cast<const float16_t *>(src);
    std::byte *Tight_bytes = static_cast<std::byte *>(tight_dst);
    const size_t dst_elem_bytes = (ctx.variant == KernelVariant::kF16_F16_F32)
            ? sizeof(float)
            : sizeof(float16_t);

    // Plain FP16 pack: one o-block is `K` rows of `pack_nr` FP16 cols.
    // units: float16_t elements (not bytes) — pointer arithmetic below is
    // on `float16_t *`.  No K-pair interleave and no odd-K pad (the native
    // FP16 FMA consumes one K-lane per step, unlike the bf16 VDPBF16PS
    // K-pair pack whose stride is `K_pair * pack_nr * 2`), so the stride
    // is exactly `K * pack_nr`.
    const size_t o_blk_stride = static_cast<size_t>(K) * ctx.pack_nr;

    const int subtile_cols = (ctx.subtile_cols_per_expert[expert_idx] > 0)
            ? ctx.subtile_cols_per_expert[expert_idx]
            : ctx.subtile_cols;

    // Balanced MR partition — identical scheme to the bf16 sibling.
    const int n_calls = (M + ctx.max_mr - 1) / ctx.max_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;
    const bool oblk_outer = se_oblock_outer(ctx.single_expert, n_calls);
    const auto mr_of = [mr_base, n_big](int c) {
        return (c < n_big) ? mr_base + 1 : mr_base;
    };
    const auto m_off_of = [mr_base, n_big](int c) {
        return c * mr_base + std::min(c, n_big);
    };

    // Per-col bias byte stride.  The FP16 microkernel accepts a bf16,
    // fp32, or f16 bias.  Only the fp32 case is 4 bytes; bf16 and
    // f16 are both 2-byte element widths, so the non-fp32 arm uses a
    // 2-byte stride (`sizeof(float16_t) == sizeof(bfloat16_t)`).  This is
    // the bias element width, NOT the f16 dst width — they merely coincide
    // at 2 bytes.
    const size_t bias_elem_bytes = (ctx.bias_kind == BiasKind::fp32)
            ? sizeof(float)
            : sizeof(float16_t);

    const bool is_gated_act = (ctx.act_kind == ActivationKind::swiglu_oai_mul)
            || (ctx.act_kind == ActivationKind::silu_and_mul)
            || (ctx.act_kind == ActivationKind::gelu_and_mul);

    if (is_gated_act) {
        for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
            const int sub_n = std::min(subtile_cols, n_tile - sub_off);
            const int sub_col_base = col_start + sub_off;
            const int n_blocks = sub_n / ctx.pack_nr;

            const float16_t *Bpacked_blk_base = Bpacked_full
                    + static_cast<size_t>(sub_col_base / ctx.pack_nr)
                            * o_blk_stride;
            const char *bias_blk_base = (bias != nullptr)
                    ? static_cast<const char *>(bias)
                            + static_cast<size_t>(sub_col_base)
                                    * bias_elem_bytes
                    : nullptr;

            // At M=32 the FP16 kernel spans four to six MR calls. Walking
            // o-blocks first keeps each deep-K weight slab hot across those
            // calls, matching the dense BF16/DQ-INT8 execution order.
            for_each_call_block(
                    oblk_outer, n_calls, n_blocks, [&](int c, int b) {
                const int mr_now = mr_of(c);
                const int m_off = m_off_of(c);
                const f16_ukernel_fn_t kfn = ctx.kernel_table_f16[mr_now];

                const float16_t *A_chunk = A + static_cast<size_t>(m_off) * lda;
                // Gated-act epilogues are F16-dst-only, so the half-width tight
                // arena is a plain float16_t matrix (`tight_ldc` in elements).
                // Index it as float16_t* so the element→byte scaling is implicit
                // — avoids a sizeof multiply against a byte pointer.
                // /2: the output is half-width (one post-activation col per
                // gate/up pair), so the wide input column `sub_col_base` maps to
                // output column `sub_col_base / 2`.  Offset is in float16_t
                // elements, not bytes.
                float16_t *Tight_row_base = static_cast<float16_t *>(tight_dst)
                        + static_cast<size_t>(m_off) * tight_ldc
                        + (sub_col_base / 2);

                const float16_t *Bpacked_blk = Bpacked_blk_base
                        + static_cast<size_t>(b) * o_blk_stride;
                const void *bias_blk = (bias_blk_base != nullptr)
                        ? static_cast<const void *>(bias_blk_base
                                  + static_cast<size_t>(b) * ctx.pack_nr
                                          * bias_elem_bytes)
                        : nullptr;
                float16_t *Tight_row = Tight_row_base
                        + static_cast<size_t>(b) * (ctx.pack_nr / 2);

                kfn(A_chunk, lda, Bpacked_blk, bias_blk, ctx.bias_kind,
                        /*Cout=*/nullptr, /*ldc=*/0,
                        static_cast<void *>(Tight_row), tight_ldc, K);
            });
        }
        return;
    }

    // Act = none — wide output to the [M, N] dst arena.
    for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
        const int sub_n = std::min(subtile_cols, n_tile - sub_off);
        const int sub_col_base = col_start + sub_off;
        const int n_blocks = sub_n / ctx.pack_nr;

        const float16_t *Bpacked_blk_base = Bpacked_full
                + static_cast<size_t>(sub_col_base / ctx.pack_nr)
                        * o_blk_stride;
        const char *bias_blk_base = (bias != nullptr)
                ? static_cast<const char *>(bias)
                        + static_cast<size_t>(sub_col_base) * bias_elem_bytes
                : nullptr;

        for_each_call_block(oblk_outer, n_calls, n_blocks, [&](int c, int b) {
            const int mr_now = mr_of(c);
            const int m_off = m_off_of(c);
            const f16_ukernel_fn_t kfn = ctx.kernel_table_f16[mr_now];

            const float16_t *A_chunk = A + static_cast<size_t>(m_off) * lda;
            std::byte *Wide_row_base = Tight_bytes
                    + (static_cast<size_t>(m_off) * tight_ldc + sub_col_base)
                            * dst_elem_bytes;

            const float16_t *Bpacked_blk
                    = Bpacked_blk_base + static_cast<size_t>(b) * o_blk_stride;
            const void *bias_blk = (bias_blk_base != nullptr)
                    ? static_cast<const void *>(bias_blk_base
                              + static_cast<size_t>(b) * ctx.pack_nr
                                      * bias_elem_bytes)
                    : nullptr;
            std::byte *Wide_row = Wide_row_base
                    + static_cast<size_t>(b) * ctx.pack_nr * dst_elem_bytes;

            kfn(A_chunk, lda, Bpacked_blk, bias_blk, ctx.bias_kind,
                    static_cast<void *>(Wide_row), tight_ldc,
                    /*Cout_tight=*/nullptr, /*ldc_tight=*/0, K);
        });
    }
}
} // namespace

// ─────────────────────────────────────────────────────────────────────
// Deep-K single-thread K-blocking, enabled by `ZENDNNL_GRP_MATMUL_KBLOCK` or
// by `ctx.kblock_auto`.  BF16 family, act=none, bias-free only.
//
// Problem it fixes: the plain `dispatch_tile` bf16 path streams the FULL K
// reduction per o-block and loops M in `ceil(M / max_mr)` blocks, so once an
// o-block's B strip (K * pack_nr * 2 bytes) exceeds L2 it is re-streamed from
// DRAM once per M-block.
//
// Fix: split K into L2-resident chunks and iterate
//   for k_chunk: for o_block: for m_block
// so each (k_chunk, o_block) B slab is loaded once and reused across every
// M-block while resident in L2 — summed over k_chunks, each o-block's B is
// read exactly once.  Partial sums accumulate in a per-thread f32 C scratch
// that the epilogue narrows to the caller's dst dtype.
//
// Scope guards (all enforced by the caller's routing predicate):
//   * BF16 family, ActivationKind::none (no fused activation epilogue here).
//   * bias == nullptr (bias-free; keeps the epilogue a plain narrow).
//   * K even (K-pair aligned chunks; the deep-K shapes are all even).
//   * M > max_mr AND the o-block B strip exceeds the L2 budget
//     (otherwise the plain path is already optimal — no benefit).
namespace {
// dst[r][c] += src[r][c] over `rows` x `cols` f32 (cols a multiple of 16).
ZENDNNL_TARGET("avx512f")
inline void add_f32_rows(
        float *dst, int ldd, const float *src, int lds, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float *d = dst + static_cast<size_t>(r) * ldd;
        const float *s = src + static_cast<size_t>(r) * lds;
        for (int c = 0; c < cols; c += 16)
            _mm512_storeu_ps(d + c,
                    _mm512_add_ps(
                            _mm512_loadu_ps(d + c), _mm512_loadu_ps(s + c)));
    }
}

ZENDNNL_TARGET("avx512f,avx512bf16,avx512bw,avx512vl,fma")
inline void dispatch_tile_bf16_kblocked(const CallContext &ctx, int expert_idx,
        int M, int K, int n_tile, int col_start, const void *src, int lda,
        void *tight_dst, int tight_ldc) {
    const bfloat16_t *Bpacked_full = ctx.packed_ptrs[expert_idx];
    const auto *A = static_cast<const bfloat16_t *>(src);
    const int pack_nr = ctx.pack_nr;
    const int NV = ctx.NV;
    const int max_mr = ctx.max_mr;
    const size_t dst_elem_bytes = (ctx.variant == KernelVariant::kBF16_BF16_F32)
            ? sizeof(float)
            : sizeof(bfloat16_t);

    // Packed-B strides (bf16 elements).  Mirror of `dispatch_tile`.
    const int K_pair_total = (K + 1) / 2;
    const size_t o_blk_stride
            = static_cast<size_t>(K_pair_total) * pack_nr * kVNNIPair;
    const int kp_stride_bf16 = pack_nr * kVNNIPair; // == NV*16*kVNNIPair

    const int n_blocks = n_tile / pack_nr;

    // Balanced MR partition — identical scheme to `dispatch_tile`.
    const int n_calls = (M + max_mr - 1) / max_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;

    // f32-store, act=none kernels for the two MR sizes we emit.
    const bf16_ukernel_fn_t fn_small = select_bf16_ukernel(
            mr_base, NV, ActivationKind::none, OutputType::kF32);
    const bf16_ukernel_fn_t fn_big = (n_big > 0)
            ? select_bf16_ukernel(
                      mr_base + 1, NV, ActivationKind::none, OutputType::kF32)
            : fn_small;
    // A prompt tile runs the 12-row prompt kernels, whose accumulate form
    // lives with them; the decode-tuned accumulate kernels stop at 8 rows.
    const auto select_accumulate = [&](int mr) {
        return ctx.prompt_kc > 0
                ? select_ukernel_accumulate_f32(mr, NV)
                : decode_specialized::select_plain_bf16_f32_accumulate(mr, NV);
    };
    const bf16_ukernel_fn_t accumulate_small = select_accumulate(mr_base);
    const bf16_ukernel_fn_t accumulate_big
            = (n_big > 0) ? select_accumulate(mr_base + 1) : accumulate_small;
    const bool direct_accumulate
            = accumulate_small != nullptr && accumulate_big != nullptr;

    // Per-thread f32 accumulator [M × n_tile].  thread_local so the
    // allocation amortises across tiles/calls on the same worker.
    thread_local std::vector<float> c_accum;
    c_accum.resize(static_cast<size_t>(M) * n_tile);

    // L2 budget (same model as `pick_l2_subtile_cols`).  Size the
    // K-chunk so one o-block's B slab plus a max_mr A strip fits.
    const int64_t l2_budget = ck_l2_budget_bytes();
    int kc = static_cast<int>(
            l2_budget / (static_cast<int64_t>(pack_nr + max_mr) * 2));
    kc &= ~1; // even (K-pair aligned)
    if (kc < 2) kc = 2;
    const bool dense_decode_w2
            = pack_nr == 32 && M > 8 && (K == 15360 || K == 14336);
    if (dense_decode_w2) {
        // Dense Gemma/Llama decode W2 rotates through cold layer weights.
        // A 512-element K chunk keeps each NR32 slab resident while all
        // M-blocks consume it, avoiding the bandwidth cliff of the larger
        // generic L2-derived chunk.
        kc = 512;
    }
    // A prompt tile is tall: its A chunk and f32 accumulator overflow L2, so
    // it walks MR calls outside the column blocks instead, and caps the
    // chunk so the tile's whole B chunk (kc x n_tile) stays in L2 while
    // every MR call sweeps it.
    const bool prompt_order = ctx.prompt_kc > 0;
    if (prompt_order) {
        const int64_t fit = l2_budget / (static_cast<int64_t>(n_tile) * 2);
        kc = static_cast<int>(std::min<int64_t>(ctx.prompt_kc, fit)) & ~1;
        kc = std::max(kc, kPromptKcMin);
    }

    // Small on-stack partial for k_chunk > 0 (max_mr × pack_nr f32).
    float ctmp[kMaxMR * 64];

    auto mr_call = [&](int c, int m_off, int b, int kc_off, int kc_now,
                           const bfloat16_t *Bpk, bool first) {
        const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
        const bf16_ukernel_fn_t kfn = (c < n_big) ? fn_big : fn_small;
        const bfloat16_t *A_chunk
                = A + static_cast<size_t>(m_off) * lda + kc_off;
        float *Cdst = c_accum.data() + static_cast<size_t>(m_off) * n_tile
                + static_cast<size_t>(b) * pack_nr;
        if (first) {
            kfn(A_chunk, lda, Bpk, /*bias=*/nullptr, BiasKind::none,
                    /*Cout=*/Cdst, /*ldc=*/n_tile, /*Cout_tight=*/nullptr,
                    /*ldc_tight=*/0, kc_now);
        } else if (direct_accumulate) {
            const bf16_ukernel_fn_t accumulate_kfn
                    = (c < n_big) ? accumulate_big : accumulate_small;
            accumulate_kfn(A_chunk, lda, Bpk, /*bias=*/nullptr, BiasKind::none,
                    /*Cout=*/Cdst, /*ldc=*/n_tile,
                    /*Cout_tight=*/nullptr, /*ldc_tight=*/0, kc_now);
        } else {
            kfn(A_chunk, lda, Bpk, /*bias=*/nullptr, BiasKind::none,
                    /*Cout=*/ctmp, /*ldc=*/pack_nr, /*Cout_tight=*/nullptr,
                    /*ldc_tight=*/0, kc_now);
            add_f32_rows(Cdst, n_tile, ctmp, pack_nr, mr_now, pack_nr);
        }
    };

    for (int kc_off = 0; prompt_order && kc_off < K; kc_off += kc) {
        const int kc_now = std::min(kc, K - kc_off);
        const size_t k_off_pk
                = static_cast<size_t>(kc_off / 2) * kp_stride_bf16;
        int m_off = 0;
        for (int c = 0; c < n_calls; ++c) {
            for (int b = 0; b < n_blocks; ++b) {
                const bfloat16_t *Bpk = Bpacked_full
                        + static_cast<size_t>(col_start / pack_nr + b)
                                * o_blk_stride
                        + k_off_pk;
                mr_call(c, m_off, b, kc_off, kc_now, Bpk, kc_off == 0);
            }
            m_off += (c < n_big) ? mr_base + 1 : mr_base;
        }
    }

    for (int kc_off = 0; !prompt_order && kc_off < K; kc_off += kc) {
        const int kc_now = std::min(kc, K - kc_off);
        const int kp0 = kc_off / 2;
        const bool first = (kc_off == 0);
        for (int b = 0; b < n_blocks; ++b) {
            const bfloat16_t *Bpk = Bpacked_full
                    + static_cast<size_t>(col_start / pack_nr + b)
                            * o_blk_stride
                    + static_cast<size_t>(kp0) * kp_stride_bf16;
            int m_off = 0;
            for (int c = 0; c < n_calls; ++c) {
                const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
                const bf16_ukernel_fn_t kfn = (c < n_big) ? fn_big : fn_small;
                const bfloat16_t *A_chunk
                        = A + static_cast<size_t>(m_off) * lda + kc_off;
                if (first) {
                    float *Cdst = c_accum.data()
                            + static_cast<size_t>(m_off) * n_tile
                            + static_cast<size_t>(b) * pack_nr;
                    kfn(A_chunk, lda, Bpk, /*bias=*/nullptr, BiasKind::none,
                            /*Cout=*/Cdst, /*ldc=*/n_tile,
                            /*Cout_tight=*/nullptr, /*ldc_tight=*/0, kc_now);
                } else if (direct_accumulate) {
                    float *Cdst = c_accum.data()
                            + static_cast<size_t>(m_off) * n_tile
                            + static_cast<size_t>(b) * pack_nr;
                    const bf16_ukernel_fn_t accumulate_kfn
                            = (c < n_big) ? accumulate_big : accumulate_small;
                    accumulate_kfn(A_chunk, lda, Bpk, /*bias=*/nullptr,
                            BiasKind::none, /*Cout=*/Cdst, /*ldc=*/n_tile,
                            /*Cout_tight=*/nullptr, /*ldc_tight=*/0, kc_now);
                } else {
                    kfn(A_chunk, lda, Bpk, /*bias=*/nullptr, BiasKind::none,
                            /*Cout=*/ctmp, /*ldc=*/pack_nr,
                            /*Cout_tight=*/nullptr, /*ldc_tight=*/0, kc_now);
                    for (int mm = 0; mm < mr_now; ++mm) {
                        float *dst_row = c_accum.data()
                                + static_cast<size_t>(m_off + mm) * n_tile
                                + static_cast<size_t>(b) * pack_nr;
                        const float *src_row
                                = ctmp + static_cast<size_t>(mm) * pack_nr;
                        for (int cc = 0; cc < pack_nr; cc += 16) {
                            const __m512 dstv = _mm512_loadu_ps(dst_row + cc);
                            const __m512 srcv = _mm512_loadu_ps(src_row + cc);
                            _mm512_storeu_ps(
                                    dst_row + cc, _mm512_add_ps(dstv, srcv));
                        }
                    }
                }
                m_off += mr_now;
            }
        }
    }

    // Epilogue: narrow the f32 accumulator to the caller's dst dtype.
    if (dst_elem_bytes == sizeof(float)) {
        auto *dst = static_cast<float *>(tight_dst);
        for (int m = 0; m < M; ++m) {
            const float *acc_row
                    = c_accum.data() + static_cast<size_t>(m) * n_tile;
            float *dst_row
                    = dst + static_cast<size_t>(m) * tight_ldc + col_start;
            for (int col = 0; col < n_tile; col += 16) {
                _mm512_storeu_ps(dst_row + col, _mm512_loadu_ps(acc_row + col));
            }
        }
    } else {
        auto *dst = static_cast<bfloat16_t *>(tight_dst);
        for (int m = 0; m < M; ++m) {
            const float *acc_row
                    = c_accum.data() + static_cast<size_t>(m) * n_tile;
            bfloat16_t *dst_row
                    = dst + static_cast<size_t>(m) * tight_ldc + col_start;
            for (int col = 0; col < n_tile; col += 32) {
                const __m512 lo = _mm512_loadu_ps(acc_row + col);
                const __m512 hi = _mm512_loadu_ps(acc_row + col + 16);
                const __m512bh out = _mm512_cvtne2ps_pbh(hi, lo);
                _mm512_storeu_si512(reinterpret_cast<__m512i *>(dst_row + col),
                        (__m512i)out);
            }
        }
    }
}

} // namespace

namespace {

bool fill_src_tables(CallContext &ctx, bool src_u8) {
    const OutputType dst_dt = dst_dt_for_variant(ctx.variant);
    if (is_w4a8_variant(ctx.variant)) {
        w4a8_ukernel_fn_t table[kMaxMR + 1] = {};
        w4a8_ukernel_fn_t table_offset[kMaxMR + 1] = {};
        if (fill_kernel_table_s4(ctx.NV, ctx.act_kind, ctx.src_scale_groups,
                    dst_dt, table, table_offset,
                    src_u8 ? S4SourceKind::kU8Zp128 : S4SourceKind::kS8)
                != status_t::success)
            return false;
        std::copy(std::begin(table), std::end(table),
                std::begin(ctx.kernel_table_s4));
        std::copy(std::begin(table_offset), std::end(table_offset),
                std::begin(ctx.kernel_table_s4_offset));
    } else {
        int8_ukernel_fn_t table[kMaxMR + 1] = {};
        if (fill_kernel_table_int8(ctx.NV, ctx.compute_int, ctx.act_kind,
                    dst_dt, table, ctx.group_size, src_u8)
                != status_t::success)
            return false;
        std::copy(std::begin(table), std::end(table),
                std::begin(ctx.kernel_table_int8));
    }
    ctx.src_u8 = src_u8;
    return true;
}

} // namespace

bool u8_src_pays() {
    const int ovr
            = test_api::s_ck_u8_src_override.load(std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    return grp_machine::dot_rates().u8_src_pays;
}

bool use_u8_src(CallContext &ctx) {
    return u8_src_eligible(ctx) && fill_src_tables(ctx, /*src_u8=*/true);
}

void use_s8_src(CallContext &ctx) {
    if (ctx.src_u8) fill_src_tables(ctx, /*src_u8=*/false);
}

void copy_s8_rows_as_u8(const int8_t *src, int64_t lda, uint8_t *dst,
        int64_t ldd, int rows, int cols) {
    constexpr uint64_t kFlip = 0x8080808080808080ULL;
    for (int r = 0; r < rows; ++r) {
        const auto *s = reinterpret_cast<const unsigned char *>(src + r * lda);
        unsigned char *d = dst + r * ldd;
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            uint64_t w;
            std::memcpy(&w, s + c, sizeof(w));
            w ^= kFlip;
            std::memcpy(d + c, &w, sizeof(w));
        }
        for (; c < cols; ++c)
            d[c] = static_cast<unsigned char>(s[c] ^ 0x80u);
    }
}

void dispatch_tile(const CallContext &ctx, int expert_idx, int M, int K,
        int n_tile, int col_start, const void *src, int lda, const void *bias,
        void *tight_dst, int tight_ldc, const void *src_scale,
        const int32_t *src_zp, const void *wei_scale,
        const Int8EpilogueConfig &int8_epilogue, const int32_t *src_group_sums,
        const void *wei_zp) {

    // ── Per-tile contract assertions (debug-only) ────────────────────
    //
    // Every assert below mirrors a contract that `prepare_for_call`
    // already validated.  Caller bug = call `dispatch_tile` without a
    // successful `prepare_for_call` first; these asserts turn silent
    // wrong output (zeros / garbage) into a loud debug-build failure.
    //
    // Asserts are debug-only; in release the contracts are upheld by
    // the upstream gates documented next to each assert.
    //
    // 1. ctx.enabled — set true by `prepare_for_call` on success only.
    //    A `dispatch_tile` call with `ctx.enabled == false` means the
    //    caller skipped the post-prepare check and is asking the
    //    dispatcher to use a CallContext that was either never
    //    populated (default-constructed) or rejected.
    assert(ctx.enabled
         && "dispatch_tile: ctx.enabled is false — caller must check "
            "prepare_for_call's status (or kctx.enabled) and route to "
            "DLP on failure");
    // 2. ctx.variant — set by `prepare_for_call` via `resolve_variant`.
    //    `kUnsupported` would mean the routing table couldn't classify
    //    the (src, wei, dst) tuple but the caller proceeded anyway.
    assert(ctx.variant != KernelVariant::kUnsupported
         && "dispatch_tile: ctx.variant is kUnsupported — "
            "prepare_for_call should have returned failure on this "
            "dtype tuple and the caller should have routed to DLP");
    // 3. Tile-shape contract: every per-thread tile produced by
    //    `aligned_n_split` must be a multiple of `pack_nr` (the kernel
    //    processes B in `pack_nr`-wide blocks via truncating
    //    `n_blocks = sub_n / pack_nr`; any non-multiple tail would be
    //    SILENTLY DROPPED, leaving dst columns uninitialised).
    //
    //    This holds for the current production envelope because:
    //      a. `prepare_for_call` refuses (N % pack_nr != 0);
    //      b. for any `n_thr ≤ N / pack_nr`, `aligned_n_split` always
    //         finds an aligned candidate that satisfies the 2× imbalance
    //         bound, never falling back to its unaligned even-split path;
    //      c. `tile_thread_count` upstream caps `n_thr` by
    //         `N / min_n_tile`, and `min_n_tile >= nr_align >= pack_nr`,
    //         so condition (b) holds.  `min_n_tile` used to be the fixed
    //         `kDecodeNTile` (256) which cleared `pack_nr` trivially; the
    //         adaptive N-tile sizers can now size it down to the 32-col
    //         tile floor, so `nr_aligned_min_n_tile` (N-tile planner) is
    //         what maintains the `>= nr_align` half of this invariant.
    assert(ctx.pack_nr > 0
            && "dispatch_tile: pack_nr is zero — call prepare_for_call");
    assert((n_tile % ctx.pack_nr) == 0
         && "dispatch_tile: n_tile not a multiple of pack_nr — tail "
            "cols would be silently dropped by truncating n_blocks");
    assert((col_start % ctx.pack_nr) == 0
         && "dispatch_tile: col_start not a multiple of pack_nr — "
            "B-pack offset (col_start / pack_nr) would mis-align");

    // ── DQ-INT8 fast path — hand off to the int8 sibling and return.
    // The bf16 loop body below is straight-line; keeping the int8
    // branch up front avoids loading the bf16-specific stride
    // constants we don't need on this path.
    if (is_int8_variant(ctx.variant)) {
        const bool simple_dynamic_epilogue = !int8_epilogue.src_quant_per_tensor
                && !int8_epilogue.wei_scale_per_tensor
                && int8_epilogue.binary_mul == nullptr
                && int8_epilogue.binary_add == nullptr
                && int8_epilogue.unary_postop == Int8UnaryPostOp::kNone;
        const int64_t int8_kblock_l2_budget = ck_l2_budget_bytes();
        const int64_t int8_oblock_bytes
                = static_cast<int64_t>(K) * 32 + 32 * sizeof(int32_t);
        const bool int8_kblock_policy
                = grp_matmul_int8_kblock_enabled(ctx.kblock_auto);
        // group_size 0 is the full-K compensation row this kernel reads.
        // Q8 (group_size 32) stores a compensation row after every group.
        const bool int8_kblock_shape = ctx.group_size == 0
                && ctx.variant == KernelVariant::kS8_S8_BF16_SYM
                && ctx.compute_int == IntCompute::kS8_Sym && ctx.pack_nr == 32
                && ctx.act_kind == ActivationKind::none && bias == nullptr
                && src_zp == nullptr && simple_dynamic_epilogue
                && M > ctx.max_mr && int8_oblock_bytes > int8_kblock_l2_budget;
        if (int8_kblock_policy && int8_kblock_shape) {
            if (test_api::s_capture_kblock.load(std::memory_order_relaxed)) {
                test_api::s_last_kblock_used.store(
                        true, std::memory_order_relaxed);
            }
            dispatch_tile_int8_kblocked(ctx, expert_idx, M, K, n_tile,
                    col_start, src, lda, tight_dst, tight_ldc, src_scale,
                    wei_scale);
            return;
        }
        dispatch_tile_int8(ctx, expert_idx, M, K, n_tile, col_start, src, lda,
                bias, tight_dst, tight_ldc, src_scale, src_zp, wei_scale,
                int8_epilogue);
        return;
    }
    if (is_w4a8_variant(ctx.variant)) {
        dispatch_tile_w4(ctx, expert_idx, M, K, n_tile, col_start, src, lda,
                bias, tight_dst, tight_ldc, src_scale, wei_scale,
                src_group_sums, wei_zp);
        return;
    }
    // FP16 fast path — native AVX-512-FP16.  Non-quant like the bf16
    // path, so the scale / zp / wei_scale args are unused (asserted null
    // below for the bf16 branch; the f16 branch returns before that).
    if (is_f16_variant(ctx.variant)) {
        if (bias == nullptr
                && dispatch_tile_f16_kblocked(ctx, expert_idx, M, K, n_tile,
                        col_start, src, lda, tight_dst, tight_ldc))
            return;
        dispatch_tile_f16(ctx, expert_idx, M, K, n_tile, col_start, src, lda,
                bias, tight_dst, tight_ldc);
        return;
    }
    // BF16 callers must NOT pass src_scale / src_zp / wei_scale —
    // those are DQ-INT8-only.  If they did, fail loudly in debug so
    // the contract violation is visible; in release the BF16 body
    // simply ignores them.
    assert(src_scale == nullptr
            && "dispatch_tile: src_scale must be null on the BF16 path");
    assert(src_zp == nullptr
            && "dispatch_tile: src_zp must be null on the BF16 path");
    assert(wei_scale == nullptr
            && "dispatch_tile: wei_scale must be null on the BF16 path");
    (void)src_scale;
    (void)src_zp;
    (void)wei_scale;

    // ── Deep-K K-blocking fast path ──────────────────────────────────
    // Only for the bias-free act=none bf16 case where one o-block's B
    // strip exceeds L2 and M spans more than one MR-block — the exact
    // regime where the plain path re-streams B from DRAM per M-block.
    // Everything else falls through to the straight-line path below.
    const bool prompt_kblock = ctx.prompt_kc > 0 && ctx.prompt_kc < K;
    if ((prompt_kblock || grp_matmul_kblock_enabled(ctx.kblock_auto))
            && ctx.act_kind == ActivationKind::none && bias == nullptr
            && (K % 2) == 0 && M > ctx.max_mr) {
        const int64_t kblock_l2_budget = ck_l2_budget_bytes();
        if (prompt_kblock
                || static_cast<int64_t>(K) * ctx.pack_nr * 2
                        > kblock_l2_budget) {
            if (test_api::s_capture_kblock.load(std::memory_order_relaxed)) {
                test_api::s_last_kblock_used.store(
                        true, std::memory_order_relaxed);
            }
            dispatch_tile_bf16_kblocked(ctx, expert_idx, M, K, n_tile,
                    col_start, src, lda, tight_dst, tight_ldc);
            return;
        }
    }

    const bfloat16_t *Bpacked_full = ctx.packed_ptrs[expert_idx];
    const auto *A = static_cast<const bfloat16_t *>(src);
    // Use byte-arithmetic for the dst since the element width depends on
    // the kernel variant (BF16=2B for kBF16_BF16_BF16; FP32=4B for
    // kBF16_BF16_F32).  The kernel internally re-casts the void* to its
    // template `DstT`; ldc / ldc_tight stay in element units (not bytes).
    std::byte *Tight_bytes = static_cast<std::byte *>(tight_dst);
    const size_t dst_elem_bytes = (ctx.variant == KernelVariant::kBF16_BF16_F32)
            ? sizeof(float)
            : sizeof(bfloat16_t);

    // Per-tile constants (hoisted once outside the sub-tile / m-chunk /
    // n-block loops below).
    const int K_pair = (K + 1) / 2;
    const size_t o_blk_stride
            = static_cast<size_t>(K_pair) * ctx.pack_nr * kVNNIPair;

    // Per-expert L2-friendly subtile width (see `subtile_cols_per_expert`
    // doc in dispatch.hpp).  Falls back to the representative global
    // value if the per-expert slot wasn't populated (M <= 0 at prep
    // time, which the caller's sweep over num_ops upstream should have
    // already filtered out).
    const int subtile_cols = (ctx.subtile_cols_per_expert[expert_idx] > 0)
            ? ctx.subtile_cols_per_expert[expert_idx]
            : ctx.subtile_cols;

    // Balanced MR partition: split M into `n_calls = ceil(M / max_mr)`
    // kernel calls, spreading the rows so each call's MR is either
    // `mr_base` or `mr_base + 1`.  First `n_big` calls take mr_base+1.
    // Avoids thin-tail MR=1 / MR=2 calls that would otherwise cap FMA
    // ILP (MR<4 is latency-limited at 4-5 cycle FMA dep).
    //
    // Worked examples (max_mr=8):
    //   M= 9 → 2 calls [5, 4]   (naive would be [8, 1] — MR=1 tail)
    //   M=17 → 3 calls [6, 6, 5] (naive [8, 8, 1])
    //   M=24 → 3 calls [8, 8, 8] (no tail either way)
    //   M= 5 → 1 call  [5]
    const int n_calls = (M + ctx.max_mr - 1) / ctx.max_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;

    // Per-col byte stride for bias pointer arithmetic — the dispatcher
    // stores the bias dtype in `ctx.bias_kind` so we advance the pointer
    // by the right element width when slicing into the per-sub-tile /
    // per-block bias window.
    const size_t bias_elem_bytes = (ctx.bias_kind == BiasKind::fp32)
            ? sizeof(float)
            : sizeof(bfloat16_t); // bf16 or f16 (both 2B); none unused when bias==null

    // Hoist the act-kind branch out of the loops — straight-line code
    // with one indirect call per microkernel invocation.  All three
    // gated-activation kinds (swiglu_oai_mul, silu_and_mul,
    // gelu_and_mul) share the same tight-arena contract: halved-width
    // output, BF16-only, 16 cols per (g, u) pair × NV/2 pairs per
    // kernel call.  They differ only in which in-register pair-store
    // helper the microkernel calls (resolved at `select_bf16_ukernel` time
    // and baked into `ctx.kernel_table[]`), so the dispatcher loop body
    // is identical.
    const bool is_gated_act = (ctx.act_kind == ActivationKind::swiglu_oai_mul)
            || (ctx.act_kind == ActivationKind::silu_and_mul)
            || (ctx.act_kind == ActivationKind::gelu_and_mul);
    if (is_gated_act) {
        // Walk the per-thread N range in L2-friendly sub-tiles.  Each
        // sub-tile's B strip (= K × subtile_cols × 2 bytes) fits in L2
        // alongside the input slice, so the microkernel never hits L3 /
        // DRAM mid-call for weight.
        for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
            const int sub_n = std::min(subtile_cols, n_tile - sub_off);
            const int sub_col_base = col_start + sub_off;
            const int n_blocks = sub_n / ctx.pack_nr;

            const bfloat16_t *Bpacked_blk_base = Bpacked_full
                    + static_cast<size_t>(sub_col_base / ctx.pack_nr)
                            * o_blk_stride;
            const char *bias_blk_base = (bias != nullptr)
                    ? static_cast<const char *>(bias)
                            + static_cast<size_t>(sub_col_base)
                                    * bias_elem_bytes
                    : nullptr;

            // ── Loop nesting: o-block outer vs MR-call outer ──────────
            // MR-call outer makes every MR call re-walk the whole
            // sub-tile's B strip, so the reuse distance for one o-block's
            // B is the entire sub-tile (512 KB on Llama-8B gate+up).
            // o-block outer shrinks that distance to a single A chunk
            // (64 KB), keeping the o-block's 256 KB of B hot across all
            // MR calls.  `dispatch_tile_bf16_kblocked` already nests this
            // way; this brings the straight-line path in line with it.
            //
            // Restricted to single-expert calls: the reordering was only
            // characterised on dense FFN shapes (num_ops == 1).  MoE tiles
            // are narrower because N is split across more concurrent work,
            // so the o-block count per sub-tile — and with it the whole
            // benefit — can differ.  Both orders compute the same result,
            // so this is a scope guard, not a correctness one.
            if (se_oblock_outer(ctx.single_expert, n_calls)) {
                for (int b = 0; b < n_blocks; ++b) {
                    const bfloat16_t *Bpacked_blk = Bpacked_blk_base
                            + static_cast<size_t>(b) * o_blk_stride;
                    const void *bias_blk = (bias_blk_base != nullptr)
                            ? static_cast<const void *>(bias_blk_base
                                      + static_cast<size_t>(b) * ctx.pack_nr
                                              * bias_elem_bytes)
                            : nullptr;
                    int m_off_b = 0;
                    for (int c = 0; c < n_calls; ++c) {
                        const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
                        const bf16_ukernel_fn_t kfn = ctx.kernel_table[mr_now];
                        const bfloat16_t *A_chunk
                                = A + static_cast<size_t>(m_off_b) * lda;
                        bfloat16_t *Tight_row
                                = static_cast<bfloat16_t *>(tight_dst)
                                + static_cast<size_t>(m_off_b) * tight_ldc
                                + (sub_col_base / 2)
                                + static_cast<size_t>(b) * (ctx.pack_nr / 2);
                        kfn(A_chunk, lda, Bpacked_blk, bias_blk, ctx.bias_kind,
                                /*Cout=*/nullptr, /*ldc=*/0,
                                static_cast<void *>(Tight_row), tight_ldc, K);
                        m_off_b += mr_now;
                    }
                }
                continue;
            }

            int m_off = 0;
            for (int c = 0; c < n_calls; ++c) {
                const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
                const bf16_ukernel_fn_t kfn = ctx.kernel_table[mr_now];

                const bfloat16_t *A_chunk
                        = A + static_cast<size_t>(m_off) * lda;
                // Gated-act epilogues are BF16-dst-only (gated by
                // select_bf16_ukernel for swiglu_oai_mul, silu_and_mul, and
                // gelu_and_mul — see `is_gated_act` above), so the half-width
                // tight arena is a plain bfloat16_t matrix (`tight_ldc` in
                // elements) regardless of `dst_elem_bytes`.  Index it as
                // bfloat16_t* so the element→byte scaling is implicit —
                // avoids a sizeof multiply against a byte pointer.
                bfloat16_t *Tight_row_base
                        = static_cast<bfloat16_t *>(tight_dst)
                        + static_cast<size_t>(m_off) * tight_ldc
                        + (sub_col_base / 2);

                for (int b = 0; b < n_blocks; ++b) {
                    const bfloat16_t *Bpacked_blk = Bpacked_blk_base
                            + static_cast<size_t>(b) * o_blk_stride;
                    const void *bias_blk = (bias_blk_base != nullptr)
                            ? static_cast<const void *>(bias_blk_base
                                      + static_cast<size_t>(b) * ctx.pack_nr
                                              * bias_elem_bytes)
                            : nullptr;
                    // 16 cols per (g, u) pair × NV/2 pairs per kernel call.
                    bfloat16_t *Tight_row = Tight_row_base
                            + static_cast<size_t>(b) * (ctx.pack_nr / 2);

                    kfn(A_chunk, lda, Bpacked_blk, bias_blk, ctx.bias_kind,
                            /*Cout=*/nullptr, /*ldc=*/0,
                            static_cast<void *>(Tight_row), tight_ldc, K);
                }
                m_off += mr_now;
            }
        }
        return;
    }

    // Act = none — write the full pack_nr cols straight to the wide
    // [M, N] arena (Tight is the wide destination in this mode).
    for (int sub_off = 0; sub_off < n_tile; sub_off += subtile_cols) {
        const int sub_n = std::min(subtile_cols, n_tile - sub_off);
        const int sub_col_base = col_start + sub_off;
        const int n_blocks = sub_n / ctx.pack_nr;

        const bfloat16_t *Bpacked_blk_base = Bpacked_full
                + static_cast<size_t>(sub_col_base / ctx.pack_nr)
                        * o_blk_stride;
        const char *bias_blk_base = (bias != nullptr)
                ? static_cast<const char *>(bias)
                        + static_cast<size_t>(sub_col_base) * bias_elem_bytes
                : nullptr;

        if (se_oblock_outer(ctx.single_expert, n_calls)) {
            for (int b = 0; b < n_blocks; ++b) {
                const bfloat16_t *Bpacked_blk = Bpacked_blk_base
                        + static_cast<size_t>(b) * o_blk_stride;
                const void *bias_blk = (bias_blk_base != nullptr)
                        ? static_cast<const void *>(bias_blk_base
                                  + static_cast<size_t>(b) * ctx.pack_nr
                                          * bias_elem_bytes)
                        : nullptr;
                int m_off_b = 0;
                for (int c = 0; c < n_calls; ++c) {
                    const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
                    const bf16_ukernel_fn_t kfn = ctx.kernel_table[mr_now];
                    const bfloat16_t *A_chunk
                            = A + static_cast<size_t>(m_off_b) * lda;
                    std::byte *Wide_row = Tight_bytes
                            + (static_cast<size_t>(m_off_b) * tight_ldc
                                      + sub_col_base
                                      + static_cast<size_t>(b) * ctx.pack_nr)
                                    * dst_elem_bytes;
                    kfn(A_chunk, lda, Bpacked_blk, bias_blk, ctx.bias_kind,
                            Wide_row, tight_ldc,
                            /*Cout_tight=*/nullptr, /*ldc_tight=*/0, K);
                    m_off_b += mr_now;
                }
            }
            continue;
        }

        int m_off = 0;
        for (int c = 0; c < n_calls; ++c) {
            const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
            const bf16_ukernel_fn_t kfn = ctx.kernel_table[mr_now];

            const bfloat16_t *A_chunk = A + static_cast<size_t>(m_off) * lda;
            // Wide / act=none path: dst element width depends on the
            // resolved variant (BF16=2B, FP32=4B).  Use byte arithmetic
            // here so the same loop body serves both store epilogues; the
            // kernel re-casts the void* row to its template `DstT`.
            std::byte *Wide_row_base = Tight_bytes
                    + (static_cast<size_t>(m_off) * tight_ldc + sub_col_base)
                            * dst_elem_bytes;

            for (int b = 0; b < n_blocks; ++b) {
                const bfloat16_t *Bpacked_blk = Bpacked_blk_base
                        + static_cast<size_t>(b) * o_blk_stride;
                const void *bias_blk = (bias_blk_base != nullptr)
                        ? static_cast<const void *>(bias_blk_base
                                  + static_cast<size_t>(b) * ctx.pack_nr
                                          * bias_elem_bytes)
                        : nullptr;
                std::byte *Wide_row = Wide_row_base
                        + static_cast<size_t>(b) * ctx.pack_nr * dst_elem_bytes;

                kfn(A_chunk, lda, Bpacked_blk, bias_blk, ctx.bias_kind,
                        Wide_row, tight_ldc,
                        /*Cout_tight=*/nullptr, /*ldc_tight=*/0, K);
            }
            m_off += mr_now;
        }
    }
}

bool block_api_supported(const CallContext &ctx, const void *bias,
        const void *src_zp, const Int8EpilogueConfig &int8_epilogue) {
    if (ctx.act_kind != ActivationKind::none || bias != nullptr) return false;
    if (ctx.variant == KernelVariant::kS8_S8_BF16_SYM)
        return ctx.group_size == 0 && ctx.compute_int == IntCompute::kS8_Sym
                && ctx.pack_nr == kNRMin && src_zp == nullptr
                && !int8_epilogue.src_quant_per_tensor
                && !int8_epilogue.wei_scale_per_tensor
                && int8_epilogue.binary_mul == nullptr
                && int8_epilogue.binary_add == nullptr
                && int8_epilogue.unary_postop == Int8UnaryPostOp::kNone;
    return (ctx.variant == KernelVariant::kBF16_BF16_BF16
                   || ctx.variant == KernelVariant::kBF16_BF16_F32)
            && (ctx.pack_nr == kNRMin || ctx.pack_nr == kNRMax);
}

int block_k_step(const CallContext &ctx) {
    return is_int8_variant(ctx.variant) ? kVNNIInt8Quad : kVNNIPair;
}

size_t block_dst_elem_bytes(const CallContext &ctx) {
    return ctx.variant == KernelVariant::kBF16_BF16_F32 ? sizeof(float)
                                                        : sizeof(bfloat16_t);
}

size_t block_scale_elem_bytes(const CallContext &ctx) {
    return ctx.scale_kind == ScaleKind::kBf16 ? sizeof(bfloat16_t)
                                              : sizeof(float);
}

namespace {

// Packed-block byte strides; must match `bytes_per_oblock` in pack.cpp, as
// `dispatch_tile_int8_kblocked` and `dispatch_tile_bf16_kblocked` do.
inline size_t int8_block_weight_bytes(int K) {
    return static_cast<size_t>(K / kVNNIInt8Quad) * kNRMin * kVNNIInt8Quad;
}
inline size_t int8_block_stride_bytes(int K) {
    return int8_block_weight_bytes(K) + kNRMin * sizeof(int32_t);
}

} // namespace

void accumulate_block(const CallContext &ctx, int expert_idx, int M, int K,
        int block, int k_begin, int k_end, const void *src, int lda, void *acc,
        bool add, int mr_cap) {
    assert(k_begin % block_k_step(ctx) == 0 && k_begin < k_end && k_end <= K);
    const int pack_nr = ctx.pack_nr;
    const int max_mr = (mr_cap > 0) ? std::min(mr_cap, ctx.max_mr) : ctx.max_mr;
    const int n_calls = (M + max_mr - 1) / max_mr;
    const int mr_base = M / n_calls;
    const int n_big = M - mr_base * n_calls;

    if (is_int8_variant(ctx.variant)) {
        const int8_t *Bpk = ctx.packed_ptrs_int8[expert_idx]
                + static_cast<size_t>(block) * int8_block_stride_bytes(K)
                + static_cast<size_t>(k_begin / kVNNIInt8Quad) * kNRMin
                        * kVNNIInt8Quad;
        const auto *A = static_cast<const uint8_t *>(src) + k_begin;
        auto *C = static_cast<int32_t *>(acc);
        int m_off = 0;
        for (int c = 0; c < n_calls; ++c) {
            const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
            const int8_accum_ukernel_fn_t kfn
                    = select_int8_accum_ukernel(mr_now, ctx.src_u8);
            assert(kfn != nullptr);
            kfn(A + static_cast<size_t>(m_off) * lda, lda, Bpk, k_end - k_begin,
                    C + static_cast<size_t>(m_off) * pack_nr, pack_nr, add);
            m_off += mr_now;
        }
        return;
    }

    const size_t o_blk_stride
            = static_cast<size_t>((K + 1) / 2) * pack_nr * kVNNIPair;
    const bfloat16_t *Bpk = ctx.packed_ptrs[expert_idx]
            + static_cast<size_t>(block) * o_blk_stride
            + static_cast<size_t>(k_begin / kVNNIPair) * pack_nr * kVNNIPair;
    const auto *A = static_cast<const bfloat16_t *>(src) + k_begin;
    auto *C = static_cast<float *>(acc);
    // The f32-store kernel overwrites its output, so `add` goes through a
    // per-call partial.
    float part[kMaxMR * kNRMax];
    int m_off = 0;
    for (int c = 0; c < n_calls; ++c) {
        const int mr_now = (c < n_big) ? mr_base + 1 : mr_base;
        const bf16_ukernel_fn_t kfn = select_bf16_ukernel(
                mr_now, ctx.NV, ActivationKind::none, OutputType::kF32);
        assert(kfn != nullptr);
        float *Crow = C + static_cast<size_t>(m_off) * pack_nr;
        kfn(A + static_cast<size_t>(m_off) * lda, lda, Bpk, /*bias=*/nullptr,
                BiasKind::none, add ? part : Crow, pack_nr,
                /*Cout_tight=*/nullptr, /*ldc_tight=*/0, k_end - k_begin);
        if (add) {
            for (int i = 0; i < mr_now * pack_nr; ++i)
                Crow[i] += part[i];
        }
        m_off += mr_now;
    }
}

void finish_block(const CallContext &ctx, int expert_idx, int M, int K,
        int block, const void *acc, const void *src_scale,
        const void *wei_scale, void *dst, int ldc) {
    const int pack_nr = ctx.pack_nr;
    const size_t col = static_cast<size_t>(block) * pack_nr;
    if (is_int8_variant(ctx.variant)) {
        const int8_t *Bblock = ctx.packed_ptrs_int8[expert_idx]
                + static_cast<size_t>(block) * int8_block_stride_bytes(K);
        const auto *comp = reinterpret_cast<const int32_t *>(
                Bblock + int8_block_weight_bytes(K));
        const void *wei_scale_block = static_cast<const char *>(wei_scale)
                + col * block_scale_elem_bytes(ctx);
        finish_int8_accum_nv2_sym_bf16(static_cast<const int32_t *>(acc),
                pack_nr, M, comp, src_scale, wei_scale_block, ctx.scale_kind,
                static_cast<bfloat16_t *>(dst) + col, ldc);
        return;
    }
    const auto *a = static_cast<const float *>(acc);
    if (ctx.variant == KernelVariant::kBF16_BF16_F32) {
        auto *out = static_cast<float *>(dst);
        for (int m = 0; m < M; ++m)
            std::memcpy(out + static_cast<size_t>(m) * ldc + col,
                    a + static_cast<size_t>(m) * pack_nr,
                    pack_nr * sizeof(float));
        return;
    }
    auto *out = static_cast<bfloat16_t *>(dst);
    for (int m = 0; m < M; ++m) {
        const float *row = a + static_cast<size_t>(m) * pack_nr;
        bfloat16_t *o = out + static_cast<size_t>(m) * ldc + col;
        for (int c = 0; c < pack_nr; ++c)
            o[c] = bfloat16_t(row[c]);
    }
}

} // namespace custom_kernel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
