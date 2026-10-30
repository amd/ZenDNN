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

/// Custom-kernel enable knobs and the effective per-call custom-kernel
/// verdict that the arena, cache and executor decisions share.

#ifndef ZENDNNL_GROUP_MATMUL_CK_ENABLE_HPP
#define ZENDNNL_GROUP_MATMUL_CK_ENABLE_HPP

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_constants.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_env.hpp"
#include "lowoha_operators/matmul/group_matmul/common/grp_matmul_n_split.hpp"
#include "lowoha_operators/matmul/group_matmul/custom_kernel/dispatch.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL = { "0", "1" } — cached, default ON.
//   Master switch for the hand-rolled AVX-512-BF16 microkernel
//   (custom_kernel/).  ON: ALGO 3 n_tile_execute dispatches per-tile
//   GEMM through VDPBF16PS, with swiglu_oai_mul applied in-register
//   (writes activated I cols at caller's ldc — covers both wide and
//   tight fused-MoE layouts in one path).  OFF: per-tile GEMM goes
//   through the standard AOCL DLP / BRGEMM dispatch, fused activation
//   runs as a separate per-tile pass.
//
//   Default ON history: an earlier revision flipped this to OFF
//   because the CK pack cache (see `custom_kernel/pack.cpp`) is
//   keyed by the raw weight pointer, and frameworks that recycle
//   freed allocator addresses (e.g. PyTorch CPU allocator) could
//   silently serve stale packed bytes for a new tensor at the
//   same address.  That hazard is now addressed at the library
//   level: the CK path honours `ZENDNNL_MATMUL_WEIGHT_CACHE` with
//   the same semantics as the AOCL DLP path.
//     - `ZENDNNL_MATMUL_WEIGHT_CACHE=2` (default): in-place reuse
//       where the path supports it; otherwise pointer-keyed LRU cache
//       stays warm across calls (production MoE serving with a stable
//       model — the common case).  Mode `1` forces out-of-place LRU.
//     - `ZENDNNL_MATMUL_WEIGHT_CACHE=0`: per-call caller-owned
//       packed buffers are allocated fresh and freed after the
//       call, so a recycled pointer can never hit a stale entry.
//       CK remains engaged, just without the cache.
//   See `custom_kernel/dispatch.cpp::prepare_for_call` (owned-ptr
//   path) and `prepack/prepack_custom_kernel.cpp` (warm-pack skip
//   when WEIGHT_CACHE=0).  Default-ON is therefore safe for both
//   stable-pointer and recycled-pointer regimes; callers that
//   want to bypass CK entirely (e.g. parity bisection against
//   AOCL DLP) can still set `ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL=0`.
//
//   The dispatcher refuses cleanly and falls back to the standard
//   AOCL DLP path for any expert that violates the CK contract
//   (non-bf16, transA, alpha≠1, β≠0, N % pack_nr ≠ 0, non-const
//   weights, etc. — see `custom_kernel/dispatch.cpp::prepare_for_call`
//   for the full gate cascade), so callers outside the supported
//   envelope see no behaviour change regardless of this knob.
inline bool get_grp_matmul_custom_kernel() {
    const int ovr = test_api::s_grp_matmul_custom_kernel_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL");
        int parsed = 0;
        // Strict parse AND domain check.  `e[0] != '0'` used to read
        // "off" / "false" / "no" as ON, i.e. the opposite of what the
        // operator wrote; `parse_env_int_strict` alone still accepted
        // any integer, so `2` or `-1` would enable the kernel.  The
        // documented domain is {0, 1} -- anything else is junk and
        // resolves to the default.
        if (!parse_env_int_strict(e, parsed)) return true;
        if (parsed != 0 && parsed != 1) return true;
        return parsed != 0;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8 = { "0", "1" } — cached, default ON.
//   Independent sub-switch for the hand-rolled AVX-512 VNNI int8
//   microkernel (`custom_kernel/ukernel/int8_microkernel.{hpp,cpp}`).
//   Lets deployments compare the DQ-INT8 fast path against the AOCL DLP
//   `aocl_gemm_s8s8s32obf16_sym_quant` reference without affecting
//   the bf16 CK path.
//
//   Cascade with the master `_CUSTOM_KERNEL` switch:
//     * `_CUSTOM_KERNEL=0`                       → both regimes off
//                                                  (every CK route falls
//                                                  back to standard DLP /
//                                                  BRGEMM dispatch).
//     * `_CUSTOM_KERNEL=1 && _CUSTOM_KERNEL_INT8=0` → bf16 CK on,
//                                                  int8 CK off (DQ-INT8
//                                                  N-tile calls fall back
//                                                  to AOCL DLP sym_quant).
//     * `_CUSTOM_KERNEL=1 && _CUSTOM_KERNEL_INT8=1` → both on.  DEFAULT.
//
//   Defaults ON so the master knob means one thing for every dtype: a
//   deployment that leaves `_CUSTOM_KERNEL` at its default gets the custom
//   kernel, not the custom kernel for bf16 and the DLP fallback for int8.
//   The split default was a standing trap -- an operator A/B-ing
//   `_CUSTOM_KERNEL` on a W8A8 model changed nothing about which int8
//   kernel ran and drew conclusions from two identical configurations.
//
//   This knob previously defaulted OFF on throughput grounds: the int8 CK
//   can be slower than the DLP sym-quant path on some shapes.  Do not flip
//   it back on that basis alone -- the two paths are not interchangeable on
//   accuracy for every W8A8 configuration, and the CK path is the one this
//   default is validated against.
//
//   Set `0` to route DQ-INT8 to the DLP sym-quant path, for a parity
//   bisection or to measure the throughput difference.
//
//   Implemented as a STATIC sub-knob: even if the master `_CUSTOM_KERNEL`
//   knob is on, eligibility code paths that route a DQ-INT8 call to
//   CK first consult this getter — when it returns false the call
//   routes to the AOCL DLP sym-quant fallback exactly as it does
//   under `_CUSTOM_KERNEL=0`.
//
//   Tests can pin the value via `s_grp_matmul_custom_kernel_int8_override`
//   (sentinel `-1` = no override).
inline bool get_grp_matmul_custom_kernel_int8() {
    const int ovr = test_api::s_grp_matmul_custom_kernel_int8_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_INT8");
        int parsed = 0;
        // Strict parse AND domain check, mirroring the master knob: junk
        // input resolves to the default (now ON) rather than being read as
        // a request to disable.
        if (!parse_env_int_strict(e, parsed)) return true;
        if (parsed != 0 && parsed != 1) return true;
        return parsed != 0;
    }();
    return v;
}

// ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_F16 = { "0", "1" } — cached, default ON.
//   Independent sub-switch for the hand-rolled native AVX-512-FP16
//   microkernel (`custom_kernel/ukernel/f16_microkernel.{hpp,cpp}`).
//   Lets deployments A/B the f16×f16→{f16,f32} fast path against the
//   AOCL DLP F16 reference without affecting the bf16 / int8 CK paths.
//
//   Cascade with the master `_CUSTOM_KERNEL` switch (same shape as the
//   `_INT8` sub-knob):
//     * `_CUSTOM_KERNEL=0`                      → every CK route off.
//     * `_CUSTOM_KERNEL=1 && _CUSTOM_KERNEL_F16=0` → bf16 CK on, f16 CK
//                                                 off (f16 N-tile calls
//                                                 fall back to AOCL DLP
//                                                 F16).  int8 is unaffected
//                                                 and stays ON unless
//                                                 `_INT8=0` turns it off.
//     * `_CUSTOM_KERNEL=1 && _CUSTOM_KERNEL_F16=1` → f16 CK on (default).
//
//   Even when ON, the f16 CK also needs `avx512f16_available()` true —
//   which folds TWO independent conditions with DIFFERENT routing when
//   they fail:
//     * TOOLCHAIN missing the FP16 intrinsics (built with GCC < 12, so
//       the compile-time gate `ZENDNNL_GRP_F16_CK_AVAILABLE == 0`) while
//       the CPU DOES have the ISA: `prepare_for_call` refuses the f16 CK
//       and the call falls back to AOCL DLP F16 (which runs natively on
//       the FP16-capable CPU) — same routing as `_CUSTOM_KERNEL_F16=0`.
//     * CPU missing the AVX-512-FP16 ISA (`get_avx512_f16_status()`
//       false): this is NOT a CK-vs-DLP fallback.  `group_matmul_direct`
//       hard-rejects the WHOLE call for any f16 operand up front with
//       `status_t::isa_unsupported` (the AOCL DLP F16 path cannot serve
//       it either), before the CK dispatch is even reached.  The caller
//       must handle the isa_unsupported status; there is no f16 GEMM on
//       such a host.
//
//   Tests can pin the value via `s_grp_matmul_custom_kernel_f16_override`
//   (sentinel `-1` = no override).
inline bool get_grp_matmul_custom_kernel_f16() {
    const int ovr = test_api::s_grp_matmul_custom_kernel_f16_override.load(
            std::memory_order_relaxed);
    if (ovr >= 0) return ovr != 0;
    static const bool v = []() {
        const char *e = std::getenv("ZENDNNL_GRP_MATMUL_CUSTOM_KERNEL_F16");
        int parsed = 0;
        // Strict parse AND domain check.  `e[0] != '0'` used to read
        // "off" / "false" / "no" as ON, i.e. the opposite of what the
        // operator wrote; `parse_env_int_strict` alone still accepted
        // any integer, so `2` or `-1` would enable the kernel.  The
        // documented domain is {0, 1} -- anything else is junk and
        // resolves to the default.
        if (!parse_env_int_strict(e, parsed)) return true;
        if (parsed != 0 && parsed != 1) return true;
        return parsed != 0;
    }();
    return v;
}

/// The variant whose pack serves a (K, N) weight of this dtype tuple, or
/// `kUnsupported`: the family sub-toggle and ISA, the pack width, the INT8
/// and W4A8 K alignment, the INT8 accumulation depth, and an AOCL-prepacked
/// weight the kernel cannot convert (every family but BF16, FP16 and static
/// U8xS8).  A non-positive K skips the K gates, and the pack-width gate also
/// needs a positive N.  The master knob and the per-call quant metadata are
/// the caller's.
custom_kernel::KernelVariant grp_matmul_ck_weight_variant(data_type_t src,
        data_type_t wei, data_type_t dst, bool dynamic_quant,
        data_type_t compute, bool static_u8s8, bool aocl_prepacked, int K,
        int N);

/// The EFFECTIVE per-call custom-kernel verdict: the master knob folded
/// with the family sub-toggle and the per-group disqualifier.
///
/// This exists because `get_grp_matmul_custom_kernel` reads the master
/// knob alone, while `prepare_n_tile_custom_kernel` additionally honours
/// `..._INT8` / `..._F16`, and `n_tile_execute` additionally refuses a per-group
/// `{G, N}` weight scale.  Any decision that has to AGREE with the kernel's
/// real engagement must use this, not the master knob.
///
/// The fused-MoE arena choice is the case that matters.  Granting a tight
/// arena on the master knob and then refusing the kernel on a sub-knob leaves
/// a tight destination with no tight-aware writer, and the split-halves
/// fallback collapses the whole call to the serial Sequential strategy -- so
/// `..._INT8=0` was an order-of-magnitude decode regression rather than the
/// "int8 on DLP, bf16 on CK" split it reads like.
///
/// A representative ACTIVE expert is used: slot 0 may be an inactive padding
/// placeholder whose dtypes were never filled in.
template <typename ParamsVec>
inline bool group_matmul_custom_kernel_candidate(const ParamsVec &params,
        const std::vector<int> &M, const std::vector<int> &N,
        grp_matmul_gated_act_t fused_act) {
    size_t rep = params.size();
    for (size_t i = 0; i < params.size(); ++i) {
        if (i < M.size() && M[i] <= 0) { continue; }
        rep = i;
        break;
    }
    if (rep >= params.size()) { return false; }

    const auto &d = params[rep].dtypes;
    if (!get_grp_matmul_custom_kernel()) { return false; }
    // Start from the dispatcher's positive support table. A master-knob-only
    // verdict licenses mixed in-place and tight arenas for unrelated families
    // (for example F32), even though prepare_for_call must reject them.
    const auto variant = custom_kernel::resolve_variant(
            d.src, d.wei, d.dst, params[rep].dynamic_quant, d.compute);
    if (variant == custom_kernel::KernelVariant::kUnsupported) { return false; }

    const bool is_int8_call = custom_kernel::is_int8_variant(variant);
    if (is_int8_call && !get_grp_matmul_custom_kernel_int8()) { return false; }
    const bool is_f16_call = custom_kernel::is_f16_variant(variant);
    if (is_f16_call && !get_grp_matmul_custom_kernel_f16()) { return false; }

    // A per-group `{G, N}` weight scale on ANY active expert disqualifies the
    // int8 custom kernel for the whole call -- it consumes per-token source
    // scales only.  W4A8 normally has `{G, N}` weights, so exempt that side,
    // but still reject a per-group `{M, G}` SOURCE scale: the s4 microkernel
    // also reads only one source scale per row.
    for (size_t i = 0; i < params.size(); ++i) {
        if (i >= M.size() || i >= N.size() || M[i] <= 0) { continue; }
        if (!params[i].postop_.empty()
                && !(params[i].dtypes.src == data_type_t::u8
                        && params[i].dtypes.wei == data_type_t::s8
                        && !params[i].dynamic_quant
                        && int8_ck_postops_supported(params[i], M[i], N[i]))) {
            return false;
        }
        // The static U8xS8 epilogue and a gated activation cannot both run:
        // the custom kernel has one epilogue slot and would otherwise drop
        // one operation. Keep this shared verdict aligned with
        // n_tile_supported so arena, cache, and execution policy agree.
        if (fused_act != grp_matmul_gated_act_t::none
                && !params[i].postop_.empty()) {
            return false;
        }
        const auto &expert_d = params[i].dtypes;
        if (custom_kernel::resolve_variant(expert_d.src, expert_d.wei,
                    expert_d.dst, params[i].dynamic_quant, expert_d.compute)
                != variant) {
            return false;
        }
        if (is_w4a8_config(params[i])) {
            const auto &ss = params[i].quant_params.src_scale;
            if (ss.dims.size() == 2 && ss.dims[1] > 1) { return false; }
            continue;
        }
        const auto &ws = params[i].quant_params.wei_scale;
        if (ws.dims.size() == 2 && ws.dims[0] > 1) { return false; }
    }
    return true;
}

/// Convenience overload for the ALGO executors.  Uses the fused-MoE hint
/// when one is published, and otherwise falls back to the caller's own
/// observed stride -- correct for a caller that owns its destination and
/// keeps one layout across prompt and decode.
template <typename ParamsVec>
inline int n_tile_decode_nr_align(const std::vector<int> &M,
        const std::vector<int> &N, const std::vector<int> &ldc,
        grp_matmul_gated_act_t act, const ParamsVec &params) {
    // Classify from the first ACTIVE expert, not slot 0.  An inactive slot
    // carries arbitrary placeholder strides, and both the dispatcher and
    // `n_tile_execute` pick their representative the same way -- reading slot 0
    // here instead could call a wide arena tight, warm alignment-2 keys, and
    // leave decode querying alignment-1 keys.  That reinstates the very
    // first-call reorder this alignment agreement exists to remove.
    size_t rep = M.size();
    for (size_t i = 0; i < M.size(); ++i) {
        if (M[i] > 0) {
            rep = i;
            break;
        }
    }
    // All-inactive (or no metadata): nothing to classify from, so keep the
    // backend default rather than inventing a verdict.
    const bool have_rep = rep < M.size() && rep < N.size() && rep < ldc.size();
    const bool observed_tight = have_rep && ldc[rep] < N[rep];
    const bool tight
            = tls_decode_arena_known ? tls_decode_arena_tight : observed_tight;
    return n_tile_decode_nr_align(act,
            group_matmul_custom_kernel_candidate(params, M, N, act), tight);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_CK_ENABLE_HPP
