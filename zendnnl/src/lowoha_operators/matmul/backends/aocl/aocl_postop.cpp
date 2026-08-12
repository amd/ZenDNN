/*******************************************************************************
# * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lowoha_operators/matmul/backends/aocl/aocl_postop.hpp"
#include "common/zendnnl_exceptions.hpp"
#include "common/zendnnl_global.hpp"
#include "lowoha_operators/matmul/lru_cache/lru_cache.hpp"
#include "lowoha_operators/matmul/lru_cache/zendnnl_key.hpp"
#include "memory/memory_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

using namespace zendnnl::error_handling;

namespace zendnnl {
namespace lowoha {
namespace matmul {
// =============================================================================
// Per-layer scratch capacities. The AOCL DLP API itself caps any single
// post-op chain at AOCL_DLP_MAX_POST_OPS (= 8) total seq_vector entries;
// chains beyond that are rejected by the AOCL DLP backend (see
// dependencies/aocldlp/include/classic/aocl_gemm_post_ops.h and the
// metadata->seq_length > AOCL_DLP_MAX_POST_OPS guard in
// dependencies/aocldlp/classic/frame/dlp_gemm_post_ops.c). The AOCL DLP
// backend itself uses fixed-size dlp_gemm_post_op[AOCL_DLP_MAX_POST_OPS]
// stack arrays in every kernel, so a variable-sized holder on our side
// would not unlock any new workloads.
//
// kMaxSeqOps mirrors that total cap: it bounds total_ops (user post-op
// chain + optional bias + INT8 scales + zp_comp), which is exactly what
// gets written into seq_vector and counted as seq_length. Each per-type
// array is sized to the same total cap because in the worst case the
// entire chain may be one type (e.g. 8 binary_add).
// =============================================================================
namespace {

constexpr std::size_t kMaxPostOps = AOCL_DLP_MAX_POST_OPS;
constexpr std::size_t kMaxSeqOps = kMaxPostOps;
constexpr std::size_t kMaxEltwise = kMaxPostOps;
constexpr std::size_t kMaxMatrixAdd = kMaxPostOps;
constexpr std::size_t kMaxMatrixMul = kMaxPostOps;
constexpr std::size_t kMaxBias = kMaxPostOps;
constexpr std::size_t kMaxScale = kMaxPostOps;

// Per-layer post-op metadata holder. One contiguous allocation owns the
// dlp_metadata_t plus every sub-buffer it can ever point at, so the LRU
// cache can free the entire holder with a single zendnnl_aligned_free on
// eviction (matching lru_cache_t's pointer-eviction contract).
//
// All members are trivial; the holder is allocated via zendnnl_aligned_alloc
// then zeroed with memset, and init_metadata_holder() pins the invariant sub-
// pointers (sf/zp/scl/etc.) and DLP_F32 defaults that the build path
// expects to be in place.
struct dlp_postop_metadata_holder_t {
    dlp_metadata_t metadata;

    DLP_POST_OP_TYPE seq_vector[kMaxSeqOps];
    dlp_post_op_eltwise eltwise[kMaxEltwise];
    // Stable per-eltwise alpha/beta storage. Build path copies parameterized
    // eltwise values here and wires e.algo.alpha/beta to these addresses; patch path
    // refreshes the values without touching the pointers, so e.algo.alpha
    // /beta stay valid across cache hits regardless of the caller's
    // matmul_params lifetime.
    float eltwise_alpha[kMaxEltwise];
    float eltwise_beta[kMaxEltwise];
    dlp_post_op_matrix_add matrix_add[kMaxMatrixAdd];
    dlp_sf_t matrix_add_sf[kMaxMatrixAdd];
    dlp_post_op_matrix_mul matrix_mul[kMaxMatrixMul];
    dlp_sf_t matrix_mul_sf[kMaxMatrixMul];
    dlp_post_op_bias bias[kMaxBias];
    dlp_sf_t zp_comp_bias_sf;
    int32_t zp_comp_neg_src_zp;
    dlp_scale_t scale[kMaxScale];
    dlp_sf_t scale_sf[kMaxScale];
    dlp_zp_t scale_zp[kMaxScale];

    // Unified A-matrix quant op (new AOCL DLP API). Serves two mutually-
    // exclusive paths:
    //   - bf16/f32 -> s8 on-the-fly quant: quant_scale_factors carries the
    //     forward inverse-scale, dequant_scale_factors the src (post) scale,
    //     zero_point the src zp (NULL == symmetric). Replaces the old
    //     a_pre_quant + a_post_quant pair.
    //   - s8 x s8 symmetric quant: only dequant_scale_factors (the a-scale)
    //     is wired; quant_scale_factors / zero_point stay NULL. Replaces the
    //     a_scl half of the old post_op_grp.
    dlp_quant_op_t a_quant_op;
    dlp_qparam_t a_quant_scl; // quant_scale_factors (fwd inverse-scale)
    dlp_qparam_t a_dequant_scl; // dequant_scale_factors (src / a scale)
    dlp_qparam_t a_quant_zp; // zero_point (asymmetric src)
    float a_quant_inv_scale; // scalar backing store for a_quant_scl.data
    // Heap-owned inverse-scale buffer for the BF16/INT8 per-token-symmetric
    // path (length = M floats, M = src_scale_nelems for that path). nullptr
    // for every other path. Owned by cleanup_dlp_post_op() — see is_per_call
    // below and the function comment in this file.
    float *a_quant_inv_scales_dyn;

    // Unified B-matrix quant op (new AOCL DLP API). Serves two mutually-
    // exclusive paths:
    //   - WOQ (bf16 x s4/u4): dequant_scale_factors carries the weight scale,
    //     zero_point the weight zp (u4 only; NULL for s4). Replaces the old
    //     pre_ops.
    //   - s8 x s8 symmetric quant: dequant_scale_factors carries the weight
    //     scale. Replaces the b_scl half of the old post_op_grp.
    dlp_quant_op_t b_quant_op;
    dlp_qparam_t b_dequant_scl; // dequant_scale_factors (weight scale)
    dlp_qparam_t b_quant_zp; // zero_point (WOQ asymmetric u4)

    // Set true by the build path when the layer has no post-op metadata
    // (plain matmul with no chain). Cache hits read this and short-circuit
    // to nullptr, preserving the original per-call return semantics across
    // every subsequent call for this key.
    bool no_metadata;

    // True when this holder is a per-call allocation (currently only the
    // BF16/INT8 per-token-symmetric path) whose lifetime is owned by
    // cleanup_dlp_post_op() at the kernel call site rather than by the
    // LRU cache. False for cached holders; cleanup_dlp_post_op is a no-op
    // for those.
    bool is_per_call;
};

// Pin the metadata's invariant sub-pointer fields (the ones the build path
// expects to already point into the holder's embedded sub-buffers) and the
// DLP_F32 defaults. Called once per holder, immediately after the memset.
void init_metadata_holder(dlp_postop_metadata_holder_t *h) {
    for (std::size_t i = 0; i < kMaxMatrixAdd; ++i) {
        h->matrix_add[i].sf = &h->matrix_add_sf[i];
        h->matrix_add_sf[i].scale_factor_type = DLP_F32;
    }
    for (std::size_t i = 0; i < kMaxMatrixMul; ++i) {
        h->matrix_mul[i].sf = &h->matrix_mul_sf[i];
        h->matrix_mul_sf[i].scale_factor_type = DLP_F32;
    }
    for (std::size_t i = 0; i < kMaxScale; ++i) {
        h->scale[i].sf = &h->scale_sf[i];
        h->scale[i].zp = &h->scale_zp[i];
    }
    // Pin the unified quant-op sub-pointers to the holder's embedded qparam
    // slots. quant_scale_factors / dequant_scale_factors always point at
    // their slots; zero_point is (re)pointed per build — set to the embedded
    // slot for asymmetric paths and NULL for symmetric ones. The scalar
    // forward inverse-scale backing store is pinned into a_quant_scl.data.
    h->a_quant_op.quant_scale_factors = &h->a_quant_scl;
    h->a_quant_op.dequant_scale_factors = &h->a_dequant_scl;
    h->a_quant_op.zero_point = &h->a_quant_zp;
    h->a_quant_scl.data = &h->a_quant_inv_scale;
    h->a_quant_scl.stor_type = DLP_F32;
    h->b_quant_op.dequant_scale_factors = &h->b_dequant_scl;
    h->b_quant_op.zero_point = &h->b_quant_zp;
    h->zp_comp_bias_sf.scale_factor = &h->zp_comp_neg_src_zp;
    h->zp_comp_bias_sf.scale_factor_len = 1;
    h->zp_comp_bias_sf.scale_factor_type = DLP_S32;
    h->zp_comp_bias_sf.scale_factor_dim = DLP_PARAM_DIM_PER_TENSOR;
}

// Map zendnnl data_type_t to DLP_TYPE. File-scope so both the cold-path
// build inside create_dlp_post_op() and the hit-path patch_mutable_fields()
// can use it (the dynamic-quant zp type is refreshed per hit, which needs
// this mapping outside the build path).
auto to_dlp_type = [](data_type_t dt) -> DLP_TYPE {
    switch (dt) {
        case data_type_t::f32: return DLP_F32;
        case data_type_t::bf16: return DLP_BF16;
        case data_type_t::s32: return DLP_S32;
        case data_type_t::f16: return DLP_F16;
        case data_type_t::s8: return DLP_S8;
        case data_type_t::u8: return DLP_U8;
        // WOQ weight types: b_quant_op->src_type must carry the true 4-bit
        // width or DLP reads the packed nibbles as a wider type.
        case data_type_t::s4: return DLP_S4;
        case data_type_t::u4: return DLP_U4;
        default: return DLP_F32;
    }
};

// Infer the SCALE post-op granularity (scale_factor_dim) from the scale
// vector length. The AOCL DLP SCALE post-op now *requires* the caller to
// set sf->scale_factor_dim explicitly (PER_TENSOR / PER_CHANNEL / PER_TOKEN)
// and rejects DLP_PARAM_DIM_INVALID; it also strictly validates that the
// length matches the chosen dim (len==1 for PER_TENSOR, len==n for
// PER_CHANNEL, len==m for PER_TOKEN). See dlp_gemm_post_ops.c and the
// scale_factor_dim contract in aocl_gemm_post_ops.h.
//
// len == 1            -> PER_TENSOR
// len == M (rows)     -> PER_TOKEN   (source / per-token activations)
// len == N (columns)  -> PER_CHANNEL (weight / dst / row-broadcast operands)
//
// vec_dim selects the granularity for the non-scalar case and disambiguates
// the M == N corner case (e.g. PER_TOKEN for source scales, PER_CHANNEL for
// weight / dst / binary_mul broadcast scales).
auto infer_scale_dim
        = [](std::size_t len, int64_t M, int64_t N,
                  DLP_PARAM_DIM_TYPE vec_dim) -> DLP_PARAM_DIM_TYPE {
    if (len == 1) { return DLP_PARAM_DIM_PER_TENSOR; }
    if (vec_dim == DLP_PARAM_DIM_PER_TOKEN
            && len == static_cast<std::size_t>(M)) {
        return DLP_PARAM_DIM_PER_TOKEN;
    }
    if (vec_dim == DLP_PARAM_DIM_PER_CHANNEL
            && len == static_cast<std::size_t>(N)) {
        return DLP_PARAM_DIM_PER_CHANNEL;
    }
    // Fall back to the caller-requested vector granularity; DLP will reject a
    // genuine (dim, len) mismatch, surfacing a misconfigured scale loudly.
    return vec_dim;
};

} // namespace

// Per-call teardown for create_dlp_post_op()'s return value.
//
// Cached holders (the common path) are owned by the per-thread LRU cache,
// which evicts them with zendnnl_aligned_free; this function is a no-op for them.
//
// Per-call holders (currently only the BF16/INT8 per-token-sym path) are
// flagged with is_per_call=true during build and are not added to the
// cache. For those, this function releases the heap-owned inverse-scale
// buffer and then the holder itself, in that order.
//
// The metadata pointer is the first field of dlp_postop_metadata_holder_t
// (see the struct layout above), so reinterpret_cast recovers the holder
// address. Safe to call with nullptr — matches create_dlp_post_op()'s
// no-metadata return contract.
void cleanup_dlp_post_op(dlp_metadata_t *metadata) {
    if (!metadata) { return; }
    auto *h = reinterpret_cast<dlp_postop_metadata_holder_t *>(metadata);
    if (!h->is_per_call) { return; }
    std::free(h->a_quant_inv_scales_dyn);
    // Holder is zendnnl_aligned_alloc'd in create_dlp_post_op; release it with
    // the matching wrapper (_aligned_free on Windows).
    zendnnl_aligned_free(h);
}

namespace {

// Per-thread LRU cache of post-op metadata holders.
//
// Keyed by Key_matmul + extra_input_hash (carrying our postop_signature),
// so layer identity is captured by the key — non-deterministic execution
// orders (MoE, speculative decoding, conditional skip, early exit) are
// correct by construction: every call hashes its own key and looks up
// the matching holder.
//
// Per-thread (thread_local) because the holder's metadata fields are
// mutated per-call by patch_mutable_fields (per-call binary_add operand
// pointers, CLIP bounds, dynamic-quant inverse scale, etc.); a shared
// holder would race across concurrent inference threads. Per-thread also
// avoids the contention of a global mutex on the matmul hot path.
//
// Capacity comes from matmul_config_t::lru_cache_capacity (default
// uint32_max, i.e. eviction disabled — same default as the existing weight
// caches). On thread exit, the cache's destructor evict()'s every holder,
// freeing each via zendnnl_aligned_free.
lru_cache_t<Key_matmul, dlp_postop_metadata_holder_t *> &
get_postop_metadata_cache() {
    thread_local lru_cache_t<Key_matmul, dlp_postop_metadata_holder_t *> c;
    return c;
}

// Compute the structural signature folded into Key_matmul::extra_input_hash
// alongside weight_ptr/N/K/algo. This captures every input that drives
// build-path branches whose results are baked into cached metadata but
// NOT refreshed by patch_mutable_fields:
//   - ordered post-op type sequence
//   - dtype config (src/wei/dst/bias)
//   - presence of each quant buffer
//   - zp_comp_ndim
//   - bias presence
//   - for binary_add/binary_mul: po.dtype (drives m.stor_type),
//     po.leading_dim (drives m.ldm), and po.dims (drives whether the cold
//     path maps the operand to BIAS/SCALE broadcast vs MATRIX_ADD/MUL) —
//     collisions on these between weight-tied layers would silently produce
//     wrong results or segfaults because the patch path only refreshes
//     mutable pointers into the slot type chosen at build time.
//   - source and weight scale granularity. Per-tensor, per-token/channel,
//     and per-group scales produce different quant-op topology/outer_dim
//     values that patch_mutable_fields does not refresh.
//
// Intentionally NOT in the signature:
//   - po.alpha/po.beta (parameterized eltwise values): patched per-call via holder-owned
//     eltwise_alpha/eltwise_beta floats.
//   - po.buff (binary operand pointer): patched per-call.
//   - M: dynamic batch sizes must not invalidate per-layer cache hits.
//   - scale element counts within the same granularity (the per-call lengths
//     and group_size are refreshed by patch_mutable_fields).
std::size_t compute_postop_signature(const matmul_params &lowoha_param,
        const matmul_data_types &dtypes, int zp_comp_ndim, const void *bias,
        bool is_w4a8, int M, int N) {
    std::size_t sig = 0;
    for (const auto &po : lowoha_param.postop_) {
        sig = sig * 31u + static_cast<std::size_t>(po.po_type);
        if (po.po_type == post_op_type_t::binary_add
                || po.po_type == post_op_type_t::binary_mul) {
            sig = sig * 31u + static_cast<std::size_t>(po.dtype);
            sig = sig * 31u
                    + static_cast<std::size_t>(
                            static_cast<uint32_t>(po.leading_dim));
            sig = sig * 31u + po.dims.size();
            for (int64_t d : po.dims) {
                sig = sig * 31u + static_cast<std::size_t>(d);
            }
            // A {1, N} binary_mul is routed to SCALE (broadcast) only when
            // M > 1, otherwise to MATRIX_MUL (see binary_mul_uses_scale_path).
            // Fold the routing class in so M == 1 and M > 1 metadata stay
            // distinct.
            if (po.po_type == post_op_type_t::binary_mul && po.dims.size() == 2
                    && po.dims[0] == 1) {
                sig = sig * 31u + (M > 1 ? 1u : 0u);
            }
        }
    }
    sig = sig * 31u + static_cast<std::size_t>(dtypes.src);
    sig = sig * 31u + static_cast<std::size_t>(dtypes.wei);
    sig = sig * 31u + static_cast<std::size_t>(dtypes.dst);
    sig = sig * 31u + static_cast<std::size_t>(dtypes.bias);
    sig = sig * 31u + (lowoha_param.quant_params.src_scale.buff ? 1u : 0u);
    sig = sig * 31u + (lowoha_param.quant_params.src_zp.buff ? 1u : 0u);
    sig = sig * 31u + (lowoha_param.quant_params.wei_scale.buff ? 1u : 0u);
    sig = sig * 31u + (lowoha_param.quant_params.wei_zp.buff ? 1u : 0u);
    sig = sig * 31u + (lowoha_param.quant_params.dst_scale.buff ? 1u : 0u);
    sig = sig * 31u + (lowoha_param.quant_params.dst_zp.buff ? 1u : 0u);
    sig = sig * 31u + static_cast<std::size_t>(zp_comp_ndim);
    sig = sig * 31u + (bias ? 1u : 0u);
    // Quant scale granularity is structural metadata. In particular,
    // per-token {M,1} and per-group {M,G} source scales both have more than
    // one element, but require different a_quant_op outer_dim values. The
    // same applies to per-channel {1,N} vs per-group {G,N} weight scales.
    // Hash the category rather than the exact element count so dynamic M and
    // group-size changes can still hit and use the mutable-field refresh path.
    const auto scale_layout = [](size_t nelems, int outer_dim) {
        if (nelems <= 1) return 0u; // per-tensor
        if (nelems == static_cast<size_t>(outer_dim)) {
            return 1u; // per-token (A) / per-channel (B)
        }
        return 2u; // per-group
    };
    sig = sig * 31u
            + scale_layout(
                    get_num_elements(lowoha_param.quant_params.src_scale.dims),
                    M);
    sig = sig * 31u
            + scale_layout(
                    get_num_elements(lowoha_param.quant_params.wei_scale.dims),
                    N);
    // W4A8 forces the sym-quant wiring even at a collapsed (1-element)
    // source scale; a genuine s8s8 layer with the same collapsed scale on
    // the same key wires the non-sym path, so the two must not share a
    // holder.
    sig = sig * 31u + (is_w4a8 ? 1u : 0u);
    return sig;
}

} // namespace

// Free every holder owned by the calling thread's post-op metadata cache.
// Intended for use between gtest cases (each test is a fresh "model" with
// new weight pointers); not meant to be called on the matmul hot path.
void clear_aocl_postop_metadata_cache() {
    get_postop_metadata_cache().clear();
}

std::size_t get_aocl_postop_metadata_cache_size() {
    return get_postop_metadata_cache().get_size();
}

// A {1, N} binary_mul only needs the AOCL DLP SCALE post-op (a per-N multiply,
// out[:, j] *= vec[j]) when it must broadcast a single row across MULTIPLE
// output rows, i.e. when M > 1. When M == 1 the {1, N} operand is a dense
// single-row matrix and is routed to MATRIX_MUL instead. This decision is
// independent of the operand dtype.
//
// Why avoid SCALE for the M == 1 case: the DLP SCALE kernel
// (POST_OPS_DOWNSCALE_*) loads its scale vector with _mm512_loadu_ps((float*)
// scale_factor + j), i.e. it always reads f32 and ignores scale_factor_type,
// so a bf16/f16 operand is byte-reinterpreted as f32 and produces huge/inf/nan
// output. MATRIX_MUL dispatches on stor_type and widens bf16/f16 to f32
// correctly. The fused WOQ linear*mul+add path produces exactly this case at
// M == 1 (the dense {M, N} mul operand is {1, N}); see ZENAI-3714 / ZENAI-3717.
//
// NOTE: this predicate depends on M, which is otherwise excluded from the
// routing decision. compute_postop_signature() folds the same M > 1 bit into
// the signature for {1, N} binary_mul operands so that M == 1 and M > 1 calls
// for the same layer get distinct cached metadata.
static inline bool binary_mul_uses_scale_path(
        const matmul_post_op &po, int64_t n, int64_t M) {
    return po.po_type == post_op_type_t::binary_mul && po.dims.size() == 2
            && po.dims[0] == 1 && static_cast<int64_t>(po.dims[1]) == n
            && M > 1;
}

// Fill DLP post-op array entries (eltwise, binary_add, binary_mul) for one
// matmul.
//
// Precondition: the caller (create_dlp_post_op) has ensured that the
// union of (zp_comp + INT8 scales + bias + user post-ops) fits within
// the holder's per-type arrays — all sized to kMaxSeqOps
// (== AOCL_DLP_MAX_POST_OPS). This function performs NO bounds checking
// on the per-array indices it bumps: seq_vector, eltwise, matrix_add,
// matrix_mul, and (for CLIP only) the holder's eltwise_alpha /
// eltwise_beta float arrays. Overflowing any of them — for example by
// passing a longer chain than the AOCL DLP cap supports — would corrupt
// the next-adjacent holder field; the responsibility for refusing such
// a chain sits in create_dlp_post_op (the only caller of this helper),
// not here.
//
// CLIP bounds (po.alpha / po.beta) are copied into holder-owned float
// storage and dlp_post_op_eltwise::algo.alpha/beta are wired to those
// holder addresses. This avoids the lifetime hazard of caching pointers
// into the caller's matmul_post_op vector across cache hits; the patch
// path refreshes the values through the same holder addresses on every
// hit.
static void setup_dlp_postops(dlp_metadata_t *md,
        dlp_postop_metadata_holder_t *h,
        const std::vector<matmul_post_op> &postops, int &op_index,
        int &eltwise_index, int &matrix_add_index, int &matrix_mul_index,
        int &bias_index, int &scale_index, int n_cols, int M) {
    // Write one ELTWISE slot. stor_type describes the storage of alpha/beta;
    // when the op carries neither, leave the field at its zero-init default
    // (DLP_INVALID) — the AOCL DLP kernel only consults stor_type when at
    // least one of alpha/beta is present.
    auto put_eltwise = [&](DLP_ELT_ALGO_TYPE algo, void *alpha = nullptr,
                               void *beta = nullptr) {
        auto &e = md->eltwise[eltwise_index++];
        md->seq_vector[op_index++] = ELTWISE;
        e.algo.algo_type = algo;
        if (alpha || beta) { e.algo.stor_type = DLP_F32; }
        e.algo.alpha = alpha;
        e.algo.beta = beta;
        e.sf = nullptr;
    };

    // Write one MATRIX_{ADD,MUL} slot.
    auto put_matrix = [&](auto *arr, int &idx, DLP_POST_OP_TYPE tag,
                              const matmul_post_op &po) {
        auto &m = arr[idx++];
        md->seq_vector[op_index++] = tag;
        m.matrix = po.buff;
        m.ldm = po.leading_dim;
        m.stor_type = to_dlp_type(po.dtype);
        // sf is pre-allocated by the caller; matrix add/mul operands have no
        // per-element scale, so point at the shared ONE_F32 constant.
        m.sf->scale_factor = get_void_ptr(ONE_F32);
        m.sf->scale_factor_len = 1;
    };

    for (const auto &po : postops) {
        switch (po.po_type) {
            case post_op_type_t::relu: put_eltwise(RELU); break;
            case post_op_type_t::leaky_relu: {
                const std::size_t i = static_cast<std::size_t>(eltwise_index);
                h->eltwise_alpha[i] = po.alpha;
                put_eltwise(PRELU, &h->eltwise_alpha[i]);
                break;
            }
            case post_op_type_t::gelu_tanh: put_eltwise(GELU_TANH); break;
            case post_op_type_t::gelu_erf: put_eltwise(GELU_ERF); break;
            case post_op_type_t::sigmoid: put_eltwise(SIGMOID); break;
            case post_op_type_t::swish: {
                const std::size_t i = static_cast<std::size_t>(eltwise_index);
                h->eltwise_alpha[i] = po.alpha;
                put_eltwise(SWISH, &h->eltwise_alpha[i]);
                break;
            }
            case post_op_type_t::tanh: put_eltwise(TANH); break;
            // clip(x; lo, hi): bounds from matmul_post_op::alpha (lower), ::beta
            // (upper). Copy the values into holder-owned floats and wire algo.
            // alpha/beta to those holder addresses so the pointers stay valid
            // across cache hits even after the caller's matmul_post_op vector is
            // destroyed.
            case post_op_type_t::clip: {
                const std::size_t i = static_cast<std::size_t>(eltwise_index);
                h->eltwise_alpha[i] = po.alpha;
                h->eltwise_beta[i] = po.beta;
                put_eltwise(CLIP, &h->eltwise_alpha[i], &h->eltwise_beta[i]);
                break;
            }
            case post_op_type_t::mish: put_eltwise(MISH); break;
            case post_op_type_t::binary_add:
                if (po.dims.size() == 2 && po.dims[0] == 1
                        && po.dims[1] == static_cast<int>(n_cols)) {
                    md->seq_vector[op_index++] = BIAS;
                    md->bias[bias_index].bias = po.buff;
                    md->bias[bias_index].stor_type = to_dlp_type(po.dtype);
                    md->bias[bias_index].sf = nullptr;
                    md->bias[bias_index].zp = nullptr;
                    md->bias[bias_index].bias_len = po.dims[1];
                    bias_index++;
                } else {
                    put_matrix(
                            md->matrix_add, matrix_add_index, MATRIX_ADD, po);
                }
                break;
            case post_op_type_t::binary_mul:
                // A {1, N} operand broadcasts a per-column vector across output
                // rows only when M > 1 -> DLP SCALE (per-channel multiply). When
                // M == 1 it is a dense single-row operand -> MATRIX_MUL. This is
                // dtype-independent; see binary_mul_uses_scale_path.
                if (binary_mul_uses_scale_path(po, n_cols, M)) {
                    md->seq_vector[op_index++] = SCALE;
                    dlp_scale_t &sc = md->scale[scale_index++];
                    sc.sf->scale_factor = const_cast<void *>(po.buff);
                    sc.sf->scale_factor_len = n_cols;
                    sc.sf->scale_factor_type = to_dlp_type(po.dtype);
                    // Row-broadcast {1, N}: one scale per output column -> per-channel.
                    sc.sf->scale_factor_dim = DLP_PARAM_DIM_PER_CHANNEL;
                    static int32_t bmul_bcast_zp = 0;
                    sc.zp->zero_point = &bmul_bcast_zp;
                    sc.zp->zero_point_type = DLP_S32;
                    sc.zp->zero_point_len = 1;
                } else {
                    put_matrix(
                            md->matrix_mul, matrix_mul_index, MATRIX_MUL, po);
                }
                break;
            default:
                // Skip unsupported post-ops
                break;
        }
    }
}

// Helper function to set up WOQ (Weight-Only Quantization) metadata.
// b_quant_op->dequant_scale_factors / zero_point are pre-wired by
// init_metadata_holder() at the holder's embedded buffers; this function
// fills them in.
static void setup_woq_pre_ops(dlp_metadata_t *dlp_metadata,
        dlp_postop_metadata_holder_t *h, const matmul_params &lowoha_param,
        int64_t K, int64_t N, data_type_t wei_dt) {
    // WOQ (bf16 x s4/u4) now flows through the unified b_quant_op: the low-
    // precision weight is dequantized up to the BF16 compute domain using
    // dequant_scale_factors (+ zero_point for asymmetric u4). Replaces the
    // old dlp_metadata->pre_ops path.
    dlp_metadata->b_quant_op = &h->b_quant_op;

    const auto &wei_scale = lowoha_param.quant_params.wei_scale;
    const auto &wei_zp = lowoha_param.quant_params.wei_zp;

    // DEQUANTIZE: B is expanded from s4/u4 up to bf16 before accumulation, so
    // DLP routes this metadata through the pre-op list. The pre-op translator
    // rejects any other kind (dlp_gemm_translate_to_pre_ops_list).
    dlp_metadata->b_quant_op->quant_op_kind = DLP_QUANT_OP_DEQUANTIZE;
    dlp_metadata->b_quant_op->src_type = to_dlp_type(wei_dt);
    dlp_metadata->b_quant_op->dst_type = DLP_BF16;
    // WOQ never forward-quantizes B; only the dequant direction is used.
    dlp_metadata->b_quant_op->quant_scale_factors = nullptr;

    // Determine group_size and granularity from wei_scale.dims:
    //   - Per-tensor:  dims = {} or {1}     → group_size = K, PER_TENSOR
    //   - Per-channel: dims = {N} or {1, N} → group_size = K, PER_CHANNEL
    //   - Per-group:   dims = {G, N}        → group_size = K / G, PER_GROUP
    int64_t group_size = K; // Default per-tensor / per-channel (single K group)
    DLP_PARAM_DIM_TYPE outer_dim = DLP_PARAM_DIM_PER_TENSOR;
    const auto &dims = wei_scale.dims;
    const size_t scale_len = get_num_elements(wei_scale.dims);
    if (!dims.empty() && !(dims.size() == 1 && dims[0] == 1)) {
        if (dims.size() == 2 && dims[1] == N && dims[0] > 1) {
            group_size = K / dims[0]; // Per-group: dims = {G, N}
            outer_dim = DLP_PARAM_DIM_PER_GROUP;
        } else {
            outer_dim = DLP_PARAM_DIM_PER_CHANNEL; // dims = {N} or {1, N}
        }
    }

    // Validation: group_size must divide K evenly
    if (K % group_size != 0) {
        log_error("WOQ: group_size (", group_size, ") must divide K (", K,
                ") evenly");
        group_size = K; // Fallback to per-tensor
        outer_dim = DLP_PARAM_DIM_PER_TENSOR;
    }
    dlp_metadata->b_quant_op->group_size = static_cast<int>(group_size);

    // Weight dequant scale (dequant_scale_factors slot pre-wired by
    // init_metadata_holder).
    dlp_metadata->b_quant_op->dequant_scale_factors->data
            = const_cast<void *>(wei_scale.buff);
    dlp_metadata->b_quant_op->dequant_scale_factors->len = scale_len;
    dlp_metadata->b_quant_op->dequant_scale_factors->stor_type
            = (wei_scale.dt == data_type_t::bf16) ? DLP_BF16 : DLP_F32;
    dlp_metadata->b_quant_op->dequant_scale_factors->outer_dim = outer_dim;

    // Weight zero-point (asymmetric u4 only). zero_point slot is pre-wired by
    // init_metadata_holder; the explicit re-assignment documents the u4 path
    // and the else branch nulls it for symmetric (s4) weights.
    if (wei_dt == data_type_t::u4) {
        dlp_metadata->b_quant_op->zero_point = &h->b_quant_zp;
        dlp_metadata->b_quant_op->zero_point->data
                = const_cast<void *>(wei_zp.buff);
        dlp_metadata->b_quant_op->zero_point->len
                = get_num_elements(wei_zp.dims);
        dlp_metadata->b_quant_op->zero_point->stor_type
                = (wei_zp.dt == data_type_t::s8) ? DLP_S8 : DLP_BF16;
        dlp_metadata->b_quant_op->zero_point->outer_dim = outer_dim;
    } else {
        dlp_metadata->b_quant_op->zero_point = nullptr;
    }

    apilog_info("WOQ: scale_len=", scale_len, ", group_size=", group_size);
}

// Patch the per-call mutable fields onto a previously-built cached metadata.
// Mirrors the build path's index-assignment order so the patch indices for
// matrix_add/matrix_mul/bias/eltwise align byte-for-byte with what the
// build wrote.
//
// Mutable field set:
//   - bias[0].bias OR matrix_add[0].matrix  : per-call zp_comp_acc buffer
//                                             (depending on zp_comp_ndim)
//   - bias[bias_index].bias                 : per-call user bias pointer
//   - matrix_add[i].matrix                  : per-call binary_add addend
//   - matrix_mul[i].matrix                  : per-call binary_mul operand
//   - eltwise[i] CLIP alpha/beta values     : per-call CLIP bounds, refreshed
//                                             through holder-owned float
//                                             storage (eltwise[i].algo.alpha
//                                             /beta pointers were wired to
//                                             those holder floats at build
//                                             time and remain stable)
//   - a_quant_op->quant_scale_factors->data : 1/src_scale[0] for dyn-quant
//   - a_quant_op->dequant_scale_factors->{data, len, stor_type}
//                                           : per-call src_scale buffer
//   - a_quant_op->zero_point->{data, len, stor_type}
//                                           : per-call src_zp buffer for
//                                             dyn-quant asymmetric path
//                                             (reorder_quant_buffers_t owns
//                                             this and frees it at scope
//                                             exit, so the cold-path-cached
//                                             pointer is dangling on hits)
//   - a_quant_op->{group_size, dequant_scale_factors->{data, len, stor_type}}
//                                           : M-dependent sym-quant source
//                                             metadata
//   - b_quant_op->{group_size, dequant_scale_factors->{data, len, stor_type}}
//                                           : per-call sym-quant weight
//                                             metadata
//   - b_quant_op->dequant_scale_factors->{data, len, stor_type}
//                                           : per-call WOQ weight-scale buffer
//   - b_quant_op->zero_point->{data, len, stor_type}
//                                           : per-call WOQ-asymmetric (u4)
//                                             weight-zp buffer
//   - scale[i].sf->{scale_factor,
//                   scale_factor_len,
//                   scale_factor_type}       : per-call src/wei/dst_scale
//                                              buffers for non-sym INT8.
//                                              Slot indices derived from the
//                                              same presence booleans the
//                                              cold path uses (safe-by-
//                                              construction because
//                                              compute_postop_signature folds
//                                              those same booleans into the
//                                              cache key, so identical
//                                              presence patterns map to the
//                                              same key)
//   - scale[dst].zp->{zero_point,
//                     zero_point_len,
//                     zero_point_type}       : per-call dst_zp buffer for
//                                              non-sym INT8 with asymmetric
//                                              dst; src/wei dummy-zps point
//                                              at process-lifetime statics
//                                              and need no refresh
//
// All other fields of *md are immutable across the holder's lifetime —
// they were written once during the cold-path build and remain valid for
// every subsequent cache hit.
static void patch_mutable_fields(dlp_metadata_t *md,
        dlp_postop_metadata_holder_t *h, const matmul_params &lowoha_param,
        const void *bias, const matmul_data_types &dtypes, int M, int N, int K,
        int32_t *zp_comp_acc, int zp_comp_ndim, bool is_w4a8,
        const int32_t *reorder_colsum = nullptr, int32_t neg_src_zp = 0) {
    // NOTE: keep these flag definitions in lockstep with the build path in
    // create_dlp_post_op(). The hit path mirrors the build path's per-call
    // mutable-field updates, so any divergence in classification (especially
    // is_bf16_f32_per_token_sym vs is_non_quant_src_int8) would silently
    // corrupt scale_factor on cache hits.
    bool is_int8 = dtypes.wei == data_type_t::s8;
    size_t src_scale_nelems
            = get_num_elements(lowoha_param.quant_params.src_scale.dims);
    const bool is_bf16_f32_per_token_sym = is_int8
            && (dtypes.src == data_type_t::bf16
                    || dtypes.src == data_type_t::f32)
            && !lowoha_param.quant_params.src_zp.buff && src_scale_nelems > 1
            && src_scale_nelems == static_cast<size_t>(M)
            && (dtypes.dst == data_type_t::f32
                    || dtypes.dst == data_type_t::bf16);
    bool is_non_quant_src_int8 = (dtypes.src == data_type_t::bf16
                                         || dtypes.src == data_type_t::f32)
            && is_int8 && !is_bf16_f32_per_token_sym;
    bool is_sym_quant = is_int8 && dtypes.src == data_type_t::s8
            && !lowoha_param.quant_params.src_zp.buff
            && (src_scale_nelems > 1 || is_w4a8)
            && (dtypes.dst == data_type_t::f32
                    || dtypes.dst == data_type_t::bf16);

    int matrix_add_index = 0;
    int matrix_mul_index = 0;
    int bias_index = 0;

    // INT8 zero-point compensation must be patched first to mirror the
    // build-path order; it consumes bias[0] or matrix_add[0].
    const bool colsum_mode
            = reorder_colsum && !zp_comp_acc && zp_comp_ndim == 0;
    if (colsum_mode) {
        md->bias[bias_index].bias = const_cast<int32_t *>(reorder_colsum);
        h->zp_comp_neg_src_zp = neg_src_zp;
        bias_index++;
    } else if (zp_comp_ndim == 1 && zp_comp_acc) {
        md->bias[bias_index].bias = zp_comp_acc;
        bias_index++;
    } else if (zp_comp_ndim == 2 && zp_comp_acc) {
        md->matrix_add[matrix_add_index].matrix = zp_comp_acc;
        matrix_add_index++;
    }

    // User bias (BIAS slot follows zp_comp's BIAS slot when both present).
    if (bias) {
        md->bias[bias_index].bias = const_cast<void *>(bias);
        bias_index++;
    }

    // Walk lowoha_param.postop_ in build-path order to refresh per-call
    // mutable fields:
    //   - parameterized eltwise values (Leaky ReLU, Swish, CLIP): rewritten
    //     through the holder's eltwise_alpha/beta floats. The
    //     eltwise[i].algo.alpha/beta pointers were wired to these holder
    //     addresses at build time and are stable across cache hits.
    //   - binary_add / binary_mul operand pointers (dense M×N uses
    //     matrix_add/matrix_mul; row-broadcast {1, N} uses BIAS/SCALE,
    //     matching setup_dlp_postops).
    // Other eltwise types have no per-call mutable fields. We still advance
    // eltwise_index for them so it stays in lockstep with the build path.
    int scale_index = 0;
    if (is_int8 && lowoha_param.quant_params.src_scale.buff
            && !is_non_quant_src_int8 && !is_sym_quant
            && !is_bf16_f32_per_token_sym) {
        scale_index++;
    }
    if (is_int8 && lowoha_param.quant_params.wei_scale.buff && !is_sym_quant) {
        scale_index++;
    }

    std::size_t eltwise_index = 0;
    for (const auto &po : lowoha_param.postop_) {
        switch (po.po_type) {
            case post_op_type_t::leaky_relu:
            case post_op_type_t::swish:
                h->eltwise_alpha[eltwise_index] = po.alpha;
                ++eltwise_index;
                break;
            case post_op_type_t::clip:
                h->eltwise_alpha[eltwise_index] = po.alpha;
                h->eltwise_beta[eltwise_index] = po.beta;
                ++eltwise_index;
                break;
            case post_op_type_t::relu:
            case post_op_type_t::gelu_tanh:
            case post_op_type_t::gelu_erf:
            case post_op_type_t::sigmoid:
            case post_op_type_t::tanh:
            case post_op_type_t::mish: ++eltwise_index; break;
            case post_op_type_t::binary_add:
                if (po.dims.size() == 2 && po.dims[0] == 1 && po.dims[1] == N) {
                    md->bias[bias_index].bias = po.buff;
                    bias_index++;
                } else {
                    md->matrix_add[matrix_add_index].matrix = po.buff;
                    matrix_add_index++;
                }
                break;
            case post_op_type_t::binary_mul:
                if (binary_mul_uses_scale_path(po, N, M)) {
                    md->scale[scale_index].sf->scale_factor
                            = const_cast<void *>(po.buff);
                    md->scale[scale_index].sf->scale_factor_len = N;
                    md->scale[scale_index].sf->scale_factor_type
                            = to_dlp_type(po.dtype);
                    // Row-broadcast {1, N}: one scale per output column -> per-channel.
                    md->scale[scale_index].sf->scale_factor_dim
                            = DLP_PARAM_DIM_PER_CHANNEL;
                    scale_index++;
                } else {
                    md->matrix_mul[matrix_mul_index].matrix = po.buff;
                    matrix_mul_index++;
                }
                break;
            default: break;
        }
    }

    // Dynamic source quantization (BF16/F32 → INT8): the inverse src_scale
    // value (stored in the holder's embedded float) and the dequant scale
    // factor pointer are recomputed/repointed per call. Now flows through the
    // unified a_quant_op: quant_scale_factors is the forward inverse-scale,
    // dequant_scale_factors the src (post) scale, zero_point the src zp.
    //
    // dequant_scale_factors->{len,stor_type} must be refreshed alongside
    // data: compute_postop_signature folds dtypes.src/wei/dst and the quant-
    // buffer presence booleans, but NOT the quant buffers' own dtypes
    // (src_scale.dt, etc.). Two calls with identical matmul-level dtypes but
    // different src_scale.dt (e.g. f32 vs bf16 src_scale) share a cache key
    // and would otherwise reuse the first call's stale stor_type, making
    // AOCL reinterpret the live buffer with the wrong storage type.
    //
    // The forward inverse scale must also be read through read_and_cast<float>
    // with the same 1e-20f clamp the cold path uses, otherwise a bf16 src_scale
    // on a hit would be misread as a single f32 (corrupted inverse) — main #382
    // added that numerical-safety contract and the hit path must mirror it.
    if (is_non_quant_src_int8 && lowoha_param.quant_params.src_scale.buff) {
        const data_type_t src_scale_dt = lowoha_param.quant_params.src_scale.dt;
        float s = read_and_cast<float>(lowoha_param.quant_params.src_scale.buff,
                src_scale_dt,
                /*index=*/0);
        if (s < 1e-20f) { s = 1e-20f; }
        *static_cast<float *>(md->a_quant_op->quant_scale_factors->data)
                = 1.0f / s;
        md->a_quant_op->dequant_scale_factors->data
                = const_cast<void *>(lowoha_param.quant_params.src_scale.buff);
        md->a_quant_op->dequant_scale_factors->len = 1;
        md->a_quant_op->dequant_scale_factors->stor_type
                = to_dlp_type(src_scale_dt);

        // Asymmetric dyn-quant: src_zp.buff is freshly allocated per call by
        // reorder_quant_buffers_t (RAII, freed at the matmul wrapper's scope
        // exit). The cold-path-cached zero_point pointer therefore dangles on
        // every hit — repoint it from the live src_zp buffer. len/stor_type are
        // stable for the dyn-quant path but cheap to refresh defensively.
        if (lowoha_param.quant_params.src_zp.buff
                && md->a_quant_op->zero_point) {
            md->a_quant_op->zero_point->data
                    = const_cast<void *>(lowoha_param.quant_params.src_zp.buff);
            md->a_quant_op->zero_point->len = 1;
            md->a_quant_op->zero_point->stor_type
                    = to_dlp_type(lowoha_param.quant_params.src_zp.dt);
        }
    }

    // Symmetric INT8 quant: group_size depends on M, which is intentionally
    // excluded from the cache key. Recompute on every hit. a_quant_op /
    // b_quant_op dequant_scale_factors point at per-call user buffers and
    // (for per-token a-scale) carry an M-dependent length; the cold-path-
    // cached values dangle on hits with a different M or a fresh src_scale
    // buffer. The per-tensor vs per-token distinction itself is folded into
    // the signature, so on hit we know is_sym_quant matches the cold-path mode
    // (a_quant_op is non-null iff the cold path wired it).
    if (is_sym_quant && md->a_quant_op) {
        int64_t src_group_size = (src_scale_nelems == static_cast<size_t>(M))
                ? K
                : K / (static_cast<int64_t>(src_scale_nelems) / M);
        md->a_quant_op->group_size = static_cast<int>(src_group_size);

        if (md->a_quant_op->dequant_scale_factors) {
            md->a_quant_op->dequant_scale_factors->data = const_cast<void *>(
                    lowoha_param.quant_params.src_scale.buff);
            md->a_quant_op->dequant_scale_factors->len = src_scale_nelems;
            md->a_quant_op->dequant_scale_factors->stor_type
                    = to_dlp_type(lowoha_param.quant_params.src_scale.dt);
        }

        if (md->b_quant_op && md->b_quant_op->dequant_scale_factors) {
            size_t wei_scale_nelems = get_num_elements(
                    lowoha_param.quant_params.wei_scale.dims);
            md->b_quant_op->group_size = static_cast<int>(src_group_size);
            md->b_quant_op->dequant_scale_factors->data = const_cast<void *>(
                    lowoha_param.quant_params.wei_scale.buff);
            md->b_quant_op->dequant_scale_factors->len = wei_scale_nelems;
            md->b_quant_op->dequant_scale_factors->stor_type
                    = to_dlp_type(lowoha_param.quant_params.wei_scale.dt);
        }
    }

    // WOQ (Weight-Only Quantization) b_quant_op refresh on cache hit.
    // setup_woq_pre_ops on the cold path wires b_quant_op->
    // dequant_scale_factors->data from wei_scale.buff (and zero_point->data
    // from wei_zp.buff for u4 weights). Both are per-call buffers in
    // integrator paths like zentorch — every forward allocates fresh quant
    // tensors — so the cold-path-cached pointers dangle on every subsequent
    // hit and the kernel reads stale (or freed) memory. Repoint from the live
    // params.
    //
    // dequant_scale_factors / zero_point are pre-wired by init_metadata_holder
    // to the holder's embedded dlp_qparam_t slots, so we patch into those
    // stable sub-structs rather than swap the slot pointers. The sym-quant
    // path also uses b_quant_op and is refreshed above, so the !is_sym_quant
    // guard keeps it from being double-written here.
    if (md->b_quant_op && !is_sym_quant && md->b_quant_op->dequant_scale_factors
            && lowoha_param.quant_params.wei_scale.buff) {
        const auto &wei_scale = lowoha_param.quant_params.wei_scale;
        md->b_quant_op->dequant_scale_factors->data
                = const_cast<void *>(wei_scale.buff);
        md->b_quant_op->dequant_scale_factors->len
                = get_num_elements(wei_scale.dims);
        md->b_quant_op->dequant_scale_factors->stor_type
                = (wei_scale.dt == data_type_t::bf16) ? DLP_BF16 : DLP_F32;
    }
    if (md->b_quant_op && !is_sym_quant && md->b_quant_op->zero_point
            && lowoha_param.quant_params.wei_zp.buff) {
        const auto &wei_zp = lowoha_param.quant_params.wei_zp;
        md->b_quant_op->zero_point->data = const_cast<void *>(wei_zp.buff);
        md->b_quant_op->zero_point->len = get_num_elements(wei_zp.dims);
        md->b_quant_op->zero_point->stor_type
                = (wei_zp.dt == data_type_t::s8) ? DLP_S8 : DLP_BF16;
    }

    // Non-symmetric INT8 SCALE post-op slots refresh on cache hit.
    // The cold path writes SCALE entries into scale[] in this order:
    //   src_scale  (if buff && !is_non_quant_src_int8)
    //   wei_scale  (if buff)
    //   row-broadcast binary_mul {1, N} operands (one SCALE slot each)
    //   eltwise post-ops
    //   dst_scale  (if buff) — last SCALE, after all post-ops
    // and (when dst_scale is present and dst_zp.buff is set) also wires
    // scale[dst].zp from dst_zp. Every one of those .scale_factor / .
    // zero_point pointers is a per-call user buffer in integrator paths,
    // so they all dangle on cache hits and must be repointed.
    //
    // src_scale / wei_scale sit before the post-op walk and use fixed
    // leading indices; dst_scale uses scale_index after that walk so
    // row-broadcast binary_mul SCALE slots are skipped. binary_mul bcast
    // buffers are refreshed in the post-op loop above.
    if (is_int8 && !is_sym_quant) {
        int sidx = 0;
        if (lowoha_param.quant_params.src_scale.buff
                && !is_non_quant_src_int8) {
            const auto &src_scale = lowoha_param.quant_params.src_scale;
            md->scale[sidx].sf->scale_factor
                    = const_cast<void *>(src_scale.buff);
            md->scale[sidx].sf->scale_factor_len
                    = get_num_elements(src_scale.dims);
            md->scale[sidx].sf->scale_factor_type = to_dlp_type(src_scale.dt);
            // Source scale is per-tensor (len 1) or per-token / per-row (len M).
            md->scale[sidx].sf->scale_factor_dim
                    = infer_scale_dim(get_num_elements(src_scale.dims), M, N,
                            DLP_PARAM_DIM_PER_TOKEN);
            sidx++;
        }
        if (lowoha_param.quant_params.wei_scale.buff) {
            const auto &wei_scale = lowoha_param.quant_params.wei_scale;
            md->scale[sidx].sf->scale_factor
                    = const_cast<void *>(wei_scale.buff);
            md->scale[sidx].sf->scale_factor_len
                    = get_num_elements(wei_scale.dims);
            md->scale[sidx].sf->scale_factor_type = to_dlp_type(wei_scale.dt);
            // Weight scale is per-tensor (len 1) or per-channel / per-column (len N).
            md->scale[sidx].sf->scale_factor_dim
                    = infer_scale_dim(get_num_elements(wei_scale.dims), M, N,
                            DLP_PARAM_DIM_PER_CHANNEL);
        }
        if (lowoha_param.quant_params.dst_scale.buff) {
            const auto &dst_scale = lowoha_param.quant_params.dst_scale;
            md->scale[scale_index].sf->scale_factor
                    = const_cast<void *>(dst_scale.buff);
            md->scale[scale_index].sf->scale_factor_len
                    = get_num_elements(dst_scale.dims);
            md->scale[scale_index].sf->scale_factor_type
                    = to_dlp_type(dst_scale.dt);
            // Dst scale is per-tensor (len 1) or per-channel / per-column (len N).
            md->scale[scale_index].sf->scale_factor_dim
                    = infer_scale_dim(get_num_elements(dst_scale.dims), M, N,
                            DLP_PARAM_DIM_PER_CHANNEL);
            // dst_zp shares the same slot as dst_scale in the cold path
            // (lines 1017-1023). Only patch when the caller actually supplies
            // a dst_zp buffer; the dummy-zp branch on the cold path uses a
            // process-lifetime constexpr int, so it needs no refresh.
            if (lowoha_param.quant_params.dst_zp.buff) {
                const auto &dst_zp = lowoha_param.quant_params.dst_zp;
                md->scale[scale_index].zp->zero_point
                        = const_cast<void *>(dst_zp.buff);
                md->scale[scale_index].zp->zero_point_len
                        = get_num_elements(dst_zp.dims);
                md->scale[scale_index].zp->zero_point_type
                        = to_dlp_type(dst_zp.dt);
            }
        }
    }
}

// Helper function to create post_op structure for bias and post-ops.
//
// LRU-cached, keyed by Key_matmul (weight_ptr + N + K + algo +
// extra_input_hash). On a cache hit, we patch the per-call mutable fields
// onto the cached metadata and return it. On a miss, we allocate a fresh
// holder, run the full build path, register it in the cache, and return.
//
// The cache is per-thread (thread_local lru_cache_t) so the holder's
// mutable fields can be patched safely without inter-thread races, while
// still being key-based — non-deterministic execution orders (MoE,
// speculative decoding, conditional skip, early exit) are correct by
// construction because lookup is by key, not by call position.
dlp_metadata_t *create_dlp_post_op(const matmul_params &lowoha_param,
        const void *bias, const matmul_data_types &dtypes, int N, int K, int M,
        int32_t *zp_comp_acc, int zp_comp_ndim,
        zendnnl::ops::matmul_algo_t kernel, const void *weight_ptr,
        bool is_w4a8, const int32_t *reorder_colsum, int32_t neg_src_zp) {
    // Normalize zp_comp presence at the API boundary: a zp_comp slot is
    // present iff BOTH the dimensionality and the buffer are non-null.
    // Downstream sites disagree on what "present" means today — the
    // signature (compute_postop_signature), total_ops, and the
    // bias_count/matrix_add_count bumps branch on zp_comp_ndim alone,
    // whereas the cold-path slot wiring and patch_mutable_fields branch on
    // (zp_comp_ndim && zp_comp_acc). Without this normalization, a call
    // with (ndim > 0, acc == nullptr) — reachable via an OOM in
    // cache_or_compute_zp_compensation, which sets ndim before the alloc
    // and returns nullptr on failure — would share a cache key with a
    // subsequent (ndim > 0, acc != nullptr) call on the same weight, and
    // patch_mutable_fields on the hit path would overwrite the cached
    // user-bias slot with zp_comp_acc, producing silently wrong output.
    if (!zp_comp_acc) { zp_comp_ndim = 0; }

    const bool colsum_mode
            = reorder_colsum && !zp_comp_acc && zp_comp_ndim == 0;
    const int sig_zp_ndim = colsum_mode ? 3 : zp_comp_ndim;

    // Runtime kill switch: ZENDNNL_ENABLE_POSTOP_CACHE=0 forces every
    // call through the cold path by clearing the cache here, before the
    // lookup below. The cold-path build still inserts into the cache at
    // the tail of this function, so the freshly-built holder is owned
    // and freed by the cache exactly as in the enabled path — the next
    // disabled-mode call drops it on the way in, giving exactly-once
    // ownership with no leak and no API change. Behavior with the flag
    // off is bit-equivalent to pre-cache zendnnl; intended for triage
    // and as a safety valve for integrators (zentorch, vLLM, etc.).
    // The flag is sampled once per process inside is_postop_cache_enabled
    // (static const cache), so this branch compiles to a single load.
    const bool cache_enabled = zendnnl::common::is_postop_cache_enabled();
    auto &cache = get_postop_metadata_cache();
    if (!cache_enabled) { cache.clear(); }

    // Build the cache key.
    const std::size_t sig = compute_postop_signature(
            lowoha_param, dtypes, sig_zp_ndim, bias, is_w4a8, M, N);
    const Key_matmul key(/*TransB=*/false, static_cast<unsigned int>(K),
            static_cast<unsigned int>(N),
            /*ldb=*/0, weight_ptr, static_cast<uint32_t>(kernel), sig);

    // Hot path: cache hit. Unreachable when cache_enabled is false
    // because the clear() above just emptied the cache.
    dlp_postop_metadata_holder_t *cached_holder = nullptr;
    if (cache.try_get(key, cached_holder)) {
        if (cached_holder->no_metadata) { return nullptr; }
        patch_mutable_fields(&cached_holder->metadata, cached_holder,
                lowoha_param, bias, dtypes, M, N, K, zp_comp_acc, zp_comp_ndim,
                is_w4a8, reorder_colsum, neg_src_zp);
        return &cached_holder->metadata;
    }

    // Cold path: determine dispatch flags BEFORE allocating the holder.
    // None of these computations touch the holder — they are pure
    // functions of lowoha_param/dtypes/zp_comp_ndim. Computing them
    // first lets us short-circuit the multi-KB holder allocation
    // entirely for plain-matmul layers that have no post-op metadata
    // to wire (see the early return below).

    // Check if this is a WOQ case (need metadata even if no post-ops)
    bool is_woq
            = (dtypes.wei == data_type_t::s4 || dtypes.wei == data_type_t::u4)
            && dtypes.src == data_type_t::bf16
            && kernel == zendnnl::ops::matmul_algo_t::aocl_dlp_blocked;

    // Check if this is INT8 quantization case
    bool is_int8 = dtypes.wei == data_type_t::s8;

    size_t src_scale_nelems
            = get_num_elements(lowoha_param.quant_params.src_scale.dims);
    // bf16/f32 + s8 with per-token symmetric scales: a_quant_op carries the
    // per-row src scales, while the weight scale remains a SCALE post-op.
    const bool is_bf16_f32_per_token_sym = is_int8
            && (dtypes.src == data_type_t::bf16
                    || dtypes.src == data_type_t::f32)
            && !lowoha_param.quant_params.src_zp.buff && src_scale_nelems > 1
            && src_scale_nelems == static_cast<size_t>(M)
            && (dtypes.dst == data_type_t::f32
                    || dtypes.dst == data_type_t::bf16);

    bool is_non_quant_src_int8 = (dtypes.src == data_type_t::bf16
                                         || dtypes.src == data_type_t::f32)
            && is_int8 && !is_bf16_f32_per_token_sym;

    // s8s8 *_sym_quant GEMMs consume a_quant_op / b_quant_op; bf16s8/f32s8
    // paths use a_quant_op and regular SCALE post-ops instead.
    // W4A8 (is_w4a8) always dispatches the s8s8 *_sym_quant GEMM, which
    // mandates a_quant_op/b_quant_op group metadata even when the broadcast
    // source scale collapses to a single element (M==1 with a single weight
    // group). Force the sym-quant wiring in that case; the generic
    // src_scale_nelems>1 gate would otherwise route it through the SCALE
    // post-op path and leave the GEMM without its required quant metadata.
    const bool is_sym_quant = is_int8 && dtypes.src == data_type_t::s8
            && !lowoha_param.quant_params.src_zp.buff
            && (src_scale_nelems > 1 || is_w4a8)
            && (dtypes.dst == data_type_t::f32
                    || dtypes.dst == data_type_t::bf16);

    // Count INT8 scale post-ops (s8 sym-quant scales go via quant metadata;
    // bf16/f32 per-token source scales go via a_quant_op).
    int int8_scale_count = 0;
    if (is_int8) {
        if (lowoha_param.quant_params.src_scale.buff && !is_non_quant_src_int8
                && !is_sym_quant && !is_bf16_f32_per_token_sym) {
            int8_scale_count++;
        }
        if (lowoha_param.quant_params.wei_scale.buff && !is_sym_quant) {
            int8_scale_count++;
        }
        if (lowoha_param.quant_params.dst_scale.buff
                || lowoha_param.quant_params.dst_zp.buff) {
            int8_scale_count++;
        }
    }

    // Count total operations (bias + post-ops + scales + zp_comp)
    int total_ops
            = (bias ? 1 : 0) + lowoha_param.postop_.size() + int8_scale_count;

    // Add zero-point compensation to total ops
    if (zp_comp_ndim > 0 || colsum_mode) { total_ops++; }

    // Plain matmul with no post-op chain: nothing to build, nothing to
    // cache. Re-dispatching on every call (the flag/count work above) is
    // cheap — a handful of branches over already-loaded dtype fields —
    // compared to pinning a multi-KB dlp_postop_metadata_holder_t in the
    // per-thread LRU solely to flag "no metadata here". The BF16-INT8
    // src_scale-null error path further down still allocates + caches
    // with no_metadata=true to suppress log-spam on misconfigured calls,
    // so the holder's no_metadata field itself stays.
    if (total_ops == 0 && !is_woq && !is_int8) { return nullptr; }

    // Cold path: allocate + initialize a new holder now that we know it
    // will be wired.
    // Aligned via zendnnl_aligned_alloc (not std::calloc) so the lru_cache_t
    // evictor and cleanup_dlp_post_op release this holder with the matching
    // zendnnl_aligned_free (_aligned_free on Windows). Freeing std::calloc'd
    // memory with _aligned_free corrupts the heap on Windows. sizeof is always
    // a multiple of alignof, which satisfies std::aligned_alloc on Linux.
    // Aligned storage is not zero-initialized, so memset restores the zero-init
    // that the build path (init_metadata_holder + trivial members) relied on
    // with std::calloc.
    auto *new_holder = static_cast<dlp_postop_metadata_holder_t *>(
            zendnnl_aligned_alloc(alignof(dlp_postop_metadata_holder_t),
                    sizeof(dlp_postop_metadata_holder_t)));
    if (!new_holder) {
        EXCEPTION_WITH_LOC(
                "[postop-cache] failed to allocate "
                "dlp_postop_metadata_holder_t");
    }
    std::memset(new_holder, 0, sizeof(dlp_postop_metadata_holder_t));
    init_metadata_holder(new_holder);

    dlp_metadata_t *dlp_metadata = &new_holder->metadata;

    // Count different types of operations
    int eltwise_count = 0;
    int matrix_add_count = 0;
    int matrix_mul_count = 0;
    int bias_count = bias ? 1 : 0;
    int scale_count = 0;

    // For INT8, add scale count
    if (is_int8) { scale_count = int8_scale_count; }
    // Row-broadcast binary_mul {1, N} at M > 1 uses DLP SCALE (per-N multiply);
    // the M == 1 {1, N} case stays on MATRIX_MUL (counted below).
    int binary_mul_bcast_scale_count = 0;
    for (const auto &po : lowoha_param.postop_) {
        if (binary_mul_uses_scale_path(po, N, M)) {
            binary_mul_bcast_scale_count++;
        }
    }
    scale_count += binary_mul_bcast_scale_count;

    // Add zp_comp to appropriate count
    if (zp_comp_ndim == 1 || colsum_mode) {
        bias_count++; // 1D compensation is added as bias
    } else if (zp_comp_ndim == 2) {
        matrix_add_count++; // 2D compensation is added as matrix_add
    }

    // Count post-ops by type
    for (const auto &po : lowoha_param.postop_) {
        switch (po.po_type) {
            case post_op_type_t::relu:
            case post_op_type_t::leaky_relu:
            case post_op_type_t::gelu_tanh:
            case post_op_type_t::gelu_erf:
            case post_op_type_t::sigmoid:
            case post_op_type_t::swish:
            case post_op_type_t::tanh:
            case post_op_type_t::clip:
            case post_op_type_t::mish: eltwise_count++; break;
            case post_op_type_t::binary_add:
                if (po.dims.size() == 2 && po.dims[0] == 1
                        && po.dims[1] == static_cast<int>(N)) {
                    bias_count++;
                } else {
                    matrix_add_count++;
                }
                break;
            case post_op_type_t::binary_mul:
                // Everything not on the SCALE broadcast path (dense {M, N}, and
                // the M == 1 {1, N} case) is a MATRIX_MUL operand.
                if (!binary_mul_uses_scale_path(po, N, M)) {
                    matrix_mul_count++;
                }
                break;
            default:
                // Skip unsupported post-ops
                break;
        }
    }
    // Wire seq_vector (only if we have post-ops). total_ops is bounded by
    // AOCL_DLP_MAX_POST_OPS via the AOCL DLP API contract (chains beyond
    // that are rejected by the backend), and new_holder->seq_vector is sized
    // to exactly that cap, so no in-house bounds check is needed here.
    if (total_ops > 0) {
        dlp_metadata->seq_vector = new_holder->seq_vector;
    } else {
        dlp_metadata->seq_vector = nullptr;
    }

    // Wire pre-quantization INT8 quant_op + scl/zp structs.
    //
    // Two flavors share this block:
    //   - Scalar (is_non_quant_src_int8): single inverse-scale float fits in
    //     the holder's pre-wired a_quant_inv_scale slot; the holder is
    //     cached and reused like every other path.
    //   - Per-token-sym (is_bf16_f32_per_token_sym, from main #382): inverse
    //     scales are an M-element array whose length is only known at call
    //     time (M = src_scale_nelems, can be hundreds during prefill). M
    //     doesn't fit the cache holder's fixed scalar slot, and lru_cache_t
    //     evicts holders with a bare std::free() that can't release a per-
    //     holder heap pointer. So per-token-sym is intentionally NOT cached:
    //     the holder is flagged is_per_call=true, the inv_scales[] buffer
    //     is malloc'd into the holder, and the kernel call site's matching
    //     cleanup_dlp_post_op() frees both at the end of the matmul.
    if (is_non_quant_src_int8 || is_bf16_f32_per_token_sym) {
        // Unified A quant op (new AOCL DLP API): forward-quant (bf16/f32 -> s8)
        // via quant_scale_factors (inverse-scale), dequant of the s32
        // accumulator via dequant_scale_factors (src scale), src zp via
        // zero_point (NULL == symmetric). Replaces the old a_pre_quant +
        // a_post_quant pair.
        dlp_metadata->a_quant_op = &new_holder->a_quant_op;
        // QUANTIZE: A is forward-quantized to s8 and the accumulator is corrected
        // afterwards. dlp_gemm_translate_adquantize_post_op rejects other kinds.
        dlp_metadata->a_quant_op->quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        // quant/dequant scale-factor slots pre-wired by init_metadata_holder.
        dlp_metadata->a_quant_op->quant_scale_factors
                = &new_holder->a_quant_scl;
        dlp_metadata->a_quant_op->dequant_scale_factors
                = &new_holder->a_dequant_scl;

        const DLP_PARAM_DIM_TYPE a_dim = is_bf16_f32_per_token_sym
                ? DLP_PARAM_DIM_PER_TOKEN
                : DLP_PARAM_DIM_PER_TENSOR;

        if (lowoha_param.quant_params.src_zp.buff) {
            // Asymmetric: zero_point slot pre-wired by init_metadata_holder; the
            // explicit re-assignment documents the target. The else branch nulls
            // zero_point for the symmetric variant (replaces the old `symmetric`
            // bool).
            dlp_metadata->a_quant_op->zero_point = &new_holder->a_quant_zp;
            dlp_metadata->a_quant_op->zero_point->data
                    = const_cast<void *>(lowoha_param.quant_params.src_zp.buff);
            dlp_metadata->a_quant_op->zero_point->len = 1;
            dlp_metadata->a_quant_op->zero_point->stor_type
                    = to_dlp_type(lowoha_param.quant_params.src_zp.dt);
            dlp_metadata->a_quant_op->zero_point->outer_dim = a_dim;
        } else {
            dlp_metadata->a_quant_op->zero_point = nullptr;
        }
        if (!lowoha_param.quant_params.src_scale.buff) {
            log_error("BF16-INT8: src_scale buffer is null");
            // Stash the holder in the cache so we don't leak it; mark as no-
            // metadata so subsequent hits also return nullptr deterministically.
            // Safe to cache.add this holder for the per-token-sym key too: we
            // haven't set is_per_call yet, so this becomes a regular cached
            // no_metadata sentinel and every subsequent call (per-token-sym
            // or scalar) for this key short-circuits to nullptr.
            new_holder->no_metadata = true;
            cache.add(key, new_holder);
            return nullptr;
        }
        const data_type_t src_scale_dt = lowoha_param.quant_params.src_scale.dt;
        const md_t src_quant_scale_len = is_bf16_f32_per_token_sym
                ? static_cast<md_t>(src_scale_nelems)
                : 1;

        dlp_metadata->a_quant_op->group_size = 0;
        dlp_metadata->a_quant_op->src_type = to_dlp_type(dtypes.src);
        dlp_metadata->a_quant_op->dst_type = DLP_S8;
        if (is_bf16_f32_per_token_sym) {
            // Per-token-sym: M inverse scales malloc'd into the holder's
            // heap-owned slot. Mark the holder per-call so cleanup at the
            // kernel call site releases both inv_scales and the holder.
            // The cache.add at the tail of this function is gated on
            // !is_per_call, so this holder is never inserted into the cache.
            new_holder->is_per_call = true;
            new_holder->a_quant_inv_scales_dyn = static_cast<float *>(
                    std::malloc(static_cast<size_t>(src_quant_scale_len)
                            * sizeof(float)));
            if (!new_holder->a_quant_inv_scales_dyn) {
                log_error(
                        "BF16-INT8 per-token-sym: failed to allocate "
                        "inv_scales");
                // Release the per-call holder directly. The caller's
                // cleanup_dlp_post_op(nullptr) is a documented no-op, so
                // returning nullptr is the safe error contract here.
                zendnnl_aligned_free(new_holder);
                return nullptr;
            }
            for (md_t si = 0; si < src_quant_scale_len; ++si) {
                float s = read_and_cast<float>(
                        lowoha_param.quant_params.src_scale.buff, src_scale_dt,
                        static_cast<size_t>(si));
                if (s < 1e-20f) { s = 1e-20f; }
                new_holder->a_quant_inv_scales_dyn[si] = 1.0f / s;
            }
            dlp_metadata->a_quant_op->quant_scale_factors->data
                    = new_holder->a_quant_inv_scales_dyn;
            dlp_metadata->a_quant_op->quant_scale_factors->len
                    = src_quant_scale_len;
        } else {
            // Scalar (is_non_quant_src_int8): single inverse scale written into
            // the pre-wired holder float by init_metadata_holder. Uses
            // read_and_cast + 1e-20f clamp to match main #382's numerical-safety
            // contract for non-f32 src_scale dtypes.
            float s = read_and_cast<float>(
                    lowoha_param.quant_params.src_scale.buff, src_scale_dt,
                    /*index=*/0);
            if (s < 1e-20f) { s = 1e-20f; }
            *static_cast<float *>(
                    dlp_metadata->a_quant_op->quant_scale_factors->data)
                    = 1.0f / s;
            dlp_metadata->a_quant_op->quant_scale_factors->len = 1;
        }
        dlp_metadata->a_quant_op->quant_scale_factors->stor_type = DLP_F32;
        dlp_metadata->a_quant_op->quant_scale_factors->outer_dim = a_dim;

        dlp_metadata->a_quant_op->dequant_scale_factors->data
                = const_cast<void *>(lowoha_param.quant_params.src_scale.buff);
        dlp_metadata->a_quant_op->dequant_scale_factors->len
                = src_quant_scale_len;
        dlp_metadata->a_quant_op->dequant_scale_factors->stor_type
                = to_dlp_type(src_scale_dt);
        dlp_metadata->a_quant_op->dequant_scale_factors->outer_dim = a_dim;
    }
    if (is_sym_quant) {
        int64_t src_group_size = (src_scale_nelems == static_cast<size_t>(M))
                ? K
                : K / (static_cast<int64_t>(src_scale_nelems) / M);

        // s8 x s8 symmetric quant (new AOCL DLP API): the A-side dequant scale
        // goes into a_quant_op.dequant_scale_factors, the weight (B) dequant
        // scale into b_quant_op.dequant_scale_factors. Both are dequant-only
        // (A and B are already s8, no forward-quant or zero-point). Replaces
        // the old post_op_grp.
        const bool a_per_token = (src_scale_nelems == static_cast<size_t>(M));

        dlp_metadata->a_quant_op = &new_holder->a_quant_op;
        // QUANTIZE: operands are already s8; the scales only dequantize/correct
        // the s32 accumulator. Both the grouped post-op translator and the
        // sym-quant reorder reject any other kind.
        dlp_metadata->a_quant_op->quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        dlp_metadata->a_quant_op->src_type = DLP_S8;
        dlp_metadata->a_quant_op->dst_type = DLP_S8;
        dlp_metadata->a_quant_op->group_size = static_cast<int>(src_group_size);
        dlp_metadata->a_quant_op->quant_scale_factors = nullptr;
        dlp_metadata->a_quant_op->zero_point = nullptr;
        dlp_metadata->a_quant_op->dequant_scale_factors
                = &new_holder->a_dequant_scl;
        dlp_metadata->a_quant_op->dequant_scale_factors->data
                = const_cast<void *>(lowoha_param.quant_params.src_scale.buff);
        dlp_metadata->a_quant_op->dequant_scale_factors->len = src_scale_nelems;
        dlp_metadata->a_quant_op->dequant_scale_factors->stor_type
                = to_dlp_type(lowoha_param.quant_params.src_scale.dt);
        dlp_metadata->a_quant_op->dequant_scale_factors->outer_dim = a_per_token
                ? DLP_PARAM_DIM_PER_TOKEN
                : DLP_PARAM_DIM_PER_GROUP;

        size_t wei_scale_nelems
                = get_num_elements(lowoha_param.quant_params.wei_scale.dims);
        const bool b_per_channel = (wei_scale_nelems == static_cast<size_t>(N));

        dlp_metadata->b_quant_op = &new_holder->b_quant_op;
        dlp_metadata->b_quant_op->quant_op_kind = DLP_QUANT_OP_QUANTIZE;
        dlp_metadata->b_quant_op->src_type = DLP_S8;
        dlp_metadata->b_quant_op->dst_type = DLP_S8;
        dlp_metadata->b_quant_op->group_size = static_cast<int>(src_group_size);
        dlp_metadata->b_quant_op->quant_scale_factors = nullptr;
        dlp_metadata->b_quant_op->zero_point = nullptr;
        dlp_metadata->b_quant_op->dequant_scale_factors
                = &new_holder->b_dequant_scl;
        dlp_metadata->b_quant_op->dequant_scale_factors->data
                = const_cast<void *>(lowoha_param.quant_params.wei_scale.buff);
        dlp_metadata->b_quant_op->dequant_scale_factors->len = wei_scale_nelems;
        dlp_metadata->b_quant_op->dequant_scale_factors->stor_type
                = to_dlp_type(lowoha_param.quant_params.wei_scale.dt);
        dlp_metadata->b_quant_op->dequant_scale_factors->outer_dim
                = b_per_channel ? DLP_PARAM_DIM_PER_CHANNEL
                                : DLP_PARAM_DIM_PER_GROUP;
    }

    // Wire scale array for INT8 (scale[i].sf and scale[i].zp pre-wired by
    // init_metadata_holder at the holder's embedded sub-arrays).
    if (scale_count > 0) { dlp_metadata->scale = new_holder->scale; }

    // Wire bias array
    if (bias_count > 0) { dlp_metadata->bias = new_holder->bias; }

    // Wire eltwise array
    if (eltwise_count > 0) { dlp_metadata->eltwise = new_holder->eltwise; }

    // Wire matrix_add array (sf pre-wired + DLP_F32 default by init_metadata_holder)
    if (matrix_add_count > 0) {
        dlp_metadata->matrix_add = new_holder->matrix_add;
    }

    // Wire matrix_mul array (sf pre-wired + DLP_F32 default by init_metadata_holder)
    if (matrix_mul_count > 0) {
        dlp_metadata->matrix_mul = new_holder->matrix_mul;
    }

    int op_index = 0;
    int eltwise_index = 0;
    int matrix_add_index = 0;
    int matrix_mul_index = 0;
    int bias_index = 0;
    int scale_index = 0;

    // For INT8: Add zero-point compensation FIRST (before scales)
    if (colsum_mode) {
        new_holder->zp_comp_neg_src_zp = neg_src_zp;
        dlp_metadata->seq_vector[op_index++] = BIAS;
        dlp_metadata->bias[bias_index].bias
                = const_cast<int32_t *>(reorder_colsum);
        dlp_metadata->bias[bias_index].stor_type = DLP_S32;
        dlp_metadata->bias[bias_index].sf = &new_holder->zp_comp_bias_sf;
        dlp_metadata->bias[bias_index].zp = nullptr;
        dlp_metadata->bias[bias_index].bias_len = N;
        bias_index++;
    } else if (zp_comp_ndim == 1 && zp_comp_acc) {
        dlp_metadata->seq_vector[op_index++] = BIAS;
        dlp_metadata->bias[bias_index].bias = zp_comp_acc;
        dlp_metadata->bias[bias_index].stor_type = DLP_S32;
        dlp_metadata->bias[bias_index].sf = nullptr;
        dlp_metadata->bias[bias_index].zp = nullptr;
        dlp_metadata->bias[bias_index].bias_len = N;
        bias_index++;
    } else if (zp_comp_ndim == 2 && zp_comp_acc) {
        dlp_metadata->seq_vector[op_index++] = MATRIX_ADD;
        dlp_metadata->matrix_add[matrix_add_index].matrix = zp_comp_acc;
        dlp_metadata->matrix_add[matrix_add_index].stor_type = DLP_S32;
        dlp_metadata->matrix_add[matrix_add_index].ldm = N;
        // Point scale factor at the shared ONE_F32 constant (default 1.0, read-only).
        dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor
                = get_void_ptr(ONE_F32);
        dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor_len = 1;
        dlp_metadata->matrix_add[matrix_add_index].sf->scale_factor_type
                = DLP_F32;
        matrix_add_index++;
    }

    // For INT8: Add source scale unless it is handled by a_quant_op.
    if (is_int8 && lowoha_param.quant_params.src_scale.buff
            && !is_non_quant_src_int8 && !is_sym_quant
            && !is_bf16_f32_per_token_sym) {
        dlp_metadata->seq_vector[op_index++] = SCALE;
        dlp_metadata->scale[scale_index].sf->scale_factor
                = const_cast<void *>(lowoha_param.quant_params.src_scale.buff);
        dlp_metadata->scale[scale_index].sf->scale_factor_type
                = to_dlp_type(lowoha_param.quant_params.src_scale.dt);
        dlp_metadata->scale[scale_index].sf->scale_factor_len
                = get_num_elements(lowoha_param.quant_params.src_scale.dims);
        // Source scale is per-tensor (len 1) or per-token / per-row (len M).
        dlp_metadata->scale[scale_index].sf->scale_factor_dim = infer_scale_dim(
                get_num_elements(lowoha_param.quant_params.src_scale.dims), M,
                N, DLP_PARAM_DIM_PER_TOKEN);
        // Set dummy zero point
        static int32_t dummy_zp = 0;
        dlp_metadata->scale[scale_index].zp->zero_point = &dummy_zp;
        dlp_metadata->scale[scale_index].zp->zero_point_type = DLP_S32;
        dlp_metadata->scale[scale_index].zp->zero_point_len = 1;
        scale_index++;
    }

    // For INT8: Add weight scale (skip for sym_quant, handled by b_quant_op).
    if (is_int8 && lowoha_param.quant_params.wei_scale.buff && !is_sym_quant) {
        dlp_metadata->seq_vector[op_index++] = SCALE;
        dlp_metadata->scale[scale_index].sf->scale_factor
                = const_cast<void *>(lowoha_param.quant_params.wei_scale.buff);
        dlp_metadata->scale[scale_index].sf->scale_factor_type
                = to_dlp_type(lowoha_param.quant_params.wei_scale.dt);
        dlp_metadata->scale[scale_index].sf->scale_factor_len
                = get_num_elements(lowoha_param.quant_params.wei_scale.dims);
        // Weight scale is per-tensor (len 1) or per-channel / per-column (len N).
        dlp_metadata->scale[scale_index].sf->scale_factor_dim = infer_scale_dim(
                get_num_elements(lowoha_param.quant_params.wei_scale.dims), M,
                N, DLP_PARAM_DIM_PER_CHANNEL);
        // Set dummy zero point
        static int32_t dummy_zp_wei = 0;
        dlp_metadata->scale[scale_index].zp->zero_point = &dummy_zp_wei;
        dlp_metadata->scale[scale_index].zp->zero_point_type = DLP_S32;
        dlp_metadata->scale[scale_index].zp->zero_point_len = 1;
        scale_index++;
    }

    // Add bias if present
    if (bias) {
        dlp_metadata->seq_vector[op_index++] = BIAS;
        dlp_metadata->bias[bias_index].bias = const_cast<void *>(bias);

        // Set storage type based on bias data type
        dlp_metadata->bias[bias_index].stor_type = to_dlp_type(dtypes.bias);
        dlp_metadata->bias[bias_index].sf = nullptr; // No scale factor for bias
        dlp_metadata->bias[bias_index].zp = nullptr; // No zero point for bias
        dlp_metadata->bias[bias_index].bias_len = N;
        bias_index++;
    }

    // Add post-ops
    setup_dlp_postops(dlp_metadata, new_holder, lowoha_param.postop_, op_index,
            eltwise_index, matrix_add_index, matrix_mul_index, bias_index,
            scale_index, N, M);

    // For INT8: Add destination scale at the end (after eltwise post-ops)
    if (is_int8 && lowoha_param.quant_params.dst_scale.buff) {
        dlp_metadata->seq_vector[op_index++] = SCALE;
        dlp_metadata->scale[scale_index].sf->scale_factor
                = const_cast<void *>(lowoha_param.quant_params.dst_scale.buff);
        dlp_metadata->scale[scale_index].sf->scale_factor_type
                = to_dlp_type(lowoha_param.quant_params.dst_scale.dt);
        dlp_metadata->scale[scale_index].sf->scale_factor_len
                = get_num_elements(lowoha_param.quant_params.dst_scale.dims);
        // Dst scale is per-tensor (len 1) or per-channel / per-column (len N).
        dlp_metadata->scale[scale_index].sf->scale_factor_dim = infer_scale_dim(
                get_num_elements(lowoha_param.quant_params.dst_scale.dims), M,
                N, DLP_PARAM_DIM_PER_CHANNEL);
        // Set destination zero-point if present
        if (lowoha_param.quant_params.dst_zp.buff) {
            dlp_metadata->scale[scale_index].zp->zero_point
                    = const_cast<void *>(lowoha_param.quant_params.dst_zp.buff);
            dlp_metadata->scale[scale_index].zp->zero_point_type
                    = to_dlp_type(lowoha_param.quant_params.dst_zp.dt);
            dlp_metadata->scale[scale_index].zp->zero_point_len
                    = get_num_elements(lowoha_param.quant_params.dst_zp.dims);
        } else {
            static int32_t dummy_dst_zp = 0;
            dlp_metadata->scale[scale_index].zp->zero_point = &dummy_dst_zp;
            dlp_metadata->scale[scale_index].zp->zero_point_type = DLP_S32;
            dlp_metadata->scale[scale_index].zp->zero_point_len = 1;
        }
        scale_index++;
    }

    dlp_metadata->seq_length = op_index;
    dlp_metadata->num_eltwise = eltwise_count;

    // Setup pre-ops for WOQ (Weight-Only Quantization)
    if (is_woq) {
        setup_woq_pre_ops(
                dlp_metadata, new_holder, lowoha_param, K, N, dtypes.wei);
    }

    // Register the fully-built holder in the per-thread cache; subsequent
    // calls with the same key take the hot path above.
    //
    // Per-call holders (currently only the BF16/INT8 per-token-sym path)
    // are owned by the kernel call site's cleanup_dlp_post_op() and must
    // NOT enter the cache: their inv_scales[M] buffer is length-variable
    // and would leak if released by lru_cache_t's bare std::free eviction.
    if (!new_holder->is_per_call) { cache.add(key, new_holder); }

    return dlp_metadata;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
