/********************************************************************************
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

/**
 * @file ggml_weight_unpack.cpp
 * @brief GGML weight unpacking for ZenDNN kernels
 */

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>

#include "ggml_weight_unpack.hpp"
#include "lowoha_matmul_utils.hpp"
#if ZENDNNL_DEPENDS_AOCLDLP
#include "lowoha_operators/matmul/backends/aocl/aocl_kernel.hpp"
#else
#include "lowoha_operators/matmul/backends/reference/reference_kernel.hpp"
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {

namespace {

struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};
struct block_q4_0 {
    uint16_t d;
    uint8_t qs[16];
};
struct block_q4_0x8 {
    uint16_t d[8];
    uint8_t qs[128];
};
// GGML block_q4_K: 256 weights, 8 sub-blocks of 32.  144 bytes, no padding.
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K size");
static_assert(offsetof(block_q4_K, qs) == 16, "block_q4_K qs offset");

// GGML block_q4_Kx8: the x86 CPU_REPACK of 8 consecutive Q4_K rows.
// d and dmin are per row. scales[sb * 12 .. +11] is sub-block sb of all
// 8 rows, in the same 6-bit packing as block_q4_K. qs interleaves the 8
// rows 8 bytes at a time. The repack does not XOR the nibbles.
struct block_q4_Kx8 {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[96];
    uint8_t qs[1024];
};
static_assert(
        sizeof(block_q4_Kx8) == 8 * sizeof(block_q4_K), "block_q4_Kx8 size");
static_assert(
        offsetof(block_q4_Kx8, scales) == 32, "block_q4_Kx8 scales offset");
static_assert(offsetof(block_q4_Kx8, qs) == 128, "block_q4_Kx8 qs offset");

// Internal block ids passed to the unpackers.  The public selector is
// pack_format.ggml_type (0 = infer, 1 = Q4_K).
constexpr int kGgmlBlockQ4_0 = 2;
constexpr int kGgmlBlockQ8_0 = 8;
constexpr int kGgmlBlockQ4_K = 12;

inline void q4k_scale_min(const uint8_t *sc, int sb, int &sc6, int &mn6) {
    if (sb < 4) {
        sc6 = sc[sb] & 63;
        mn6 = sc[sb + 4] & 63;
        return;
    }
    sc6 = (sc[sb + 4] & 0x0F) | ((sc[sb - 4] >> 6) << 4);
    mn6 = (sc[sb + 4] >> 4) | ((sc[sb] >> 6) << 4);
}

// Same signed-nibble byte order as the Q4_0 unpack: dst[0..7] pairs the
// low half, dst[8..15] pairs the high half, each nibble already minus 8.
inline void store_signed_q4_group(int8_t *dst, const uint8_t q[32]) {
    for (int i = 0; i < 8; ++i) {
        const auto nib = [](uint8_t code) {
            return static_cast<uint8_t>((code - 8) & 0x0F);
        };
        const uint8_t lo0 = nib(q[2 * i]);
        const uint8_t lo1 = nib(q[2 * i + 1]);
        const uint8_t hi0 = nib(q[16 + 2 * i]);
        const uint8_t hi1 = nib(q[17 + 2 * i]);
        dst[i] = static_cast<int8_t>(lo0 | (lo1 << 4));
        dst[8 + i] = static_cast<int8_t>(hi0 | (hi1 << 4));
    }
}

inline uint32_t fp32_to_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(f));
    return bits;
}

inline float fp32_from_bits(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(bits));
    return f;
}

// fp16 -> fp32, from llama.cpp
float fp16_to_fp32(uint16_t h) {
    const uint32_t w = static_cast<uint32_t>(h) << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
    const float normalized
            = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float denormalized
            = fp32_from_bits((two_w >> 17) | magic_mask) - 0.5f;
    const uint32_t result = sign
            | (two_w < (UINT32_C(1) << 27) ? fp32_to_bits(denormalized)
                                           : fp32_to_bits(normalized));
    return fp32_from_bits(result);
}

// fp32 -> bf16, from llama.cpp
uint16_t fp32_to_bf16(float s) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000)
        return (u.i >> 16) | 64; // NaN: force quiet
    return (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
}

void write_scale(void *scale_buffer, bool use_bf16, int64_t idx, float val) {
    if (use_bf16)
        static_cast<uint16_t *>(scale_buffer)[idx] = fp32_to_bf16(val);
    else
        static_cast<float *>(scale_buffer)[idx] = val;
}

size_t align_up(size_t size, size_t alignment = 64) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// ── Why eviction is DISABLED (UINT32_MAX capacity) ────────────────
// group_matmul_direct warms EVERY expert's weight per call — the active
// ones AND the currently-cold (M==0) ones — then hands the cached buffer
// pointers to the GEMM, which dereferences them AFTER the warm loop
// finishes.  A bounded capacity below the warmed pool size (up to
// 2 * total_experts entries for a fused gate/up + down layer) would let a
// later expert's add() evict() an EARLIER, still-in-use expert's buffer
// mid-call and zendnnl_aligned_free() it → use-after-free when the GEMM reads it.
// Pinning to an infinite capacity sidesteps that without per-pointer
// refcounts, mirroring pack_cache_singleton() in custom_kernel/pack.cpp;
// it explicitly overrides any global ZENDNNL_LRU_CACHE_CAPACITY setting.
// The footprint is model-bounded (one entry per distinct weight
// pointer/shape); weight-rotating workloads should call
// clear_ggml_weight_unpack_cache() during a quiescent window.
lru_cache_t<Key_matmul, void *> &get_ggml_reordered_weight_cache() {
    static lru_cache_t<Key_matmul, void *> cache(
            std::numeric_limits<uint32_t>::max());
    return cache;
}

std::mutex &get_ggml_reordered_weight_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct uncached_unpack_slot_t {
    void *buf = nullptr;
    ~uncached_unpack_slot_t() { zendnnl_aligned_free(buf); }
};

void *adopt_uncached_unpack(void *fresh) {
    static thread_local uncached_unpack_slot_t slot;
    zendnnl_aligned_free(slot.buf);
    slot.buf = fresh;
    return fresh;
}

size_t ggml_scale_bytes(int64_t N, int64_t K, bool use_bf16_scales) {
    return static_cast<size_t>(N * (K / 32)
            * (use_bf16_scales ? sizeof(uint16_t) : sizeof(float)));
}

// Custom-kernel INT8 pack size. Q8_0 (K multiple of 32, N multiple of 32)
// stores one compensation row per K-group.  0 when N is not CK-packable.
static size_t ggml_ck_int8_pack_bytes(int64_t N, int64_t K) {
    if (N <= 0 || K <= 0) return 0;
    constexpr int k_nr_min = 32;
    constexpr int k_nr_max = 64;
    constexpr int k_vnni = 4;
    int pack_nr = 0;
    if ((N % k_nr_min) == 0)
        pack_nr = k_nr_min;
    else if ((N % k_nr_max) == 0)
        pack_nr = k_nr_max;
    else
        return 0;
    const int K_quad = (static_cast<int>(K) + k_vnni - 1) / k_vnni;
    const int n_blocks = static_cast<int>(N) / pack_nr;
    const int comp_rows
            = ((K % 32) == 0 && (N % 32) == 0) ? static_cast<int>(K / 32) : 1;
    const size_t weight_bytes_per_oblock = static_cast<size_t>(K_quad)
            * static_cast<size_t>(pack_nr) * static_cast<size_t>(k_vnni);
    const size_t comp_bytes_per_oblock = static_cast<size_t>(comp_rows)
            * static_cast<size_t>(pack_nr) * sizeof(int32_t);
    return static_cast<size_t>(n_blocks)
            * (weight_bytes_per_oblock + comp_bytes_per_oblock);
}

// Writable weight capacity for the unpack buffer. Q8_0 that the custom
// kernel can pack (N and K multiples of 32) is sized for that one in-place
// image. Anything else also reserves the AOCL blocked image. Scales sit
// AFTER this capacity.
static size_t ggml_sym_quant_weight_capacity(int64_t N, int64_t K) {
    const size_t weight_bytes = static_cast<size_t>(N) * static_cast<size_t>(K);
    size_t capacity = weight_bytes;
    const size_t ck_bytes = ggml_ck_int8_pack_bytes(N, K);
    const bool q8_ck = ck_bytes > 0 && (K % 32) == 0 && (N % 32) == 0;
    if (ck_bytes > 0) { capacity = std::max(capacity, align_up(ck_bytes)); }
    if (q8_ck) { return capacity; }
#if ZENDNNL_DEPENDS_AOCLDLP
    dlp_metadata_t symq_meta {};
    dlp_quant_op_t symq_b_quant_op {};
    symq_b_quant_op.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
    // GGML group size is fixed at 32; matches the {K/32, N} scale layout.
    symq_b_quant_op.group_size = 32;
    symq_meta.b_quant_op = &symq_b_quant_op;
    const size_t blocked_req
            = aocl_get_reorder_buf_size_s8s8s32os32_sym_quant('r', 't', 'B',
                    static_cast<md_t>(K), static_cast<md_t>(N), &symq_meta);
    if (blocked_req > 0) {
        capacity = std::max(capacity, align_up(blocked_req));
    }
#endif
    return capacity;
}

// Writable capacity for a packed Q4_0 weight.  The resident image is nibble
// packed (N*K/2), and the in-place reorder writes the s8×s4 blocked layout
// from aocl_get_reorder_buf_size_s8s4s32os32.  Scales sit AFTER this capacity.
// Q4_0 does not widen to s8, so the custom-kernel s8 pack size is not reserved.
static size_t ggml_s4_sym_quant_weight_capacity(int64_t N, int64_t K) {
    const size_t packed_bytes
            = static_cast<size_t>(N) * static_cast<size_t>(K / 2);
    size_t capacity = packed_bytes;
#if ZENDNNL_DEPENDS_AOCLDLP
    dlp_metadata_t symq_meta {};
    dlp_quant_op_t symq_b_quant_op {};
    symq_b_quant_op.quant_op_kind = DLP_QUANT_OP_QUANTIZE;
    symq_b_quant_op.group_size = 32;
    symq_meta.b_quant_op = &symq_b_quant_op;
    const size_t blocked_req = aocl_get_reorder_buf_size_s8s4s32os32('r', 't',
            'B', static_cast<md_t>(K), static_cast<md_t>(N), &symq_meta);
    if (blocked_req > 0) {
        capacity = std::max(capacity, align_up(blocked_req));
    }
#endif
    return capacity;
}

// Unpack a GGML block-quantized weight into a freshly-allocated buffer laid out
// as [ raw weights, with writable capacity >= the matching sym-quant blocked
// size | bf16 scales ({K/32, N}) ].  Q8_0 (ggml_type 8) unpacks straight to s8.
// Q4_0 (ggml_type 2) unpacks to packed signed s4 and stays there; the s8×s4
// GEMM widens nibbles inside the microkernel.
//
// On success `*owned_buffer` holds the allocation (caller frees), `*out_s8`
// points at its weight region (s8 or packed s4), `*out_scales` at its bf16
// scale region (past the weight capacity), and `*out_weight_capacity` is the
// writable weight extent for `wei_buffer_capacity_bytes`.
static size_t ggml_layout_hash(int ggml_type, bool is_superblock) {
    return (static_cast<size_t>(ggml_type) << 1)
            | (is_superblock ? size_t {1} : size_t {0});
}

static status_t ggml_unpack_to_s8_buffer(const void *weight, int64_t N,
        int64_t K, int ggml_type, bool is_superblock, void **owned_buffer,
        int8_t **out_s8, void **out_scales, size_t *out_weight_capacity) {
    if (is_superblock && ggml_type != kGgmlBlockQ4_0
            && ggml_type != kGgmlBlockQ4_K) {
        log_error("GGML superblock unpack is only valid for Q4_0 and Q4_K");
        return status_t::failure;
    }
    const bool q4k = ggml_type == kGgmlBlockQ4_K;
    const size_t weight_bytes = static_cast<size_t>(N) * static_cast<size_t>(K);
    // Q4_K stores S as f32, the same value used to form M' = M-8*S.
    // A bf16 scale would make the kernel multiply a rounded S against an
    // offset built from the original S.
    const size_t scale_bytes = ggml_scale_bytes(N, K, /*use_bf16_scales=*/!q4k);
    // Q4_K stores a second {K/32, N} table, the per-group offset, as f32.
    const size_t min_bytes = q4k ? ggml_scale_bytes(N, K, false) : 0;
    const size_t meta_bytes = scale_bytes + min_bytes;
    const size_t weight_capacity = (ggml_type == kGgmlBlockQ4_0 || q4k)
            ? ggml_s4_sym_quant_weight_capacity(N, K)
            : ggml_sym_quant_weight_capacity(N, K);
    const size_t canon_size = align_up(weight_capacity + meta_bytes);

    void *canon = zendnnl_aligned_alloc(64, canon_size);
    if (!canon) {
        log_error("GGML unpack failed: s8 buffer allocation failed");
        return status_t::failure;
    }

    if (ggml_type == 8) {
        // Q8_0: native s8 unpack writes scales at weight_bytes.  When the
        // in-place capacity is larger, slide the scales past that region.
        int8_t *w = nullptr;
        void *s = nullptr;
        if (ggml_unpack_weight_buffer(
                    weight, 8, /*use_bf16_scales=*/true, N, K, &w, &s, canon)
                != 0) {
            zendnnl_aligned_free(canon);
            log_error("GGML Q8_0 unpack failed");
            return status_t::failure;
        }
        if (weight_capacity != weight_bytes) {
            std::memmove(static_cast<uint8_t *>(canon) + weight_capacity,
                    static_cast<uint8_t *>(canon) + weight_bytes, meta_bytes);
        }
        *out_s8 = static_cast<int8_t *>(canon);
        *out_scales = static_cast<uint8_t *>(canon) + weight_capacity;
        *out_weight_capacity = weight_capacity;
        *owned_buffer = canon;
        return status_t::success;
    }

    // Q4_0 / Q4_K: unpack packed signed nibbles straight into the resident
    // buffer.  Scales (and the Q4_K offset) are written immediately after
    // the packed bytes; slide them past the reorder capacity so an in-place
    // s8×s4 reorder cannot clobber them.
    if (ggml_type != kGgmlBlockQ4_0 && ggml_type != kGgmlBlockQ4_K) {
        zendnnl_aligned_free(canon);
        log_error("GGML unpack failed: unsupported block type ", ggml_type);
        return status_t::failure;
    }
    int8_t *packed_s4 = nullptr;
    void *tmp_scales = nullptr;
    if (ggml_unpack_weight_buffer(weight, ggml_type, /*use_bf16_scales=*/!q4k,
                N, K, &packed_s4, &tmp_scales, canon, is_superblock)
            != 0) {
        zendnnl_aligned_free(canon);
        log_error(q4k ? "GGML Q4_K unpack failed" : "GGML Q4_0 unpack failed");
        return status_t::failure;
    }
    const size_t packed_bytes
            = static_cast<size_t>(N) * static_cast<size_t>(K / 2);
    if (weight_capacity != packed_bytes) {
        std::memmove(static_cast<uint8_t *>(canon) + weight_capacity,
                static_cast<uint8_t *>(canon) + packed_bytes, meta_bytes);
    }

    apilog_info(q4k ? "GGML Q4_K unpack: kept packed s4 N="
                    : "GGML Q4_0 unpack: kept packed s4 N=",
            N, ", K=", K, " packed_bytes=", packed_bytes,
            " capacity=", weight_capacity);

    *out_s8 = static_cast<int8_t *>(canon);
    *out_scales = static_cast<uint8_t *>(canon) + weight_capacity;
    *out_weight_capacity = weight_capacity;
    *owned_buffer = canon;
    return status_t::success;
}

} // anonymous namespace

namespace {
size_t quant_param_nelems(const std::vector<int64_t> &dims) {
    size_t nelems = 1;
    for (auto d : dims) {
        nelems *= static_cast<size_t>(d);
    }
    return nelems;
}
} // namespace

bool ggml_is_sym_quant(const matmul_params &params) {
    // Q8_0 weights arrive as s8; Q4_0 weights arrive as s4 and stay packed s4
    // through the unpack.  Accept both here so the pre-unpack gate does not
    // reject a Q4_0 packed weight.
    const bool wei_is_ggml_int = params.dtypes.wei == data_type_t::s8
            || params.dtypes.wei == data_type_t::s4;
    return wei_is_ggml_int && params.dtypes.src == data_type_t::s8
            && !params.quant_params.src_zp.buff
            && quant_param_nelems(params.quant_params.src_scale.dims) > 1
            && (params.dtypes.dst == data_type_t::f32
                    || params.dtypes.dst == data_type_t::bf16);
}

bool ggml_packed_gemm_supported(const matmul_params &params) {
    if (ggml_is_sym_quant(params)) { return true; }
    // Float entry. Source quantization produces the sym-quant shape above.
    // A scalar source scale (one element) stays rejected: the unpacked weight
    // scale is per-group, and a scalar source scale cannot apply it.
    if (!params.dynamic_quant) { return false; }
    if (params.dtypes.src != data_type_t::f32
            && params.dtypes.src != data_type_t::bf16) {
        return false;
    }
    if (params.dtypes.wei != data_type_t::s8
            && params.dtypes.wei != data_type_t::s4
            && params.dtypes.wei != data_type_t::none) {
        return false;
    }
    if (params.quant_params.src_zp.buff) { return false; }
    if (params.dtypes.dst != data_type_t::f32
            && params.dtypes.dst != data_type_t::bf16) {
        return false;
    }
    // GGML Q8_0/Q4_0 per-group quantization is symmetric-only. Keep the
    // pre-unpack contract aligned with `unpack_ggml_raw_s8_and_cache`, which
    // normalizes the materialized operation to s8 compute.
    if (params.dtypes.compute != data_type_t::s8) { return false; }
    return quant_param_nelems(params.quant_params.src_scale.dims) > 1;
}

status_t validate_ggml_packed_inputs(const matmul_params &params,
        bool is_weights_const, int Batch_B, bool transB) {
    if (params.packing.pack_format_b != 1) { return status_t::success; }
    if (!is_weights_const) {
        log_error(
                "GGML packed weights require constant weights for cached "
                "out-of-place unpack");
        return status_t::failure;
    }
    if (Batch_B > 1) {
        log_error(
                "GGML packed weights with Batch_B > 1 are not supported by "
                "the single-buffer unpack cache");
        return status_t::failure;
    }
    // GGML packs weights as a fixed N x K row-major layout, the transpose
    // of the logical B matrix (transB=true).  With transB=false the later
    // reorder would read the unpacked buffer as K x N.
    if (!transB) {
        log_error(
                "GGML packed weights require transB=true; the unpacked "
                "layout is N x K (transposed B)");
        return status_t::failure;
    }
    return status_t::success;
}

int64_t ggml_unpack_weight_buffer_size(
        int ggml_type, bool use_bf16_scales, int64_t N, int64_t K) {
    if (N <= 0 || K <= 0 || K % 32 != 0) return -1;
    int64_t ng = K / 32;
    int64_t scale_bytes = N * ng * (use_bf16_scales ? 2 : 4);
    if (ggml_type == kGgmlBlockQ8_0) return N * K + scale_bytes;
    if (ggml_type == kGgmlBlockQ4_0) return N * (K / 2) + scale_bytes;
    if (ggml_type == kGgmlBlockQ4_K) {
        // S and M' are both f32.  use_bf16_scales is ignored for this type.
        const int64_t f32_scale_bytes = static_cast<int64_t>(N) * ng * 4;
        return N * (K / 2) + f32_scale_bytes + f32_scale_bytes;
    }
    return -1;
}

int ggml_unpack_weight_buffer(const void *weight_data, int ggml_type,
        bool use_bf16_scales, int64_t N, int64_t K, int8_t **wei_ptr,
        void **scl_ptr, void *unpack_buffer, bool is_superblock) {
    if (!weight_data || !wei_ptr || !scl_ptr) return -1;
    if (N <= 0 || K <= 0 || K % 32 != 0) return -1;
    // block_q4_0x8 and block_q4_Kx8 both interleave 8 rows.
    if (is_superblock
            && (ggml_type != kGgmlBlockQ4_0 && ggml_type != kGgmlBlockQ4_K))
        return -1;
    if (is_superblock && N % 8 != 0) return -1;
    if (ggml_type == kGgmlBlockQ4_K && K % 256 != 0) return -1;
    // Q4_K's unpacked image is the nibble bytes plus two scale tables, which
    // is larger than the 144-byte source block.  Writing that into the source
    // overruns it.  Q4_0 and Q8_0 still allow the legacy in-place form.
    if (ggml_type == kGgmlBlockQ4_K && unpack_buffer == nullptr) return -1;
    // Q4_K keeps S in f32 so M' = M-8*S uses the scale the kernel multiplies.
    if (ggml_type == kGgmlBlockQ4_K) use_bf16_scales = false;

    // If no destination is supplied, preserve the legacy inplace behavior.
    // A supplied destination lets callers keep packed GGML bytes immutable.
    void *buf = unpack_buffer ? unpack_buffer : const_cast<void *>(weight_data);

    const int64_t ng = K / 32;
    const int64_t num_blocks = N * ng;

    const int64_t weight_bytes
            = (ggml_type == kGgmlBlockQ8_0) ? N * K : N * (K / 2);
    *wei_ptr = static_cast<int8_t *>(buf);
    *scl_ptr = static_cast<void *>(static_cast<int8_t *>(buf) + weight_bytes);

    // Q8_0: copy int8 weights directly, one scale per group.
    if (ggml_type == kGgmlBlockQ8_0) {
        auto *blocks = static_cast<const block_q8_0 *>(weight_data);

        std::unique_ptr<int8_t[]> tmp_weights(new int8_t[N * K]);
        const size_t scale_bytes = num_blocks
                * (use_bf16_scales ? sizeof(uint16_t) : sizeof(float));
        std::unique_ptr<uint8_t[]> tmp_scales(new uint8_t[scale_bytes]);

        const int64_t rows4 = (N / 4) * 4;

#pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < rows4; row += 4) {
            for (int64_t g = 0; g < ng; g++) {
                const block_q8_0 &b0 = blocks[(row + 0) * ng + g];
                const block_q8_0 &b1 = blocks[(row + 1) * ng + g];
                const block_q8_0 &b2 = blocks[(row + 2) * ng + g];
                const block_q8_0 &b3 = blocks[(row + 3) * ng + g];

                write_scale(tmp_scales.get(), use_bf16_scales, g * N + row + 0,
                        fp16_to_fp32(b0.d));
                write_scale(tmp_scales.get(), use_bf16_scales, g * N + row + 1,
                        fp16_to_fp32(b1.d));
                write_scale(tmp_scales.get(), use_bf16_scales, g * N + row + 2,
                        fp16_to_fp32(b2.d));
                write_scale(tmp_scales.get(), use_bf16_scales, g * N + row + 3,
                        fp16_to_fp32(b3.d));

                std::memcpy(&tmp_weights[(row + 0) * K + g * 32], b0.qs, 32);
                std::memcpy(&tmp_weights[(row + 1) * K + g * 32], b1.qs, 32);
                std::memcpy(&tmp_weights[(row + 2) * K + g * 32], b2.qs, 32);
                std::memcpy(&tmp_weights[(row + 3) * K + g * 32], b3.qs, 32);
            }
        }

        for (int64_t row = rows4; row < N; row++) {
            for (int64_t g = 0; g < ng; g++) {
                const block_q8_0 &b = blocks[row * ng + g];
                write_scale(tmp_scales.get(), use_bf16_scales, g * N + row,
                        fp16_to_fp32(b.d));
                std::memcpy(&tmp_weights[row * K + g * 32], b.qs, 32);
            }
        }

        uint8_t *raw_buf = static_cast<uint8_t *>(buf);
        std::memcpy(raw_buf, tmp_weights.get(), N * K);
        std::memcpy(raw_buf + N * K, tmp_scales.get(), scale_bytes);
        return 0;
    }

    // Q4_0 superblock (block_q4_0x8): 8 rows interleaved.  Nibbles are already
    // signed, so unlike plain Q4_0 this path does not subtract the +8 bias.
    // The s4 GEMM sign-extends those nibbles.
    if (ggml_type == kGgmlBlockQ4_0 && is_superblock) {
        const int64_t weight_q4_bytes = N * (K / 2);
        const size_t scale_bytes = num_blocks
                * (use_bf16_scales ? sizeof(uint16_t) : sizeof(float));

        std::unique_ptr<int8_t[]> weight_tmp(new int8_t[weight_q4_bytes]);
        std::unique_ptr<uint8_t[]> scale_tmp(new uint8_t[scale_bytes]);

        auto *blocks = static_cast<const block_q4_0x8 *>(weight_data);

#pragma omp parallel for schedule(static)
        for (int64_t r8 = 0; r8 < N / 8; r8++) {
            for (int64_t g = 0; g < ng; g++) {
                block_q4_0x8 local;
                std::memcpy(&local, &blocks[r8 * ng + g], sizeof(local));

                for (int ri = 0; ri < 8; ri++) {
                    const int64_t row = r8 * 8 + ri;

                    write_scale(scale_tmp.get(), use_bf16_scales, g * N + row,
                            fp16_to_fp32(local.d[ri]));

                    uint8_t src[16];
                    for (int i = 0; i < 8; i++) {
                        src[i] = local.qs[ri * 8 + i];
                        src[i + 8] = local.qs[64 + ri * 8 + i];
                    }

                    int8_t *dst = &weight_tmp[(row * K + g * 32) / 2];
                    for (int i = 0; i < 8; i++) {
                        uint8_t lo0 = src[2 * i] & 0x0F;
                        uint8_t lo1 = src[2 * i + 1] & 0x0F;
                        uint8_t hi0 = src[2 * i] >> 4;
                        uint8_t hi1 = src[2 * i + 1] >> 4;

                        dst[i] = static_cast<int8_t>(lo0 | (lo1 << 4));
                        dst[8 + i] = static_cast<int8_t>(hi0 | (hi1 << 4));
                    }
                }
            }
        }

        uint8_t *raw_buf = static_cast<uint8_t *>(buf);
        std::memcpy(raw_buf, weight_tmp.get(), weight_q4_bytes);
        std::memcpy(raw_buf + weight_q4_bytes, scale_tmp.get(), scale_bytes);
        return 0;
    }

    // Q4_K superblock (block_q4_Kx8).  Same signed nibble and M-8*S offset
    // as plain Q4_K.  Only the location of each value differs, and unlike
    // Q4_0x8 the quants are still unsigned.
    if (ggml_type == kGgmlBlockQ4_K && is_superblock) {
        const int64_t weight_q4_bytes = N * (K / 2);
        const size_t scale_bytes = static_cast<size_t>(num_blocks)
                * (use_bf16_scales ? sizeof(uint16_t) : sizeof(float));
        const size_t min_bytes
                = static_cast<size_t>(num_blocks) * sizeof(float);
        std::unique_ptr<int8_t[]> weight_tmp(new int8_t[weight_q4_bytes]);
        std::unique_ptr<uint8_t[]> scale_tmp(new uint8_t[scale_bytes]);
        std::unique_ptr<uint8_t[]> min_tmp(new uint8_t[min_bytes]);
        auto *blocks = static_cast<const block_q4_Kx8 *>(weight_data);
        const int64_t blocks_per_row = K / 256;

#pragma omp parallel for schedule(static)
        for (int64_t r8 = 0; r8 < N / 8; r8++) {
            for (int64_t b = 0; b < blocks_per_row; b++) {
                block_q4_Kx8 local;
                std::memcpy(&local, &blocks[r8 * blocks_per_row + b],
                        sizeof(local));
                for (int ri = 0; ri < 8; ri++) {
                    const int64_t row = r8 * 8 + ri;
                    const float d = fp16_to_fp32(local.d[ri]);
                    const float dmin = fp16_to_fp32(local.dmin[ri]);
                    for (int sb = 0; sb < 8; sb++) {
                        int sc6 = 0;
                        int mn6 = 0;
                        q4k_scale_min(&local.scales[sb * 12], ri, sc6, mn6);
                        const float scale = d * static_cast<float>(sc6);
                        const float min = dmin * static_cast<float>(mn6);
                        const float min_signed = min - 8.0f * scale;
                        const int64_t g = b * 8 + sb;
                        write_scale(scale_tmp.get(), use_bf16_scales,
                                g * N + row, scale);
                        static_cast<float *>(
                                static_cast<void *>(min_tmp.get()))[g * N + row]
                                = min_signed;

                        uint8_t q[32];
                        const int chunk = sb / 2;
                        const bool high = (sb & 1) != 0;
                        for (int l = 0; l < 32; l++) {
                            // Row ri, qs byte p, 8-byte interleave.
                            const int p = 32 * chunk + l;
                            const uint8_t byte
                                    = local.qs[(p / 8) * 64 + ri * 8 + p % 8];
                            q[l] = high ? static_cast<uint8_t>(byte >> 4)
                                        : static_cast<uint8_t>(byte & 0x0F);
                        }
                        store_signed_q4_group(
                                &weight_tmp[(row * K + g * 32) / 2], q);
                    }
                }
            }
        }

        uint8_t *raw_buf = static_cast<uint8_t *>(buf);
        std::memcpy(raw_buf, weight_tmp.get(),
                static_cast<size_t>(weight_q4_bytes));
        std::memcpy(raw_buf + weight_q4_bytes, scale_tmp.get(), scale_bytes);
        std::memcpy(raw_buf + weight_q4_bytes + scale_bytes, min_tmp.get(),
                min_bytes);
        return 0;
    }

    // Q4_K (block_q4_K): unsigned nibbles.  Store q-8 and the f32 offset
    // M-8*S.  The s4 GEMM sign-extends the stored nibble.
    if (ggml_type == kGgmlBlockQ4_K) {
        const int64_t weight_q4_bytes = N * (K / 2);
        const size_t scale_bytes = static_cast<size_t>(num_blocks)
                * (use_bf16_scales ? sizeof(uint16_t) : sizeof(float));
        std::unique_ptr<int8_t[]> weight_tmp(new int8_t[weight_q4_bytes]);
        std::unique_ptr<uint8_t[]> scale_tmp(new uint8_t[scale_bytes]);
        const size_t min_bytes
                = static_cast<size_t>(num_blocks) * sizeof(float);
        std::unique_ptr<uint8_t[]> min_tmp(new uint8_t[min_bytes]);
        auto *blocks = static_cast<const block_q4_K *>(weight_data);
        const int64_t blocks_per_row = K / 256;

#pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < N; row++) {
            for (int64_t b = 0; b < blocks_per_row; b++) {
                block_q4_K local;
                std::memcpy(&local, &blocks[row * blocks_per_row + b],
                        sizeof(local));
                const float d = fp16_to_fp32(local.d);
                const float dmin = fp16_to_fp32(local.dmin);
                for (int sb = 0; sb < 8; sb++) {
                    int sc6 = 0;
                    int mn6 = 0;
                    q4k_scale_min(local.scales, sb, sc6, mn6);
                    const float scale = d * static_cast<float>(sc6);
                    const float min = dmin * static_cast<float>(mn6);
                    // Signed nibble is q-8, so the remaining offset is M-8*S.
                    const float min_signed = min - 8.0f * scale;
                    const int64_t g = b * 8 + sb;
                    write_scale(scale_tmp.get(), use_bf16_scales, g * N + row,
                            scale);
                    static_cast<float *>(
                            static_cast<void *>(min_tmp.get()))[g * N + row]
                            = min_signed;

                    uint8_t q[32];
                    const int chunk = sb / 2;
                    const bool high = (sb & 1) != 0;
                    for (int l = 0; l < 32; l++) {
                        const uint8_t byte = local.qs[32 * chunk + l];
                        q[l] = high ? static_cast<uint8_t>(byte >> 4)
                                    : static_cast<uint8_t>(byte & 0x0F);
                    }
                    store_signed_q4_group(
                            &weight_tmp[(row * K + g * 32) / 2], q);
                }
            }
        }

        uint8_t *raw_buf = static_cast<uint8_t *>(buf);
        std::memcpy(raw_buf, weight_tmp.get(),
                static_cast<size_t>(weight_q4_bytes));
        std::memcpy(raw_buf + weight_q4_bytes, scale_tmp.get(), scale_bytes);
        std::memcpy(raw_buf + weight_q4_bytes + scale_bytes, min_tmp.get(),
                min_bytes);
        return 0;
    }

    // Q4_0: unsigned nibbles 0-15, subtract 8 to convert to signed S4.  The
    // s4 GEMM sign-extends, so the bias MUST come off here or every code
    // lands 8 away from its true value.
    if (ggml_type == kGgmlBlockQ4_0) {
        const int64_t weight_q4_bytes = N * (K / 2);
        const size_t scale_bytes = num_blocks
                * (use_bf16_scales ? sizeof(uint16_t) : sizeof(float));

        std::unique_ptr<int8_t[]> weight_tmp(new int8_t[weight_q4_bytes]);
        std::unique_ptr<uint8_t[]> scale_tmp(new uint8_t[scale_bytes]);

        auto *blocks = static_cast<const block_q4_0 *>(weight_data);

#pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < N; row++) {
            for (int64_t g = 0; g < ng; g++) {
                block_q4_0 local;
                std::memcpy(&local, &blocks[row * ng + g], sizeof(local));

                write_scale(scale_tmp.get(), use_bf16_scales, g * N + row,
                        fp16_to_fp32(local.d));

                int8_t *dst = &weight_tmp[(row * K + g * 32) / 2];
                for (int i = 0; i < 8; i++) {
                    uint8_t lo0 = local.qs[2 * i] & 0x0F;
                    uint8_t lo1 = local.qs[2 * i + 1] & 0x0F;
                    uint8_t hi0 = local.qs[2 * i] >> 4;
                    uint8_t hi1 = local.qs[2 * i + 1] >> 4;

                    lo0 = (lo0 - 8) & 0x0F;
                    lo1 = (lo1 - 8) & 0x0F;
                    hi0 = (hi0 - 8) & 0x0F;
                    hi1 = (hi1 - 8) & 0x0F;

                    dst[i] = static_cast<int8_t>(lo0 | (lo1 << 4));
                    dst[8 + i] = static_cast<int8_t>(hi0 | (hi1 << 4));
                }
            }
        }

        uint8_t *raw_buf = static_cast<uint8_t *>(buf);
        std::memcpy(raw_buf, weight_tmp.get(), weight_q4_bytes);
        std::memcpy(raw_buf + weight_q4_bytes, scale_tmp.get(), scale_bytes);
        return 0;
    }

    return -1; // unsupported type
}

// Out-of-place GGML unpack: cache the UNPACKED s8 weight (N x K) + {K/32, N}
// bf16 scales and hand that buffer back UN-reordered (mem_format 'n',
// pack_format_b cleared).  This function never reorders.  The GEMM reorders
// later IN PLACE into this library-owned allocation (weight_cache_type=2 +
// wei_buffer_capacity_bytes covering the sym-quant blocked image).  Scales
// sit past that capacity so the write-back cannot clobber them.  The packed
// GGML allocation from the caller (often read-only) is never written.
// n_tile_execute still reorders each N-tile from the raw bytes while the buffer
// remains unreordered (`execute_tile` caps its tile WC to out-of-place).
static status_t unpack_ggml_raw_s8_and_cache(const void *&weight, int N, int K,
        int ldb, char trans, matmul_params &params, int ggml_type,
        bool is_superblock, bool is_weights_const) {
    // Deterministic from (N, K, ggml type): recomputed on cache hit so scale
    // offsets and wei_buffer_capacity_bytes stay consistent without storing
    // them in the key.  Q4_0 sizes the s8×s4 blocked image; Q8_0 sizes s8.
    const bool keep_packed_s4
            = ggml_type == kGgmlBlockQ4_0 || ggml_type == kGgmlBlockQ4_K;
    const bool q4k = ggml_type == kGgmlBlockQ4_K;
    const size_t weight_capacity = keep_packed_s4
            ? ggml_s4_sym_quant_weight_capacity(N, K)
            : ggml_sym_quant_weight_capacity(N, K);

    Key_matmul cache_key(trans == 't', static_cast<unsigned int>(K),
            static_cast<unsigned int>(N), static_cast<unsigned int>(ldb),
            weight, static_cast<uint32_t>(matmul_algo_t::none),
            ggml_layout_hash(ggml_type, is_superblock));

    lru_cache_t<Key_matmul, void *> &weight_cache
            = get_ggml_reordered_weight_cache();
    std::mutex &cache_mutex = get_ggml_reordered_weight_cache_mutex();

    void *cached_buffer = nullptr;
    if (is_weights_const) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (weight_cache.try_get(cache_key, cached_buffer)) {
            apilog_info("GGML raw unpack cache hit: N=", N, ", K=", K,
                    keep_packed_s4 ? " dtype=s4" : " dtype=s8");
        }
    }

    if (!cached_buffer) {
        apilog_info("GGML raw unpack cache miss: N=", N, ", K=", K,
                keep_packed_s4 ? " dtype=s4" : " dtype=s8");
        // Canonical [ weight capacity | bf16 scales ] buffer.  Q4_0 stays
        // packed signed s4; Q8_0 is native s8.  Either layout is the raw
        // per-group weight the N-tile DLP path reorders, with enough room
        // for a later full-weight in-place sym-quant reorder.
        void *owned = nullptr;
        int8_t *unpacked_weights = nullptr;
        void *unpacked_scales = nullptr;
        size_t owned_capacity = 0;
        if (ggml_unpack_to_s8_buffer(weight, N, K, ggml_type, is_superblock,
                    &owned, &unpacked_weights, &unpacked_scales,
                    &owned_capacity)
                != status_t::success) {
            return status_t::failure;
        }
        (void)unpacked_weights;
        (void)unpacked_scales;
        (void)owned_capacity;
        if (!is_weights_const) {
            cached_buffer = adopt_uncached_unpack(owned);
        } else {
            std::lock_guard<std::mutex> lock(cache_mutex);
            if (weight_cache.try_get(cache_key, cached_buffer)) {
                zendnnl_aligned_free(owned); // another thread filled it first
            } else {
                weight_cache.add(cache_key, owned);
                cached_buffer = owned;
            }
        }
    }

    // Hand back the raw weight (Q8_0: s8 N x K; Q4_0: packed s4) + {K/32, N}
    // bf16 scales, UN-reordered, so the per-group N-tile DLP path reorders
    // it per-tile.
    weight = cached_buffer;
    params.mem_format_b = 'n';
    // Clear the GGML packed flag.  m_tile_supported / n_tile_supported
    // reject pack_format_b != 0 on the mem_format 'n' path, so leaving it set
    // would veto ALGO 3.  Q4_0 keeps dtypes.wei = s4 so the s8×s4 GEMM sees
    // packed nibbles; Q8_0 is s8.  An f32 destination keeps this off the
    // bf16-only W4A8 route.
    params.packing.pack_format_b = 0;
    params.packing.is_superblock = false;
    params.dtypes.wei = keep_packed_s4 ? data_type_t::s4 : data_type_t::s8;
    // Q4_K must stay identifiable after the packed flag is cleared: the s4
    // kernel reads wei_zp as the per-group offset only while ggml_type is 1.
    if (q4k) {
        params.packing.ggml_type = 1;
        const size_t scale_bytes
                = ggml_scale_bytes(N, K, /*use_bf16_scales=*/false);
        params.quant_params.wei_zp.buff
                = static_cast<const uint8_t *>(cached_buffer) + weight_capacity
                + scale_bytes;
        params.quant_params.wei_zp.dt = data_type_t::f32;
    } else {
        params.packing.ggml_type = 0;
        params.quant_params.wei_zp = {};
    }
    // This function runs only for pack_format_b == 1 (GGML int4/int8). The
    // caller's packed buffer stays read-only; this new buffer is writable and
    // sized for the AOCL blocked image, so the reorder may use it in place.
    // bf16 / fp32 never enter here and keep the caller's weight_cache_type.
    params.weight_cache_type = 2;
    params.wei_buffer_capacity_bytes = weight_capacity;
    params.quant_params.wei_scale.buff = static_cast<const void *>(
            static_cast<const uint8_t *>(cached_buffer) + weight_capacity);
    params.quant_params.wei_scale.dt
            = q4k ? data_type_t::f32 : data_type_t::bf16;
    const int64_t ng = static_cast<int64_t>(K) / 32;
    params.quant_params.wei_scale.dims = {ng, static_cast<int64_t>(N)};
    if (q4k) {
        params.quant_params.wei_zp.dims = {ng, static_cast<int64_t>(N)};
    }
    // Symmetric-int8 compute discriminator.  Q8_0 is a plain per-group s8
    // matrix; Q4_0 is packed s4 with the same per-group scales.  This is the
    // field the AOCL DLP sym-quant PREPACK gate keys on (wei s8 or s4,
    // compute s8).  Without it the ahead-of-time warmer skips the family and
    // the per-group N-tile reorder is packed lazily.  Per-group is
    // symmetric-only, so s8 (never u8).  The runtime per-group call still
    // runs on the AOCL execute_tile path (`ck_per_group` forces it — the custom
    // kernel has no per-group support in this path).
    params.dtypes.compute = data_type_t::s8;

    apilog_info("GGML raw unpack output: weight_capacity=", weight_capacity,
            ", scale_groups=", ng, ", wei=", (keep_packed_s4 ? "s4" : "s8"),
            ", scale_dt=", (q4k ? "f32" : "bf16"),
            ", mem_format=n, weight_cache_type=2");
    return status_t::success;
}

status_t unpack_ggml_weights_and_cache(const void *&weight, int N, int K,
        int ldb, char trans, matmul_params &params, bool is_weights_const) {
    // Infer the GGML block format from the weight dtype (contract:
    // pack_format_b == 1 means GGML; s4 -> Q4_0, s8/none -> Q8_0).  Q4_0 stays
    // packed signed s4.  The result stays raw; the
    // GEMM reorders it later.  `none` is the Q8_0 default the fused-MoE Op2
    // down-weight scratch params carry (they are built without a weight
    // dtype); it is kept as an explicit alias so that path is not disturbed.
    //
    // Reject any POSITIVELY-wrong dtype (bf16/f32/u4/...) instead of silently
    // widening the block stride to Q8_0 and reading past the packed blob.
    // Unpack does not look at the activation, so this is the dtype check
    // every caller gets, including cold-expert warming and fused-MoE Op2.
    if (params.dtypes.wei != data_type_t::s4
            && params.dtypes.wei != data_type_t::s8
            && params.dtypes.wei != data_type_t::none) {
        log_error(
                "GGML packed weights support only s4 (Q4_0) or s8 (Q8_0) "
                "weight "
                "dtypes, got ",
                dtype_info(params.dtypes.wei));
        return status_t::failure;
    }
    int ggml_type = (params.dtypes.wei == data_type_t::s4) ? kGgmlBlockQ4_0
                                                           : kGgmlBlockQ8_0;
    const bool is_superblock = params.packing.is_superblock;
    if (params.packing.ggml_type == 1) {
        if (params.dtypes.wei != data_type_t::s4
                && params.dtypes.wei != data_type_t::none) {
            log_error("GGML Q4_K requires an s4 weight, got ",
                    dtype_info(params.dtypes.wei));
            return status_t::failure;
        }
        if (K % 256 != 0) {
            log_error("GGML Q4_K requires K divisible by 256, got K=", K);
            return status_t::failure;
        }
        // The unpack writes N rows of K, element (n, k) at n*K + k.  The s4
        // packer reaches that element only for a transposed weight with
        // ldb == K.  Any other trans or ldb reads a different element, or
        // past the allocation when ldb is larger than K.
        if (trans != 't' || ldb != K) {
            log_error(
                    "GGML Q4_K requires a transposed weight with ldb=K, got "
                    "trans=",
                    trans, ", ldb=", ldb, ", K=", K);
            return status_t::failure;
        }
        ggml_type = kGgmlBlockQ4_K;
    }
    if (is_superblock && params.dtypes.wei != data_type_t::s4) {
        log_error("GGML is_superblock requires an s4 weight, got ",
                dtype_info(params.dtypes.wei));
        return status_t::failure;
    }
    if (is_superblock && (N % 8) != 0) {
        log_error("GGML superblock requires N divisible by 8, got N=", N);
        return status_t::failure;
    }

    apilog_info("GGML unpack: N=", N, ", K=", K, ", ggml_type=", ggml_type,
            ", is_superblock=", (is_superblock ? 1 : 0),
            ", weight_address=", static_cast<const void *>(weight));

    return unpack_ggml_raw_s8_and_cache(weight, N, K, ldb, trans, params,
            ggml_type, is_superblock, is_weights_const);
}

void clear_ggml_weight_unpack_cache() {
    std::lock_guard<std::mutex> lock(get_ggml_reordered_weight_cache_mutex());
    get_ggml_reordered_weight_cache().clear();
}

size_t ggml_weight_unpack_cache_size() {
    // get_size() takes the cache's own internal mutex; do not also hold the
    // module mutex here to avoid a lock-order dependency.
    return static_cast<size_t>(get_ggml_reordered_weight_cache().get_size());
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
