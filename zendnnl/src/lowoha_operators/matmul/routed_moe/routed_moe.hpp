/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file routed_moe.hpp
 * @brief Complete descriptor and control surface for routed MoE.
 *
 * The public @c routed_fused_moe_direct entry point accepts projection and
 * routing descriptors, then normalizes eligible calls into
 * @c routed_moe_params for the specialized executor.
 */

#ifndef LOWOHA_ROUTED_MOE_HPP
#define LOWOHA_ROUTED_MOE_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common/data_types.hpp"
#include "common/error_status.hpp"
#include "common/zendnnl_api.hpp"
#include "lowoha_operators/matmul/group_matmul/group_matmul_direct.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

// The sibling public headers in this namespace pick `status_t` and
// `data_type_t` up transitively from `lowoha_common.hpp`, which drags in
// the config/JSON/LRU chain.  This header names them explicitly instead
// so it stays self-contained and cheap to include: the routed-MoE
// contract needs two enums, not the operator-config machinery.  Both
// aliases resolve to exactly the types the rest of the namespace uses.
using status_t = zendnnl::interface::status_t;
using data_type_t = zendnnl::interface::data_type_t;

/**
 * @brief One expert-stacked projection used by routed MoE.
 *
 * Weights are tightly stacked by expert. One stored expert matrix has
 * `trans_weight ? output_size : input_size` rows and leading dimension `ldb`.
 * Bias, when present, is tightly stacked as [num_experts, output_size].
 */
struct group_matmul_projection_params {
    int output_size = 0;
    int input_size = 0;
    bool trans_weight = true;
    const void *weight = nullptr;
    /// Bytes the caller guarantees are writable at each expert weight slice.
    /// 0 (default) means exactly that expert's logical weight extent.  The
    /// generic fallback forwards this to group Matmul.  The routed fast
    /// executor uses a larger value as the source stride between experts while
    /// producing its normal contiguous out-of-place pack; it does not yet
    /// reorder into the caller's buffer.
    size_t wei_buffer_capacity_bytes = 0;
    int ldb = 0;
    const void *bias = nullptr;
    float alpha = 1.0f;
    float beta = 0.0f;
    bool weight_is_const = true;
    matmul_params params;
};

/**
 * @brief Dense token-to-expert routing metadata.
 */
struct group_matmul_routing_params {
    const int32_t *topk_ids = nullptr;
    int topk_ids_stride = 0;
    const float *topk_weights = nullptr;
    int topk_weights_stride = 0;
    const int32_t *expert_map = nullptr;
    int expert_map_size = 0;
    bool skip_weighted = false;
    bool reduce_output = true;
};

/**
 * @brief Route a token matrix and execute one or two expert projections.
 *
 * The source is a single logical [num_tokens, primary.input_size] matrix.
 * Routing duplicates its rows into expert groups internally. Eligible DA8W8
 * and symmetric per-group W4A8 calls use the routed executor for both prefill
 * and decode; unsupported configurations use the vector group-matmul
 * implementation.
 *
 * `primary.params.num_threads` controls routing and the fast executor.
 * Projection weights and optional biases are tightly stacked by expert.
 */
ZENDNNL_API status_t routed_fused_moe_direct(const char layout_src,
        const bool trans_src, const void *token_src, const int token_src_ld,
        const int num_tokens, const int num_experts, const int topk,
        void *moe_output, const int moe_output_ld,
        const group_matmul_projection_params &primary,
        const group_matmul_routing_params &routing,
        const group_matmul_projection_params *secondary = nullptr,
        const grp_matmul_gated_act_params *gated_act = nullptr);

/**
 * @brief Gated activation applied between the two projections.
 *
 * The enumerators intentionally match the numbering of
 * @c grp_matmul_gated_act_t so a caller can translate one to the other
 * without a lookup table.  Values the executor cannot run yet are
 * declared here (rather than omitted) so a caller can name them and get
 * a deterministic @c status_t::unimplemented instead of a parse error;
 * adding support later does not change the enumeration.
 */
enum class routed_moe_activation_t : int32_t {
    undef = 0,
    /// dst = silu(gate) * up, weights laid out as split halves [gate | up].
    silu_and_mul = 1,
    /// dst = gelu(gate) * up, split halves [gate | up].  GELU is the erf
    /// form 0.5 * x * (1 + erf(x / sqrt(2))), computed with the same vector
    /// approximation as @c grp_matmul_gated_act_t::gelu_and_mul.
    gelu_and_mul = 2,
    /// Reserved: SwiGLU-OAI on an interleaved [g0,u0,g1,u1,...] layout.
    /// Currently @c unimplemented.
    swiglu_oai_mul = 3
};

/**
 * @brief Quantization scheme of the routed-MoE weights and activations.
 *
 * As with the activation enum, unsupported schemes are named so callers
 * receive @c status_t::unimplemented rather than being silently treated
 * as the supported one.
 */
enum class routed_moe_quant_t : int32_t {
    undef = 0,
    /// Symmetric per-output-channel int8 weights, dynamic per-token int8
    /// activations, no zero points.  This is the scheme the executor
    /// implements.
    sym_per_oc_w8a8_dynamic_per_token = 1,
    /// Reserved: symmetric per-group (sub-K) int8 weights.
    sym_per_group_w8a8_dynamic_per_token = 2,
    /// Reserved: asymmetric int8 weights (weight zero points).
    asym_per_oc_w8a8_dynamic_per_token = 3,
    /// Reserved: 4-bit weights with int8 activations.
    sym_per_oc_w4a8_dynamic_per_token = 4,
    /// Symmetric signed-4-bit weights with one scale per K group and output
    /// channel, plus dynamically quantized per-token activations.  This is a
    /// distinct value from the historical per-output-channel W4 spelling
    /// above: callers compiled against that documented meaning must never be
    /// silently reinterpreted as per-group quantization.
    sym_per_group_w4a8_dynamic_per_token = 5
};

/**
 * @brief Capabilities of the routed-MoE executor in this library build.
 *
 * Used by internal validation tests. Public callers use
 * @c routed_fused_moe_direct, which owns fast-path eligibility and fallback.
 *
 * The `*_mask` fields are bitmasks over the corresponding enumeration:
 * bit `1u << static_cast<int>(value)` is set when `value` is supported.
 */
struct routed_moe_capability {
    /// Token-block quantum of the sorted/padded routing representation.
    int32_t block_m = 0;
    /// Output-channel tile width the micro-kernels are written for.
    int32_t block_n = 0;
    /// VNNI k-grouping of the packed weight layout.
    int32_t vnni_step = 0;
    /// Largest row count a single W8 micro-kernel instantiation handles.
    int32_t max_kernel_rows = 0;

    /// @c hidden_size must be a positive multiple of this.
    int64_t hidden_size_align = 0;
    /// @c intermediate_size must be a positive multiple of this.
    int64_t intermediate_size_align = 0;

    uint32_t activation_mask = 0; ///< over routed_moe_activation_t
    uint32_t quant_mask = 0; ///< over routed_moe_quant_t
    uint32_t src_dtype_mask = 0; ///< over data_type_t (src == dst)
    uint32_t wei_dtype_mask = 0; ///< over data_type_t
    uint32_t scale_dtype_mask = 0; ///< over data_type_t

    int32_t supports_expert_map = 0; ///< global->local routing id remap
    int32_t supports_bias = 0; ///< per-projection bias
    int32_t supports_router_weight_on_input = 0;

    /// Absolute representation ceilings. A concrete call may have a lower
    /// coupled limit (for example, num_tokens * topk must also fit int32).
    int64_t max_topk = 0;
    int64_t max_local_experts = 0;

    // Append-only W4 capability details.
    int32_t max_s4_kernel_rows = 0;
    int64_t s4_group_size_align = 0;
};

/**
 * @brief Normalized parameters for the routed-MoE fast executor.
 *
 * Describes one complete MoE block:
 *
 *   1. quantize @c src [num_tokens, hidden_size] per token to int8;
 *   2. for every routed (token, slot), run the gate/up projection
 *      against @c gate_up_weight and fold the gated activation
 *      (@c silu_and_mul or @c gelu_and_mul) in;
 *   3. requantize that intermediate per row;
 *   4. run the down projection against @c down_weight, scale by the
 *      router weight and scatter;
 *   5. reduce the @c topk contributions of each token into @c dst.
 *
 * Layout contract (validated, never assumed):
 *
 *   - @c src / @c dst : [num_tokens, hidden_size], row strides
 *     @c src_stride / @c dst_stride (in elements, >= hidden_size).
 *   - @c gate_up_weight : [num_local_experts, 2 * intermediate_size,
 *     hidden_size], row-major and tightly packed within an expert.
 *     `gate_up_stride_expert` may add trailing padding between experts.
 *     The output-channel axis is split halves: rows
 *     [0, intermediate_size) are the gate projection and rows
 *     [intermediate_size, 2 * intermediate_size) the up projection.
 *   - @c down_weight : [num_local_experts, hidden_size,
 *     intermediate_size], same packing rule; `down_stride_expert` may add
 *     trailing padding between experts. S8 weights occupy one byte per
 *     logical element.  S4 weights use the library's canonical packed-nibble
 *     convention (even logical index in the low nibble).
 *   - For per-output-channel W8, @c gate_up_scale is
 *     [num_local_experts, 2 * intermediate_size] and @c down_scale is
 *     [num_local_experts, hidden_size].  For per-group W4 they are
 *     [num_local_experts, hidden_size / gate_up_group_size,
 *     2 * intermediate_size] and
 *     [num_local_experts, intermediate_size / down_group_size, hidden_size].
 *     Scale tensors may be f32 or bf16 and must be tightly contiguous.
 *   - @c topk_ids : [num_tokens, topk] int32 expert ids, row stride
 *     @c topk_ids_stride.  Ids index the LOCAL expert range
 *     [0, num_local_experts) unless @c expert_map is supplied.
 *   - @c topk_weights : [num_tokens, topk] f32 router weights, row
 *     stride @c topk_weights_stride.
 *
 * Expert parallelism: a caller sharding experts across ranks may either
 * pre-map routing ids into the local range itself (leave @c expert_map
 * null), or pass a @c expert_map of @c expert_map_size entries mapping
 * a global id to a local id, using a negative entry for "not resident on
 * this rank".  Slots that map to a negative id contribute nothing to
 * their token's reduction.
 *
 * Weight identity: @c gate_up_cache_key / @c down_cache_key supply the
 * identity component of the library's permanent packed-weight cache.  The
 * complete internal key also includes tensor role (gate/up or down), expert
 * count, output/input geometry, packed-layout version and dtype.  Reusing one
 * explicit identity for a different geometry therefore creates a distinct
 * pack rather than aliasing an undersized buffer.  Pass null to use the weight
 * pointer as the identity component, which is appropriate for model
 * parameters at stable addresses.
 */
struct routed_moe_params {
    // ── problem geometry ────────────────────────────────────────────
    int64_t num_tokens = 0; ///< M
    int64_t hidden_size = 0; ///< K
    int64_t intermediate_size = 0; ///< N (per gate/up half)
    int64_t num_local_experts = 0; ///< E resident on this rank
    int64_t topk = 0; ///< experts selected per token

    // ── activations ─────────────────────────────────────────────────
    const void *src = nullptr;
    int64_t src_stride = 0; ///< elements between rows of src
    data_type_t src_dt = data_type_t::none;

    void *dst = nullptr;
    int64_t dst_stride = 0; ///< elements between rows of dst
    data_type_t dst_dt = data_type_t::none;

    // ── weights ─────────────────────────────────────────────────────
    const void *gate_up_weight = nullptr;
    const void *down_weight = nullptr;
    data_type_t wei_dt = data_type_t::none;
    /// Elements between experts.  0 selects the tight default; a larger value
    /// adds trailing padding after each expert:
    /// 2 * intermediate_size * hidden_size for gate/up and
    /// hidden_size * intermediate_size for down.
    int64_t gate_up_stride_expert = 0;
    int64_t down_stride_expert = 0;
    /// Elements between output-channel rows.  0 selects the tight
    /// default (hidden_size / intermediate_size respectively).
    int64_t gate_up_stride_oc = 0;
    int64_t down_stride_oc = 0;

    // ── weight scales ───────────────────────────────────────────────
    const void *gate_up_scale = nullptr;
    const void *down_scale = nullptr;
    data_type_t scale_dt = data_type_t::none;
    /// Elements between experts; 0 selects the tight default.
    int64_t gate_up_scale_stride_expert = 0;
    int64_t down_scale_stride_expert = 0;

    // ── optional bias (currently rejected as unimplemented) ─────────
    const void *gate_up_bias = nullptr;
    const void *down_bias = nullptr;
    data_type_t bias_dt = data_type_t::none;

    // ── routing ─────────────────────────────────────────────────────
    const int32_t *topk_ids = nullptr;
    int64_t topk_ids_stride = 0; ///< elements between rows; 0 => topk
    const float *topk_weights = nullptr;
    int64_t topk_weights_stride = 0; ///< elements between rows; 0 => topk
    const int32_t *expert_map = nullptr; ///< optional global->local map
    int64_t expert_map_size = 0; ///< entries in expert_map

    // ── configuration ───────────────────────────────────────────────
    routed_moe_activation_t activation = routed_moe_activation_t::undef;
    routed_moe_quant_t quant_scheme = routed_moe_quant_t::undef;
    /// Pre-multiply src rows by the router weight instead of scaling the
    /// down-projection output.  Currently rejected as unimplemented.
    int32_t apply_router_weight_on_input = 0;
    /// OMP threads for the executor; <= 0 means "ask the runtime".
    int32_t num_threads = 0;

    // ── packed-weight cache identity ────────────────────────────────
    const void *gate_up_cache_key = nullptr;
    const void *down_cache_key = nullptr;

    // ── append-only W4 per-group contract ───────────────────────────
    //
    // These fields were appended rather than changing the meaning of any
    // existing enum value or member.  They are consumed only when
    // quant_scheme == sym_per_group_w4a8_dynamic_per_token and must otherwise
    // remain zero.
    int64_t gate_up_group_size = 0; ///< K elements per W13 scale group
    int64_t down_group_size = 0; ///< K elements per W2 scale group
};

static_assert(sizeof(routed_moe_activation_t) == sizeof(int32_t),
        "routed-MoE activation enum size changed");
static_assert(sizeof(routed_moe_quant_t) == sizeof(int32_t),
        "routed-MoE quant enum size changed");
static_assert(std::is_standard_layout<routed_moe_capability>::value,
        "routed-MoE capability must remain standard-layout");
static_assert(std::is_trivially_copyable<routed_moe_capability>::value,
        "routed-MoE capability must remain trivially copyable");
static_assert(std::is_standard_layout<routed_moe_params>::value,
        "routed-MoE params must remain standard-layout");
static_assert(std::is_trivially_copyable<routed_moe_params>::value,
        "routed-MoE params must remain trivially copyable");

/**
 * @brief Report what the routed-MoE executor in this build supports.
 *
 * @param cap  Caller-allocated result, overwritten on success.
 * @return @c status_t::success, @c status_t::isa_unsupported when this
 *         CPU/OS/build cannot execute the required AVX-512 VNNI+BF16 kernel
 *         (all capability masks are cleared), or @c status_t::op_bad_io when
 *         @c cap is null.
 */
ZENDNNL_API status_t group_matmul_routed_moe_query(routed_moe_capability *cap);

/**
 * @brief Decide whether the routed-MoE executor can run @p params.
 *
 * Writes nothing.  The only caller data it reads is the routing id array,
 * which it bound-checks in O(num_tokens * topk). The fast executor invokes
 * these same checks before touching caller buffers or packed-cache state.
 *
 * @return @c status_t::success when the routed executor will execute
 *         this call.  Otherwise a specific rejection:
 *         @c op_bad_io (null pointer or non-positive extent),
 *         @c memory_bad_size
 *         (dimension not a supported multiple or a derived count/byte extent
 *         overflows the internal representation), @c memory_bad_stride
 *         (stride too small / non-tight weights), @c memory_bad_index
 *         (routing id or expert-map entry out of range),
 *         @c memory_bad_quant (weight dtype inconsistent with the selected
 *         scheme) or @c unimplemented (named-but-unsupported quantization
 *         scheme, activation, bias, router-weight-on-input, or dtype), or
 *         @c isa_unsupported when the required AVX-512 ISA is unavailable.
 */
ZENDNNL_API status_t group_matmul_routed_moe_validate(
        const routed_moe_params &params);

/**
 * @brief Drop every packed weight held by either whole-MoE fast path.
 *
 * The standalone routed executor and the vector-ABI private ALGO4 executor
 * maintain separate cache representations. This is their canonical shared
 * generation boundary: a host that reloads or frees a model must call it
 * first, so neither cache can retain a pack keyed by recycled storage.
 *
 * Safe to call when nothing is cached.  This is a model-lifecycle operation:
 * callers must not invoke it concurrently with ANY routed-MoE execution, not
 * merely executions using the weights being retired.  The implementation
 * serializes flush against active routed calls as a memory-safety backstop;
 * after the function returns, all calls that could reference old packs have
 * completed and the host may release the corresponding weights.
 */
ZENDNNL_API void group_matmul_routed_moe_flush_weight_cache();

/**
 * @brief Bytes per packed output-channel row for a given input width.
 *
 * Despite the API's historical `row_bytes` name, this is an AMORTIZED byte
 * count per logical output channel, not the stride of a physically contiguous
 * packed row.  Packing is block-major.  For every 32-channel block it stores:
 *
 *   [in_channels / 4][32][4] int8 quants, then [32] int32 compensation.
 *
 * Thus a block occupies 32 * (in_channels + sizeof(int32_t)) bytes and the
 * complete tensor occupies num_experts * out_channels times the returned
 * value.  The value is exposed for destination sizing only.
 *
 * @return the row size in bytes, or 0 if the arguments are unsupported.
 */
ZENDNNL_API int64_t group_matmul_routed_moe_packed_row_bytes(
        int64_t in_channels, data_type_t wei_dt);

/**
 * @brief Pack int8 MoE weights into the executor's VNNI layout.
 *
 * Normally unnecessary — the routed executor packs on first use and
 * caches the result.  Exposed so the layout can be inspected and
 * diffed byte-for-byte by tests.
 *
 * @param src          [num_experts, out_channels, in_channels] int8.
 * @param dst          [num_experts][out_channels / 32] blocks in the physical
 *                     layout documented above.  Its required byte size is
 *                     num_experts * out_channels * packed_row_bytes.
 * @param num_threads  <= 0 means "ask the runtime".
 */
ZENDNNL_API status_t group_matmul_routed_moe_pack_weights(const void *src,
        void *dst, int64_t num_experts, int64_t out_channels,
        int64_t in_channels, data_type_t wei_dt, int num_threads);

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_ROUTED_MOE_HPP
