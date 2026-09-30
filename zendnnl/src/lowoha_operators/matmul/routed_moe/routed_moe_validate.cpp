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
 * @file routed_moe_validate.cpp
 * @brief Capability query and parameter validation for routed MoE.
 *
 * Every rule the executor depends on is stated here exactly once, so a
 * host framework can gate on this function instead of maintaining its
 * own copy of the shape rules.  The executor calls the same static
 * checks before it touches any data, which is what makes "validate
 * returned success" a genuine guarantee rather than a convention.
 *
 * Nothing in this file knows about any particular model: the rules are
 * expressed only in terms of the geometry and dtypes in the POD.
 */

#include "lowoha_operators/matmul/routed_moe/routed_moe_internal.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#if ZENDNNL_ROUTED_MOE_KERNELS_COMPILED
#include "common/zendnnl_cpuid_compat.hpp"
#include "lowoha_operators/matmul/matmul_native/common/cost_model.hpp"
#endif

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace routed_moe {

namespace {

bool checked_add_i64(int64_t a, int64_t b, int64_t &result) {
    if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b) {
        return false;
    }
    result = a + b;
    return true;
}

bool checked_mul_i64(int64_t a, int64_t b, int64_t &result) {
    if (a < 0 || b < 0
            || (a != 0 && b > std::numeric_limits<int64_t>::max() / a)) {
        return false;
    }
    result = a * b;
    return true;
}

bool checked_size(int64_t elements, size_t element_size, size_t &bytes) {
    if (elements < 0
            || static_cast<uint64_t>(elements) > static_cast<uint64_t>(
                       std::numeric_limits<size_t>::max() / element_size)) {
        return false;
    }
    bytes = static_cast<size_t>(elements) * element_size;
    return true;
}

bool checked_elements_2(
        int64_t a, int64_t b, size_t element_size, size_t &elements) {
    int64_t product = 0;
    size_t bytes = 0;
    if (!checked_mul_i64(a, b, product)
            || !checked_size(product, element_size, bytes)) {
        return false;
    }
    elements = static_cast<size_t>(product);
    return true;
}

bool checked_elements_3(int64_t a, int64_t b, int64_t c, size_t element_size,
        size_t &elements) {
    int64_t ab = 0;
    int64_t product = 0;
    size_t bytes = 0;
    if (!checked_mul_i64(a, b, ab) || !checked_mul_i64(ab, c, product)
            || !checked_size(product, element_size, bytes)) {
        return false;
    }
    elements = static_cast<size_t>(product);
    return true;
}

bool checked_extent(
        int64_t rows, int64_t stride, int64_t width, size_t element_size) {
    int64_t last_row = 0;
    int64_t extent = 0;
    size_t bytes = 0;
    return rows > 0 && stride >= width && width > 0
            && checked_mul_i64(rows - 1, stride, last_row)
            && checked_add_i64(last_row, width, extent)
            && checked_size(extent, element_size, bytes);
}

bool is_w4(const routed_moe_params &p) {
    return p.quant_scheme
            == routed_moe_quant_t::sym_per_group_w4a8_dynamic_per_token;
}

bool checked_s4_storage(int64_t num_experts, int64_t out_channels,
        int64_t in_channels, int64_t group_size, size_t &raw_bytes,
        size_t &packed_bytes) {
    if (group_size <= 0 || in_channels % group_size != 0 || in_channels % 2 != 0
            || out_channels % block_n != 0) {
        return false;
    }
    int64_t logical_elements = 0;
    int64_t groups = in_channels / group_size;
    int64_t comp_bytes_per_oc = 0;
    int64_t packed_bytes_per_oc = 0;
    int64_t packed_elements = 0;
    if (!checked_mul_i64(num_experts, out_channels, logical_elements)
            || !checked_mul_i64(logical_elements, in_channels, logical_elements)
            || !checked_size(logical_elements / 2, 1, raw_bytes)
            || !checked_mul_i64(groups, static_cast<int64_t>(sizeof(int32_t)),
                    comp_bytes_per_oc)
            || !checked_add_i64(
                    in_channels / 2, comp_bytes_per_oc, packed_bytes_per_oc)
            || !checked_mul_i64(num_experts, out_channels, packed_elements)
            || !checked_mul_i64(
                    packed_elements, packed_bytes_per_oc, packed_elements)
            || !checked_size(packed_elements, 1, packed_bytes)) {
        return false;
    }
    return packed_bytes <= std::numeric_limits<size_t>::max() - 63;
}

} // namespace

int effective_num_threads(int32_t requested) {
    const int max_threads = omp_get_max_threads();
    const int runtime_max = max_threads > 0 ? max_threads : 1;
    return requested > 0 ? std::min(requested, runtime_max) : runtime_max;
}

bool isa_supported() {
#if ZENDNNL_ROUTED_MOE_KERNELS_COMPILED
    const auto &uarch = native::detect_uarch();
    if (!uarch.avx512f || !uarch.avx512vnni || !uarch.avx512bf16) {
        return false;
    }

    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    zendnnl_cpuid_count(7, 0, eax, ebx, ecx, edx);
    constexpr unsigned avx512dq = 1u << 17;
    constexpr unsigned avx512bw = 1u << 30;
    constexpr unsigned avx512vl = 1u << 31;
    return (ebx & (avx512dq | avx512bw | avx512vl))
            == (avx512dq | avx512bw | avx512vl);
#else
    return false;
#endif
}

status_t checked_pack_sizes(int64_t num_experts, int64_t out_channels,
        int64_t in_channels, size_t &src_bytes, size_t &dst_bytes,
        int64_t &blocks_per_expert, int64_t &packed_bytes_per_oc) {
    if (num_experts <= 0 || out_channels <= 0 || in_channels <= 0) {
        return status_t::op_bad_io;
    }
    if (out_channels % block_n != 0 || in_channels % vnni_step != 0
            || in_channels > max_gemm_reduction) {
        return status_t::memory_bad_size;
    }

    packed_bytes_per_oc = packed_row_bytes_unchecked(in_channels);
    blocks_per_expert = out_channels / block_n;

    size_t src_elements = 0;
    size_t dst_elements = 0;
    int64_t total_blocks = 0;
    if (!checked_elements_3(num_experts, out_channels, in_channels,
                sizeof(int8_t), src_elements)
            || !checked_elements_3(num_experts, out_channels,
                    packed_bytes_per_oc, sizeof(int8_t), dst_elements)
            || !checked_mul_i64(num_experts, blocks_per_expert, total_blocks)) {
        return status_t::memory_bad_size;
    }

    src_bytes = src_elements;
    dst_bytes = dst_elements;
    // zendnnl_aligned_alloc rounds up by alignment - 1 internally.
    if (dst_bytes > std::numeric_limits<size_t>::max() - 63) {
        return status_t::memory_bad_size;
    }
    return status_t::success;
}

status_t checked_problem_sizes(
        const routed_moe_params &p, checked_problem_sizes_t &sizes) {
    int64_t expert_padding = 0;
    int64_t blocks_numerator = 0;
    int64_t routing_elements_i64 = 0;
    int64_t temporary = 0;

    if (!checked_mul_i64(p.num_tokens, p.topk, sizes.numel)
            || sizes.numel > max_flat_routes
            || p.num_local_experts > max_local_experts
            || !checked_mul_i64(
                    p.num_local_experts, block_m - 1, expert_padding)
            || !checked_add_i64(sizes.numel, expert_padding, sizes.max_padded)
            || sizes.max_padded > max_flat_routes
            || !checked_add_i64(
                    sizes.max_padded, block_m - 1, blocks_numerator)) {
        return status_t::memory_bad_size;
    }
    sizes.max_blocks = blocks_numerator / block_m;

    // routing = sorted + expert ids + offsets + inactive + count + cursor.
    routing_elements_i64 = sizes.max_padded;
    if (!checked_add_i64(
                routing_elements_i64, sizes.max_blocks, routing_elements_i64)
            || !checked_add_i64(sizes.max_blocks, 1, temporary)
            || !checked_add_i64(
                    routing_elements_i64, temporary, routing_elements_i64)
            || !checked_add_i64(
                    routing_elements_i64, sizes.numel, routing_elements_i64)
            || !checked_mul_i64(p.num_local_experts, 2, temporary)
            || !checked_add_i64(
                    routing_elements_i64, temporary, routing_elements_i64)) {
        return status_t::memory_bad_size;
    }
    size_t routing_bytes = 0;
    if (!checked_size(routing_elements_i64, sizeof(int32_t), routing_bytes)) {
        return status_t::memory_bad_size;
    }
    sizes.routing_elements = static_cast<size_t>(routing_elements_i64);

    if (!checked_mul_i64(2, p.intermediate_size, sizes.gate_up_oc)
            || !checked_elements_2(p.num_tokens, p.hidden_size,
                    sizeof(uint16_t), sizes.src_elements)
            || !checked_elements_2(sizes.numel, p.intermediate_size,
                    sizeof(uint16_t), sizes.intermediate_elements)
            || !checked_elements_2(sizes.numel, p.hidden_size, sizeof(uint16_t),
                    sizes.down_elements)) {
        return status_t::memory_bad_size;
    }

    const int64_t id_stride
            = p.topk_ids_stride != 0 ? p.topk_ids_stride : p.topk;
    const int64_t weight_stride
            = p.topk_weights_stride != 0 ? p.topk_weights_stride : p.topk;
    if (!checked_extent(
                p.num_tokens, p.src_stride, p.hidden_size, sizeof(uint16_t))
            || !checked_extent(
                    p.num_tokens, p.dst_stride, p.hidden_size, sizeof(uint16_t))
            || !checked_extent(p.num_tokens, id_stride, p.topk, sizeof(int32_t))
            || !checked_extent(
                    p.num_tokens, weight_stride, p.topk, sizeof(float))) {
        return status_t::memory_bad_size;
    }
    if (p.expert_map != nullptr) {
        size_t map_bytes = 0;
        if (p.expert_map_size > std::numeric_limits<int32_t>::max()
                || !checked_size(
                        p.expert_map_size, sizeof(int32_t), map_bytes)) {
            return status_t::memory_bad_size;
        }
    }

    int64_t gate_up_expert_stride = 0;
    int64_t down_expert_stride = 0;
    if (!checked_mul_i64(sizes.gate_up_oc, p.hidden_size, gate_up_expert_stride)
            || !checked_mul_i64(
                    p.hidden_size, p.intermediate_size, down_expert_stride)) {
        return status_t::memory_bad_size;
    }
    const int64_t effective_gate_up_expert_stride = p.gate_up_stride_expert != 0
            ? p.gate_up_stride_expert
            : gate_up_expert_stride;
    const int64_t effective_down_expert_stride = p.down_stride_expert != 0
            ? p.down_stride_expert
            : down_expert_stride;
    if (effective_gate_up_expert_stride < gate_up_expert_stride
            || effective_down_expert_stride < down_expert_stride) {
        return status_t::memory_bad_stride;
    }

    const size_t scale_bytes = static_cast<size_t>(size_of(p.scale_dt));
    if (scale_bytes == 0) { return status_t::memory_bad_quant; }

    if (is_w4(p)) {
        // Native S4 packing currently assumes tightly stacked experts.
        if (effective_gate_up_expert_stride != gate_up_expert_stride
                || effective_down_expert_stride != down_expert_stride) {
            return status_t::memory_bad_stride;
        }
        const int64_t gate_groups = p.hidden_size / p.gate_up_group_size;
        const int64_t down_groups = p.intermediate_size / p.down_group_size;
        int64_t gate_scale_expert_stride = 0;
        int64_t down_scale_expert_stride = 0;
        if (!checked_mul_i64(
                    gate_groups, sizes.gate_up_oc, gate_scale_expert_stride)
                || !checked_mul_i64(
                        down_groups, p.hidden_size, down_scale_expert_stride)) {
            return status_t::memory_bad_size;
        }
        if ((p.gate_up_scale_stride_expert != 0
                    && p.gate_up_scale_stride_expert
                            != gate_scale_expert_stride)
                || (p.down_scale_stride_expert != 0
                        && p.down_scale_stride_expert
                                != down_scale_expert_stride)) {
            return status_t::memory_bad_stride;
        }
        size_t ignored = 0;
        if (!checked_elements_3(p.num_local_experts, gate_groups,
                    sizes.gate_up_oc, scale_bytes, ignored)
                || !checked_elements_3(p.num_local_experts, down_groups,
                        p.hidden_size, scale_bytes, ignored)) {
            return status_t::memory_bad_size;
        }

        size_t raw_bytes = 0;
        if (!checked_s4_storage(p.num_local_experts, sizes.gate_up_oc,
                    p.hidden_size, p.gate_up_group_size, raw_bytes,
                    sizes.gate_up_packed_bytes)
                || !checked_s4_storage(p.num_local_experts, p.hidden_size,
                        p.intermediate_size, p.down_group_size, raw_bytes,
                        sizes.down_packed_bytes)) {
            return status_t::memory_bad_size;
        }
        return status_t::success;
    }

    if ((p.gate_up_scale_stride_expert != 0
                && p.gate_up_scale_stride_expert != sizes.gate_up_oc)
            || (p.down_scale_stride_expert != 0
                    && p.down_scale_stride_expert != p.hidden_size)) {
        return status_t::memory_bad_stride;
    }

    size_t ignored = 0;
    if (!checked_extent(p.num_local_experts, effective_gate_up_expert_stride,
                gate_up_expert_stride, sizeof(int8_t))
            || !checked_extent(p.num_local_experts,
                    effective_down_expert_stride, down_expert_stride,
                    sizeof(int8_t))
            || !checked_elements_2(
                    p.num_local_experts, sizes.gate_up_oc, scale_bytes, ignored)
            || !checked_elements_2(
                    p.num_local_experts, p.hidden_size, scale_bytes, ignored)) {
        return status_t::memory_bad_size;
    }

    size_t src_pack_bytes = 0;
    int64_t blocks = 0;
    int64_t packed_per_oc = 0;
    status_t status = checked_pack_sizes(p.num_local_experts, sizes.gate_up_oc,
            p.hidden_size, src_pack_bytes, sizes.gate_up_packed_bytes, blocks,
            packed_per_oc);
    if (status != status_t::success) { return status; }
    return checked_pack_sizes(p.num_local_experts, p.hidden_size,
            p.intermediate_size, src_pack_bytes, sizes.down_packed_bytes,
            blocks, packed_per_oc);
}

status_t validate_static(const routed_moe_params &p) {
    // ── geometry must be fully specified and positive ───────────────
    if (p.num_tokens <= 0 || p.hidden_size <= 0 || p.intermediate_size <= 0
            || p.num_local_experts <= 0 || p.topk <= 0) {
        return status_t::op_bad_io;
    }

    // ── every buffer the executor dereferences ──────────────────────
    if (p.src == nullptr || p.dst == nullptr || p.gate_up_weight == nullptr
            || p.down_weight == nullptr || p.gate_up_scale == nullptr
            || p.down_scale == nullptr || p.topk_ids == nullptr
            || p.topk_weights == nullptr) {
        return status_t::op_bad_io;
    }

    // ── configuration that is named but not implemented yet ─────────
    //
    // These are deliberately distinguishable from malformed input: a
    // caller that asks for SwiGLU-OAI on a well-formed problem gets
    // `unimplemented` and can fall back, rather than being told its
    // tensors are wrong.
    if (p.activation == routed_moe_activation_t::undef
            || p.quant_scheme == routed_moe_quant_t::undef) {
        return status_t::op_bad_io;
    }
    if (p.activation != routed_moe_activation_t::silu_and_mul
            && p.activation != routed_moe_activation_t::gelu_and_mul) {
        return status_t::unimplemented;
    }
    const bool w8 = p.quant_scheme
            == routed_moe_quant_t::sym_per_oc_w8a8_dynamic_per_token;
    const bool w4 = is_w4(p);
    if (!w8 && !w4) { return status_t::unimplemented; }
    if (p.apply_router_weight_on_input != 0) { return status_t::unimplemented; }
    if (p.gate_up_bias != nullptr || p.down_bias != nullptr
            || p.bias_dt != data_type_t::none) {
        return status_t::unimplemented;
    }

    // ── dtypes ──────────────────────────────────────────────────────
    if (p.src_dt != data_type_t::bf16 || p.dst_dt != data_type_t::bf16) {
        return status_t::unimplemented;
    }
    if (p.scale_dt != data_type_t::f32 && p.scale_dt != data_type_t::bf16) {
        return status_t::unimplemented;
    }
    // A dtype inconsistent with the selected implemented scheme is malformed,
    // not a request for another implementation.
    if ((w8 && p.wei_dt != data_type_t::s8)
            || (w4 && p.wei_dt != data_type_t::s4)) {
        return status_t::memory_bad_quant;
    }
    if (w4) {
        if (p.gate_up_group_size <= 0 || p.down_group_size <= 0
                || p.gate_up_group_size % s4_group_size_align != 0
                || p.down_group_size % s4_group_size_align != 0
                || p.hidden_size % p.gate_up_group_size != 0
                || p.intermediate_size % p.down_group_size != 0) {
            return status_t::memory_bad_quant;
        }
    } else if (p.gate_up_group_size != 0 || p.down_group_size != 0) {
        return status_t::memory_bad_quant;
    }

    // ── alignment ───────────────────────────────────────────────────
    //
    // Both reduction dimensions are consumed 32 elements at a time (the
    // gate/up pass reduces over hidden_size, the down pass over
    // intermediate_size), and both appear as an output-channel extent
    // tiled by block_n.
    if (p.hidden_size % block_n != 0 || p.intermediate_size % block_n != 0
            || p.hidden_size > max_gemm_reduction
            || p.intermediate_size > max_gemm_reduction) {
        return status_t::memory_bad_size;
    }

    // Never advertise or accept a call that could reach an instruction the
    // current CPU/OS/compiler combination cannot execute.
    if (!isa_supported()) { return status_t::isa_unsupported; }

    // ── activation strides ──────────────────────────────────────────
    if (p.src_stride < p.hidden_size || p.dst_stride < p.hidden_size) {
        return status_t::memory_bad_stride;
    }

    // ── weight strides: tight rows, optional trailing expert padding ─
    //
    // The packer walks 32 consecutive output-channel rows of a expert as
    // one block, so a padded output-channel stride would interleave
    // foreign bytes into the pack.  The expert stride may be larger than
    // the logical matrix so a descriptor caller can place trailing padding
    // after each expert; the packer skips that padding on its one-time read.
    if (p.gate_up_stride_oc != 0 && p.gate_up_stride_oc != p.hidden_size) {
        return status_t::memory_bad_stride;
    }
    if (p.down_stride_oc != 0 && p.down_stride_oc != p.intermediate_size) {
        return status_t::memory_bad_stride;
    }

    // ── routing strides ─────────────────────────────────────────────
    if (p.topk_ids_stride != 0 && p.topk_ids_stride < p.topk) {
        return status_t::memory_bad_stride;
    }
    if (p.topk_weights_stride != 0 && p.topk_weights_stride < p.topk) {
        return status_t::memory_bad_stride;
    }

    // ── expert map ──────────────────────────────────────────────────
    if (p.expert_map != nullptr && p.expert_map_size <= 0) {
        return status_t::op_bad_io;
    }
    if (p.expert_map == nullptr && p.expert_map_size != 0) {
        return status_t::op_bad_io;
    }

    checked_problem_sizes_t sizes;
    return checked_problem_sizes(p, sizes);
}

status_t validate_routing_ids(const routed_moe_params &p) {
    const int64_t id_stride
            = p.topk_ids_stride != 0 ? p.topk_ids_stride : p.topk;
    const int64_t limit
            = p.expert_map != nullptr ? p.expert_map_size : p.num_local_experts;

    for (int64_t m = 0; m < p.num_tokens; ++m) {
        const int32_t *row = p.topk_ids + m * id_stride;
        for (int64_t t = 0; t < p.topk; ++t) {
            const int32_t id = row[t];
            if (id < 0 || static_cast<int64_t>(id) >= limit) {
                return status_t::memory_bad_index;
            }
            if (p.expert_map != nullptr) {
                const int32_t local = p.expert_map[id];
                // A negative entry means "not resident here" and is a
                // legal way to mask a slot out.
                if (static_cast<int64_t>(local) >= p.num_local_experts) {
                    return status_t::memory_bad_index;
                }
            }
        }
    }
    return status_t::success;
}

} // namespace routed_moe

status_t group_matmul_routed_moe_query(routed_moe_capability *cap) {
    if (cap == nullptr) { return status_t::op_bad_io; }

    // Clear every advertised bit before the runtime gate.  A caller reusing a
    // previously populated object cannot mistake stale masks for support.
    *cap = routed_moe_capability {};
    if (!routed_moe::isa_supported()) { return status_t::isa_unsupported; }

    cap->block_m = static_cast<int32_t>(routed_moe::block_m);
    cap->block_n = static_cast<int32_t>(routed_moe::block_n);
    cap->vnni_step = static_cast<int32_t>(routed_moe::vnni_step);
    cap->max_kernel_rows = static_cast<int32_t>(routed_moe::max_kernel_rows);

    cap->hidden_size_align = routed_moe::block_n;
    cap->intermediate_size_align = routed_moe::block_n;

    cap->activation_mask
            = routed_moe::act_bit(routed_moe_activation_t::silu_and_mul)
            | routed_moe::act_bit(routed_moe_activation_t::gelu_and_mul);
    cap->quant_mask
            = routed_moe::quant_bit(
                      routed_moe_quant_t::sym_per_oc_w8a8_dynamic_per_token)
            | routed_moe::quant_bit(
                    routed_moe_quant_t::sym_per_group_w4a8_dynamic_per_token);
    cap->src_dtype_mask = routed_moe::dt_bit(data_type_t::bf16);
    cap->wei_dtype_mask = routed_moe::dt_bit(data_type_t::s8)
            | routed_moe::dt_bit(data_type_t::s4);
    cap->scale_dtype_mask = routed_moe::dt_bit(data_type_t::f32)
            | routed_moe::dt_bit(data_type_t::bf16);

    cap->supports_expert_map = 1;
    cap->supports_bias = 0;
    cap->supports_router_weight_on_input = 0;

    cap->max_topk = routed_moe::max_topk;
    cap->max_local_experts = routed_moe::max_local_experts;
    cap->max_s4_kernel_rows
            = static_cast<int32_t>(routed_moe::max_s4_kernel_rows);
    cap->s4_group_size_align = routed_moe::s4_group_size_align;

    return status_t::success;
}

status_t group_matmul_routed_moe_validate(const routed_moe_params &params) {
    const status_t s = routed_moe::validate_static(params);
    if (s != status_t::success) { return s; }
    return routed_moe::validate_routing_ids(params);
}

int64_t group_matmul_routed_moe_packed_row_bytes(
        int64_t in_channels, data_type_t wei_dt) {
    if (wei_dt != data_type_t::s8 || in_channels <= 0
            || in_channels % routed_moe::vnni_step != 0
            || in_channels > routed_moe::max_gemm_reduction) {
        return 0;
    }
    return routed_moe::packed_row_bytes_unchecked(in_channels);
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
