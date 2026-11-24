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

/// N-tile flat parallel's microkernels over `nr32_pack.hpp` packs.
///
/// A call multiplies up to `max_mr` source rows by one step of columns and
/// stores what the epilogue makes of them.  A step is two vectors of
/// columns: 16 each for bf16, s8 and s4 (one zmm of each block row), 32 each
/// for f16.  Where the two vectors' weights, scales, sums and biases come
/// from is the caller's choice, which is how one pack serves every
/// activation:
///   * plain and swiglu_pairs: a whole block, the second vector right after
///     the first (f16: one 32-column vector);
///   * silu_split and gelu_split: the gate columns and the up columns of a
///     `[gate | up]` weight, `I` columns apart; with `I` off the vector
///     grid, the up columns start inside a vector and the `unaligned`
///     kernels assemble them from two.
/// A split step's vectors are half blocks for bf16, s8 and s4, so it reads
/// every other line of the two blocks.  Their `wide` kernels instead read a
/// whole gate block and the whole up block `b_next` bytes on, four vectors,
/// for 32 outputs, so each reads its weights as dense streams.
/// The math of each output element is the custom kernel's: the same dot
/// products over K in the same order, the same dequantization and the same
/// activation, so outputs match it bit for bit.

#ifndef ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_NR32_UKERNEL_HPP
#define ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_NR32_UKERNEL_HPP

#include <cstddef>
#include <cstdint>

#include "nr32_pack.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace ntile_flat_parallel {
namespace nr32 {

/// How a quantized weight's kernels apply scales along K.
struct scale_groups {
    /// The weight has more than one scale group (s4 kernels sum by group
    /// whatever their count).
    bool weight = false;
    /// Each source row has one scale per weight scale group, not one.
    bool source = false;

    /// Whether s8 kernels sum by group.
    bool any() const { return weight || source; }
};

/// Rows a call of `wei`'s kernels takes (`format_desc::max_mr`): 12 for s8,
/// 6 for s4, 8 for bf16 and f16.
int max_mr(data_type_t wei);

/// What a call stores from a step's two vectors.
enum class epilogue : uint8_t {
    /// Both vectors, as computed.
    plain,
    /// SwiGLU-OAI over the (gate, up) pairs of adjacent columns.
    swiglu_pairs,
    /// silu(gate) * up and gelu(gate) * up, the first vector being the gate.
    silu_split,
    gelu_split,
};

/// Whether `ep` writes one output per (gate, up) pair.
constexpr bool gated(epilogue ep) {
    return ep != epilogue::plain;
}

/// Whether `ep` reads gate and up from separate columns.
constexpr bool split(epilogue ep) {
    return ep == epilogue::silu_split || ep == epilogue::gelu_split;
}

enum class bias_kind : uint8_t { none, bf16, f16, f32 };

/// One kernel call.
struct ukernel_args {
    /// Source rows, `lda` elements apart: bf16, f16, or s8 bytes (XOR-ed
    /// with 0x80 for the `src_u8` s8 kernels).
    const void *a = nullptr;
    int lda = 0;
    int K = 0;
    /// The first vector's weights in the first block row.  Each next block
    /// row is one row stride on (128 bytes, 64 for f16).
    const uint8_t *b = nullptr;
    /// Bytes from the first vector's weights to the second's, in any row.
    ptrdiff_t b_next = 0;
    /// `unaligned` kernels: the second vector's columns start `shift` lanes
    /// into the vector at `b + b_next` and continue into the one at
    /// `b_next_hi`.
    const uint8_t *b_next_hi = nullptr;
    int shift = 0;
    /// s8 and s4: the first vector's sums and f32 scales, `col_group`
    /// columns from one scale group to the next.
    const int32_t *comp = nullptr;
    const float *scale = nullptr;
    int col_group = 0;
    int group_size = 0;
    /// Columns from the first vector's sums, scales and bias to the second's.
    int col_next = 0;
    /// s8 and s4: one f32 scale per source row, or with `src_groups`, one per
    /// weight scale group of each row, rows `src_groups` scales apart.
    const float *src_scale = nullptr;
    int src_groups = 0;
    /// s4 with `src_groups`: the expert's largest weight scale magnitude
    /// (`packed_expert::max_scale`).
    float max_scale = 0.0f;
    /// The first vector's bias, or null.
    const void *bias = nullptr;
    bias_kind bias_t = bias_kind::none;
    /// Row 0's first output column; rows `ldc` elements apart.  bf16, or f16
    /// for f16 weights.
    void *c = nullptr;
    int ldc = 0;
    /// Output columns to store: the step's full width, fewer at the end of
    /// a weight whose columns end inside a step.
    int cols = 0;
};

using ukernel_fn = void (*)(const ukernel_args &);

/// Whether this CPU, and the build, runs `wei`'s kernels.
bool isa_supported(data_type_t wei);

/// The kernel for `mr` rows of `wei` with epilogue `ep`, or nullptr.
/// `src_u8` (a format with `u8_src` kernels: s8) reads source bytes already
/// XOR-ed with 0x80; `unaligned` (split epilogues only) assembles misaligned
/// up columns; `groups` (quantized formats) says how the scales are grouped
/// along K.
ukernel_fn select(data_type_t wei, int mr, epilogue ep, bool src_u8,
        bool unaligned, scale_groups groups = {});

/// Rows a call of `wei`'s wide split kernels takes
/// (`format_desc::max_mr_wide`): 6 for s8 and bf16, 3 for s4, 0 for a type
/// without them.
int max_mr_wide(data_type_t wei);

/// The wide split kernel for `mr` rows of `wei` with split epilogue `ep`, or
/// nullptr.  `two_chains` (`mr` up to the format's `two_chain_rows`: bf16,
/// three) sums even and odd K pairs in two accumulators, as the narrow
/// kernels do at three rows or fewer.
ukernel_fn select_wide(data_type_t wei, int mr, epilogue ep, bool src_u8,
        bool two_chains, scale_groups groups = {});

ukernel_fn select_s8(int mr, epilogue ep, bool src_u8, bool unaligned);
ukernel_fn select_s8_groups(
        int mr, epilogue ep, bool src_u8, bool unaligned, bool src_groups);
ukernel_fn select_bf16(int mr, epilogue ep, bool unaligned);
ukernel_fn select_f16(int mr, epilogue ep, bool unaligned);
ukernel_fn select_s4(int mr, epilogue ep, bool unaligned, bool src_groups);
ukernel_fn select_s8_wide(int mr, epilogue ep, bool src_u8);
ukernel_fn select_s8_groups_wide(
        int mr, epilogue ep, bool src_u8, bool src_groups);
ukernel_fn select_bf16_wide(int mr, epilogue ep, bool two_chains);
ukernel_fn select_s4_wide(int mr, epilogue ep, bool src_groups);
bool f16_kernels_built();

} // namespace nr32
} // namespace ntile_flat_parallel
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // ZENDNNL_GROUP_MATMUL_NTILE_FLAT_PARALLEL_NR32_UKERNEL_HPP
