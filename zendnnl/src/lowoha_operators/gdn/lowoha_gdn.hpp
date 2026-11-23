/********************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LOWOHA_GDN_HPP
#define LOWOHA_GDN_HPP

#include "common/zendnnl_api.hpp"
#include "lowoha_operators/gdn/lowoha_gdn_common.hpp"

namespace zendnnl {
namespace lowoha {
namespace gdn {

/**
 * @brief Execute Gated DeltaNet over one batch of decode and prefill
 *        sequences: causal conv1d, gating, delta-rule recurrence and output.
 *
 * The batch is vLLM's non-speculative GDN batch as is. Its first
 * @c params.num_decodes sequences are single-token continuations and run the
 * fused decode kernel; the remaining sequences are prefills and run the
 * prefill kernel (token-recurrent, or the chunked delta rule once the
 * prefills average 16 tokens with bf16 sources, 32 with f16 and 24 with
 * f32). Sequence i owns token rows
 * [cu_seqlens[i], cu_seqlens[i + 1]) of every per-token tensor and pool slot
 * state_indices[i]; see @c gdn_params for cu_seqlens, num_decodes and
 * has_initial_state.
 *
 * Every sequence that runs leaves its final conv and SSM states in its pool
 * slot, unless @c params.update_state is false: that prefill-only call leaves
 * both pools unchanged, and its sequences may share a slot. A sequence whose
 * slot equals @c params.pad_slot_id touches neither pool: a padded decode
 * writes a zero output row and a padded prefill leaves its output rows
 * untouched, as does a prefill with no tokens. Otherwise two sequences of one
 * call must not share a slot.
 *
 * A prefill continued from the state an earlier call left gives the output
 * bits of one call over all its tokens when state_dt is f32 and both calls
 * run the same kernel, pinned prefill_algo and the same rounding options
 * (round_conv_out, round_core_out): the recurrent form at any cut, the
 * chunked form at a cut on one of the sequence's 32-token chunk boundaries.
 * The final state matches too when both calls write it (update_state).
 *
 * With params.spec_state_indices set, the call verifies speculative draft
 * tokens instead (see gdn_params): at the same num_threads it gives the bits
 * of as many decode calls over the same sequences on a conv state of the same
 * strides, and it stores the state after every token.
 *
 * @param mixed_qkv     [num_tokens][qkv_stride] packed q, k, v (pre-conv when
 *                      params.conv_width > 0), or q alone when params.k and
 *                      params.v are set
 * @param a, b          [num_tokens][a_stride / b_stride], num_v_heads used:
 *                      the gate projections, or g and beta when
 *                      params.gate_input is precomputed
 * @param A_log         [num_v_heads] (not read with precomputed gates)
 * @param dt_bias       [num_v_heads] (not read with precomputed gates)
 * @param conv_weight   [conv_dim][conv_width] depthwise weights (unpacked)
 * @param conv_bias     [conv_dim] or nullptr
 * @param conv_state    conv state pool [slot][tap][channel] (strided)
 * @param ssm_state     SSM state pool [slot][v_head][...] (strided)
 * @param state_indices [batch] pool slot per sequence (int32); not read by
 *                      speculative verify
 * @param output        [num_tokens][num_v_heads][head_v_dim] (strided)
 * @param params        GDN parameters
 *
 * @return status_t::success or status_t::failure
 */
ZENDNNL_API status_t gdn_direct(const void *mixed_qkv, const void *a,
        const void *b, const void *A_log, const void *dt_bias,
        const void *conv_weight, const void *conv_bias, void *conv_state,
        void *ssm_state, const int32_t *state_indices, void *output,
        gdn_params &params);

/**
 * @brief Whether gdn_direct runs its native AVX-512 kernels for @p params on
 *        this CPU.
 *
 * The native kernels need AVX512-F, AVX512-BW/VL and AVX512-BF16, head_k_dim
 * and head_v_dim multiples of 16 up to 256, and conv_width up to 8; other
 * calls run the scalar reference kernel. Only those fields of @p params are
 * read, and a field left at 0 does not restrict the answer, so a
 * default-constructed gdn_params asks about the CPU alone. params.kernel and
 * ZENDNNL_GDN_KERNEL are not considered.
 */
ZENDNNL_API bool gdn_native_supported(const gdn_params &params);

} // namespace gdn
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_GDN_HPP
