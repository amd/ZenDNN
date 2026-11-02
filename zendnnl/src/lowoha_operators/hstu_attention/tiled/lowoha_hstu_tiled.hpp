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

/// Entry points of the tiled kernel, one per data-type policy of `run_tiled`
/// (lowoha_hstu_tiled_driver.hpp): BF16 in bf16/, dynamic INT8 in int8/.

#ifndef LOWOHA_HSTU_TILED_HPP
#define LOWOHA_HSTU_TILED_HPP

#include "lowoha_operators/hstu_attention/lowoha_hstu_attention.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Causal mask, BF16 Q/K/V and an AVX512-BF16/BW/VL CPU.
bool hstu_attention_tiled_supported(const hstu_attention_params &params);

/// BF16 policy. K^T and V are packed once per (256-key block, head) in the
/// `vdpbf16ps` pair layout; query tiles, sized to the thread count and run in
/// SDPA order on a dynamic schedule, visit only the key blocks of their causal
/// band.
/// Q*K^T runs on an AVX512-BF16 tile kernel whose epilogue applies silu
/// (`group_matmul_act_avx512::silu_avx512`, or with `fast_silu` a cheaper
/// approximation of relative error <= 1.2e-4), the row scale and the band
/// mask in registers and stores P in BF16; P*V runs on the packed V blocks
/// (`hstu_gemm_ck/`).
status_t hstu_attention_tiled_bf16(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params,
        bool fast_silu = false);

/// AVX512-VNNI/BW/VL, which the INT8 policy needs on top of what
/// `hstu_attention_tiled_supported` checks.
bool hstu_attention_tiled_int8_isa();

/// Dynamic-INT8 policy: INT8 Q*K^T (`is_qk_quant`) and / or P*V
/// (`is_pv_quant`).
/// - INT8 Q*K^T: Q is quantized per token in its tile copy (alpha folded
///   into the scale) and K per token while packing, with the packed column
///   sums of the S8 K^T; Q*K^T runs as U8 (Q + 128) x S8 on `vpdpbusd`, and
///   its epilogue dequantizes and applies silu and the band mask.
/// - INT8 P*V: V is quantized per (key block, channel) while packing, and
///   P = mask * silu per (row, key block) to S8 + 128 after the tile's
///   Q*K^T of the block; P*V runs on `ck::int8::gemm_block`, dequantized
///   by the P scale * scale(i) and the V scales and added to O in FP32.
/// Otherwise the half runs on BF16 as in `hstu_attention_tiled_bf16`, and
/// `fast_silu` selects the same silu as there. Returns `unimplemented` for
/// calls the tiled kernel does not support, without VNNI, without a flag, or
/// with `is_qk_quant` and qk_dim > kInt8MaxQkDim.
status_t hstu_attention_tiled_int8(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params,
        bool fast_silu = false);

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_TILED_HPP
