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

#ifndef LOWOHA_HSTU_ATTENTION_TILED_HPP
#define LOWOHA_HSTU_ATTENTION_TILED_HPP

#include "common/config_params.hpp"
#include "lowoha_operators/hstu_attention/lowoha_hstu_attention.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Causal mask, BF16 Q/K/V and an AVX512-BF16/BW/VL CPU.
bool hstu_attention_tiled_supported(const hstu_attention_params &params);

/// Tiled kernel. K^T and V are packed once per (256-key block, head) in the
/// `vdpbf16ps` pair layout; query tiles, sized to the thread count and run in
/// SDPA order on a dynamic schedule, visit only the key blocks of their causal
/// band.
/// Q*K^T runs on an AVX512-BF16 tile kernel whose epilogue applies silu
/// (`group_matmul_act_avx512::silu_avx512`, or with `fast_silu` a cheaper
/// approximation of relative error <= 1.2e-4), the row scale and the band
/// mask in registers and stores P in BF16; P*V runs on the packed V blocks
/// (`hstu_gemm_ck/`).
status_t hstu_attention_tiled(const void *query, const void *key,
        const void *value, void *output, const hstu_attention_params &params,
        bool fast_silu = false);

/// ZENDNNL_HSTU_ATTENTION_FAST_SILU: -1 when `value` is unset, empty or not a
/// boolean (0/false/off/no, 1/true/on/yes), else 0 or 1.
inline int parse_fast_silu_env(const char *value) {
    bool on = false;
    return zendnnl::common::parse_cache_bool(value, on) ? int(on) : -1;
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_ATTENTION_TILED_HPP
