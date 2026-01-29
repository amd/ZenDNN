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

#ifndef _LOWOHA_CONV_ONEDNN_KERNEL_HPP
#define _LOWOHA_CONV_ONEDNN_KERNEL_HPP

#include "lowoha_conv_utils.hpp"
#include "lowoha_conv_common.hpp"

#if ZENDNNL_DEPENDS_ONEDNN
#include "dnnl.hpp"
#include "conv_cache_key.hpp"  // For Key_conv
using namespace dnnl;
#endif

namespace zendnnl {
namespace lowoha {
namespace conv {

#if ZENDNNL_DEPENDS_ONEDNN

/**
 * @brief Reorder and cache convolution weights
 *
 * This function handles weight reordering from HWIO format to OneDNN's
 * optimal blocked format. It implements conditional caching based on
 * whether weights are constant.
 *
 * ReorderAndCacheWeights pattern:
 * - If is_weights_const=true: Uses LRU cache for reordered weights
 * - If is_weights_const=false: Reorders directly without caching
 *
 * @param key               Cache key identifying the weight configuration
 * @param src_weights_mem   Source weights memory (HWIO format)
 * @param dst_weights_mem   Destination weights memory (OneDNN blocked format)
 * @param eng               OneDNN engine
 * @param strm              OneDNN stream for execution
 * @param is_weights_const  If true, enable caching; if false, reorder directly
 *
 * @return true on success, false on failure
 */
bool reorderAndCacheWeights(
    const Key_conv& key,
    dnnl::memory& src_weights_mem,
    dnnl::memory& dst_weights_mem,
    const dnnl::engine& eng,
    const bool is_weights_const
);

/**
 * @brief Wrapper function for OneDNN-based Conv
 *
 * This function implements convolution using OneDNN backend.
 * It handles:
 * - Data layout conversion (NHWC to NCHW)
 * - Memory descriptor creation
 * - Convolution primitive setup
 * - Post-operation fusion (Relu, etc.)
 * - Weight caching (when is_weights_const = true)
 * - Execution and synchronization
 *
 * @param input            Input tensor [N, H, W, C] in NHWC format
 * @param filter           Filter tensor [KH, KW, C_in, C_out]
 * @param bias             Optional bias [C_out]
 * @param output           Output tensor [N, H_out, W_out, C_out]
 * @param is_weights_const Flag indicating if weights are constant (enables caching)
 * @param params           Convolution parameters
 *
 * @return status_t::success on successful execution, status_t::failure otherwise
 */
status_t conv_onednn_wrapper(
    const void *input,
    const void *filter,
    const void *bias,
    void *output,
    const bool is_weights_const,
    conv_params &params
);

#endif // ZENDNNL_DEPENDS_ONEDNN

} // namespace conv
} // namespace lowoha
} // namespace zendnnl

#endif // _LOWOHA_CONV_ONEDNN_KERNEL_HPP
