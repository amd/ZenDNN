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
#ifndef _GDN_TENSOR_FACTORY_HPP_
#define _GDN_TENSOR_FACTORY_HPP_

// Declares only the tensor-creation helpers of the GDN benchmark driver; it
// does not pull in `gdn_utils.hpp` (and its `GdnConfig`), so it stays out of
// the benchdnn.hpp include cycle.
#include "benchdnn.hpp"
#include "tensor_helper/tensor_factory.hpp"

namespace zendnnl {
namespace benchdnn {
namespace gdn {

using namespace zendnnl::tensor_helper;

/**
 * @brief Sizes of the tensors of one gdn_direct call.
 */
struct gdn_tensor_sizes {
    int64_t num_k_heads;
    int64_t num_v_heads;
    int64_t head_k_dim;
    int64_t head_v_dim;
    int64_t conv_width;
    int64_t tokens; ///< rows of the per-token tensors
    int64_t slots; ///< slots of the conv and SSM state pools
    zendnnl::common::data_type_t src_dt;
    zendnnl::common::data_type_t state_dt;
    bool fused_norm;

    int64_t conv_dim() const {
        return 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
    }
    /// Conv-state taps per slot (one placeholder tap when there are none).
    int64_t conv_taps() const { return conv_width > 1 ? conv_width - 1 : 1; }
};

/**
 * @brief The tensors of one gdn_direct call, in vLLM's layouts. Every
 *        tensor is dense; inputs and states hold uniform random values and
 *        the output starts at zero.
 */
struct gdn_tensors {
    tensor_t mixed_qkv; ///< [tokens][conv_dim], src_dt
    tensor_t a; ///< [tokens][num_v_heads], src_dt
    tensor_t b; ///< [tokens][num_v_heads], src_dt
    tensor_t A_log; ///< [num_v_heads], f32
    tensor_t dt_bias; ///< [num_v_heads], src_dt
    tensor_t conv_weight; ///< [conv_dim][conv_width], src_dt
    tensor_t conv_bias; ///< [conv_dim], src_dt
    tensor_t conv_state; ///< [slots][conv_taps][conv_dim] (SD), src_dt
    tensor_t ssm_state; ///< [slots][num_v_heads][head_k_dim * head_v_dim]
    tensor_t output; ///< [tokens][num_v_heads][head_v_dim], src_dt
    tensor_t z; ///< fused norm gate, like output
    tensor_t norm_weight; ///< fused norm weight [head_v_dim], f32
};

/**
 * @brief Creates every tensor of one gdn_direct call.
 *
 * @param tensor_factory Factory object for tensor creation.
 * @param sizes          Shapes and data types of the call.
 * @param out            Output tensors; z and norm_weight stay empty
 *                       unless sizes.fused_norm.
 * @return int OK (0) on success, NOT_OK (1) on invalid sizes.
 */
int create_gdn_tensors(tensor_factory_t &tensor_factory,
        const gdn_tensor_sizes &sizes, gdn_tensors &out);

} // namespace gdn
} // namespace benchdnn
} // namespace zendnnl

#endif
