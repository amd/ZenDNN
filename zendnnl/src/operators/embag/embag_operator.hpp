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
#ifndef _EMBAG_OPERATOR_HPP_
#define _EMBAG_OPERATOR_HPP_

#include "common/zendnnl_global.hpp"
#include "embag_context.hpp"
#include "embag_operator_impl.hpp"
#include "operators/common/operator.hpp"

namespace zendnnl {
namespace ops {

/** @class embag_operator_t
 *  @brief Implements embedding bag operator.
 *
 * @par Synopsys
 *
 * Given a weights matrix of size (num_embeddings x embedding_dim), an indices vector, and (optional) offsets vector,
 * the embedding bag operator computes the sum (or mean/max, depending on mode) of embeddings for each bag.
 * Each bag is defined by a range of indices, specified by offsets. The output is a bags x embedding_dim tensor,
 * where each row is the aggregated embedding for a bag.
 *
 * In order to enable chaining, the first parameter in @c operator_t template
 * should be the class itself, and the second parameter should be this operator's
 * context, derived from @c operator_context_t.
 *
 * @par Quantization support
 *
 * The operator supports both fp32 and bf16 computations.
 *
 * @par Parameters, Inputs, Outputs
 *
 * The operator has following parameters and input/outputs:
 * - Parameter(s)
 *   1. (mandatory) weights : A (num_embeddings x embedding_dim) 2D tensor.
 * - Inputs
 *   1. (mandatory) indices : A 1D tensor of indices into the weights.
 *   2. (optional) offsets : A 1D tensor specifying the start of each bag in indices.
 * - Output(s)
 *   1. (mandatory) output : A (num_bags x embedding_dim) 2D tensor containing aggregated embeddings per bag.
 */
class embag_operator_t final
    : public operator_t<embag_operator_t, embag_context_t, embag_impl_t> {
public:
    /** @brief Self type **/
    using self_type = embag_operator_t;
    /** @brief Parent type **/
    using parent_type
            = operator_t<embag_operator_t, embag_context_t, embag_impl_t>;
    /** @brief context type **/
    using context_type = parent_type::context_type;
    /** @brief impl type **/
    using impl_type = parent_type::impl_type;
    /** @brief impl pointer type **/
    using impl_sptr_type = parent_type::impl_sptr_type;
};

} //namespace ops

// Keep `interface` undef'd for the rest of the TU (do NOT push/pop-restore):
// `interface` is a public zendnnl namespace that consumers reference (e.g.
// `using namespace zendnnl::interface;`) after including this header, so the
// Windows <windows.h> `interface` macro must stay undefined here -- restoring
// it would re-shadow the namespace and break downstream consumers on Windows.
#ifdef interface
#undef interface
#endif
namespace interface {
using embag_operator_t = zendnnl::ops::embag_operator_t;
}

} //namespace zendnnl
#endif
