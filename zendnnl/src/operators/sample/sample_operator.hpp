/********************************************************************************
# * Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef _SAMPLE_OPERATOR_HPP_
#define _SAMPLE_OPERATOR_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator.hpp"
#include "operators/sample/sample_context.hpp"
#include "operators/sample/sample_operator_impl.hpp"

namespace zendnnl {
namespace ops {
/** @class sample_operator_t
 *  @brief A sample operator class for demonstration and starting point for new
 *  operators.
 *
 * @par Synopsys
 *
 * Invokes a fp32(bf16) kernel if input type is fp32(bf16). The kernel prints
 * its name.
 *
 * An new operator can be developed by taking this class as a boiler-plate code,
 * or a starting point.
 *
 * In order to elable chaining, the first parameter in @c operator_t template
 * should be the class itself, and the second parameter should be this operators
 * context, derived from @c operator_context_t.
 *
 * @par Parameters, Inputs, Outputs
 *
 * The operator has following parameters and input/outputs
 * - Parameter(s)
 *   1. (mandatory) sample_param  : An arbitrary tensor.
 * - Inputs
 *   1. (mandatory) sample_input  : An arbitrary tensor of type(f32,bf16).
 * - Output(s)
 *   1. (mandatory) sample_output : An arbitrary tensor.
 */
class sample_operator_t final
    : public operator_t<sample_operator_t, sample_context_t, sample_impl_t> {
public:
    /** @brief Self type **/
    using self_type = sample_operator_t;
    /** @brief Parent type **/
    using parent_type
            = operator_t<sample_operator_t, sample_context_t, sample_impl_t>;
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
using sample_operator_t = zendnnl::ops::sample_operator_t;
} // namespace interface

} //namespace zendnnl
#endif
