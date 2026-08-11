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
#ifndef _SAMPLE_CONTEXT_HPP_
#define _SAMPLE_CONTEXT_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_context.hpp"

namespace zendnnl {
namespace ops {

/** @class sample_context_t
 *  @brief sample context for @c sample_operator_t.
 *
 * In order to elable chaining, the first parameter in @c op_context_t template
 * should be the class itself.
 * @sa sample_operator_t
 */
class sample_context_t final : public op_context_t<sample_context_t> {
public:
    using parent_type = op_context_t<sample_context_t>;

    /** @brief Returns sample context information */
    std::string context_info() override;

protected:
    /** @brief validate parameters */
    status_t validate() override;
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
using sample_context_t = zendnnl::ops::sample_context_t;
} // namespace interface

} //namespace zendnnl
#endif
