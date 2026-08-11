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
#ifndef _COMPARE_OPERATOR_HPP_
#define _COMPARE_OPERATOR_HPP_

#include "common/zendnnl_global.hpp"
#include "compare_context.hpp"
#include "compare_operator_impl.hpp"
#include "operators/common/operator.hpp"

namespace zendnnl {
namespace ops {

/** @class compare_operator_t
 *  @brief Implements compare operator.
 *
 */
class compare_operator_t final
    : public operator_t<compare_operator_t, compare_context_t, compare_impl_t> {
public:
    /** @brief Self type **/
    using self_type = compare_operator_t;
    /** @brief Parent type **/
    using parent_type
            = operator_t<compare_operator_t, compare_context_t, compare_impl_t>;
    /** @brief context type **/
    using context_type = parent_type::context_type;
    /** @brief impl type **/
    using impl_type = parent_type::impl_type;
    /** @brief impl pointer type **/
    using impl_sptr_type = parent_type::impl_sptr_type;

    compare_stats_t get_compare_stats() { return impl->get_compare_stats(); }
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
using compare_operator_t = zendnnl::ops::compare_operator_t;
}

} //namespace zendnnl
#endif
