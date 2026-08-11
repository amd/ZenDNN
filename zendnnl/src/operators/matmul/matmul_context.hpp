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
#ifndef _MATMUL_CONTEXT_HPP_
#define _MATMUL_CONTEXT_HPP_

#include <memory>
#include <optional>
#include <vector>

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_context.hpp"

#if ZENDNNL_DEPENDS_AOCLDLP
#include "operators/matmul/aocl_dlp/matmul_aocl_dlp_utils.hpp"
#endif

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;

/** @class matmul_context_t
 *  @brief context for @c matmul_operator_t.
 *
 * In order to elable chaining, the first parameter in @c op_context_t template
 * should be the class itself.
 * @sa matmul_operator_t
 */
class matmul_context_t final : public op_context_t<matmul_context_t> {
public:
    /** @brief parent type */
    using parent_type = op_context_t<matmul_context_t>;

    /** @brief constructor */
    matmul_context_t();

    /** TODO: Add a interface to support different backends */
    /** @brief get post op pointer */
#if ZENDNNL_DEPENDS_AOCLDLP
    dlp_metadata_t *get_aocl_dlp_post_op_ptr_unsafe() const;

    /** @brief get reordered weights pointer */
    void *get_aocl_dlp_reordered_weights_ptr_unsafe() const;
#endif

    /** @brief Set parameter alpha value.*/
    matmul_context_t &set_alpha(float alpha_);

    /** @brief Get parameter alpha value.*/
    float get_alpha() const;

    /** @brief Set parameter beta value.*/
    matmul_context_t &set_beta(float beta_);

    /** @brief Get parameter beta value.*/
    float get_beta() const;

    /** @brief preprocess */
    status_t preprocess() override;

    /** @brief Generate object hash including matmul-specific parameters. */
    std::size_t hash() override;

protected:
    /** @brief validate parameters */
    status_t validate() override;

    /** @brief Returns matmul context information */
    std::string context_info() override;

#if ZENDNNL_DEPENDS_AOCLDLP
    std::shared_ptr<aocl_dlp_utils_t> aocl_dlp_utils_ptr; /**< aocl dlp utils */
#endif
    friend class matmul_operator_t;
    friend class matmul_impl_t;

private:
    float _alpha; /**< alpha parameter */
    float _beta; /**< beta parameter */
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
using matmul_context_t = zendnnl::ops::matmul_context_t;
} // namespace interface

} //namespace zendnnl
#endif
