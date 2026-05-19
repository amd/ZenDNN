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
#ifndef _SDPA_ENCODER_REF_KERNEL_HPP_
#define _SDPA_ENCODER_REF_KERNEL_HPP_

#include "common/zendnnl_global.hpp"
#include "operators/common/operator_kernel.hpp"
#include "sdpa_encoder_context.hpp"

namespace zendnnl {
namespace ops {
/** @class sdpa_encoder_ref_kernel_t
 *  @brief A unified SDPA encoder reference kernel.
 *
 * Single entry point for the reference SDPA encoder. The kernel dispatches
 * internally on the Q/K/V data type; the QKV dtypes currently supported are
 *   - @c data_type_t::f32  (mask, if present, must be FP32)
 *   - @c data_type_t::bf16 (mask, if present, may be FP32 or BF16; converted
 *                           to FP32 element-wise at add time)
 *   - @c data_type_t::f16  (mask, if present, may be FP32 or F16; converted
 *                           to FP32 element-wise at add time)
 * Storage is at the QKV dtype on input and output; all arithmetic
 * (Q@K^T, scaling, optional masking, softmax, scores@V) is performed in FP32
 * internally so that softmax is numerically stable for low-precision inputs.
 *
 * Because every reduced-precision element is widened to FP32 at the I/O
 * boundary using the dtype's float conversion operators, the F16 reference
 * kernel does not require an F16-capable ISA (AVX512-FP16 / AVX-NE-CONVERT)
 * and runs on every CPU the rest of the operator framework supports.
 *
 * The supported-dtype list is owned by @c sdpa_encoder_impl_t::kernel_factory();
 * @c execute() rejects anything outside that list defensively.
 */
class sdpa_encoder_ref_kernel_t final : public
  op_kernel_t<sdpa_encoder_context_t> {
 public:
  /** @brief Default destructor */
  ~sdpa_encoder_ref_kernel_t() = default;

  /** @brief Execute */
  status_t execute(const context_type &context_,
                   tensor_map_type &inputs_,
                   tensor_map_type &outputs_) override;
};

} //namespace ops
} //namespace zendnnl

extern "C" {
  /** @fn get_sdpa_encoder_ref_kernel
   *  @brief returns a raw pointer to a newly-allocated sdpa_encoder_ref_kernel_t instance
   */
  zendnnl::ops::sdpa_encoder_ref_kernel_t *get_sdpa_encoder_ref_kernel();
} //extern "C"

#endif //_SDPA_ENCODER_REF_KERNEL_HPP_
