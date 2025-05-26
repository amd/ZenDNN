/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdint>
#include "error_handling.hpp"
#include "embag_fp32_avx512_utils.hpp"
#include "embag_avx512_kernels.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;

status_t embag_f32_avx512_kernel_t::execute(const context_type& context_,
                                               tensor_map_type& inputs_,
                                               tensor_map_type& outputs_) {
  LOG_DEBUG_INFO("Executing embag_f32_avx512_kernel_t");
  //get the parameters
  auto table_tensor   = context_.get_param("table");
  auto indices_tensor = inputs_.find("indices")->second;
  auto offsets_tensor = inputs_.find("offsets")->second;
  auto dst_tensor     = outputs_.find("output")->second;

  float const* input   = table_tensor->get_raw_handle<float>();
  int32_t*     indices = indices_tensor.get_raw_handle<int32_t>();
  int32_t*     offsets = offsets_tensor.get_raw_handle<int32_t>();
  float*       dst     = dst_tensor.get_raw_handle<float>();

  const int64_t  width               = (table_tensor->get_sizes()).at(1);
  const int32_t  indsz               = (indices_tensor.get_sizes()).at(0);
  int32_t        offsz               = (offsets_tensor.get_sizes()).at(0);
  const int32_t  dstsz               = (dst_tensor.get_sizes()).at(0);
  const int32_t  padidx              = context_.get_padding_index();
  const uint32_t nthr                = context_.get_core_count();
  const uint32_t scatter_offset      = context_.get_scatter_offset();
  const uint32_t scatter_stride      = context_.get_scatter_stride();
  const bool     include_last_offset = context_.get_include_last_offset();

  // add scatter_offset
  uint32_t stride  = scatter_stride*width;
  dst             += scatter_offset*width;

  if (include_last_offset) {
    offsz -= 1;
  }

  if (256 == width) {
    if (padidx >= 0) {
      #pragma omp parallel for num_threads(nthr) //proc_bind(master)
      for (auto oi = 0; oi < offsz; ++oi) {
        auto ofirst = offsets[oi];
        auto olast  = 0;
        if (!include_last_offset) {
          olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
        }
        else {
          olast  = offsets[oi+1];
        }
        zenmmAVX512_ext_ps256 sum;
        for (auto i = ofirst; i < olast; ++i) {
          if (indices[i] != padidx) {
            sum.fetch_add_ps(input + (indices[i] * width));
          }
        }
        sum.store_ps(dst + oi*stride);
      }
    }
    else {
      #pragma omp parallel for num_threads(nthr) //proc_bind(master)
      for (auto oi = 0; oi < offsz; ++oi) {
        auto ofirst = offsets[oi];
        auto olast=0;
        if (include_last_offset==0) {
          olast  = oi < (offsz -1) ? offsets[oi+1] : indsz;
        }
        else {
          olast  = offsets[oi+1];
        }
        zenmmAVX512_ext_ps256 sum;
        for (auto i = ofirst; i < olast; ++i) {
          sum.fetch_add_ps(input + (indices[i] * width));
        }
        sum.store_ps(dst + oi*stride);
      }
    }
    return status_t::success;
  }
  log_error("embag_f32_avx512_kernel_t for given case is unimplemented");
  return status_t::unimplemented;
}

using namespace zendnnl::error_handling;
status_t embag_bf16_avx512_kernel_t::execute(const context_type& context_,
                                               tensor_map_type& inputs_,
                                               tensor_map_type& outputs_) {
  LOG_DEBUG_INFO("Executing embag_bf16_avx512_kernel_t");
  log_error("embag_bf16_avx512_kernel is unimplemented");

  return status_t::unimplemented;
}

extern "C" {
  std::shared_ptr<embag_f32_avx512_kernel_t> get_embag_f32_avx512_kernel() {
    return std::make_shared<embag_f32_avx512_kernel_t>();
  }

  std::shared_ptr<embag_bf16_avx512_kernel_t> get_embag_bf16_avx512_kernel() {
    return std::make_shared<embag_bf16_avx512_kernel_t>();
  }
}

} //namespace ops
} //namespace zendnnl

