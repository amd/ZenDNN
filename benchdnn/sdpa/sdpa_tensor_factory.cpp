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

#include "sdpa_tensor_factory.hpp"

namespace zendnnl {
namespace benchdnn {
namespace sdpa {

using index_type = tensor_factory_t::index_type;

namespace {

// Returns the physical allocation shape for a logical [B, H, S, D] tensor
// in the requested QKV layout. The factory always allocates row-major over
// the returned shape, so the choice of shape is what selects BHSD vs BSHD
// memory ordering.
std::vector<index_type> qkv_alloc_shape(int64_t B, int64_t H, int64_t S,
                                        int64_t D, qkv_layout_t layout) {
  if (layout == qkv_layout_t::bshd) {
    return {
      static_cast<index_type>(B),
      static_cast<index_type>(S),
      static_cast<index_type>(H),
      static_cast<index_type>(D)
    };
  }
  return {
    static_cast<index_type>(B),
    static_cast<index_type>(H),
    static_cast<index_type>(S),
    static_cast<index_type>(D)
  };
}

} // anonymous namespace

int create_qkv_tensor(tensor_factory_t &tensor_factory,
                      int64_t B, int64_t H, int64_t S, int64_t D,
                      zendnnl::common::data_type_t dt,
                      qkv_layout_t layout,
                      const std::string &name,
                      tensor_t &out) {
  if (B <= 0 || H <= 0 || S <= 0 || D <= 0) {
    commonlog_error("create_qkv_tensor: dimensions must all be > 0 "
                    "(got B=", B, ", H=", H, ", S=", S, ", D=", D, ")");
    return NOT_OK;
  }
  if (dt != data_type_t::f32 && dt != data_type_t::bf16 &&
      dt != data_type_t::f16) {
    commonlog_error("create_qkv_tensor: dtype must be f32, bf16 or f16, got ",
                    datatypeToStr(dt));
    return NOT_OK;
  }

  // Small range keeps softmax inputs well-behaved (no NaNs/overflows in bf16).
  out = tensor_factory.uniform_dist_tensor(
          qkv_alloc_shape(B, H, S, D, layout), dt, 0.1f, name);
  return OK;
}

int create_output_tensor(tensor_factory_t &tensor_factory,
                         int64_t B, int64_t H, int64_t S_q, int64_t D,
                         zendnnl::common::data_type_t dt,
                         qkv_layout_t layout,
                         tensor_t &out) {
  if (B <= 0 || H <= 0 || S_q <= 0 || D <= 0) {
    commonlog_error("create_output_tensor: dimensions must all be > 0 "
                    "(got B=", B, ", H=", H, ", S_q=", S_q, ", D=", D, ")");
    return NOT_OK;
  }
  if (dt != data_type_t::f32 && dt != data_type_t::bf16 &&
      dt != data_type_t::f16) {
    commonlog_error("create_output_tensor: dtype must be f32, bf16 or f16, got ",
                    datatypeToStr(dt));
    return NOT_OK;
  }

  out = tensor_factory.zero_tensor(
          qkv_alloc_shape(B, H, S_q, D, layout), dt, "sdpa_output");
  return OK;
}

int create_mask_tensor(tensor_factory_t &tensor_factory,
                       int64_t B, int64_t H, int64_t S_q, int64_t S_kv,
                       int mask_ndims,
                       zendnnl::common::data_type_t mask_dt,
                       tensor_t &out) {
  if (mask_ndims == 0) {
    out = tensor_t();
    return OK;
  }
  if (mask_ndims != 2 && mask_ndims != 4) {
    commonlog_error("create_mask_tensor: mask_ndims must be 0, 2, or 4 "
                    "(got ", mask_ndims, ")");
    return NOT_OK;
  }
  if (mask_dt != data_type_t::f32 && mask_dt != data_type_t::bf16 &&
      mask_dt != data_type_t::f16) {
    commonlog_error("create_mask_tensor: mask_dt must be f32, bf16 or f16, got ",
                    datatypeToStr(mask_dt));
    return NOT_OK;
  }
  if (S_q <= 0 || S_kv <= 0) {
    commonlog_error("create_mask_tensor: S_q and S_kv must be > 0 "
                    "(got S_q=", S_q, ", S_kv=", S_kv, ")");
    return NOT_OK;
  }

  std::vector<index_type> shape;
  if (mask_ndims == 2) {
    shape = {
      static_cast<index_type>(S_q),
      static_cast<index_type>(S_kv)
    };
  }
  else {
    if (B <= 0 || H <= 0) {
      commonlog_error("create_mask_tensor: 4D mask requires B>0 and H>0 "
                      "(got B=", B, ", H=", H, ")");
      return NOT_OK;
    }
    shape = {
      static_cast<index_type>(B),
      static_cast<index_type>(H),
      static_cast<index_type>(S_q),
      static_cast<index_type>(S_kv)
    };
  }

  // Use a tiny range so additive bias doesn't completely override Q*K^T.
  out = tensor_factory.uniform_dist_tensor(shape, mask_dt, 0.01f,
                                           "sdpa_mask");
  return OK;
}

} // namespace sdpa
} // namespace benchdnn
} // namespace zendnnl
