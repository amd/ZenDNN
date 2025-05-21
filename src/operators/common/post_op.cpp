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
#include "post_op.hpp"

namespace zendnnl {
namespace ops {
using namespace zendnnl::error_handling;
post_op_t::post_op_t(post_op_type_t type_):
  type{type_} {
  LOG_DEBUG_INFO("Setting Post-op post_op_t");
  switch (type_) {
  case post_op_type_t::elu:
    elu_params.alpha = 1.0;
    break;
  case post_op_type_t::leaky_relu:
    leaky_relu_params.nslope = 0.0;
    break;
  case post_op_type_t::swish:
    swish_params.scale = 1.0;
    break;
  case post_op_type_t::clip:
    clip_params.lower = -0.5;
    clip_params.upper = 0.5;
    break;
  case post_op_type_t::binary_add:
    binary_add_params.scale       = 1.0;
    binary_add_params.tensor_name = "binary_add_tensor_";
    break;
  case post_op_type_t::binary_mul:
    binary_mul_params.scale       = 1.0;
    binary_mul_params.tensor_name = "binary_mul_tensor_";
    break;
  default:
    break;
  }
}

post_op_t::post_op_t(elu_params_t params_):
  type{post_op_type_t::elu} {
  elu_params = params_;
}

post_op_t::post_op_t(leaky_relu_params_t params_):
  type{post_op_type_t::leaky_relu} {
  leaky_relu_params = params_;
}

post_op_t::post_op_t(swish_params_t params_):
  type{post_op_type_t::swish} {
  swish_params = params_;
}

post_op_t::post_op_t(clip_params_t params_):
  type{post_op_type_t::clip} {
  clip_params = params_;
}

post_op_t::post_op_t(binary_add_params_t params_):
  type{post_op_type_t::binary_add} {
  binary_add_params.scale = params_.scale;
  binary_add_params.tensor_name = "binary_add_tensor_";
}

post_op_t::post_op_t(binary_mul_params_t params_):
  type{post_op_type_t::binary_mul} {
  binary_mul_params.scale = params_.scale;
  binary_mul_params.tensor_name = "binary_mul_tensor_";
}

} //namespace ops
} //namespace zendnnl

