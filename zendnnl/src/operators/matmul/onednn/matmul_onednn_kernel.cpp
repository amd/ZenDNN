/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "matmul_onednn_kernel.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::common;
using namespace zendnnl::memory;
using namespace zendnnl::error_handling;

matmul_onednn_kernel_t::~matmul_onednn_kernel_t() {
}

#if ZENDNNL_DEPENDS_ONEDNN
status_t matmul_onednn_kernel_t::preprocess(const context_type &context_,
    tensor_map_type &inputs_,
    tensor_map_type &outputs_,
    onednn_utils_t::onednn_matmul_params &params,
    std::unordered_map<int, dnnl::memory> &matmul_args,
    dnnl::primitive_attr &matmul_attr,
    const dnnl::engine &eng) {
  log_info("Preprocessing in onednn_utils_t::preprocess");

  auto     input_tensor_iter  = inputs_.find("matmul_input");
  auto     output_tensor_iter = outputs_.find("matmul_output");

  if (input_tensor_iter == inputs_.end() ||
      output_tensor_iter == outputs_.end()) {
    log_error("Missing required matmul tensors");
    return status_t::failure;
  }

  auto     input_tensor  = input_tensor_iter->second;
  auto     output_tensor = output_tensor_iter->second;
  auto     weight_tensor = context_.get_param("weights").value();
  auto     bias_tensor_  = context_.get_param("bias");
  auto     post_ops_list = context_.get_post_op();

  params.alpha                 = context_.get_alpha();
  params.beta                  = context_.get_beta();

  params.src.buffer            = input_tensor.get_raw_handle_unsafe();
  params.dst.buffer            = output_tensor.get_raw_handle_unsafe();
  params.weights.buffer        = weight_tensor.get_raw_handle_unsafe();

  auto input_dim               = input_tensor.get_dim();
  auto weight_dim              = weight_tensor.get_dim();

  params.src.dtype             = input_tensor.get_data_type();
  params.weights.dtype         = weight_tensor.get_data_type();
  params.dst.dtype             = output_tensor.get_data_type();

  params.src.format_tag        = input_tensor.get_order();
  params.weights.format_tag    = weight_tensor.get_order();
  params.dst.format_tag        = output_tensor.get_order();

  params.src.is_transposed     = (input_dim == 2)  ? (input_tensor.get_order() ==
                                 "ba") : (input_tensor.get_order() == "acb");
  params.weights.is_transposed = (weight_dim == 2) ? (weight_tensor.get_order() ==
                                 "ba") : (weight_tensor.get_order() == "acb");

  auto src_dims                = input_tensor.get_size();
  auto weights_dims            = weight_tensor.get_size();
  auto dst_dims                = output_tensor.get_size();
  params.src.dims.assign(src_dims.begin(), src_dims.end());
  params.weights.dims.assign(weights_dims.begin(), weights_dims.end());
  params.dst.dims.assign(dst_dims.begin(), dst_dims.end());

  auto src_stride_dims         = input_tensor.get_stride();
  auto weights_stride_dims     = weight_tensor.get_stride();
  auto dst_stride_dims         = output_tensor.get_stride();
  params.src.strides.assign(src_stride_dims.begin(), src_stride_dims.end());
  params.weights.strides.assign(weights_stride_dims.begin(),
                                weights_stride_dims.end());
  params.dst.strides.assign(dst_stride_dims.begin(), dst_stride_dims.end());

  if (bias_tensor_) {
    auto bias_tensor             = bias_tensor_.value();
    params.bias.buffer           = bias_tensor.get_raw_handle_unsafe();
    params.bias.dtype            = bias_tensor.get_data_type();
    params.bias.format_tag       = bias_tensor.get_order();
    auto bias_dims               = bias_tensor.get_size();
    params.bias.dims.assign(bias_dims.begin(), bias_dims.end());
    auto bias_stride_dims        = bias_tensor.get_stride();
    params.bias.strides.assign(bias_stride_dims.begin(), bias_stride_dims.end());
  }

  dnnl::post_ops matmul_pops;
  int post_op_index = 0;

  if (params.alpha != 1.0f) {
    matmul_attr.set_scales_mask(DNNL_ARG_SRC, 0);
    auto alpha_mem = dnnl::memory({{1}, dnnl::memory::data_type::f32, {1}}, eng,
    const_cast<float *>(&params.alpha));
    matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, alpha_mem});
  }

  if (params.beta != 0.0f) {
    matmul_pops.append_sum(params.beta);
    post_op_index++;
  }

  if (!post_ops_list.empty()) {
    [[maybe_unused]] int mul_index = 0, add_index = 0;
    [[maybe_unused]] float po_alpha = 0.0f, po_beta = 0.0f;
    for (size_t po = 0; po < post_ops_list.size(); po++) {
      post_op_t zen_po = post_ops_list[po];

      switch (zen_po.type) {
      case post_op_type_t::elu: {
        po_alpha = zen_po.elu_params.alpha;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_elu, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::relu: {
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_relu, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::leaky_relu: {
        po_alpha = zen_po.leaky_relu_params.nslope;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_relu, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::gelu_tanh: {
        po_alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, po_alpha,
                                   po_beta);
        break;
      }
      case post_op_type_t::gelu_erf: {
        po_alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_gelu_erf, po_alpha,
                                   po_beta);
        break;
      }
      case post_op_type_t::tanh: {
        po_alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_tanh, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::square: {
        po_alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_square, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::abs: {
        po_alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_abs, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::sqrt: {
        po_alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_sqrt, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::exp: {
        po_alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_exp, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::log: {
        po_alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_log, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::sigmoid: {
        po_alpha = 1.0f;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_logistic, po_alpha,
                                   po_beta);
        break;
      }
      case post_op_type_t::swish: {
        po_alpha = zen_po.swish_params.scale;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_swish, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::clip: {
        po_alpha = zen_po.clip_params.lower;
        po_beta = zen_po.clip_params.upper;
        matmul_pops.append_eltwise(dnnl::algorithm::eltwise_clip, po_alpha, po_beta);
        break;
      }
      case post_op_type_t::binary_add: {
        std::string key_add = "binary_add_tensor_" + std::to_string(add_index++);
        auto buffer_it      = inputs_.find(key_add);
        if (buffer_it == inputs_.end()) {
          log_error("Missing binary add tensor for post-op");
          return status_t::failure;
        }

        auto buff_tensor    = buffer_it->second;
        onednn_utils_t::onednn_tensor_params binary_tensor;
        auto buff_dims           = buff_tensor.get_size();
        binary_tensor.dims.assign(buff_dims.begin(), buff_dims.end());
        binary_tensor.buffer     = buff_tensor.get_raw_handle_unsafe();
        binary_tensor.dtype      = buff_tensor.get_data_type();
        binary_tensor.format_tag = buff_dims.size() == 3 ? "abc" : "ab";
        auto dnnl_buff_tensor    = onednn_utils_t::to_dnnl_tensor(binary_tensor, eng);
        matmul_pops.append_binary(dnnl::algorithm::binary_add,
                                  dnnl_buff_tensor.get_desc());
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index) | DNNL_ARG_SRC_1, dnnl_buff_tensor});
        break;
      }
      case post_op_type_t::binary_mul: {
        std::string key_mul = "binary_mul_tensor_" + std::to_string(mul_index++);
        auto buffer_it      = inputs_.find(key_mul);
        if (buffer_it == inputs_.end()) {
          log_error("Missing binary mul tensor for post-op");
          return status_t::failure;
        }

        auto buff_tensor    = buffer_it->second;
        onednn_utils_t::onednn_tensor_params binary_tensor;
        auto buff_dims           = buff_tensor.get_size();
        binary_tensor.dims.assign(buff_dims.begin(), buff_dims.end());
        binary_tensor.buffer     = buff_tensor.get_raw_handle_unsafe();
        binary_tensor.dtype      = buff_tensor.get_data_type();
        binary_tensor.format_tag = buff_dims.size() == 3 ? "abc" : "ab";
        auto dnnl_buff_tensor    = onednn_utils_t::to_dnnl_tensor(binary_tensor, eng);
        matmul_pops.append_binary(dnnl::algorithm::binary_mul,
                                  dnnl_buff_tensor.get_desc());
        matmul_args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(post_op_index) | DNNL_ARG_SRC_1, dnnl_buff_tensor});
        break;
      }
      default:
        break;
      }
      post_op_index++;
    }
  }

  if (matmul_pops.len() > 0) {
    matmul_attr.set_post_ops(matmul_pops);
  }

  return status_t::success;
}

void matmul_onednn_kernel_t::execute_matmul(const
    onednn_utils_t::onednn_matmul_params &params,
    std::unordered_map<int, dnnl::memory> &matmul_args,
    dnnl::primitive_attr &matmul_attr, dnnl::engine &eng) {

  dnnl::stream eng_stream(eng);
  dnnl::memory  dnnl_input_tensor    = onednn_utils_t::to_dnnl_tensor(params.src,
                                       eng);
  dnnl::memory  dnnl_weight_tensor   = onednn_utils_t::to_dnnl_tensor(
                                         params.weights, eng);
  dnnl::memory  dnnl_output_tensor   = onednn_utils_t::to_dnnl_tensor(params.dst,
                                       eng);
  [[maybe_unused]] dnnl::memory  dnnl_bias_tensor  =
    onednn_utils_t::to_dnnl_tensor(params.bias, eng);

  auto dnnl_input_desc  = dnnl_input_tensor.get_desc();
  auto dnnl_output_desc = dnnl_output_tensor.get_desc();
  auto dnnl_weight_desc = dnnl_weight_tensor.get_desc();

  // Create primitive descriptor and primitive
  dnnl::matmul::primitive_desc matmul_pd;
  if (params.bias.buffer != nullptr) {
    matmul_pd = dnnl::matmul::primitive_desc(eng, dnnl_input_desc,
                dnnl_weight_desc, dnnl_bias_tensor.get_desc(),
                dnnl_output_desc, matmul_attr);
  }
  else {
    matmul_pd = dnnl::matmul::primitive_desc(eng, dnnl_input_desc,
                dnnl_weight_desc,
                dnnl_output_desc, matmul_attr);
  }
  auto matmul_prim = dnnl::matmul(matmul_pd);

  // Set up arguments
  matmul_args.insert({DNNL_ARG_SRC, dnnl_input_tensor});
  matmul_args.insert({DNNL_ARG_WEIGHTS, dnnl_weight_tensor});
  if (params.bias.buffer != nullptr) {
    matmul_args.insert({DNNL_ARG_BIAS, dnnl_bias_tensor});
  }
  matmul_args.insert({DNNL_ARG_DST, dnnl_output_tensor});

  // Execute primitive
  matmul_prim.execute(eng_stream, matmul_args);
  eng_stream.wait();
}
#endif

status_t matmul_onednn_kernel_t::execute(const context_type &context_,
    tensor_map_type &inputs_, tensor_map_type &outputs_) {
#if ZENDNNL_DEPENDS_ONEDNN
  log_info("matmul onednn kernel");

  dnnl::engine eng(dnnl::engine::kind::cpu, 0);

  std::unordered_map<int, dnnl::memory> matmul_args;
  dnnl::primitive_attr matmul_attr;

  onednn_utils_t::onednn_matmul_params params;
  status_t preprocess_status = preprocess(context_, inputs_, outputs_, params,
                                          matmul_args, matmul_attr, eng);
  if (preprocess_status != status_t::success) {
    return preprocess_status;
  }

  execute_matmul(params, matmul_args, matmul_attr, eng);

  return status_t::success;
#else
  log_error("onednn dependency is disabled");
  return status_t::failure;
#endif
}

} //namespace ops
} //namespace zendnnl

extern "C" {
  zendnnl::ops::matmul_onednn_kernel_t *get_matmul_onednn_kernel() {
    return new zendnnl::ops::matmul_onednn_kernel_t();
  }
}
