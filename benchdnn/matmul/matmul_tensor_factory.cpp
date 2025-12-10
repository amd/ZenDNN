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

#include "matmul_tensor_factory.hpp"

namespace zendnnl {
namespace benchdnn {
namespace matmul {

int create_weights_tensor(tensor_factory_t &tensor_factory, MatmulConfig cfg,
                          std::vector<tensor_t> &weights, const global_options &options, bool isLOWOHA) {

  zendnnl::common::data_type_t dt = cfg.dt[1];

  for (auto i = 0; i < cfg.n_values.size(); i++) {

    size_t k = (i == 0) ? cfg.k : cfg.n_values[i - 1];
    size_t n = cfg.n_values[i];
    tensor_t weights_tensor;

    // Apply reorder for regular API, not for LOWOHA
    if (cfg.kernel_name == "aocl_blis_blocked" && !isLOWOHA) {
      auto wei_scale = (dt == data_type_t::s8) ? tensor_factory.uniform_dist_tensor({1, n},
                       data_type_t::f32, 0.2) : tensor_t();

      // Create input tensor with contigious layout.
      auto input_tensor = tensor_factory.uniform_dist_tensor({k, n},
                          dt,
                          1.0, "reorder_input", cfg.isTransB, wei_scale);
      // Reorder context creation with backend aocl.
      auto reorder_context = reorder_context_t()
                             .set_algo_format("aocl")
                             .set_source_dtype(dt)
                             .create();

      if (! reorder_context.check()) {
        testlog_error("reorder context creation failed");
        return NOT_OK;
      }

      // Reorder operator creation with name, context and input.
      auto reorder_operator = reorder_operator_t()
                              .set_name("reorder_operator")
                              .set_context(reorder_context)
                              .create()
                              .set_input("reorder_input", input_tensor);

      // Check if reorder operation creation is successful.
      if (reorder_operator.is_bad_object()) {
        testlog_error("operator ", reorder_operator.get_name(), " creation failed");
        return NOT_OK;
      }

      // Compute the reorder size
      size_t reorder_size         = reorder_operator.get_reorder_size();
      // Extract the input buffer size
      size_t input_buffer_size    = input_tensor.get_buffer_sz_bytes();

      // Inplace reorder takes place when reorder buffer size is same as input buffer size
      if (reorder_size == input_buffer_size) {
        // Assign input_tensor to buffer_params as a tensor_t variant
        StorageParam buffer_params = input_tensor;

        // Blocked Tensor creation with seperate view for input tensor.
        weights_tensor = tensor_factory.copy_tensor({k, n},
                         dt,
                         buffer_params, cfg.isTransB, true,
                         "weights_" + std::to_string(i), std::move(wei_scale));
      }
      else {
        // Compute the reorder size and create a buffer with reorderd size
        void *reorder_weights = aligned_alloc(64, reorder_size);

        // Create a Pair of storage params [reorder size and reorder weights] and
        // use it in tensor creation
        StorageParam buffer_params = std::make_pair(reorder_size, reorder_weights);

        // Blocked Tensor creation with seperate view for input tensor.
        weights_tensor = tensor_factory.copy_tensor({k, n},
                         dt,
                         buffer_params, cfg.isTransB, true,
                         "weights_" + std::to_string(i), std::move(wei_scale));
      }
    }
    else {
      if (options.ndims > 2) {
        weights_tensor = tensor_factory.uniform_dist_tensor({cfg.bs, k, n},
                         dt,
                         1.0, "weights_" + std::to_string(i), cfg.isTransB);
      }
      else {
        auto wei_scale = (dt == data_type_t::s8) ? tensor_factory.uniform_dist_tensor({1, n},
                         data_type_t::f32, 0.2) : tensor_t();
        weights_tensor = tensor_factory.uniform_dist_tensor({k, n},
                         dt,
                         1.0, "weights_" + std::to_string(i), cfg.isTransB, wei_scale);
      }
    }
    weights.push_back(weights_tensor);
  }
  return OK;
}

int create_bias_tensor(tensor_factory_t tensor_factory, const MatmulConfig &cfg,
                       std::vector<tensor_t> &bias, const global_options &options) {
  if (cfg.isBiasEnabled) {
    zendnnl::common::data_type_t dt = cfg.bias_dt;
    for (auto i = 0; i < cfg.n_values.size(); i++) {
      tensor_t bias_tensor;
      if (options.ndims > 2) {
        bias_tensor = tensor_factory.uniform_dist_tensor({1, 1, cfg.n_values[i]},
                      dt,
                      -10.0, "bias_" + std::to_string(i));
      }
      else {
        bias_tensor = tensor_factory.uniform_dist_tensor({1, cfg.n_values[i]},
                      dt,
                      -10.0, "bias_" + std::to_string(i));

      }
      bias.push_back(bias_tensor);
    }
  }
  return OK;
}

int create_input_tensor(tensor_factory_t &tensor_factory,
                        const MatmulConfig &cfg, tensor_t &input, const global_options &options) {
  if (options.ndims > 2) {
    input = tensor_factory.uniform_dist_tensor({cfg.bs, cfg.m, cfg.k},
            cfg.dt[0],
            1.0, "matmul_input", cfg.isTransA);
  }
  else {
    auto src_scale = (cfg.dt[0] == data_type_t::s8 ||
                      cfg.dt[0] == data_type_t::u8) ? tensor_factory.uniform_dist_tensor({1, 1},
                          data_type_t::f32, 0.3) : tensor_t();
    auto src_zp = cfg.dt[0] == data_type_t::u8 ? tensor_factory.uniform_tensor({1, 1},
                  data_type_t::s8, 16) : tensor_t();
    input = tensor_factory.uniform_dist_tensor({cfg.m, cfg.k},
            cfg.dt[0],
            1.0, "matmul_input", cfg.isTransA, src_scale, src_zp);
  }
  input.set_name("matmul_input");
  return OK;
}

int create_output_tensor(tensor_factory_t &tensor_factory,
                         const MatmulConfig &cfg, std::vector<tensor_t> &output,
                         const global_options &options) {
  // Create output tensor with zero initialization.
  size_t m = cfg.m;
  zendnnl::common::data_type_t dt = cfg.dt[2];
  auto dst_scale = !(dt == data_type_t::f32 ||
                     dt == data_type_t::bf16) ? tensor_factory.uniform_dist_tensor({1, 1},
                         data_type_t::f32, 1.2) : tensor_t();
  auto dst_zp  = dt == data_type_t::u8 ? tensor_factory.uniform_tensor({1, 1},
                 data_type_t::u8, 53) : tensor_t();
  for (auto i = 0; i < cfg.n_values.size(); i++) {
    size_t n = cfg.n_values[i];
    tensor_t output_tensor;
    if (options.ndims > 2) {
      output_tensor = tensor_factory.zero_tensor({cfg.bs, m, n},
                      dt, "matmul_output_" + std::to_string(i));
    }
    else {
      output_tensor = tensor_factory.zero_tensor({m, n},
                      dt, "matmul_output_" + std::to_string(i), dst_scale, dst_zp);
    }
    output_tensor.set_name("matmul_output_" + std::to_string(i));
    output.push_back(output_tensor);
  }

  return OK;
}

int create_binary_post_ops_tensors(tensor_factory_t &tensor_factory,
                                   const MatmulConfig &cfg,
                                   std::vector<std::vector<tensor_t>> &binary_post_ops_tensors) {
  for (auto i = 0; i < cfg.n_values.size(); i++) {
    std::vector<tensor_t> binary_tensors;
    for (const auto &post_op : cfg.binary_post_ops_pos) {
      // Create a tensor for each binary post-op
      auto binary_tensor = tensor_factory.uniform_dist_tensor({cfg.m, cfg.n_values[i]},
                           cfg.dt[2],
                           2.0, "binary_post_op_" + std::to_string(post_op));
      binary_tensors.push_back(binary_tensor);
    }
    binary_post_ops_tensors.push_back(binary_tensors);
  }
  return OK;
}

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl