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

    std::vector<uint64_t> scale_size;
    uint64_t group_size = 0;
    uint64_t num_groups = 1;

    if (dt == data_type_t::s4 || dt == data_type_t::u4) {
      // If no even divisors found, fall back to K only if K is even
      // Otherwise, fall back to per-channel for per-group cases
      cfg.group_size = cfg.group_size ? cfg.group_size : cfg.k;
      if (cfg.scale_granularity == "group" && cfg.group_size % 2 != 0) {
        cfg.scale_granularity = "channel";  // Fall back to per-channel
        commonlog_warning("Defaulting to 'per-channel'.");
      }
      std::string scale_granularity = cfg.scale_granularity;

      if (scale_granularity == "tensor") {
        scale_size = {1, 1};
      }
      else if (scale_granularity == "channel") {
        // scale=per-channel
        scale_size = {1, n};
      }
      else if (scale_granularity == "group") {
        group_size = cfg.group_size;
        num_groups = k / group_size;
        scale_size = {num_groups, n};
      }
    }
    else if (dt == data_type_t::s8) {
      // s8 weight supports per-tensor, per-channel, and per-group (along K)
      // scales. One common pairing with per-group dynamic source quantization
      // is matched group sizes (for example, gtest INT8_DYNAMIC_GEMM_* uses
      //   wei {G, N} + src {M, G}
      // with matched G), but other src granularities (per-channel weight +
      // per-token / per-group src) are valid and documented in matmul.md.
      // This factory only configures weight scales; any cross-tensor
      // (src vs. wei) granularity compatibility must be enforced by the
      // caller or a higher-level validation path.
      cfg.group_size = cfg.group_size ? cfg.group_size : k;
      std::string scale_granularity = cfg.scale_granularity;
      if (scale_granularity == "tensor") {
        scale_size = {1, 1};
      }
      else if (scale_granularity == "group") {
        if (cfg.group_size == 0 || k % cfg.group_size != 0) {
          commonlog_warning(
            "weight group_size=", cfg.group_size, " does not divide K=", k,
            "; falling back to per-channel for s8 weights.");
          scale_size = {1, n};
        }
        else {
          group_size = cfg.group_size;
          num_groups = k / group_size;
          scale_size = {num_groups, n};
        }
      }
      else {
        // per-channel (default)
        scale_size = {1, n};
      }
    }

    auto scale_dtype = cfg.scale_dt;

    // Apply reorder for regular API, not for LOWOHA
    if (cfg.kernel_name == "aocl_dlp_blocked" && !isLOWOHA) {
      auto wei_scale = (dt == data_type_t::s8 || dt == data_type_t::s4) ?
                       tensor_factory.uniform_dist_tensor(scale_size, scale_dtype, 0.2) :
                       tensor_t();

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
        auto wei_scale = (dt == data_type_t::s8 || dt == data_type_t::s4 ||
                          dt == data_type_t::u4) ?
                         tensor_factory.uniform_dist_tensor(scale_size, scale_dtype, 0.2) :
                         tensor_t();
        auto wei_zp = dt == data_type_t::u4 ? tensor_factory.uniform_dist_tensor(
                        scale_size, data_type_t::bf16,
                        0.2) : tensor_t();
        weights_tensor = tensor_factory.uniform_dist_tensor({k, n},
                         dt,
                         1.0, "weights_" + std::to_string(i), cfg.isTransB, wei_scale, wei_zp);
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
                        MatmulConfig &cfg, tensor_t &input, const global_options &options,
                        bool isLOWOHA) {
  // Dynamic source quantization is a LOWOHA-only feature. On the regular
  // matmul API the operator validates/consumes quantization metadata, so
  // silently attaching a src-scale tensor to a bf16/f32 input would change
  // behavior or fail. Disable the flag here so no scale is attached and the
  // downstream gates also short-circuit.
  if (cfg.src_dynamic_quant && !isLOWOHA) {
    commonlog_warning(
      "src_dynamic_quant is supported only on the LOWOHA path "
      "(--lowoha=true). Disabling for this non-LOWOHA run.");
    cfg.src_dynamic_quant = false;
  }
  if (options.ndims > 2) {
    if (cfg.src_dynamic_quant) {
      commonlog_warning(
        "src_dynamic_quant is not supported for BMM (ndims > 2). Disabling.");
      // Disable so downstream gates (e.g. set_lowoha_matmul_params) don't
      // attempt dyn-quant without a src-scale tensor for 3D inputs.
      cfg.src_dynamic_quant = false;
    }
    input = tensor_factory.uniform_dist_tensor({cfg.bs, cfg.m, cfg.k},
            cfg.dt[0],
            1.0, "matmul_input", cfg.isTransA);
  }
  else {
    tensor_t src_scale;
    tensor_t src_zp;

    // Dynamic source quantization (W8A8, symmetric). Only fires when the
    // dtype combo matches the runtime gate in reorder_quantization.cpp:
    // src in {bf16, f32} and wei == s8. The scale tensor is zero-initialized;
    // the runtime fills it during the dynamic-quant pass. Compute target
    // is fixed to s8 in set_lowoha_matmul_params.
    if (cfg.src_dynamic_quant) {
      const bool src_ok = (cfg.dt[0] == data_type_t::bf16 ||
                           cfg.dt[0] == data_type_t::f32);
      const bool wei_ok = (cfg.dt[1] == data_type_t::s8);
      if (src_ok && wei_ok) {
        const uint64_t M = cfg.m;
        const uint64_t K = cfg.k;
        std::vector<uint64_t> src_scale_dims;
        const std::string &gran = cfg.src_scale_granularity;
        if (gran == "per-tensor") {
          src_scale_dims = {1, 1};
        }
        else if (gran == "per-token") {
          src_scale_dims = {M, 1};
        }
        else if (gran == "per-group") {
          const uint64_t gs = cfg.src_group_size;
          if (gs == 0) {
            commonlog_warning(
              "src_group_size=0 is not valid for per-group src-scale "
              "granularity; falling back to per-token.");
            src_scale_dims = {M, 1};
          }
          else if (K % gs != 0) {
            commonlog_warning(
              "src_group_size=", gs, " does not divide K=", K,
              "; falling back to per-token src-scale granularity.");
            src_scale_dims = {M, 1};
          }
          else {
            src_scale_dims = {M, K / gs};
          }
        }
        else {
          commonlog_warning(
            "Unknown src_scale_granularity '", gran,
            "'. Falling back to per-tensor.");
          src_scale_dims = {1, 1};
        }
        src_scale = tensor_factory.uniform_tensor(src_scale_dims,
                                                  cfg.src_scale_dt,
                                                  0.0f, "matmul_src_scale");
      }
      else {
        commonlog_warning(
          "src_dynamic_quant=true requires src in {bf16, f32} and wei=s8. "
          "Got src=", datatypeToStr(cfg.dt[0]),
          ", wei=", datatypeToStr(cfg.dt[1]),
          ". Ignoring src_dynamic_quant.");
      }
    }

    // Existing static int8 source path (s8/u8 src). Mutually exclusive with
    // the dynamic-quant branch above at the dtype level.
    if (cfg.dt[0] == data_type_t::s8 || cfg.dt[0] == data_type_t::u8) {
      src_scale = tensor_factory.uniform_dist_tensor({1, 1},
                  data_type_t::f32, 0.3);
      if (cfg.dt[0] == data_type_t::u8) {
        src_zp = tensor_factory.uniform_tensor({1, 1}, data_type_t::u8, 16);
      }
    }

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
                     dt == data_type_t::bf16 ||
                     dt == data_type_t::f16) ? tensor_factory.uniform_dist_tensor({1, 1},
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
                           cfg.post_op_dt,
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