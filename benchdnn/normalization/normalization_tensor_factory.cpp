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

#include "normalization_tensor_factory.hpp"
#include "normalization_utils.hpp"
#include <cmath>

namespace zendnnl {
namespace benchdnn {
namespace normalization {

int create_input_tensor(tensor_factory_t &tensor_factory,
                        const NormalizationConfig &cfg, tensor_t &input) {
  input = tensor_factory.uniform_dist_tensor(cfg.shape, cfg.src_dt,
          2.0f, "input_tensor");
  return OK;
}

int create_output_tensor(tensor_factory_t &tensor_factory,
                         const NormalizationConfig &cfg, tensor_t &output) {
  output = tensor_factory.zero_tensor(cfg.shape, cfg.dst_dt, "output_tensor");
  return OK;
}

int create_gamma_tensor(tensor_factory_t &tensor_factory,
                        const NormalizationConfig &cfg, tensor_t &gamma) {
  if (!cfg.use_scale) {
    gamma = tensor_t();
    return OK;
  }

  uint64_t param_size;
  if (cfg.norm_type == "batch_norm") {
    param_size = compute_num_channels(cfg);
  }
  else {
    param_size = compute_norm_size(cfg);
  }

  gamma = tensor_factory.uniform_dist_tensor({param_size}, cfg.gamma_dt,
          1.0f, "gamma_tensor");
  return OK;
}

int create_beta_tensor(tensor_factory_t &tensor_factory,
                       const NormalizationConfig &cfg, tensor_t &beta) {
  if (!cfg.use_shift ||
      cfg.norm_type == "rms_norm" ||
      cfg.norm_type == "fused_add_rms_norm") {
    beta = tensor_t();
    return OK;
  }

  uint64_t param_size;
  if (cfg.norm_type == "batch_norm") {
    param_size = compute_num_channels(cfg);
  }
  else {
    param_size = compute_norm_size(cfg);
  }

  beta = tensor_factory.uniform_dist_tensor({param_size}, cfg.beta_dt,
         1.0f, "beta_tensor");
  return OK;
}

int create_running_mean_tensor(tensor_factory_t &tensor_factory,
                               const NormalizationConfig &cfg,
                               tensor_t &running_mean) {
  if (cfg.norm_type != "batch_norm") {
    running_mean = tensor_t();
    return OK;
  }

  uint64_t num_channels = compute_num_channels(cfg);
  running_mean = tensor_factory.uniform_dist_tensor({num_channels},
                 data_type_t::f32, 2.0f, "running_mean_tensor");
  return OK;
}

int create_running_var_tensor(tensor_factory_t &tensor_factory,
                              const NormalizationConfig &cfg,
                              tensor_t &running_var) {
  if (cfg.norm_type != "batch_norm") {
    running_var = tensor_t();
    return OK;
  }

  uint64_t num_channels = compute_num_channels(cfg);
  running_var = tensor_factory.uniform_dist_tensor({num_channels},
                data_type_t::f32, 1.0f, "running_var_tensor");

  // Variance must be non-negative; ensure all values are positive to avoid
  // NaN in the 1/sqrt(var + epsilon) computation.
  float *var_ptr = static_cast<float *>(
                     running_var.get_raw_handle_unsafe());
  if (!var_ptr) {
    commonlog_error("Failed to get raw handle for running_var tensor");
    return NOT_OK;
  }
  for (uint64_t i = 0; i < num_channels; ++i) {
    var_ptr[i] = std::fabs(var_ptr[i]) + 0.1f;
  }

  return OK;
}

int create_residual_tensor(tensor_factory_t &tensor_factory,
                           const NormalizationConfig &cfg,
                           tensor_t &residual) {
  if (cfg.norm_type != "fused_add_rms_norm") {
    residual = tensor_t();
    return OK;
  }

  residual = tensor_factory.uniform_dist_tensor(cfg.shape, cfg.src_dt,
             2.0f, "residual_tensor");
  return OK;
}

} // namespace normalization
} // namespace benchdnn
} // namespace zendnnl
