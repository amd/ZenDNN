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

#include "lowoha_normalization_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace normalization {

status_t validate_normalization_inputs(
  const void *input,
  const void *output,
  const void *gamma,
  const void *beta,
  const void *running_mean,
  const void *running_var,
  const void *residual,
  const norm_params &params
) {
  // Validate pointers
  if (!input) {
    log_error("Normalization: Input pointer is null");
    return status_t::failure;
  }
  if (!output) {
    log_error("Normalization: Output pointer is null");
    return status_t::failure;
  }

  // Validate normalization type
  if (params.norm_type == norm_type_t::NONE) {
    log_error("Normalization: norm_type not specified");
    return status_t::failure;
  }

  // Validate data types
  if (params.src_dt != data_type_t::f32 && params.src_dt != data_type_t::bf16) {
    log_error("Normalization: Unsupported source data type");
    return status_t::failure;
  }
  if (params.dst_dt != data_type_t::f32 && params.dst_dt != data_type_t::bf16) {
    log_error("Normalization: Unsupported destination data type");
    return status_t::failure;
  }

  // Validate batch and norm_size
  if (params.batch == 0) {
    log_error("Normalization: batch must be > 0");
    return status_t::failure;
  }
  if (params.norm_size == 0) {
    log_error("Normalization: norm_size must be > 0");
    return status_t::failure;
  }

  // Validate scale (gamma) parameter
  if (params.use_scale) {
    if (!gamma) {
      log_error("Normalization: use_scale=true but gamma pointer is null");
      return status_t::failure;
    }
    if (params.gamma_dt != data_type_t::f32 &&
        params.gamma_dt != data_type_t::bf16) {
      log_error("Normalization: Unsupported gamma data type (",
                static_cast<int>(params.gamma_dt), "). Supported: f32, bf16");
      return status_t::failure;
    }
  }

  // Validate shift (beta) parameter
  // Beta is never used by RMSNorm or FusedAddRMSNorm regardless of use_shift
  if (params.use_shift &&
      params.norm_type != norm_type_t::RMS_NORM &&
      params.norm_type != norm_type_t::FUSED_ADD_RMS_NORM) {
    if (!beta) {
      log_error("Normalization: use_shift=true but beta pointer is null "
                "(required for ", norm_type_to_str(params.norm_type), ")");
      return status_t::failure;
    }
    if (params.beta_dt != data_type_t::f32 && params.beta_dt != data_type_t::bf16) {
      log_error("Normalization: Unsupported beta data type (",
                static_cast<int>(params.beta_dt), "). Supported: f32, bf16");
      return status_t::failure;
    }
  }

  // FusedAddRMSNorm-specific validations
  if (params.norm_type == norm_type_t::FUSED_ADD_RMS_NORM) {
    if (!residual) {
      log_error("Normalization: FUSED_ADD_RMS_NORM requires a non-null "
                "writable residual buffer");
      return status_t::failure;
    }
  }

  // BatchNorm-specific validations
  if (params.norm_type == norm_type_t::BATCH_NORM) {
    if (!running_mean) {
      log_error("Normalization: BatchNorm inference requires running_mean "
                "(pre-computed from training)");
      return status_t::failure;
    }
    if (!running_var) {
      log_error("Normalization: BatchNorm inference requires running_var "
                "(pre-computed from training)");
      return status_t::failure;
    }
  }

  // Epsilon validation
  if (params.epsilon <= 0.0f) {
    log_error("Normalization: epsilon must be > 0, got ", params.epsilon);
    return status_t::failure;
  }

  return status_t::success;
}

std::string norm_type_to_str(norm_type_t type) {
  switch (type) {
  case norm_type_t::LAYER_NORM:
    return "LayerNorm";
  case norm_type_t::BATCH_NORM:
    return "BatchNorm";
  case norm_type_t::RMS_NORM:
    return "RMSNorm";
  case norm_type_t::FUSED_ADD_RMS_NORM:
    return "FusedAddRMSNorm";
  default:
    return "Unknown";
  }
}

} // namespace normalization
} // namespace lowoha
} // namespace zendnnl

