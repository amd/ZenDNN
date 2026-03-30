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

#include "reorder_tensor_factory.hpp"
#include "reorder_utils.hpp"
#include <cmath>

namespace zendnnl {
namespace benchdnn {
namespace reorder {

static std::vector<uint64_t> to_uint64_dims(const std::vector<int64_t> &dims) {
  std::vector<uint64_t> result(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    result[i] = static_cast<uint64_t>(dims[i]);
  }
  return result;
}

std::vector<int64_t> compute_quant_dims(const ReorderConfig &cfg) {
  int64_t M = static_cast<int64_t>(cfg.rows);
  int64_t N = static_cast<int64_t>(cfg.cols);
  bool is_3d = (cfg.batch_size > 1);

  if (cfg.scale_granularity == "per_tensor") {
    return is_3d ? std::vector<int64_t> {1, 1, 1}
           :
           std::vector<int64_t> {1, 1};
  }
  else if (cfg.scale_granularity == "per_channel_row") {
    return is_3d ? std::vector<int64_t> {1, M, 1}
           :
           std::vector<int64_t> {M, 1};
  }
  else if (cfg.scale_granularity == "per_channel_col") {
    return is_3d ? std::vector<int64_t> {1, 1, N}
           :
           std::vector<int64_t> {1, N};
  }
  else if (cfg.scale_granularity == "per_group_row") {
    const int64_t group_size = static_cast<int64_t>(cfg.group_size);
    if (group_size > 0 && group_size <= M && (M % group_size) == 0) {
      const int64_t G = M / group_size;
      return is_3d ? std::vector<int64_t> {1, G, N}
             :
             std::vector<int64_t> {G, N};
    }
    commonlog_warning(
      "Invalid group_size ", group_size,
      " for per_group_row (rows=", M,
      "). Falling back to per_channel_row quantization.");
    return is_3d ? std::vector<int64_t> {1, M, 1}
           :
           std::vector<int64_t> {M, 1};
  }
  else if (cfg.scale_granularity == "per_group_col") {
    const int64_t group_size = static_cast<int64_t>(cfg.group_size);
    if (group_size > 0 && group_size <= N && (N % group_size) == 0) {
      const int64_t G = N / group_size;
      return is_3d ? std::vector<int64_t> {1, M, G}
             :
             std::vector<int64_t> {M, G};
    }
    commonlog_warning(
      "Invalid group_size ", group_size,
      " for per_group_col (cols=", N,
      "). Falling back to per_channel_col quantization.");
    return is_3d ? std::vector<int64_t> {1, 1, N}
           :
           std::vector<int64_t> {1, N};
  }

  commonlog_warning("Unknown scale_granularity '", cfg.scale_granularity,
                    "'. Defaulting to per_tensor.");
  return is_3d ? std::vector<int64_t> {1, 1, 1}
         :
         std::vector<int64_t> {1, 1};
}

int create_src_tensor(tensor_factory_t &tensor_factory,
                      const ReorderConfig &cfg,
                      tensor_t &src, bool is_lowoha) {
  if (!is_lowoha) {
    src = tensor_factory.uniform_tensor({cfg.rows, cfg.cols},
                                        cfg.dt, 1.0, "reorder_input");
  }
  else {
    if (cfg.batch_size > 1) {
      src = tensor_factory.uniform_dist_tensor(
      {cfg.batch_size, cfg.rows, cfg.cols},
      cfg.src_dtype, 1.0, "reorder_src");
    }
    else {
      src = tensor_factory.uniform_dist_tensor(
      {cfg.rows, cfg.cols},
      cfg.src_dtype, 1.0, "reorder_src");
    }
  }
  if (!src.check()) {
    commonlog_error("Failed to create source tensor");
    return NOT_OK;
  }
  return OK;
}

int create_dst_tensor(tensor_factory_t &tensor_factory,
                      const ReorderConfig &cfg,
                      tensor_t &dst) {
  if (cfg.batch_size > 1) {
    dst = tensor_factory.uniform_tensor(
    {cfg.batch_size, cfg.rows, cfg.cols},cfg.dst_dtype, 0, "reorder_dst");
  }
  else {
    dst = tensor_factory.uniform_tensor(
    {cfg.rows, cfg.cols},
    cfg.dst_dtype, 0, "reorder_dst");
  }
  if (!dst.check()) {
    commonlog_error("Failed to create destination tensor");
    return NOT_OK;
  }
  return OK;
}

int create_scale_tensor(tensor_factory_t &tensor_factory,
                        const ReorderConfig &cfg, tensor_t &scale) {
  auto quant_dims = compute_quant_dims(cfg);
  auto quant_dims_u64 = to_uint64_dims(quant_dims);

  if (!cfg.dynamic_quant) {
    scale = tensor_factory.uniform_dist_tensor(
              quant_dims_u64, data_type_t::f32, 10.0f, "scale");
  }
  else {
    scale = tensor_factory.uniform_tensor(
              quant_dims_u64, data_type_t::f32, 0, "scale");
  }
  if (!scale.check()) {
    commonlog_error("Failed to create scale tensor");
    return NOT_OK;
  }
  if (!cfg.dynamic_quant) {
    float *scale_ptr = static_cast<float *>(scale.get_raw_handle_unsafe());
    size_t scale_nelem = scale.get_nelem();
    for (size_t i = 0; i < scale_nelem; ++i) {
      scale_ptr[i] = 0.01f + std::fabs(scale_ptr[i]);
    }
  }
  return OK;
}

int create_zp_tensor(tensor_factory_t &tensor_factory,
                     const ReorderConfig &cfg, tensor_t &zp) {
  auto quant_dims = compute_quant_dims(cfg);
  auto quant_dims_u64 = to_uint64_dims(quant_dims);

  zp = tensor_factory.zero_tensor(quant_dims_u64, data_type_t::s32);
  if (!zp.check()) {
    commonlog_error("Failed to create zero-point tensor");
    return NOT_OK;
  }

  if (!cfg.dynamic_quant) {
    int32_t *zp_ptr = static_cast<int32_t *>(zp.get_raw_handle_unsafe());
    size_t zp_nelem = zp.get_nelem();
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(0, 128);
    for (size_t i = 0; i < zp_nelem; ++i) {
      zp_ptr[i] = dist(gen);
    }
  }
  return OK;
}

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl
