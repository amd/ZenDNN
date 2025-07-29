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

#include "matmul_parser.hpp"

namespace zendnnl {
namespace benchdnn {
namespace matmul {

void inputParser(std::ifstream &infile, std::vector<MatmulConfig> &configs,
                 bool &isPipeline) {
  std::string line;

  // Parse each line of the input file into a MatmulConfig object
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    // Split the line into fields and validate
    auto fields = split(line, ',');
    if (fields.size() < 9) {
      commonlog_error(
        "Invalid line (expected 9 fields): [m, k, n, iterations, input_dtype:weights_dtype:output_dtype, isBiasEnabled, bias_dtype, postOp, kernel name, warmup iterations (optional)]");
      continue;
    }
    MatmulConfig cfg;
    try {
      // Parse matrix dimensions and iteration count
      cfg.m = std::stoi(fields[0]);
      cfg.k = std::stoi(fields[1]);
      // Parse n values (colon-separated for multi-layer)
      auto n_values = split(fields[2], ':');
      for (const auto &n : n_values) {
        cfg.n_values.push_back(std::stoi(n));
      }
      // Set isPipeline to true if more than one n value is present
      isPipeline = (n_values.size() > 1) ? true : isPipeline;
      cfg.iters = std::stoi(fields[3]);
      // Parse data types (input:weights:output)
      auto dt = split(fields[4], ':');
      if (fields[4].size() > 0) {
        auto i = 0;
        for (; i < dt.size(); i++) {
          cfg.dt.push_back(strToDatatype(dt[i]));
        }
        for (; i < 3; i++) {
          cfg.dt.push_back(data_type_t::f32);
        }
        if (dt.size() < 3) {
          commonlog_warning("Less than 3 data types specified. Defaulting missing types to f32.");
        }
      }
      else {
        cfg.dt.push_back(data_type_t::f32);
        cfg.dt.push_back(data_type_t::f32);
        cfg.dt.push_back(data_type_t::f32);
        commonlog_warning("No data types specified. Defaulting all to f32.");
      }
      // Parse bias flag and bias data type
      cfg.isBiasEnabled = (fields[5] == "true");
      if (cfg.isBiasEnabled) {
        if (!fields[6].empty()) {
          cfg.bias_dt = strToDatatype(fields[6]);
        }
        else {
          commonlog_warning("No data type specified for bias. Defaulting it to f32.");
          cfg.bias_dt = data_type_t::f32;
        }
      }
      // Parse post-operations (e.g., relu, gelu, binary ops)
      auto postOps = split(fields[7], ':');
      auto binary_post_op_pos = 0;
      for (auto i = 0; i < postOps.size(); i++) {
        if (!postOps[i].empty()) {
          cfg.post_ops.push_back(strToPostOps(postOps[i]));
          // Track positions of binary post-operations
          if (postOps[i].find("binary_") == 0) {
            cfg.binary_post_ops_pos.push_back(binary_post_op_pos);
          }
          binary_post_op_pos++;
        }
      }
      // Parse kernel name (default to 'aocl_blis' if empty)
      if (fields[8].empty()) {
        commonlog_warning("No kernel name specified. Defaulting to 'aocl_blis'.");
        cfg.kernel_name = "aocl_blis";
      }
      else {
        cfg.kernel_name = fields[8];
      }
      // Parse warmup iterations if provided, otherwise use 20% of main iterations
      if (fields.size() == 10) {
        cfg.warmup_iters = std::stoi(fields[9]);
      }
      else {
        cfg.warmup_iters = 0.2 * cfg.iters;
      }

      configs.push_back(cfg);
    }
    catch (const std::exception &e) {
      commonlog_error(e.what());
      continue;
    }
  }
}

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl