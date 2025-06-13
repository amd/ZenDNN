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

#include "reorder_parser.hpp"

namespace zendnnl {
namespace benchdnn {
namespace reorder {

void inputParser(std::ifstream &infile, std::vector<ReorderConfig> &configs) {
  std::string line;

  // Parse each line of the input file into a ReorderConfig object
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }

    // Split the line into fields and validate
    auto fields = split(line, ',');
    if (fields.size() < 6) {
      commonlog_error(
        "Invalid line (expected 6 fields): [rows, cols, iterations, dtype, kernel name, isInplace, warmup_iters (optional)]");
      continue;
    }
    ReorderConfig cfg;
    try {
      // Parse tensor dimensions and iteration count
      cfg.rows = std::stoi(fields[0]);
      cfg.cols = std::stoi(fields[1]);
      cfg.iters = std::stoi(fields[2]);
      cfg.dt = strToDatatype(fields[3]);
      // Parse kernel name (default to 'aocl' if empty)
      if (fields[4].empty()) {
        commonlog_warning("No kernel name specified. Defaulting to 'aocl'.");
        cfg.kernel_name = "aocl";
      }
      else {
        cfg.kernel_name = fields[4];
      }
      // Parse in-place flag
      cfg.isInplace = (fields[5] == "true");
      // Parse warmup iterations if provided, otherwise use 20% of main iterations
      if (fields.size() == 7) {
        cfg.warmup_iters = std::stoi(fields[6]);
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

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl