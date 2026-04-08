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

#include "normalization_benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace normalization {

int bench(const std::string &in_filename, const std::string &out_filename,
          const bool isLOWOHA, size_t cache_size) {

  if (!isLOWOHA) {
    testlog_error("Regular (non-LOWOHA) normalization benchmark is not supported. "
                  "Please use --lowoha=true to run the normalization benchmark.");
    return NOT_OK;
  }

  std::ifstream infile(in_filename);
  if (!infile.is_open()) {
    testlog_error("Error: Cannot open file ", in_filename);
    return NOT_OK;
  }

  std::vector<NormalizationConfig> normalizationConfigs;
  inputParser(infile, normalizationConfigs);

  std::vector<std::pair<NormalizationConfig, TimingStats>> normalization_results;

  int status = normalization_lowoha_benchdnn(normalizationConfigs,
               normalization_results, cache_size);
  if (status != OK) {
    testlog_error("LOWOHA Normalization benchmark failed.");
    return NOT_OK;
  }

  print_results(normalization_results, std::cout);

  std::ofstream outfile(out_filename);
  if (!outfile.is_open()) {
    testlog_error("Error: Cannot write to output file ", out_filename, "\n");
    return NOT_OK;
  }

  log_results(normalization_results, outfile);
  outfile.close();

  std::cout << "Timing results written to " << out_filename << std::endl;
  return OK;
}

} // namespace normalization
} // namespace benchdnn
} // namespace zendnnl
