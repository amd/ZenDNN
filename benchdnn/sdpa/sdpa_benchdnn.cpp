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

#include "sdpa_benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace sdpa {

int bench(const std::string &in_filename, const std::string &out_filename,
          const InputMode inputMode, const global_options &options,
          const bool isLOWOHA, size_t cache_size) {

  if (!isLOWOHA) {
    testlog_error("Regular (non-LOWOHA) SDPA benchmark is not supported. "
                  "Please use --lowoha=true (the default) to run the SDPA benchmark.");
    return NOT_OK;
  }

  std::vector<SdpaConfig> configs;

  if (inputMode == InputMode::FILE) {
    std::ifstream infile(in_filename);
    if (!infile.is_open()) {
      testlog_error("Error: Cannot open file ", in_filename);
      return NOT_OK;
    }
    inputFileParser(infile, configs);
  }
  else if (inputMode == InputMode::MODEL) {
    std::ifstream infile(in_filename);
    if (!infile.is_open()) {
      testlog_error("Error: Cannot open file ", in_filename);
      return NOT_OK;
    }
    inputModelFileParser(infile, configs, options);
  }
  else { // COMMAND_LINE
    inputCommandLineParser(configs, options);
  }

  if (configs.empty()) {
    testlog_error("No valid SDPA configurations to benchmark.");
    return NOT_OK;
  }

  std::vector<std::pair<SdpaConfig, TimingStats>> sdpa_results;

  int status = sdpa_lowoha_benchdnn(configs, sdpa_results, options, cache_size);
  if (status != OK) {
    testlog_error("LOWOHA SDPA benchmark failed.");
    return NOT_OK;
  }

  print_results(sdpa_results, std::cout, inputMode);

  std::ofstream outfile(out_filename);
  if (!outfile.is_open()) {
    testlog_error("Error: Cannot write to output file ", out_filename, "\n");
    return NOT_OK;
  }
  log_results(sdpa_results, outfile, inputMode);
  outfile.close();

  std::cout << "Timing results written to " << out_filename << std::endl;
  return OK;
}

} // namespace sdpa
} // namespace benchdnn
} // namespace zendnnl
