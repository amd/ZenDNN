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

#include "gdn_benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace gdn {

int bench(const std::string &in_filename, const std::string &out_filename,
        const InputMode inputMode, const global_options &options,
        const bool isLOWOHA, size_t cache_size) {

    if (!isLOWOHA) {
        testlog_error(
                "Regular (non-LOWOHA) GDN benchmark is not supported. "
                "Please use --lowoha=true (the default) to run the GDN "
                "benchmark.");
        return NOT_OK;
    }
    if (inputMode == InputMode::COMMAND_LINE) {
        testlog_error("--op=gdn requires --input_file=<...>.");
        return NOT_OK;
    }

    std::ifstream infile(in_filename);
    if (!infile.is_open()) {
        testlog_error("Error: Cannot open file ", in_filename);
        return NOT_OK;
    }
    std::vector<GdnConfig> configs;
    inputFileParser(infile, configs, options);
    if (configs.empty()) {
        testlog_error("No valid GDN configurations to benchmark.");
        return NOT_OK;
    }

    std::vector<std::pair<GdnConfig, TimingStats>> gdn_results;
    if (gdn_lowoha_benchdnn(configs, gdn_results, options, cache_size) != OK) {
        testlog_error("LOWOHA GDN benchmark failed.");
        return NOT_OK;
    }

    print_results(gdn_results, std::cout);

    std::ofstream outfile(out_filename);
    if (!outfile.is_open()) {
        testlog_error(
                "Error: Cannot write to output file ", out_filename, "\n");
        return NOT_OK;
    }
    log_results(gdn_results, outfile);
    outfile.close();

    std::cout << "Timing results written to " << out_filename << std::endl;
    return OK;
}

} // namespace gdn
} // namespace benchdnn
} // namespace zendnnl
