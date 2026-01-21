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

/**
 * @file benchdnn.cpp
 * @brief Main entry point for the benchdnn benchmarking utility.
 *
 * This file implements the main() function for the benchdnn utility, which parses
 * command-line arguments, reads the input configuration file, and dispatches the
 * appropriate benchmark (matmul or reorder) based on user input. Results are written
 * to a timestamped CSV file and printed to stdout.
 *
 * Usage:
 *   ./benchdnn --op=<matmul|reorder> --input_file=<filename>
 *
 */

#include "benchdnn.hpp"

/**
 * @brief Main entry point for the benchdnn benchmark utility.
 *
 * Parses command-line arguments to determine the operator (matmul or reorder) and input file.
 * Validates arguments, generates a timestamped output filename, and dispatches the benchmark.
 * Results are printed to stdout and written to a CSV file.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return int Status code (0 for success, non-zero for error).
 */
int main(int argc, char **argv) {
  // Parse command-line arguments for operator type and input file
  std::string op, input_file;
  benchdnn::global_options options;
  options.ndims = 2;
  bool isLOWOHA = true;
  benchdnn::InputMode inputMode = benchdnn::InputMode::COMMAND_LINE;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    // Parse operator argument
    if (arg.find("--op=") == 0) {
      op = arg.substr(5);
    }
    // Parse input file argument
    else if (arg.find("--input_file=") == 0) {
      input_file = arg.substr(13);
      if (inputMode == benchdnn::InputMode::COMMAND_LINE) {
        inputMode = benchdnn::InputMode::FILE;
      }
      else {
        commonlog_error("Multiple input modes specified. Please specify only one input mode.");
        return NOT_OK;
      }
    }
    else if (arg.find("--lowoha=") == 0) {
      std::string value = arg.substr(9);
      std::transform(value.begin(), value.end(), value.begin(), ::tolower);
      if (value == "true" || value == "1") {
        isLOWOHA = true;
      }
      else if (value == "false" || value == "0") {
        isLOWOHA = false;
      }
      else {
        commonlog_error("Invalid value for --lowoha. Use true/false or 1/0.");
        return NOT_OK;
      }
    }
    else if (arg.find("--input_model_file=") == 0) {
      input_file = arg.substr(19);
      if (inputMode == benchdnn::InputMode::COMMAND_LINE) {
        inputMode = benchdnn::InputMode::MODEL;
      }
      else {
        commonlog_error("Multiple input modes specified. Please specify only one input mode.");
        return NOT_OK;
      }
    }
    else {
      int status = benchdnn::parseCLArgs(options, arg);
      if (status != OK) {
        return NOT_OK;
      }
    }
  }

  // Validate required arguments
  if (op.empty()) {
    commonlog_error("Usage: ", argv[0], " --op=<matmul|reorder|embag> ...");
    return NOT_OK;
  }

  if ((inputMode == benchdnn::InputMode::MODEL ||
       inputMode == benchdnn::InputMode::FILE) && input_file.empty()) {
    commonlog_error("Input file is required for MODEL or FILE mode.");
    return NOT_OK;
  }

  if (inputMode == benchdnn::InputMode::COMMAND_LINE) {
    if ((options.ndims > 2 && options.bs == 0) || options.m == 0 ||
        options.k == 0 || options.n_values.size() < 1) {
      commonlog_error("For COMMAND_LINE mode, ", (options.ndims > 2) ? "--bs, " : "",
                      "--m, --k, and --n must be specified.");
      return NOT_OK;
    }
  }

  size_t cache_size = 0;
#if COLD_CACHE
  cache_size = benchdnn::get_cache_size();
#endif
  // Generate output filename based on current timestamp for CSV results
  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              now.time_since_epoch()) % 1000;
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << "timings_"
     << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S")
     << "_" << std::setfill('0') << std::setw(3) << ms.count() << ".csv";

  std::string out_filename = ss.str();
  std::string in_filename = input_file;

  // Dispatch to the appropriate benchmark based on operator type
  if (op == "matmul") {
    benchdnn::matmul::bench(in_filename, out_filename, inputMode,
                            options, isLOWOHA, cache_size); ///< Run matmul benchmark
  }
  else if (op == "reorder") {
    benchdnn::reorder::bench(in_filename, out_filename,
                             cache_size); ///< Run reorder benchmark
  }
  else if (op == "embag") {
    benchdnn::embag::bench(in_filename, out_filename,
                           cache_size); ///< Run embag benchmark
  }
  else {
    commonlog_error("Unsupported operator: ", op);
    commonlog_error("Supported operators: matmul, reorder, embag");
    return NOT_OK;
  }

  return OK;
}
