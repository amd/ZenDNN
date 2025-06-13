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

/**
 * @file benchdnn.cpp
 * @brief Main entry point for the benchdnn benchmarking utility.
 *
 * This file implements the main() function for the benchdnn utility, which parses
 * command-line arguments, reads the input configuration file, and dispatches the
 * appropriate benchmark (e.g., matmul) based on user input. Results are written
 * to a timestamped CSV file and printed to stdout.
 */

#include "benchdnn.hpp"

/**
 * @brief Main entry point for the benchdnn benchmark utility.
 *
 * This function reads a configuration file where each line specifies a benchmark run.
 * Each line must have the format:
 *   m, k, n, iterations, input_dtype:weights_dtype:output_dtype, postOp, kernel name
 *
 * The function parses each line, constructs a Config object, and runs the benchmark for all configurations.
 * Results are printed to stdout and written to a CSV file with a timestamped filename.
 *
 * Command-line arguments:
 *   --op=<matmul|reorder>         Operator to benchmark
 *   --input_file=<filename>       Path to the input configuration file
 *
 * Example usage:
 *   ./benchdnn --op=matmul --input_file=config.txt
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return int Status code (0 for success, non-zero for error).
 */
int main(int argc, char **argv) {
  // Parse command-line arguments for operator type and input file
  std::string op, input_file;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--op=") == 0) {
      op = arg.substr(5);
    }
    else if (arg.find("--input_file=") == 0) {
      input_file = arg.substr(13);
    }
  }

  // Validate required arguments
  if (op.empty() || input_file.empty()) {
    commonlog_error("Usage: ", argv[0],
                    " --op=<matmul|reorder> --input_file=<filename>\n");
    return NOT_OK;
  }

  // Generate output filename based on current timestamp for CSV results
  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              now.time_since_epoch()) % 1000;
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << "timings_" <<
     std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S") <<
     "_" << std::setfill('0') << std::setw(3) << ms.count() << ".csv";

  std::string out_filename = ss.str();
  std::string in_filename = input_file;

  // Dispatch to the appropriate benchmark based on operator type
  if (op == "matmul") {
    benchdnn::matmul::bench(in_filename, out_filename); ///< Run matmul benchmark
  }
  else if (op == "reorder") {
    benchdnn::reorder::bench(in_filename, out_filename); ///< Run reorder benchmark
  }
  else {
    commonlog_error("Unsupported operator: ", op);
    commonlog_error("Supported operators: matmul, reorder");
    return NOT_OK;
  }

  return OK;
}
