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
 * @file reorder_benchdnn.cpp
 * @brief Benchmarking utility for ZenDNN reorder operator.
 *
 * Handles creation, execution, and timing of reorder operations (in-place and out-of-place),
 * and outputs results to console and CSV. Supports detailed timing breakdowns.
 */

#include "reorder_benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace reorder {

/**
 * @brief Runs a single reorder operation (in-place or out-of-place) and collects timing stats.
 *
 * @param input_tensor Input tensor for reorder.
 * @param cfg Reorder configuration (dimensions, data type, kernel, etc.).
 * @param stats Timing statistics (output).
 * @param isNotWarmup If true, measures and accumulates detailed timings.
 * @return int OK (0) on success, NOT_OK (1) on failure.
 */
int run_reorder(tensor_t input_tensor, ReorderConfig cfg, TimingStats &stats,
                bool isNotWarmup = false) {
  tensor_factory_t tensor_factory;
  status_t status;
#if MEASURE_INDIVIDUAL_TIMINGS
  double other_time = 0;

  if (!isNotWarmup) {
#endif
    auto reorder_context = reorder_context_t()
                           .set_algo_format(cfg.kernel_name)
                           .create();
    if (! reorder_context.check()) {
      testlog_error("reorder context creation failed");
      return NOT_OK;
    }

    // Reorder operator creation with name, context and input.
    auto reorder_operator = reorder_operator_t()
                            .set_name("reorder_operator")
                            .set_context(reorder_context)
                            .create()
                            .set_input("reorder_input", input_tensor);

    // Check if reorder operation creation is successful.
    if (! reorder_operator.check()) {
      testlog_error("operator ", reorder_operator.get_name(), " creation failed");
      return NOT_OK;
    }

    // Compute the reorder size (bytes needed for output tensor)
    size_t reorder_size = reorder_operator.get_reorder_size();

    if (cfg.isInplace) {
      // Extract the input buffer size (bytes)
      size_t input_buffer_size = input_tensor.get_buffer_sz_bytes();

      // In-place reorder is only possible if input and output buffer sizes match
      if (reorder_size == input_buffer_size) {
        // Use input tensor as the output buffer (in-place)
        StorageParam buffer_params = input_tensor;

        // Create output tensor with a separate view but same memory as input
        auto output_tensor = tensor_factory.blocked_tensor({cfg.rows, cfg.cols},
                             cfg.dt,
                             buffer_params,
                             "reorder_output");

        // In-place Reorder operator execution.
        // New tensor with same memory view is passed as output for reorder operation.
        status = reorder_operator
                 .set_output("reorder_output", output_tensor)
                 .execute();
        if (status == status_t::success) {
          testlog_info("operator ", reorder_operator.get_name(),
                       " execution successful.");
        }
        else {
          testlog_error("operator ", reorder_operator.get_name(), " execution failed.");
          return NOT_OK;
        }
      }
      else {
        // In-place reorder not possible if buffer sizes differ
        testlog_error("Inplace reorder is not possible for given input");
        return NOT_OK;
      }
    }
    else {
      // Out-of-place reorder: allocate new buffer for output
      void *reorder_weights = aligned_alloc(64, reorder_size);

      // Create a Pair of storage params [reorder size and reorder weights] and
      // use it in tensor creation
      StorageParam buffer_params = std::make_pair(reorder_size, reorder_weights);

      // Create output tensor with blocked layout.
      auto output_tensor = tensor_factory.blocked_tensor({cfg.rows, cfg.cols},
                           cfg.dt,
                           buffer_params,
                           "reorder_output");
      // Reorder operator execution.
      status = reorder_operator
               .set_output("reorder_output", output_tensor)
               .execute();
      if (status == status_t::success) {
        testlog_info("operator ", reorder_operator.get_name(),
                     " execution successful.");
      }
      else {
        testlog_error("operator ", reorder_operator.get_name(), " execution failed.");
        return NOT_OK;
      }
      // Free reordered size buffer.
      free(reorder_weights);
    }
#if MEASURE_INDIVIDUAL_TIMINGS
  }
  else {
    auto start_context_creation = std::chrono::high_resolution_clock::now();
    auto reorder_context = reorder_context_t()
                           .set_algo_format(cfg.kernel_name)
                           .create();
    auto end_context_creation = std::chrono::high_resolution_clock::now();
    if (! reorder_context.check()) {
      testlog_error("reorder context creation failed");
      return NOT_OK;
    }

    auto start_operator_creation = std::chrono::high_resolution_clock::now();
    // Reorder operator creation with name, context and input.
    auto reorder_operator = reorder_operator_t()
                            .set_name("reorder_operator")
                            .set_context(reorder_context)
                            .create()
                            .set_input("reorder_input", input_tensor);
    auto end_operator_creation = std::chrono::high_resolution_clock::now();
    // Check if reorder operation creation is successful.
    if (! reorder_operator.check()) {
      testlog_error("operator ", reorder_operator.get_name(), " creation failed");
      return NOT_OK;
    }

    auto start_other = std::chrono::high_resolution_clock::now();
    // Compute the reorder size
    size_t reorder_size         = reorder_operator.get_reorder_size();
    auto end_other = std::chrono::high_resolution_clock::now();
    double elapsed_time = (std::chrono::duration<double, std::milli>
                           (end_other - start_other).count());
    other_time += elapsed_time;

    if (cfg.isInplace) {
      // Extract the input buffer size
      size_t input_buffer_size    = input_tensor.get_buffer_sz_bytes();

      // Inplace reorder takes place when reorder buffer size is same as input buffer size
      if (reorder_size == input_buffer_size) {
        // Assign input_tensor to buffer_params as a tensor_t variant
        StorageParam buffer_params = input_tensor;

        auto start_other = std::chrono::high_resolution_clock::now();
        // Blocked Tensor creation with seperate view for input tensor.
        auto output_tensor = tensor_factory.blocked_tensor({cfg.rows, cfg.cols},
                             cfg.dt,
                             buffer_params,
                             "reorder_output");
        auto end_other = std::chrono::high_resolution_clock::now();
        elapsed_time = (std::chrono::duration<double, std::milli>
                        (end_other - start_other).count());
        other_time += elapsed_time;

        auto start_operator_execution = std::chrono::high_resolution_clock::now();
        // Inplace Reorder operator execution.
        // New tensor with same memory view is passed as output for reorder operation.
        status = reorder_operator
                 .set_output("reorder_output", output_tensor)
                 .execute();
        auto end_operator_execution = std::chrono::high_resolution_clock::now();
        elapsed_time = (std::chrono::duration<double, std::milli>
                        (end_operator_execution - start_operator_execution).count());
        stats.operator_execution_ms += elapsed_time;
        if (status == status_t::success) {
          testlog_info("operator ", reorder_operator.get_name(),
                       " execution successful.");
        }
        else {
          testlog_error("operator ", reorder_operator.get_name(), " execution failed.");
          return NOT_OK;
        }
      }
      else {
        testlog_error("Inplace reorder is not possible for given input");
        return NOT_OK;
      }
    }
    else {
      auto start_other = std::chrono::high_resolution_clock::now();
      void *reorder_weights = aligned_alloc(64, reorder_size);

      // Create a Pair of storage params [reorder size and reorder weights] and
      // use it in tensor creation
      StorageParam buffer_params = std::make_pair(reorder_size, reorder_weights);

      // Create output tensor with blocked layout.
      auto output_tensor = tensor_factory.blocked_tensor({cfg.rows, cfg.cols},
                           cfg.dt,
                           buffer_params,
                           "reorder_output");
      auto end_other = std::chrono::high_resolution_clock::now();
      elapsed_time = (std::chrono::duration<double, std::milli>
                      (end_other - start_other).count());
      other_time += elapsed_time;

      auto start_operator_execution = std::chrono::high_resolution_clock::now();
      // Reorder operator execution.
      status = reorder_operator
               .set_output("reorder_output", output_tensor)
               .execute();
      auto end_operator_execution = std::chrono::high_resolution_clock::now();
      elapsed_time = (std::chrono::duration<double, std::milli>
                      (end_operator_execution - start_operator_execution).count());
      stats.operator_execution_ms += elapsed_time;
      if (status == status_t::success) {
        testlog_info("operator ", reorder_operator.get_name(),
                     " execution successful.");
      }
      else {
        testlog_error("operator ", reorder_operator.get_name(), " execution failed.");
        return NOT_OK;
      }
      start_other = std::chrono::high_resolution_clock::now();
      // Free reordered size buffer.
      free(reorder_weights);
      end_other = std::chrono::high_resolution_clock::now();
      elapsed_time = (std::chrono::duration<double, std::milli>
                      (end_other - start_other).count());
      other_time += elapsed_time;
    }
    double context_creation_time = (std::chrono::duration<double, std::milli>
                                    (end_context_creation - start_context_creation).count());
    double operator_creation_time = (std::chrono::duration<double, std::milli>
                                     (end_operator_creation - start_operator_creation).count());

    stats.context_creation_ms += context_creation_time;
    stats.operator_creation_ms += operator_creation_time;
    stats.other_ms += other_time;
  }
#endif
  return OK;
}

int reorder_benchdnn(std::vector<ReorderConfig> configs,
                     std::vector<std::pair<ReorderConfig, TimingStats>> &reorder_results) {
  bool skip;
  for (const auto &cfg:configs) {
    try {
      tensor_factory_t tensor_factory;
      int status;

      // Create input tensor with contigious layout.
      auto input_tensor = tensor_factory.uniform_tensor({cfg.rows, cfg.cols},
                          cfg.dt,
                          1.0, "reorder_input");

      TimingStats time_stats;
      skip = false;
      // warm-up iterations
      for (auto i = 0; i < cfg.warmup_iters; i++) {
        status = run_reorder(input_tensor, cfg, time_stats);
        if (status != OK) {
          testlog_error("run_reorder execution failed.");
          skip = true;
          break;
        }
      }
      if (skip) {
        commonlog_error("Benchmark failed for ", cfg.rows, ", ", cfg.cols, ", ",
                        cfg.iters, ", ", datatypeToStr(cfg.dt),
                        ", ", cfg.kernel_name, ", ", cfg.isInplace);
        continue;
      }

      double elapsed_ms = 0.0;
      for (auto i = 0; i < cfg.iters; i++) {
#if COLD_CACHE
        std::vector<char> buffer(CACHE_SIZE, 1);
        flush_cache(buffer);
#endif
#if !MEASURE_INDIVIDUAL_TIMINGS
        auto start = std::chrono::high_resolution_clock::now();
#endif
        status = run_reorder(input_tensor, cfg, time_stats, true);
        if (status != OK) {
          testlog_error("run_reorder execution failed.");
          skip = true;
          break;
        }
#if !MEASURE_INDIVIDUAL_TIMINGS
        auto end = std::chrono::high_resolution_clock::now();
        double time_taken = (std::chrono::duration<double, std::milli>
                             (end - start).count());
        elapsed_ms += time_taken;
#endif
      }
      if (skip) {
        commonlog_error("Benchmark failed for ", cfg.rows, ", ", cfg.cols, ", ",
                        cfg.iters, ", ", datatypeToStr(cfg.dt),
                        ", ", cfg.kernel_name, ", ", cfg.isInplace);
        continue;
      }
#if MEASURE_INDIVIDUAL_TIMINGS
      elapsed_ms = time_stats.context_creation_ms +
                   time_stats.operator_creation_ms + time_stats.operator_execution_ms +
                   time_stats.other_ms;
#endif
      time_stats.total_time_ms = elapsed_ms;
      reorder_results.emplace_back(cfg, time_stats);
    }
    catch (const exception_t &ex) {
      std::cout << ex.what() << std::endl;
      return NOT_OK;
    }
  }

  if (skip) {
    return NOT_OK;
  }
  return OK;
}

int bench(const std::string &in_filename, const std::string &out_filename) {
  // Open the input file for reading benchmark configurations
  std::ifstream infile(in_filename);
  if (!infile.is_open()) {
    testlog_error("Error: Cannot open file ", in_filename);
    return NOT_OK;
  }
  std::vector<ReorderConfig> reorderConfig;
  inputParser(infile, reorderConfig);

  std::vector<std::pair<ReorderConfig, TimingStats>> reorder_results;
  int status = reorder_benchdnn(reorderConfig, reorder_results);
  if (status != OK) {
    testlog_error("Reorder benchmark failed.");
    return NOT_OK;
  }

  // Print results to console for each configuration
  // Iterate over all benchmarked configurations and their timing statistics.
  // For each configuration, print the configuration parameters and timing results to the console.
  // This provides a summary of the benchmark results for each tested reorder scenario.
  for (const auto &result : reorder_results) {
    const auto &config = result.first;
    const auto &stat = result.second;
    std::cout <<
              config.rows << ", " <<
              config.cols << ", " <<
              config.iters << ", " <<
              datatypeToStr(config.dt) << ", " <<
              config.kernel_name << ", " <<
              config.isInplace;
    std::cout << ", Warm-up iterations: " << config.warmup_iters;
    std::cout << ", Total time: " << stat.total_time_ms << " ms";
#if MEASURE_INDIVIDUAL_TIMINGS
    std::cout << ", Context creation: " <<
              stat.context_creation_ms << " ms, Operator creation: " <<
              stat.operator_creation_ms << " ms, Operator execution: " <<
              stat.operator_execution_ms << " ms, Others: " <<
              stat.other_ms << " ms";
#endif
    std::cout << std::endl;

  }

  std::ofstream outfile(out_filename);
  if (!outfile.is_open()) {
    testlog_error("Error: Cannot write to output file ", out_filename, "\n");
    return 1;
  }

  // Write CSV header
  outfile <<
          "Rows, Cols, Iterations, Data type, Kernel name, In-place, Warmup iterations, Total time (ms)";
#if MEASURE_INDIVIDUAL_TIMINGS
  outfile <<
          ", Context Creation Time (ms), Operator Creation Time (ms), Operator Execution Time (ms), Others (ms)";
#endif
  outfile << std::endl;

  // Write results to CSV for each configuration
  for (const auto &result : reorder_results) {
    const auto &config = result.first;
    const auto &stat = result.second;
    outfile <<
            config.rows << ", " <<
            config.cols << ", " <<
            config.iters << ", " <<
            datatypeToStr(config.dt) << ", " <<
            config.kernel_name << ", " <<
            config.isInplace << ", " <<
            config.warmup_iters << ", " <<
            stat.total_time_ms;
#if MEASURE_INDIVIDUAL_TIMINGS
    outfile <<
            ", " <<
            stat.context_creation_ms << ", " <<
            stat.operator_creation_ms << ", " <<
            stat.operator_execution_ms << ", " <<
            stat.other_ms;
#endif
    outfile << std::endl;
  }

  outfile.close();
  std::cout << "Timing results written to " << out_filename << std::endl;
  return OK;
}

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl