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

#include "matmul_benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace matmul {

int run_matmul(tensor_t output_tensor, tensor_t input_tensor, tensor_t weights,
               tensor_t bias, MatmulConfig cfg, std::vector<tensor_t> binary_post_ops_tensors,
               TimingStats &stats, bool isNotWarmup) {
  status_t status;
  input_tensor.set_name("matmul_input");
  output_tensor.set_name("matmul_output");
#if MEASURE_INDIVIDUAL_TIMINGS
  if (!isNotWarmup) {
#endif
    //define matmul context
    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights)
                          .set_alpha(cfg.alpha)
                          .set_beta(cfg.beta);
    if (cfg.isBiasEnabled) {
      matmul_context.set_param("bias", bias);
    }

    for (auto i = 0; i < cfg.post_ops.size(); i++) {
      auto postOp = post_op_t{cfg.post_ops[i]};
      matmul_context.set_post_op(postOp);
    }
    matmul_context.create();

    if (! matmul_context.check()) {
      testlog_error("matmul context creation failed");
      return NOT_OK;
    }

    //define matmul operator
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul")
                           .set_context(matmul_context)
                           .create();

    if (matmul_operator.is_bad_object()) {
      testlog_error(" operator ", matmul_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    matmul_operator.set_input("matmul_input", input_tensor);
    for (size_t i = 0; i < binary_post_ops_tensors.size(); i++) {
      if (postOpsToStr(cfg.post_ops[cfg.binary_post_ops_pos[i]]) == "binary_mul") {
        // Set the input tensor for binary post-op multiplication
        // using the tensor name defined in the context.
        matmul_operator.set_input(matmul_context.get_post_op(
                                    cfg.binary_post_ops_pos[i]).binary_mul_params.tensor_name,
                                  binary_post_ops_tensors[i]);
      }
      else if (postOpsToStr(
                 cfg.post_ops[cfg.binary_post_ops_pos[i]]) == "binary_add") {
        // Set the input tensor for binary post-op addition
        // using the tensor name defined in the context.
        matmul_operator.set_input(matmul_context.get_post_op(
                                    cfg.binary_post_ops_pos[i]).binary_add_params.tensor_name,
                                  binary_post_ops_tensors[i]);
      }
    }
    matmul_operator.set_output("matmul_output", output_tensor);
    status = matmul_operator.set_forced_kernel(cfg.kernel_name).execute();
    if (status == status_t::success) {
      testlog_info("<",matmul_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",matmul_operator.get_name(),">",
                    " operator execution failed.");
      return NOT_OK;
    }
#if MEASURE_INDIVIDUAL_TIMINGS
  }
  else {
    auto start_context_creation = std::chrono::high_resolution_clock::now();
    //define matmul context
    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights)
                          .set_alpha(cfg.alpha)
                          .set_beta(cfg.beta);
    if (cfg.isBiasEnabled) {
      matmul_context.set_param("bias", bias);
    }

    for (auto i = 0; i < cfg.post_ops.size(); i++) {
      auto postOp = post_op_t{cfg.post_ops[i]};
      matmul_context.set_post_op(postOp);
    }
    matmul_context.create();
    auto end_context_creation = std::chrono::high_resolution_clock::now();

    if (! matmul_context.check()) {
      testlog_error("matmul context creation failed");
      return NOT_OK;
    }

    auto start_operator_creation = std::chrono::high_resolution_clock::now();
    //define matmul operator
    auto matmul_operator = matmul_operator_t()
                           .set_name("matmul")
                           .set_context(matmul_context)
                           .create();
    auto end_operator_creation = std::chrono::high_resolution_clock::now();

    if (matmul_operator.is_bad_object()) {
      testlog_error(" operator ", matmul_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    auto start_operator_execution = std::chrono::high_resolution_clock::now();
    matmul_operator.set_input("matmul_input", input_tensor);
    for (size_t i = 0; i < binary_post_ops_tensors.size(); i++) {
      if (postOpsToStr(cfg.post_ops[cfg.binary_post_ops_pos[i]]) == "binary_mul") {
        // Set the input tensor for binary post-op multiplication
        // using the tensor name defined in the context.
        matmul_operator.set_input(matmul_context.get_post_op(
                                    cfg.binary_post_ops_pos[i]).binary_mul_params.tensor_name,
                                  binary_post_ops_tensors[i]);
      }
      else if (postOpsToStr(
                 cfg.post_ops[cfg.binary_post_ops_pos[i]]) == "binary_add") {
        // Set the input tensor for binary post-op addition
        // using the tensor name defined in the context.
        matmul_operator.set_input(matmul_context.get_post_op(
                                    cfg.binary_post_ops_pos[i]).binary_add_params.tensor_name,
                                  binary_post_ops_tensors[i]);
      }
    }
    status = matmul_operator.set_output("matmul_output", output_tensor)
             .set_forced_kernel(cfg.kernel_name)
             .execute();
    auto end_operator_execution = std::chrono::high_resolution_clock::now();

    stats.context_creation_ms += (std::chrono::duration<double, std::milli>
                                  (end_context_creation - start_context_creation).count());
    stats.operator_creation_ms += (std::chrono::duration<double, std::milli>
                                   (end_operator_creation - start_operator_creation).count());
    stats.operator_execution_ms += (std::chrono::duration<double, std::milli>
                                    (end_operator_execution - start_operator_execution).count());
    if (status == status_t::success) {
      testlog_info("<",matmul_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",matmul_operator.get_name(),">",
                    " operator execution failed.");
      return NOT_OK;
    }
  }
#endif
  return OK;
}

int matmul_benchdnn(std::vector<MatmulConfig> configs,
                    std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>>
                    &matmul_results, const global_options &options,
                    size_t cache_size) {

  bool skip;
  for (const auto &cfg:configs) {
    try {
      skip = false;
      // Check if weight data type is u8, which is not supported
      if (cfg.dt[1] == data_type_t::u8) {
        testlog_error("Weight data type u8 is not supported");
        log_benchmark_failure(cfg);
        continue;
      }

      tensor_factory_t tensor_factory;
      tensor_t input_tensor;
      std::vector<tensor_t> weights, bias, output_tensor;

      int ret = create_weights_tensor(tensor_factory, cfg, weights, options);
      if (ret != OK) {
        testlog_error("create_weights_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_bias_tensor(tensor_factory, cfg, bias, options);
      if (ret != OK) {
        testlog_error("create_bias_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_input_tensor(tensor_factory, cfg, input_tensor, options);
      if (ret != OK) {
        testlog_error("create_input_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_output_tensor(tensor_factory, cfg, output_tensor, options);
      if (ret != OK) {
        testlog_error("create_output_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      std::vector<std::vector<tensor_t>> binary_post_ops_tensors;
      ret = create_binary_post_ops_tensors(tensor_factory, cfg,
                                           binary_post_ops_tensors);
      if (ret != OK) {
        testlog_error("create_binary_post_ops_tensors failed");
        log_benchmark_failure(cfg);
        continue;
      }

      TimingStats time_stats;
      // warm-up iterations
      for (auto j = 0; j < cfg.warmup_iters && !skip; j++) {
        for (auto i = 0; i < cfg.n_values.size(); i++) {
          int ret = run_matmul(output_tensor[i],
                               (i == 0) ? input_tensor : output_tensor[i - 1],
                               weights[i], (bias.empty() ? tensor_t() : bias[i]),
                               cfg, binary_post_ops_tensors[i], time_stats);
          if (ret != OK) {
            testlog_error("run_matmul execution failed.");
            skip = true;
            break;
          }
        }
        if (skip) {
          break;
        }
      }
      if (skip) {
        log_benchmark_failure(cfg);
        continue;
      }

      std::vector<TimingStats> time_stats_layer(cfg.n_values.size());
      std::vector<double> elapsed_ms_layer(cfg.n_values.size(), 0.0);

      for (auto j = 0; j < cfg.iters && !skip; j++) {
#if COLD_CACHE
        flush_cache(cache_size);
#endif
        for (auto i = 0; i < cfg.n_values.size(); i++) {
#if !MEASURE_INDIVIDUAL_TIMINGS
          auto start_layer = std::chrono::high_resolution_clock::now();
#endif
          TimingStats time_stats; // Per-layer, per-iteration
          int ret = run_matmul(output_tensor[i],
                               (i == 0) ? input_tensor : output_tensor[i - 1],
                               weights[i], (bias.empty() ? tensor_t() : bias[i]),
                               cfg, binary_post_ops_tensors[i], time_stats, true);
          if (ret != OK) {
            testlog_error("run_matmul execution failed.");
            skip = true;
            break;
          }
#if MEASURE_INDIVIDUAL_TIMINGS
          // Accumulate timings for each layer
          time_stats_layer[i].context_creation_ms += time_stats.context_creation_ms;
          time_stats_layer[i].operator_creation_ms += time_stats.operator_creation_ms;
          time_stats_layer[i].operator_execution_ms += time_stats.operator_execution_ms;
          time_stats_layer[i].total_time_ms += time_stats.context_creation_ms +
                                               time_stats.operator_creation_ms +
                                               time_stats.operator_execution_ms;
#else
          auto end_layer = std::chrono::high_resolution_clock::now();
          double time_taken = (std::chrono::duration<double, std::milli>
                               (end_layer - start_layer).count());
          elapsed_ms_layer[i] += time_taken;
#endif
        }
        if (skip) {
          log_benchmark_failure(cfg);
          break;
        }
      }
      if (skip) {
        continue;
      }

#if !MEASURE_INDIVIDUAL_TIMINGS
      // Store total time for each layer
      for (size_t i = 0; i < cfg.n_values.size(); i++) {
        time_stats_layer[i].total_time_ms = elapsed_ms_layer[i];
      }
#endif
      print_matmul_execution_summary(cfg, time_stats_layer, options);
      matmul_results.emplace_back(cfg, time_stats_layer);
    }
    catch (const exception_t &ex) {
      std::cout << ex.what() << std::endl;
      return NOT_OK;
    }
  }
  return OK;
}

int bench(const std::string &in_filename, const std::string &out_filename,
          const InputMode inputMode, const global_options &options, const bool isLOWOHA,
          size_t cache_size
         ) {

  std::vector<MatmulConfig> matmulConfig;
  bool isPipeline = false;
  if (inputMode == InputMode::FILE) {
    // Open the input file for reading benchmark configurations
    std::ifstream infile(in_filename);
    if (!infile.is_open()) {
      testlog_error("Error: Cannot open file ", in_filename);
      return NOT_OK;
    }
    inputFileParser(infile, matmulConfig, isPipeline, options);
  }
  else if (inputMode == InputMode::MODEL) {
    // Open the input file for reading benchmark configurations
    std::ifstream infile(in_filename);
    if (!infile.is_open()) {
      testlog_error("Error: Cannot open file ", in_filename);
      return NOT_OK;
    }
    inputModelFileParser(infile, matmulConfig, isPipeline, options);
  }
  else if (inputMode == InputMode::COMMAND_LINE) {
    inputCommandLineParser(matmulConfig, isPipeline, options);
  }

  if (isLOWOHA && isPipeline) {
    testlog_error("Error: LOWOHA and pipeline mode are not compatible.");
    return NOT_OK;
  }

  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> matmul_results;
  if (!isLOWOHA) {
    // Run the matmul benchmark with the provided configurations
    int status = matmul_benchdnn(matmulConfig, matmul_results, options, cache_size);
    if (status != OK) {
      testlog_error("Matmul benchmark failed.");
      return NOT_OK;
    }
  }
  else {
    // Run the LOWOHA benchmark with the provided configurations
    int status = matmul_lowoha_benchdnn(matmulConfig, matmul_results, options,
                                        cache_size);
    if (status != OK) {
      testlog_error("LOWOHA Matmul benchmark failed.");
      return NOT_OK;
    }
  }

  if (isPipeline) {
    // Print results to console for each configuration
    print_pipeline_results(matmul_results, std::cout, options, inputMode);

    // Export results to CSV file
    std::ofstream outfile(out_filename);
    if (!outfile.is_open()) {
      testlog_error("Error: Cannot write to output file ", out_filename, "\n");
      return 1;
    }
    log_pipeline_results(matmul_results, outfile, options, inputMode);
    outfile.close();
  }
  else {
    // Print results to console for each configuration
    print_results(matmul_results, std::cout, options, isLOWOHA, inputMode);

    // Export results to CSV file
    std::ofstream outfile(out_filename);
    if (!outfile.is_open()) {
      testlog_error("Error: Cannot write to output file ", out_filename, "\n");
      return 1;
    }
    log_results(matmul_results, outfile, options, isLOWOHA, inputMode);
    outfile.close();
  }

  std::cout << "Timing results written to " << out_filename << std::endl;
  return OK;
}

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl