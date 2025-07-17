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
#if MEASURE_INDIVIDUAL_TIMINGS
  if (!isNotWarmup) {
#endif
    //define matmul context
    auto matmul_context = matmul_context_t()
                          .set_param("weights", weights);
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
                           .set_forced_kernel(cfg.kernel_name)
                           .create();

    if (! matmul_operator.check()) {
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
    status = matmul_operator.execute();
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
                          .set_param("weights", weights);
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
                           .set_forced_kernel(cfg.kernel_name)
                           .create();
    auto end_operator_creation = std::chrono::high_resolution_clock::now();

    if (! matmul_operator.check()) {
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
                    std::vector<std::pair<MatmulConfig, TimingStats>> &matmul_results) {

  bool skip;
  for (const auto &cfg:configs) {
    try {
      skip = false;

      tensor_factory_t tensor_factory;
      tensor_t weights, bias;
      if (cfg.kernel_name == "aocl_blis_blocked") {
        // Create input tensor with contigious layout.
        auto input_tensor = tensor_factory.uniform_dist_tensor({cfg.k, cfg.n},
                            cfg.dt[1],
                            1.0, "reorder_input");

        // Reorder context creation with backend aocl.
        auto reorder_context = reorder_context_t()
                               .set_algo_format("aocl")
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

        // Compute the reorder size
        size_t reorder_size         = reorder_operator.get_reorder_size();
        // Extract the input buffer size
        size_t input_buffer_size    = input_tensor.get_buffer_sz_bytes();

        // Inplace reorder takes place when reorder buffer size is same as input buffer size
        if (reorder_size == input_buffer_size) {
          // Assign input_tensor to buffer_params as a tensor_t variant
          StorageParam buffer_params = input_tensor;

          // Blocked Tensor creation with seperate view for input tensor.
          weights = tensor_factory.blocked_tensor({cfg.k, cfg.n},
                                                  cfg.dt[1],
                                                  buffer_params,
                                                  "weights");
        }
        else {
          // Compute the reorder size and create a buffer with reorderd size
          void *reorder_weights = aligned_alloc(64, reorder_size);

          // Create a Pair of storage params [reorder size and reorder weights] and
          // use it in tensor creation
          StorageParam buffer_params = std::make_pair(reorder_size, reorder_weights);

          // Blocked Tensor creation with seperate view for input tensor.
          weights = tensor_factory.blocked_tensor({cfg.k, cfg.n},
                                                  cfg.dt[1],
                                                  buffer_params,
                                                  "weights");
        }

      }
      else {
        weights = tensor_factory.uniform_dist_tensor({cfg.k, cfg.n},
                  cfg.dt[1],
                  1.0, "weights");
      }

      if (cfg.isBiasEnabled) {
        bias    = tensor_factory.uniform_dist_tensor({cfg.n},
                  cfg.bias_dt,
                  -10.0, "bias");
        bias.set_name("bias");
      }

      auto input_tensor = tensor_factory.uniform_dist_tensor({cfg.m, cfg.k},
                          cfg.dt[0],
                          1.0, "matmul_input");
      input_tensor.set_name("matmul_input");

      auto output_tensor = tensor_factory.zero_tensor({cfg.m, cfg.n},
                           cfg.dt[2], "matmul_output");
      output_tensor.set_name("matmul_output");

      std::vector<tensor_t> binary_post_ops_tensors;
      for (const auto &post_op : cfg.binary_post_ops_pos) {
        // Create a tensor for each binary post-op
        auto binary_tensor = tensor_factory.uniform_dist_tensor({cfg.m, cfg.n},
                             cfg.dt[2],
                             2.0, "binary_post_op_" + std::to_string(post_op));
        binary_post_ops_tensors.push_back(binary_tensor);
      }

      TimingStats time_stats;
      // warm-up iterations
      for (auto i = 0; i < cfg.warmup_iters; i++) {
        int ret = run_matmul(output_tensor, input_tensor, weights, bias,
                             cfg, binary_post_ops_tensors, time_stats);
        if (ret == OK) {
          testlog_info("run_matmul execution successful.");
        }
        else {
          testlog_error("run_matmul execution failed.");
          skip = true;
          break;
        }
      }
      if (skip) {
        std::string post_op = "";
        if (!cfg.post_ops.empty()) {
          for (auto j = 0; j < cfg.post_ops.size(); j++) {
            post_op += (j > 0 ? ":" : "") + postOpsToStr(cfg.post_ops[j]);
          }
        }
        commonlog_error("Benchmark failed for ", cfg.m, ", ", cfg.k, ", ",
                        cfg.n, ", ", datatypeToStr(cfg.dt[0]), ":",
                        datatypeToStr(cfg.dt[1]), ":", datatypeToStr(cfg.dt[2]), ", ",
                        cfg.isBiasEnabled, ", ", (cfg.isBiasEnabled ? datatypeToStr(cfg.bias_dt) :""),
                        ", ",
                        post_op, ", ", cfg.kernel_name, ", ", cfg.warmup_iters);
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
        int ret = run_matmul(output_tensor, input_tensor, weights, bias,
                             cfg, binary_post_ops_tensors, time_stats, true);
        if (ret == OK) {
          testlog_info("run_matmul execution successful.");
        }
        else {
          testlog_error("run_matmul execution failed.");
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
        std::string post_op = "";
        if (!cfg.post_ops.empty()) {
          for (auto j = 0; j < cfg.post_ops.size(); j++) {
            post_op += (j > 0 ? ":" : "") + postOpsToStr(cfg.post_ops[j]);
          }
        }
        commonlog_error("Benchmark failed for ", cfg.m, ", ", cfg.k, ", ",
                        cfg.n, ", ", datatypeToStr(cfg.dt[0]), ":",
                        datatypeToStr(cfg.dt[1]), ":", datatypeToStr(cfg.dt[2]), ", ",
                        cfg.isBiasEnabled, ", ", (cfg.isBiasEnabled ? datatypeToStr(cfg.bias_dt) :""),
                        ", ",
                        post_op, ", ", cfg.kernel_name, ", ", cfg.warmup_iters);
        continue;
      }
#if MEASURE_INDIVIDUAL_TIMINGS
      elapsed_ms = time_stats.context_creation_ms +
                   time_stats.operator_creation_ms + time_stats.operator_execution_ms;
#endif
      time_stats.total_time_ms = elapsed_ms;
      matmul_results.emplace_back(cfg, time_stats);
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
  std::vector<MatmulConfig> matmulConfig;
  inputParser(infile, matmulConfig);

  std::vector<std::pair<MatmulConfig, TimingStats>> matmul_results;
  // Run the matmul benchmark with the provided configurations
  int status = matmul_benchdnn(matmulConfig, matmul_results);
  if (status != OK) {
    testlog_error("Matmul benchmark failed.");
    return NOT_OK;
  }

  // Print results to console for each configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const TimingStats &stat = result.second;
    double gops = (2 * config.m * config.k * config.n * 0.000000001);
    double gflops_val = (gops / stat.total_time_ms) * 1000;
    std::cout <<
              config.m << ", " <<
              config.k << ", " <<
              config.n << ", " <<
              config.iters << ", " <<
              datatypeToStr(config.dt[0]) << ":" <<
              datatypeToStr(config.dt[1]) << ":" <<
              datatypeToStr(config.dt[2]) << ", " <<
              config.isBiasEnabled << ", " << (config.isBiasEnabled ? datatypeToStr(
                    config.bias_dt) :"") << ", ";
    if (!config.post_ops.empty()) {
      for (auto j = 0; j < config.post_ops.size(); j++) {
        if (j > 0) {
          std::cout << ":";
        }
        std::cout << postOpsToStr(config.post_ops[j]);
      }
    }
    std::cout << ", ";
    std::cout << config.kernel_name << ", " <<
              config.warmup_iters << ", Total time: " <<
              stat.total_time_ms << " ms, GFLOPS: " << gflops_val
#if MEASURE_INDIVIDUAL_TIMINGS
              << ", Context creation: " <<
              stat.context_creation_ms << " ms, Operator creation: " <<
              stat.operator_creation_ms << " ms, Operator execution: " <<
              stat.operator_execution_ms << " ms"
#endif
              << std::endl;

  }

  std::ofstream outfile(out_filename);
  if (!outfile.is_open()) {
    testlog_error("Error: Cannot write to output file ", out_filename, "\n");
    return 1;
  }

  // Write CSV header
  outfile <<
          "M, K, N, Iterations, Data type, Bias Enabled, Bias Data type, Post Operation, Kernel name, Warmup iterations, Total time (ms), GFLOPS";
#if MEASURE_INDIVIDUAL_TIMINGS
  outfile <<
          ", Context Creation Time (ms), Operator Creation Time (ms), Operator Execution Time (ms)";
#endif
  outfile << std::endl;

  // Write results to CSV for each configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const TimingStats &stat = result.second;
    double gops = (2 * config.m * config.k * config.n * 0.000000001);
    double gflops_val = (gops / stat.total_time_ms) * 1000;
    outfile <<
            config.m << ", " <<
            config.k << ", " <<
            config.n << ", " <<
            config.iters << ", " <<
            datatypeToStr(config.dt[0]) << ":" <<
            datatypeToStr(config.dt[1]) << ":" <<
            datatypeToStr(config.dt[2]) << ", " <<
            config.isBiasEnabled << ", " << (config.isBiasEnabled ? datatypeToStr(
                  config.bias_dt) : "") << ", ";
    if (!config.post_ops.empty()) {
      for (const auto &post_op : config.post_ops) {
        outfile << postOpsToStr(post_op) << ":";
      }
      outfile.seekp(-1, std::ios_base::end); // Remove trailing colon
    }
    outfile << ", ";
    outfile << config.kernel_name << ", " <<
            config.warmup_iters << ", " <<
            stat.total_time_ms << ", " << gflops_val
#if MEASURE_INDIVIDUAL_TIMINGS
            << ", " <<
            stat.context_creation_ms << ", " <<
            stat.operator_creation_ms << ", " <<
            stat.operator_execution_ms
#endif
            << std::endl;

  }

  outfile.close();
  std::cout << "Timing results written to " << out_filename << std::endl;
  return OK;
}

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl