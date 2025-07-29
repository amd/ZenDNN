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

void log_benchmark_failure(const MatmulConfig &cfg) {
  std::string post_op = "";
  if (!cfg.post_ops.empty()) {
    for (auto j = 0; j < cfg.post_ops.size(); j++) {
      post_op += (j > 0 ? ":" : "") + postOpsToStr(cfg.post_ops[j]);
    }
  }
  commonlog_error("Benchmark failed for ", cfg.m, ", ", cfg.k, ", ",
                  cfg.n_values[0], ", ", datatypeToStr(cfg.dt[0]), ":",
                  datatypeToStr(cfg.dt[1]), ":", datatypeToStr(cfg.dt[2]), ", ",
                  cfg.isBiasEnabled, ", ", (cfg.isBiasEnabled ? datatypeToStr(cfg.bias_dt) :""),
                  ", ", post_op, ", ", cfg.kernel_name, ", ", cfg.warmup_iters);
}

int create_weights_tensor(tensor_factory_t &tensor_factory, MatmulConfig cfg,
                          std::vector<tensor_t> &weights) {

  zendnnl::common::data_type_t dt = cfg.dt[1];

  for (auto i = 0; i < cfg.n_values.size(); i++) {

    size_t k = (i == 0) ? cfg.k : cfg.n_values[i - 1];
    size_t n = cfg.n_values[i];
    tensor_t weights_tensor;

    if (cfg.kernel_name == "aocl_blis_blocked") {
      // Create input tensor with contigious layout.
      auto input_tensor = tensor_factory.uniform_dist_tensor({k, n},
                          dt,
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
        weights_tensor = tensor_factory.blocked_tensor({k, n},
                         dt,
                         buffer_params,
                         "weights_" + std::to_string(i));
      }
      else {
        // Compute the reorder size and create a buffer with reorderd size
        void *reorder_weights = aligned_alloc(64, reorder_size);

        // Create a Pair of storage params [reorder size and reorder weights] and
        // use it in tensor creation
        StorageParam buffer_params = std::make_pair(reorder_size, reorder_weights);

        // Blocked Tensor creation with seperate view for input tensor.
        weights_tensor = tensor_factory.blocked_tensor({k, n},
                         dt,
                         buffer_params,
                         "weights_" + std::to_string(i));
      }
    }
    else {
      weights_tensor = tensor_factory.uniform_dist_tensor({k, n},
                       dt,
                       1.0, "weights_" + std::to_string(i));
    }
    weights.push_back(weights_tensor);
  }
  return OK;
}

int create_bias_tensor(tensor_factory_t tensor_factory, const MatmulConfig &cfg,
                       std::vector<tensor_t> &bias) {
  if (cfg.isBiasEnabled) {
    zendnnl::common::data_type_t dt = cfg.bias_dt;
    for (auto i = 0; i < cfg.n_values.size(); i++) {
      tensor_t bias_tensor = tensor_factory.uniform_dist_tensor({cfg.n_values[i]},
                             dt,
                             -10.0, "bias_" + std::to_string(i));
      bias_tensor.set_name("bias_" + std::to_string(i));
      bias.push_back(bias_tensor);
    }
  }
  return OK;
}

int create_input_tensor(tensor_factory_t &tensor_factory,
                        const MatmulConfig &cfg, tensor_t &input) {
  input = tensor_factory.uniform_dist_tensor({cfg.m, cfg.k},
          cfg.dt[0],
          1.0, "matmul_input");
  input.set_name("matmul_input");
  return OK;
}

int create_output_tensor(tensor_factory_t &tensor_factory,
                         const MatmulConfig &cfg, std::vector<tensor_t> &output) {
  // Create output tensor with zero initialization.
  size_t m = cfg.m;
  zendnnl::common::data_type_t dt = cfg.dt[2];
  for (auto i = 0; i < cfg.n_values.size(); i++) {
    size_t n = cfg.n_values[i];
    tensor_t output_tensor = tensor_factory.zero_tensor({m, n},
                             dt, "matmul_output_" + std::to_string(i));
    output_tensor.set_name("matmul_output_" + std::to_string(i));
    output.push_back(output_tensor);
  }

  return OK;
}

int create_binary_post_ops_tensors(tensor_factory_t &tensor_factory,
                                   const MatmulConfig &cfg,
                                   std::vector<std::vector<tensor_t>> &binary_post_ops_tensors) {
  for (auto i = 0; i < cfg.n_values.size(); i++) {
    std::vector<tensor_t> binary_tensors;
    for (const auto &post_op : cfg.binary_post_ops_pos) {
      // Create a tensor for each binary post-op
      auto binary_tensor = tensor_factory.uniform_dist_tensor({cfg.m, cfg.n_values[i]},
                           cfg.dt[2],
                           2.0, "binary_post_op_" + std::to_string(post_op));
      binary_tensors.push_back(binary_tensor);
    }
    binary_post_ops_tensors.push_back(binary_tensors);
  }
  return OK;
}

int matmul_benchdnn(std::vector<MatmulConfig> configs,
                    std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>>
                    &matmul_results) {

  bool skip;
  for (const auto &cfg:configs) {
    try {
      skip = false;

      tensor_factory_t tensor_factory;
      tensor_t input_tensor;
      std::vector<tensor_t> weights, bias, output_tensor;

      int ret = create_weights_tensor(tensor_factory, cfg, weights);
      if (ret != OK) {
        testlog_error("create_bias_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_bias_tensor(tensor_factory, cfg, bias);
      if (ret != OK) {
        testlog_error("create_bias_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_input_tensor(tensor_factory, cfg, input_tensor);
      if (ret != OK) {
        testlog_error("create_input_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_output_tensor(tensor_factory, cfg, output_tensor);
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
      }
      if (skip) {
        log_benchmark_failure(cfg);
        continue;
      }

      std::vector<TimingStats> time_stats_layer(cfg.n_values.size());
      std::vector<double> elapsed_ms_layer(cfg.n_values.size(), 0.0);

      for (auto j = 0; j < cfg.iters && !skip; j++) {
#if COLD_CACHE
        std::vector<char> buffer(CACHE_SIZE, 1);
        flush_cache(buffer);
#endif
#if !MEASURE_INDIVIDUAL_TIMINGS
        auto start_total = std::chrono::high_resolution_clock::now();
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
#if !MEASURE_INDIVIDUAL_TIMINGS
        auto end_total = std::chrono::high_resolution_clock::now();
        double total_time_taken = (std::chrono::duration<double, std::milli>
                                   (end_total - start_total).count());
#endif
      }

#if !MEASURE_INDIVIDUAL_TIMINGS
      // Store total time for each layer
      for (size_t i = 0; i < cfg.n_values.size(); i++) {
        time_stats_layer[i].total_time_ms = elapsed_ms_layer[i];
      }
#endif
      matmul_results.emplace_back(cfg, time_stats_layer);
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

void write_each_config_result(const MatmulConfig &config,
                              const std::vector<TimingStats> &stat, std::ostream &outfile, int layer_num,
                              double percentage, bool isPipeline) {

  size_t m = config.m;
  size_t k = (layer_num == 0) ? config.k : config.n_values[layer_num - 1];
  size_t n = config.n_values[layer_num];
  double gops = (2 * m * k * n * 0.000000001);
  double gflops_val = (gops / (stat[layer_num].total_time_ms / config.iters)) *
                      1000;
  if (isPipeline) {
    outfile << "Layer " << layer_num << ", ";
  }
  outfile << m << ", " << k << ", " << n;
  outfile << ", " << config.iters << ", "
          << datatypeToStr(config.dt[0]) << ":"
          << datatypeToStr(config.dt[1]) << ":"
          << datatypeToStr(config.dt[2]) << ", "
          << config.isBiasEnabled << ", "
          << (config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "") << ", ";
  if (!config.post_ops.empty()) {
    outfile << postOpsToStr(config.post_ops[0]);
    for (size_t j = 1; j < config.post_ops.size(); ++j) {
      outfile << ":" << postOpsToStr(config.post_ops[j]);
    }
  }
  outfile << ", ";
  outfile << config.kernel_name << ", "
          << config.warmup_iters << ", " << stat[layer_num].total_time_ms
          << ", " << gflops_val;
  if (isPipeline) {
    outfile << ", " << percentage;
  }
#if MEASURE_INDIVIDUAL_TIMINGS
  double ctx_creation_percentage = (stat[layer_num].context_creation_ms /
                                    stat[layer_num].total_time_ms) * 100;
  double op_creation_percentage = (stat[layer_num].operator_creation_ms /
                                   stat[layer_num].total_time_ms) * 100;
  double op_execution_percentage = (stat[layer_num].operator_execution_ms /
                                    stat[layer_num].total_time_ms) * 100;
  outfile << ", "
          << stat[layer_num].context_creation_ms << " ("
          << ctx_creation_percentage << " %), "
          << stat[layer_num].operator_creation_ms << " ("
          << op_creation_percentage << " %), "
          << stat[layer_num].operator_execution_ms << " ("
          << op_execution_percentage << " %)";
#endif
  outfile << std::endl;
}

void cal_column_width(const MatmulConfig &config,
                      const std::vector<TimingStats> &stat, std::vector<size_t> &col_widths,
                      int st_index, int layer_num, double percentage,
                      bool isPipeline) {
  size_t m = config.m;
  size_t k = ((layer_num == 0) ? config.k : config.n_values[layer_num - 1]);
  size_t n = config.n_values[layer_num];
  double gops = (2 * m * k * n * 0.000000001);
  double gflops_val = (gops / (stat[layer_num].total_time_ms / config.iters)) *
                      1000;
  if (isPipeline) {
    std::string layer_str = "Layer_" + std::to_string(layer_num);
    col_widths[0] = std::max(col_widths[0], layer_str.size() + 2);
  }
  col_widths[st_index] = std::max(col_widths[st_index],
                                  std::to_string(m).size() + 2);
  col_widths[st_index + 1] = std::max(col_widths[st_index + 1],
                                      std::to_string(k).size() + 2);
  col_widths[st_index + 2] = std::max(col_widths[st_index + 2],
                                      std::to_string(n).size() + 2);
  col_widths[st_index + 3] = std::max(col_widths[st_index + 3],
                                      std::to_string(config.iters).size() + 2);
  std::string dt_str = datatypeToStr(config.dt[0]) + ":" +
                       datatypeToStr(config.dt[1]) + ":" + datatypeToStr(config.dt[2]);
  col_widths[st_index + 4] = std::max(col_widths[st_index + 4],
                                      dt_str.size() + 2);
  col_widths[st_index + 5] = std::max(col_widths[st_index + 5],
                                      std::to_string(config.isBiasEnabled).size() + 2);
  std::string bias_dt_str = config.isBiasEnabled ?
                            datatypeToStr(config.bias_dt) : "";
  col_widths[st_index + 6] = std::max(col_widths[st_index + 6],
                                      bias_dt_str.size() + 2);
  std::string postop_str;
  if (!config.post_ops.empty()) {
    postop_str += postOpsToStr(config.post_ops[0]);
    for (size_t j = 1; j < config.post_ops.size(); ++j) {
      postop_str += ":" + postOpsToStr(config.post_ops[j]);
    }
  }
  col_widths[st_index + 7] = std::max(col_widths[st_index + 7],
                                      postop_str.size() + 2);
  col_widths[st_index + 8] = std::max(col_widths[st_index + 8],
                                      config.kernel_name.size() + 2);
  col_widths[st_index + 9] = std::max(col_widths[st_index + 9],
                                      std::to_string(config.warmup_iters).size() + 2);
  col_widths[st_index + 10] = std::max(col_widths[st_index + 10],
                                       std::to_string((int)stat[0].total_time_ms).size() + 2);
  std::ostringstream gflops_ss;
  gflops_ss << std::fixed << std::setprecision(2) << gflops_val;
  col_widths[st_index + 11] = std::max(col_widths[st_index + 11],
                                       gflops_ss.str().size() + 2);
  if (isPipeline) {
    std::ostringstream perc_ss;
    perc_ss << std::fixed << std::setprecision(2) << percentage << " %";
    col_widths[st_index + 12] = std::max(col_widths[st_index + 12],
                                         perc_ss.str().size() + 2);
    st_index++;
  }
#if MEASURE_INDIVIDUAL_TIMINGS
  std::ostringstream ctx_str, op_create_str, op_exec_str;
  double ctx_creation_percentage = (stat[0].context_creation_ms /
                                    stat[0].total_time_ms) * 100;
  double op_creation_percentage = (stat[0].operator_creation_ms /
                                   stat[0].total_time_ms) * 100;
  double op_execution_percentage = (stat[0].operator_execution_ms /
                                    stat[0].total_time_ms) * 100;
  ctx_str << std::fixed << std::setprecision(2)
          << stat[0].context_creation_ms << " (" << ctx_creation_percentage << " %)";
  op_create_str << std::fixed << std::setprecision(2)
                << stat[0].operator_creation_ms << " (" << op_creation_percentage << " %)";
  op_exec_str << std::fixed << std::setprecision(2)
              << stat[0].operator_execution_ms << " (" << op_execution_percentage << " %)";
  col_widths[st_index + 12] = std::max(col_widths[st_index + 12],
                                       ctx_str.str().size() + 2);
  col_widths[st_index + 13] = std::max(col_widths[st_index + 13],
                                       op_create_str.str().size() + 2);
  col_widths[st_index + 14] = std::max(col_widths[st_index + 14],
                                       op_exec_str.str().size() + 2);
#endif
}

void fill_row(const MatmulConfig &config,
              const std::vector<TimingStats> &stat, std::vector<std::string> &row,
              int layer_num, double percentage,
              bool isPipeline) {
  size_t m = config.m;
  size_t k = ((layer_num == 0) ? config.k : config.n_values[layer_num - 1]);
  size_t n = config.n_values[layer_num];
  double gops = (2 * m * k * n * 0.000000001);
  double gflops_val = ((gops / (stat[layer_num].total_time_ms / config.iters)) *
                       1000);
  if (isPipeline) {
    row.push_back("Layer_" + std::to_string(layer_num));
  }
  row.push_back(std::to_string(m));
  row.push_back(std::to_string(k));
  row.push_back(std::to_string(n));
  row.push_back(std::to_string(config.iters));
  row.push_back(datatypeToStr(config.dt[0]) + ":" +
                datatypeToStr(config.dt[1]) + ":" + datatypeToStr(config.dt[2]));
  row.push_back(std::to_string(config.isBiasEnabled));
  row.push_back(config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "");
  std::string postop_str;
  if (!config.post_ops.empty()) {
    postop_str += postOpsToStr(config.post_ops[0]);
    for (size_t j = 1; j < config.post_ops.size(); ++j) {
      postop_str += ":" + postOpsToStr(config.post_ops[j]);
    }
  }
  row.push_back(postop_str);
  row.push_back(config.kernel_name);
  row.push_back(std::to_string(config.warmup_iters));
  std::ostringstream total_time_ss;
  total_time_ss << std::fixed << std::setprecision(2) <<
                stat[layer_num].total_time_ms;
  row.push_back(total_time_ss.str());
  std::ostringstream gflops_ss;
  gflops_ss << std::fixed << std::setprecision(2) << gflops_val;
  row.push_back(gflops_ss.str());
  if (isPipeline) {
    std::ostringstream perc_ss;
    perc_ss << std::fixed << std::setprecision(2) << percentage << " %";
    row.push_back(perc_ss.str());
  }
#if MEASURE_INDIVIDUAL_TIMINGS
  std::ostringstream ctx_str, op_create_str, op_exec_str;
  double ctx_creation_percentage = (stat[layer_num].context_creation_ms /
                                    stat[layer_num].total_time_ms) * 100;
  double op_creation_percentage = (stat[layer_num].operator_creation_ms /
                                   stat[layer_num].total_time_ms) * 100;
  double op_execution_percentage = (stat[layer_num].operator_execution_ms /
                                    stat[layer_num].total_time_ms) * 100;
  ctx_str << std::fixed << std::setprecision(2)
          << stat[layer_num].context_creation_ms << " ("
          << ctx_creation_percentage << " %)";
  op_create_str << std::fixed << std::setprecision(2)
                << stat[layer_num].operator_creation_ms << " ("
                << op_creation_percentage << " %)";
  op_exec_str << std::fixed << std::setprecision(2)
              << stat[layer_num].operator_execution_ms << " ("
              << op_execution_percentage << " %)";
  row.push_back(ctx_str.str());
  row.push_back(op_create_str.str());
  row.push_back(op_exec_str.str());
#endif
}

void log_pipeline_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile) {

  outfile << std::fixed << std::setprecision(2);
  outfile <<
          "Layer Number, M, K, N, Iterations, Data type, Bias Enabled, Bias Data type, Post Operation, Kernel name, Warmup iterations, Total time (ms) (all iters), GFLOPS, % of Total";
#if MEASURE_INDIVIDUAL_TIMINGS
  outfile <<
          ", Context Creation (ms & %), Operator Creation (ms & %), Operator Execution (ms & %)";
#endif
  outfile << std::endl;

  // Write results to CSV for each configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    double total_time = 0.0;
    for (auto i = 0; i < config.n_values.size(); i++) {
      total_time += stat[i].total_time_ms;
    }
    outfile << "Summary, " <<
            config.m << ", " <<
            config.k << ", ";
    // Output N values separated by ':'
    if (!config.n_values.empty()) {
      outfile << config.n_values[0];
      for (size_t i = 1; i < config.n_values.size(); ++i) {
        outfile << ":" << config.n_values[i];
      }
    }
    outfile << ", " <<
            config.iters << ", " <<
            datatypeToStr(config.dt[0]) << ":" <<
            datatypeToStr(config.dt[1]) << ":" <<
            datatypeToStr(config.dt[2]) << ", " <<
            config.isBiasEnabled << ", " <<
            (config.isBiasEnabled ? datatypeToStr(config.bias_dt) : "") << ", ";
    if (!config.post_ops.empty()) {
      outfile << postOpsToStr(config.post_ops[0]);
      for (size_t j = 1; j < config.post_ops.size(); ++j) {
        outfile << ":" << postOpsToStr(config.post_ops[j]);
      }
    }
    outfile << ", ";
    outfile << config.kernel_name << ", " <<
            config.warmup_iters << ", " << total_time;
    outfile << std::endl;

    for (auto i = 0; i < stat.size(); i++) {
      double percentage = (stat[i].total_time_ms / total_time) * 100;
      write_each_config_result(config, stat, outfile, i, percentage, true);
    }
  }
}

void print_pipeline_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile) {

  // Dynamic column widths calculation
  std::vector<std::string> headers = {
    "Layer", "M", "K", "N", "Iters", "Data_type", "Bias_Enabled", "Bias_dt", "PostOp", "Kernel_Name",
    "Warmup_iters", "Total_time(ms, all iters)", "GFLOPS", "%_of_Total"
  };
#if MEASURE_INDIVIDUAL_TIMINGS
  headers.push_back("Ctx_Creation(ms_%)");
  headers.push_back("Op_Creation(ms_%)");
  headers.push_back("Op_Execution(ms_%)");
#endif
  std::vector<size_t> col_widths(headers.size());
  // Initialize with header lengths
  for (size_t i = 0; i < headers.size(); ++i) {
    col_widths[i] = headers[i].size() + 2;
  }
  // Compute max width for each column based on all data rows
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    double total_time = 0.0;
    for (auto i = 0; i < config.n_values.size(); i++) {
      total_time += stat[i].total_time_ms;
    }
    // Update column widths for summary and per-layer rows
    col_widths[0] = std::max(col_widths[0], std::string("Summary").size() + 2);
    col_widths[1] = std::max(col_widths[1], std::to_string(config.m).size() + 2);
    col_widths[2] = std::max(col_widths[2], std::to_string(config.k).size() + 2);
    // N field (colon separated)
    std::string n_str;
    if (!config.n_values.empty()) {
      n_str += std::to_string(config.n_values[0]);
      for (size_t i = 1; i < config.n_values.size(); ++i) {
        n_str += ":" + std::to_string(config.n_values[i]);
      }
    }
    col_widths[3] = std::max(col_widths[3], n_str.size() + 2);
    col_widths[4] = std::max(col_widths[4],
                             std::to_string(config.iters).size() + 2);
    std::string dt_str = datatypeToStr(config.dt[0]) + ":" +
                         datatypeToStr(config.dt[1]) + ":" +
                         datatypeToStr(config.dt[2]);
    col_widths[5] = std::max(col_widths[5], dt_str.size() + 2);
    col_widths[6] = std::max(col_widths[6],
                             std::to_string(config.isBiasEnabled).size() + 2);
    std::string bias_dt_str = config.isBiasEnabled ?
                              datatypeToStr(config.bias_dt) : "";
    col_widths[7] = std::max(col_widths[7], bias_dt_str.size() + 2);
    std::string postop_str;
    if (!config.post_ops.empty()) {
      postop_str += postOpsToStr(config.post_ops[0]);
      for (size_t j = 1; j < config.post_ops.size(); ++j) {
        postop_str += ":" + postOpsToStr(config.post_ops[j]);
      }
    }
    col_widths[8] = std::max(col_widths[8], postop_str.size() + 2);
    col_widths[9] = std::max(col_widths[9], config.kernel_name.size() + 2);
    col_widths[10] = std::max(col_widths[10],
                              std::to_string(config.warmup_iters).size() + 2);
    col_widths[11] = std::max(col_widths[11],
                              std::to_string((int)total_time).size() + 2);
    col_widths[12] = std::max(col_widths[12],
                              std::string("GFLOPS").size() + 2);
    col_widths[13] = std::max(col_widths[13],
                              std::string("%_of_Total").size() + 2);
#if MEASURE_INDIVIDUAL_TIMINGS
    col_widths[14] = std::max(col_widths[14],
                              std::string("Ctx_Creation(ms_%)").size() + 2);
    col_widths[15] = std::max(col_widths[15],
                              std::string("Op_Creation(ms_%)").size() + 2);
    col_widths[16] = std::max(col_widths[16],
                              std::string("Op_Execution(ms_%)").size() + 2);
#endif
    // Per-layer rows
    for (auto i = 0; i < stat.size(); i++) {
      double percentage = (stat[i].total_time_ms / total_time) * 100;
      int st_index = 1;
      cal_column_width(config, stat, col_widths, st_index, i, percentage, true);
    }
  }

  // Helper lambda to print a row
  auto print_row = [&](const std::vector<std::string> &row) {
    for (size_t i = 0; i < row.size(); ++i) {
      outfile << std::setw(col_widths[i]) << row[i];
    }
    outfile << std::endl;
  };

  // Print table header
  outfile << std::fixed << std::setprecision(2);
  outfile << std::left;
  print_row(headers);

  // Print summary and per-layer rows for each configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    double total_time = 0.0;
    for (auto i = 0; i < config.n_values.size(); i++) {
      total_time += stat[i].total_time_ms;
    }
    // Summary row (aggregated for the pipeline)
    std::vector<std::string> summary_row;
    summary_row.push_back("Summary");
    summary_row.push_back(std::to_string(config.m));
    summary_row.push_back(std::to_string(config.k));
    // N values as colon separated string
    std::string n_str;
    if (!config.n_values.empty()) {
      n_str += std::to_string(config.n_values[0]);
      for (size_t i = 1; i < config.n_values.size(); ++i) {
        n_str += ":" + std::to_string(config.n_values[i]);
      }
    }
    summary_row.push_back(n_str);
    summary_row.push_back(std::to_string(config.iters));
    summary_row.push_back(datatypeToStr(config.dt[0]) + ":" +
                          datatypeToStr(config.dt[1]) + ":" + datatypeToStr(config.dt[2]));
    summary_row.push_back(std::to_string(config.isBiasEnabled));
    summary_row.push_back(config.isBiasEnabled ?
                          datatypeToStr(config.bias_dt) : "");
    std::string postop_str;
    if (!config.post_ops.empty()) {
      postop_str += postOpsToStr(config.post_ops[0]);
      for (size_t j = 1; j < config.post_ops.size(); ++j) {
        postop_str += ":" + postOpsToStr(config.post_ops[j]);
      }
    }
    summary_row.push_back(postop_str);
    summary_row.push_back(config.kernel_name);
    summary_row.push_back(std::to_string(config.warmup_iters));
    std::ostringstream total_time_oss;
    total_time_oss << std::fixed << std::setprecision(2) << total_time;
    summary_row.push_back(total_time_oss.str());
    summary_row.push_back("");
    summary_row.push_back("");
#if MEASURE_INDIVIDUAL_TIMINGS
    summary_row.push_back("");
    summary_row.push_back("");
    summary_row.push_back("");
#endif
    print_row(summary_row);

    // Per-layer rows (detailed timing for each layer in the pipeline)
    for (auto i = 0; i < stat.size(); i++) {
      double percentage = (stat[i].total_time_ms / total_time) * 100;
      std::vector<std::string> layer_row;
      fill_row(config, stat, layer_row, i, percentage, true);
      print_row(layer_row);
    }
  }
}

void log_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile) {

  outfile << std::fixed << std::setprecision(2);
  outfile <<
          "M, K, N, Iterations, Data type, Bias Enabled, Bias Data type, Post Operation, Kernel name, Warmup iterations, Total time (ms) (all iters), GFLOPS";
#if MEASURE_INDIVIDUAL_TIMINGS
  outfile <<
          ", Context Creation (ms & %), Operator Creation (ms & %), Operator Execution (ms & %)";
#endif
  outfile << std::endl;

  // Write results to CSV for each configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    write_each_config_result(config, stat, outfile);
  }
}

void print_results(
  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> &matmul_results,
  std::ostream &outfile) {

  // Dynamic column widths calculation
  std::vector<std::string> headers = {
    "M", "K", "N", "Iters", "Data_type", "Bias_Enabled", "Bias_dt", "PostOp", "Kernel_Name",
    "Warmup_iters", "Total_time(ms, all iters)", "GFLOPS"
  };
#if MEASURE_INDIVIDUAL_TIMINGS
  headers.push_back("Ctx_Creation(ms_%)");
  headers.push_back("Op_Creation(ms_%)");
  headers.push_back("Op_Execution(ms_%)");
#endif
  std::vector<size_t> col_widths(headers.size());
  // Initialize with header lengths
  for (size_t i = 0; i < headers.size(); ++i) {
    col_widths[i] = headers[i].size() + 2;
  }
  // Compute max width for each column based on all data rows
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    int st_index = 0;
    cal_column_width(config, stat, col_widths, st_index);
  }

  // Helper lambda to print a row for the table
  auto print_row = [&](const std::vector<std::string> &row) {
    for (size_t i = 0; i < row.size(); ++i) {
      outfile << std::setw(col_widths[i]) << row[i];
    }
    outfile << std::endl;
  };

  // Print table header
  outfile << std::fixed << std::setprecision(2);
  outfile << std::left;
  print_row(headers);

  // Print each result row for every configuration
  for (const auto &result : matmul_results) {
    const MatmulConfig &config = result.first;
    const std::vector<TimingStats> &stat = result.second;
    std::vector<std::string> row;
    fill_row(config, stat, row);
    print_row(row);
  }
}

int bench(const std::string &in_filename, const std::string &out_filename) {
  // Open the input file for reading benchmark configurations
  std::ifstream infile(in_filename);
  if (!infile.is_open()) {
    testlog_error("Error: Cannot open file ", in_filename);
    return NOT_OK;
  }
  std::vector<MatmulConfig> matmulConfig;
  bool isPipeline = false;
  inputParser(infile, matmulConfig, isPipeline);

  std::vector<std::pair<MatmulConfig, std::vector<TimingStats>>> matmul_results;
  // Run the matmul benchmark with the provided configurations
  int status = matmul_benchdnn(matmulConfig, matmul_results);
  if (status != OK) {
    testlog_error("Matmul benchmark failed.");
    return NOT_OK;
  }

  if (isPipeline) {
    // Print results to console for each configuration
    print_pipeline_results(matmul_results, std::cout);

    // Export results to CSV file
    std::ofstream outfile(out_filename);
    if (!outfile.is_open()) {
      testlog_error("Error: Cannot write to output file ", out_filename, "\n");
      return 1;
    }
    log_pipeline_results(matmul_results, outfile);
    outfile.close();
  }
  else {
    // Print results to console for each configuration
    print_results(matmul_results, std::cout);

    // Export results to CSV file
    std::ofstream outfile(out_filename);
    if (!outfile.is_open()) {
      testlog_error("Error: Cannot write to output file ", out_filename, "\n");
      return 1;
    }
    log_results(matmul_results, outfile);
    outfile.close();
  }

  std::cout << "Timing results written to " << out_filename << std::endl;
  return OK;
}

} // namespace matmul
} // namespace benchdnn
} // namespace zendnnl