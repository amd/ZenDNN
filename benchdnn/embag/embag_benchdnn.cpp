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

#include "embag_benchdnn.hpp"

namespace zendnnl {
namespace benchdnn {
namespace embag {

int run_embag(tensor_t output_tensor, tensor_t table_tensor,
              tensor_t indices_tensor,
              tensor_t offsets_tensor, tensor_t weights_tensor, EmbagConfig cfg,
              TimingStats &stats, bool isNotWarmup) {

  status_t status;
#if MEASURE_INDIVIDUAL_TIMINGS
  if (!isNotWarmup) {
#endif
    //define embag context
    auto embag_context = embag_context_t()
                         .set_param("table", table_tensor)
                         .set_algo(cfg.algo)
                         .set_padding_index(cfg.padding_index)
                         .set_scatter_stride(cfg.scatter_stride);
    if (cfg.include_last_offset) {
      embag_context.set_include_last_offset(cfg.include_last_offset);
    }
    if (cfg.is_weights) {
      embag_context.set_is_weights(cfg.is_weights);
    }
    embag_context.create();

    if (! embag_context.check()) {
      testlog_error("embag context creation failed");
      return NOT_OK;
    }

    //define embag operator
    auto embag_operator = embag_operator_t()
                          .set_name("embedding_bag")
                          .set_context(embag_context)
                          .create();
    if (! embag_operator.check()) {
      testlog_error(" operator ", embag_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    if (cfg.is_weights) {
      // Execute operator
      status = embag_operator
               .set_input("indices", indices_tensor)
               .set_input("weights", weights_tensor)
               .set_input("offsets", offsets_tensor)
               .set_output("output", output_tensor)
               .execute();
    }
    else {
      status = embag_operator
               .set_input("indices", indices_tensor)
               .set_input("offsets", offsets_tensor)
               .set_output("output", output_tensor)
               .execute();
    }
    if (status == status_t::success) {
      testlog_info("<",embag_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",embag_operator.get_name(),">", " operator execution failed.");
      return NOT_OK;
    }
#if MEASURE_INDIVIDUAL_TIMINGS
  }
  else {
    auto start_context_creation = std::chrono::high_resolution_clock::now();
    //define embag context
    auto embag_context = embag_context_t()
                         .set_param("table", table_tensor)
                         .set_algo(cfg.algo)
                         .set_padding_index(cfg.padding_index)
                         .set_scatter_stride(cfg.scatter_stride);
    if (cfg.include_last_offset) {
      embag_context.set_include_last_offset(cfg.include_last_offset);
    }
    if (cfg.is_weights) {
      embag_context.set_is_weights(cfg.is_weights);
    }
    embag_context.create();
    auto end_context_creation = std::chrono::high_resolution_clock::now();

    if (! embag_context.check()) {
      testlog_error("embag context creation failed");
      return NOT_OK;
    }

    auto start_operator_creation = std::chrono::high_resolution_clock::now();
    //define embag operator
    auto embag_operator = embag_operator_t()
                          .set_name("embedding_bag")
                          .set_context(embag_context)
                          .create();
    auto end_operator_creation = std::chrono::high_resolution_clock::now();

    if (! embag_operator.check()) {
      testlog_error(" operator ", embag_operator.get_name(), " creation failed.");
      return NOT_OK;
    }

    auto start_operator_execution = std::chrono::high_resolution_clock::now();
    if (cfg.is_weights) {
      // Execute operator
      status = embag_operator
               .set_input("indices", indices_tensor)
               .set_input("weights", weights_tensor)
               .set_input("offsets", offsets_tensor)
               .set_output("output", output_tensor)
               .execute();
    }
    else {
      status = embag_operator
               .set_input("indices", indices_tensor)
               .set_input("offsets", offsets_tensor)
               .set_output("output", output_tensor)
               .execute();
    }
    auto end_operator_execution = std::chrono::high_resolution_clock::now();

    stats.context_creation_ms += (std::chrono::duration<double, std::milli>
                                  (end_context_creation - start_context_creation).count());
    stats.operator_creation_ms += (std::chrono::duration<double, std::milli>
                                   (end_operator_creation - start_operator_creation).count());
    stats.operator_execution_ms += (std::chrono::duration<double, std::milli>
                                    (end_operator_execution - start_operator_execution).count());

    if (status == status_t::success) {
      testlog_info("<",embag_operator.get_name(),">",
                   " operator execution successful.");
    }
    else {
      testlog_error("<",embag_operator.get_name(),">", " operator execution failed.");
      return NOT_OK;
    }
  }
#endif
  return OK;
}

int embag_benchdnn(std::vector<EmbagConfig> configs,
                   std::vector<std::pair<EmbagConfig, TimingStats>> &embag_results) {
  bool skip;

  for (const auto &cfg : configs) {
    try {
      skip = false;

      tensor_factory_t tensor_factory;
      tensor_t table_tensor, indices_tensor, offsets_tensor, weights_tensor,
               output_tensor;

      // Tensor creation
      int ret = create_table_tensor(tensor_factory, cfg, table_tensor);
      if (ret != OK) {
        testlog_error("create_table_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_indices_tensor(tensor_factory, cfg, indices_tensor);
      if (ret != OK) {
        testlog_error("create_indices_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_offsets_tensor(tensor_factory, cfg, offsets_tensor);
      if (ret != OK) {
        testlog_error("create_offsets_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_weights_tensor(tensor_factory, cfg, weights_tensor);
      if (ret != OK) {
        testlog_error("create_weights_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_output_tensor(tensor_factory, cfg, output_tensor);
      if (ret != OK) {
        testlog_error("create_output_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      TimingStats time_stats;

      // Warm-up iterations
      for (int i = 0; i < cfg.warmup_iters; ++i) {
        int ret = run_embag(output_tensor, table_tensor, indices_tensor, offsets_tensor,
                            weights_tensor, cfg, time_stats);
        if (ret != OK) {
          testlog_error("run_embag execution failed during warm-up.");
          skip = true;
          break;
        }
      }

      if (skip) {
        log_benchmark_failure(cfg);
        continue;
      }

      double elapsed_ms = 0.0;

      // Benchmark iterations
      for (int i = 0; i < cfg.iters; ++i) {
#if COLD_CACHE
        std::vector<char> buffer(CACHE_SIZE, 1);
        flush_cache(buffer);
#endif
#if !MEASURE_INDIVIDUAL_TIMINGS
        auto start = std::chrono::high_resolution_clock::now();
#endif
        TimingStats iter_stats;
        int ret = run_embag(output_tensor, table_tensor, indices_tensor, offsets_tensor,
                            weights_tensor, cfg, iter_stats, true);
        if (ret != OK) {
          testlog_error("run_embag execution failed.");
          skip = true;
          break;
        }

#if MEASURE_INDIVIDUAL_TIMINGS
        time_stats.context_creation_ms += iter_stats.context_creation_ms;
        time_stats.operator_creation_ms += iter_stats.operator_creation_ms;
        time_stats.operator_execution_ms += iter_stats.operator_execution_ms;
#else
        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms += std::chrono::duration<double, std::milli>(end - start).count();
#endif
      }

      if (skip) {
        log_benchmark_failure(cfg);
        continue;
      }

#if MEASURE_INDIVIDUAL_TIMINGS
      time_stats.total_time_ms = time_stats.context_creation_ms +
                                 time_stats.operator_creation_ms +
                                 time_stats.operator_execution_ms;
#else
      time_stats.total_time_ms = elapsed_ms;
#endif
      embag_results.emplace_back(cfg, time_stats);
    }
    catch (const exception_t &ex) {
      std::cout << ex.what() << std::endl;
      return NOT_OK;
    }
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
  std::vector<EmbagConfig> embagConfig;
  inputParser(infile, embagConfig);

  std::vector<std::pair<EmbagConfig, TimingStats>> embag_results;
  int status = embag_benchdnn(embagConfig, embag_results);
  if (status != OK) {
    testlog_error("Embag benchmark failed.");
    return NOT_OK;
  }

  // Print results to console for each configuration
  print_results(embag_results, std::cout);

  std::ofstream outfile(out_filename);
  if (!outfile.is_open()) {
    testlog_error("Error: Cannot write to output file ", out_filename, "\n");
    return 1;
  }
  // Export results to CSV file
  log_results(embag_results, outfile);
  outfile.close();

  std::cout << "Timing results written to " << out_filename << std::endl;
  return OK;
}

} // namespace embag
} // namespace benchdnn
} // namespace zendnnl