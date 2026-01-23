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

#include "embag_lowoha.hpp"

namespace zendnnl {
namespace benchdnn {
namespace embag {

int embag_lowoha_benchdnn(std::vector<EmbagConfig> configs,
                          std::vector<std::pair<EmbagConfig, TimingStats>> &embag_results,
                          size_t cache_size) {

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

      // Get raw pointers for LOWOHA direct API
      const void *table_data = table_tensor.get_raw_handle_const();
      const void *indices_data = indices_tensor.get_raw_handle_const();
      const void *offsets_data = offsets_tensor.get_raw_handle_const();
      const float *weights_data = cfg.is_weights ?
                                  static_cast<const float *>(weights_tensor.get_raw_handle_const()) :
                                  nullptr;
      void *output_data = output_tensor.get_raw_handle_unsafe();

      // Setup LOWOHA embag parameters
      embag_params_t params;
      params.dtypes.table = cfg.dt[0];
      params.dtypes.output = cfg.dt[1];
      params.dtypes.indices = indices_tensor.get_data_type();
      params.dtypes.offsets = offsets_tensor.get_data_type();
      params.algo = cfg.algo;
      params.num_embeddings = cfg.num_embeddings;
      params.embedding_dim = cfg.embedding_dims;
      params.num_indices = cfg.num_indices;
      params.num_bags = cfg.num_bags;
      params.is_weights = cfg.is_weights;
      params.include_last_offset = cfg.include_last_offset;
      params.padding_idx = cfg.padding_index;
      params.fp16_scale_bias = cfg.fp16_scale_bias;
      params.num_threads = 0;  // Use default (omp_get_max_threads)
      if (cfg.scatter_stride != -1) {
        params.dst_stride = cfg.scatter_stride;
      }
      else {
        params.dst_stride = output_tensor.get_stride()[0];  // Set output stride
      }

      // Validate parameters
      if (params.num_embeddings == 0 || params.embedding_dim == 0) {
        testlog_error("LOWOHA: Invalid tensor dimensions - num_embeddings:",
                      params.num_embeddings, " embedding_dim:", params.embedding_dim);
        log_benchmark_failure(cfg);
        continue;
      }

      // Validate data types
      if (cfg.dt[0] != data_type_t::f32 && cfg.dt[0] != data_type_t::bf16 &&
          cfg.dt[0] != data_type_t::s8 && cfg.dt[0] != data_type_t::s4 &&
          cfg.dt[0] != data_type_t::u4) {
        testlog_error("LOWOHA: Unsupported table data type");
        log_benchmark_failure(cfg);
        continue;
      }
      if (cfg.dt[1] != data_type_t::f32 && cfg.dt[1] != data_type_t::bf16) {
        testlog_error("LOWOHA: Unsupported output data type");
        log_benchmark_failure(cfg);
        continue;
      }

      TimingStats time_stats;

      // Warm-up iterations
      for (int i = 0; i < cfg.warmup_iters && !skip; ++i) {
        status_t status = embedding_bag_direct(table_data, indices_data,
                                               offsets_data, weights_data,
                                               output_data, params);
        if (status != status_t::success) {
          testlog_error("LOWOHA: Embag execution failed during warm-up.");
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
      for (int i = 0; i < cfg.iters && !skip; ++i) {
#if COLD_CACHE
        flush_cache(cache_size);
#endif
        auto start = std::chrono::high_resolution_clock::now();

        status_t status = embedding_bag_direct(table_data, indices_data,
                                               offsets_data, weights_data,
                                               output_data, params);

        if (status != status_t::success) {
          testlog_error("LOWOHA: Embag execution failed during benchmark iterations.");
          skip = true;
          break;
        }

        auto end = std::chrono::high_resolution_clock::now();

        elapsed_ms += std::chrono::duration<double, std::milli>(end - start).count();
      }

      if (skip) {
        log_benchmark_failure(cfg);
        continue;
      }

      time_stats.total_time_ms = elapsed_ms;
      embag_results.emplace_back(cfg, time_stats);

    }
    catch (const exception_t &ex) {
      std::cout << ex.what() << std::endl;
      return NOT_OK;
    }
  }

  return OK;
}

} // namespace embag
} // namespace benchdnn
} // namespace zendnnl
