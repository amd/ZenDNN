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

#include "reorder_lowoha.hpp"

namespace zendnnl {
namespace benchdnn {
namespace reorder {

int reorder_lowoha_benchdnn(const std::vector<ReorderConfig> &configs,
                            std::vector<std::pair<ReorderConfig, TimingStats>> &reorder_results,
                            size_t cache_size) {

  bool skip;
  for (const auto &cfg : configs) {
    try {
      skip = false;

      tensor_factory_t tensor_factory;
      tensor_t src_tensor, dst_tensor, scale_tensor, zp_tensor;

      int ret = create_src_tensor(tensor_factory, cfg, src_tensor, true);
      if (ret != OK) {
        testlog_error("create_src_tensor failed");
        log_benchmark_failure(cfg, true);
        continue;
      }

      ret = create_dst_tensor(tensor_factory, cfg, dst_tensor);
      if (ret != OK) {
        testlog_error("create_dst_tensor failed");
        log_benchmark_failure(cfg, true);
        continue;
      }

      ret = create_scale_tensor(tensor_factory, cfg, scale_tensor);
      if (ret != OK) {
        testlog_error("create_scale_tensor failed");
        log_benchmark_failure(cfg, true);
        continue;
      }

      // Symmetric quantization (s8) has zero-point = 0 by definition,
      // so ZP tensor is only needed for asymmetric quantization (u8).
      bool needs_zp = (cfg.dst_dtype == zendnnl::common::data_type_t::u8);
      if (needs_zp) {
        ret = create_zp_tensor(tensor_factory, cfg, zp_tensor);
        if (ret != OK) {
          testlog_error("create_zp_tensor failed");
          log_benchmark_failure(cfg, true);
          continue;
        }
      }

      const void *src_data = src_tensor.get_raw_handle_const();
      void *dst_data = dst_tensor.get_raw_handle_unsafe();

      // Build reorder_params_t
      reorder_params_t params;
      params.src_dtype = cfg.src_dtype;
      params.dst_dtype = cfg.dst_dtype;
      params.algo = strToReorderAlgo(cfg.algo);
      params.num_threads = cfg.num_threads;
      params.dynamic_quant = cfg.dynamic_quant;

      if (cfg.batch_size > 1) {
        params.src_shape = {static_cast<int64_t>(cfg.batch_size),
                            static_cast<int64_t>(cfg.rows),
                            static_cast<int64_t>(cfg.cols)
                           };
      }
      else {
        params.src_shape = {static_cast<int64_t>(cfg.rows),
                            static_cast<int64_t>(cfg.cols)
                           };
      }
      params.dst_shape = params.src_shape;

      auto quant_dims = compute_quant_dims(cfg);
      params.quant_params.scale.buff = scale_tensor.get_raw_handle_unsafe();
      params.quant_params.scale.dt = data_type_t::f32;
      params.quant_params.scale.dims = quant_dims;

      if (needs_zp) {
        params.quant_params.zero_point.buff = zp_tensor.get_raw_handle_unsafe();
        params.quant_params.zero_point.dt = data_type_t::s32;
        params.quant_params.zero_point.dims = quant_dims;
      }

      TimingStats time_stats;

      // Warm-up iterations
      for (int i = 0; i < cfg.warmup_iters && !skip; ++i) {
        status_t status = reorder_direct(src_data, dst_data, params);
        if (status != status_t::success) {
          testlog_error("LOWOHA: Reorder execution failed during warm-up.");
          skip = true;
          break;
        }
      }

      if (skip) {
        log_benchmark_failure(cfg, true);
        continue;
      }

      double elapsed_ms = 0.0;

      // Benchmark iterations
      for (int i = 0; i < cfg.iters && !skip; ++i) {
#if COLD_CACHE
        flush_cache(cache_size);
#endif
        auto start = std::chrono::high_resolution_clock::now();

        status_t status = reorder_direct(src_data, dst_data, params);

        if (status != status_t::success) {
          testlog_error("LOWOHA: Reorder execution failed during benchmark.");
          skip = true;
          break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms += std::chrono::duration<double, std::milli>(end - start).count();
      }

      if (skip) {
        log_benchmark_failure(cfg, true);
        continue;
      }

      time_stats.total_time_ms = elapsed_ms;
      reorder_results.emplace_back(cfg, time_stats);

    }
    catch (const exception_t &ex) {
      std::cout << ex.what() << std::endl;
      return NOT_OK;
    }
  }

  return OK;
}

} // namespace reorder
} // namespace benchdnn
} // namespace zendnnl
