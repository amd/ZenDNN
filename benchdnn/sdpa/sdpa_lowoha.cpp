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

#include "sdpa_lowoha.hpp"

#include <chrono>

namespace zendnnl {
namespace benchdnn {
namespace sdpa {

namespace {

// Per-axis strides for a logical [B, H, S, D] tensor in the requested
// physical layout. The kernel always interprets axes as [B, H, S, D] and
// computes offsets via these strides, so swapping H<->S in memory just means
// swapping `h` and `s` here (and rescaling `b` accordingly).
//
//   bhsd memory order [B, H, S, D] -> strides {H*S*D, S*D, D,   1}
//   bshd memory order [B, S, H, D] -> strides {S*H*D, D,   H*D, 1}
//
// `stride_d == 1` in both cases, satisfying the flash kernel's
// contiguous-head_dim requirement.
struct qkv_strides {
  int64_t b, h, s, d;
};

qkv_strides derive_qkv_strides(qkv_layout_t layout,
                               int64_t H, int64_t S, int64_t D) {
  qkv_strides r;
  r.d = 1;
  if (layout == qkv_layout_t::bshd) {
    r.h = D;
    r.s = H * D;
    r.b = S * H * D;
  }
  else {
    r.s = D;
    r.h = S * D;
    r.b = H * S * D;
  }
  return r;
}

void populate_sdpa_params(sdpa_params &p, const SdpaConfig &cfg,
                          int64_t S_kv_eff) {
  p.batch      = cfg.batch;
  p.num_heads  = cfg.num_heads;
  p.seq_len    = cfg.seq_len;
  // Pass the explicit kv_seq_len value as set in the config (0 means "same as
  // seq_len" per the operator contract).
  p.kv_seq_len = cfg.kv_seq_len;
  p.head_dim   = cfg.head_dim;

  // Q is logically [B, H, S_q, D]; stride permutation depends on cfg.qkv_layout.
  auto qs = derive_qkv_strides(cfg.qkv_layout, cfg.num_heads, cfg.seq_len,
                               cfg.head_dim);
  p.q_stride_b = qs.b;
  p.q_stride_h = qs.h;
  p.q_stride_s = qs.s;
  p.q_stride_d = qs.d;

  // K and V are logically [B, H, S_kv_eff, D]; share the same layout as Q.
  auto ks = derive_qkv_strides(cfg.qkv_layout, cfg.num_heads, S_kv_eff,
                               cfg.head_dim);
  p.k_stride_b = ks.b;
  p.k_stride_h = ks.h;
  p.k_stride_s = ks.s;
  p.k_stride_d = ks.d;

  p.v_stride_b = ks.b;
  p.v_stride_h = ks.h;
  p.v_stride_s = ks.s;
  p.v_stride_d = ks.d;

  // Output matches Q's shape AND layout.
  p.o_stride_b = qs.b;
  p.o_stride_h = qs.h;
  p.o_stride_s = qs.s;
  p.o_stride_d = qs.d;

  p.mask_ndims = cfg.mask_ndims;
  if (cfg.mask_ndims == 2) {
    p.mask_sizes[0]   = cfg.seq_len;
    p.mask_sizes[1]   = S_kv_eff;
    p.mask_sizes[2]   = 0;
    p.mask_sizes[3]   = 0;
    p.mask_strides[0] = S_kv_eff;
    p.mask_strides[1] = 1;
    p.mask_strides[2] = 0;
    p.mask_strides[3] = 0;
  }
  else if (cfg.mask_ndims == 4) {
    p.mask_sizes[0]   = cfg.batch;
    p.mask_sizes[1]   = cfg.num_heads;
    p.mask_sizes[2]   = cfg.seq_len;
    p.mask_sizes[3]   = S_kv_eff;
    p.mask_strides[0] = cfg.num_heads * cfg.seq_len * S_kv_eff;
    p.mask_strides[1] = cfg.seq_len * S_kv_eff;
    p.mask_strides[2] = S_kv_eff;
    p.mask_strides[3] = 1;
  }
  else {
    for (int i = 0; i < 4; ++i) {
      p.mask_sizes[i] = 0;
      p.mask_strides[i] = 0;
    }
  }

  // BMM-backend fields are unused by the flash backend but populate them for
  // consistency.
  p.mask_type = cfg.is_causal ? mask_type_t::causal :
                (cfg.mask_ndims > 0 ? mask_type_t::custom : mask_type_t::none);
  p.mask_dims[0] = 0;
  p.mask_dims[1] = 0;
  p.mask_dims[2] = 0;

  p.qkv_dt   = cfg.qkv_dt;
  p.out_dt   = (cfg.out_dt == data_type_t::none) ? cfg.qkv_dt : cfg.out_dt;
  p.mask_dt  = cfg.mask_dt;
  p.scale    = cfg.scale;
  p.is_causal = cfg.is_causal;
  p.dropout_p = cfg.dropout_p;
  p.num_threads = cfg.num_threads;
}

} // anonymous namespace

int sdpa_lowoha_benchdnn(
  std::vector<SdpaConfig> configs,
  std::vector<std::pair<SdpaConfig, TimingStats>> &sdpa_results,
  const global_options &options,
  size_t cache_size) {

  bool skip;
  for (const auto &cfg : configs) {
    try {
      skip = false;

      std::string reason;
      if (!isSupportedSdpaConfig(cfg, reason)) {
        commonlog_error("Skipping unsupported SDPA config: ", reason);
        log_benchmark_failure(cfg);
        continue;
      }

      const int64_t S_kv_eff = (cfg.kv_seq_len > 0) ? cfg.kv_seq_len
                               : cfg.seq_len;

      tensor_factory_t tensor_factory;
      tensor_t q_tensor, k_tensor, v_tensor, o_tensor, mask_tensor;

      int ret = create_qkv_tensor(tensor_factory, cfg.batch, cfg.num_heads,
                                  cfg.seq_len, cfg.head_dim, cfg.qkv_dt,
                                  cfg.qkv_layout,
                                  "sdpa_query", q_tensor);
      if (ret != OK) {
        testlog_error("create_qkv_tensor (query) failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_qkv_tensor(tensor_factory, cfg.batch, cfg.num_heads,
                              S_kv_eff, cfg.head_dim, cfg.qkv_dt,
                              cfg.qkv_layout,
                              "sdpa_key", k_tensor);
      if (ret != OK) {
        testlog_error("create_qkv_tensor (key) failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_qkv_tensor(tensor_factory, cfg.batch, cfg.num_heads,
                              S_kv_eff, cfg.head_dim, cfg.qkv_dt,
                              cfg.qkv_layout,
                              "sdpa_value", v_tensor);
      if (ret != OK) {
        testlog_error("create_qkv_tensor (value) failed");
        log_benchmark_failure(cfg);
        continue;
      }

      const data_type_t out_dt =
        (cfg.out_dt == data_type_t::none) ? cfg.qkv_dt : cfg.out_dt;
      ret = create_output_tensor(tensor_factory, cfg.batch, cfg.num_heads,
                                 cfg.seq_len, cfg.head_dim, out_dt,
                                 cfg.qkv_layout, o_tensor);
      if (ret != OK) {
        testlog_error("create_output_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      ret = create_mask_tensor(tensor_factory, cfg.batch, cfg.num_heads,
                               cfg.seq_len, S_kv_eff, cfg.mask_ndims,
                               cfg.mask_dt, mask_tensor);
      if (ret != OK) {
        testlog_error("create_mask_tensor failed");
        log_benchmark_failure(cfg);
        continue;
      }

      const void *q_ptr   = q_tensor.get_raw_handle_const();
      const void *k_ptr   = k_tensor.get_raw_handle_const();
      const void *v_ptr   = v_tensor.get_raw_handle_const();
      void       *o_ptr   = o_tensor.get_raw_handle_unsafe();
      const void *mask_ptr = (cfg.mask_ndims > 0)
                             ? mask_tensor.get_raw_handle_const() : nullptr;

      sdpa_params params;
      populate_sdpa_params(params, cfg, S_kv_eff);

      TimingStats time_stats;

      // Warmup iterations (untimed).
      for (int i = 0; i < cfg.warmup_iters && !skip; ++i) {
        status_t status = sdpa_direct(q_ptr, k_ptr, v_ptr, mask_ptr, o_ptr,
                                      params);
        if (status != status_t::success) {
          testlog_error("LOWOHA: sdpa_direct failed during warm-up.");
          skip = true;
          break;
        }
      }

      if (skip) {
        log_benchmark_failure(cfg);
        continue;
      }

      double elapsed_ms = 0.0;

      // Timed iterations.
      for (int i = 0; i < cfg.iters && !skip; ++i) {
        if (options.cache_mode == CacheMode::COLD) {
          flush_cache(cache_size);
        }
        auto start = std::chrono::high_resolution_clock::now();

        status_t status = sdpa_direct(q_ptr, k_ptr, v_ptr, mask_ptr, o_ptr,
                                      params);

        if (status != status_t::success) {
          testlog_error("LOWOHA: sdpa_direct failed during benchmark iterations.");
          skip = true;
          break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        elapsed_ms += std::chrono::duration<double, std::milli>(end -
                      start).count();
      }

      if (skip) {
        log_benchmark_failure(cfg);
        continue;
      }

      time_stats.total_time_ms = elapsed_ms;
      sdpa_results.emplace_back(cfg, time_stats);
    }
    catch (const exception_t &ex) {
      std::cout << ex.what() << std::endl;
      return NOT_OK;
    }
  }

  return OK;
}

} // namespace sdpa
} // namespace benchdnn
} // namespace zendnnl
