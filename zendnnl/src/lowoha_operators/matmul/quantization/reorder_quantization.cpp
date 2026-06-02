/*******************************************************************************
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

#include "lowoha_operators/matmul/quantization/reorder_quantization.hpp"
#include "lowoha_operators/reorder/lowoha_reorder.hpp"
#include "lowoha_operators/reorder/lowoha_reorder_utils.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"

#include <algorithm>
#include <vector>

// Per-token bf16/f32 + s8 source path selector (manual source toggle):
//   1 = reorder compute-only (scales) + native bf16s8 / f32s8 GEMM.
//   0 = full reorder quantize A to s8 + s8s8_sym_quant GEMM (default).
// To switch paths, edit the value below and rebuild.
#define ZENDNNL_LOWOHA_DQ_BF16S8 0

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::size_of;
using zendnnl::common::op_instrumentation;
using zendnnl::lowoha::reorder::reorder_algo_t;
using zendnnl::lowoha::reorder::get_single_granularity;
using zendnnl::lowoha::reorder::granularity_type_t;

status_t reorder_quantization_wrapper(
    const void *&src, const int lda, int &reordered_lda, size_t &src_type_size,
    matmul_params &params, matmul_batch_params_t &batch_params,
    const bool transA, const int M, const int K, const int num_threads,
    reorder_quant_buffers_t &buffers) {

  const bool eligible =
      params.dynamic_quant && //TODO: Masking static quantization for now, Remove later.
      params.dtypes.wei == data_type_t::s8 &&
      (params.dtypes.src == data_type_t::bf16 ||
       params.dtypes.src == data_type_t::f32) &&
      (params.dtypes.compute == data_type_t::s8 ||
       params.dtypes.compute == data_type_t::u8);

  if (!eligible) return status_t::success;

  const bool is_dynamic = params.dynamic_quant;
  const data_type_t quant_dtype = params.dtypes.compute;
  const data_type_t orig_src_dtype = params.dtypes.src;
  const bool needs_zp = (quant_dtype == data_type_t::u8);

  status_t val_status = op_instrumentation::validate([&]() {
    if (params.quant_params.src_scale.dims.empty() ||
        params.quant_params.src_scale.dt == data_type_t::none) {
      log_error("Reorder quantization requires quant_params.src_scale "
                "dims and dt to be set");
      return status_t::failure;
    }
    if (needs_zp &&
        (params.quant_params.src_zp.dims.empty() ||
         params.quant_params.src_zp.dt == data_type_t::none)) {
      log_error("Asymmetric (u8) quantization requires quant_params.src_zp "
                "dims and dt to be set");
      return status_t::failure;
    }
    if (!is_dynamic) {
      if (!params.quant_params.src_scale.buff) {
        log_error("Static quantization requires quant_params.src_scale.buff "
                  "to be provided");
        return status_t::failure;
      }
      if (needs_zp && !params.quant_params.src_zp.buff) {
        log_error("Static quantization requires quant_params.src_zp.buff "
                  "to be provided for asymmetric (u8) quantization");
        return status_t::failure;
      }
    }
    return status_t::success;
  });
  if (val_status != status_t::success) {
    return val_status;
  }

  apilog_info("Reorder quantization: using ",
              is_dynamic ? "dynamic" : "static",
              needs_zp ? " asymmetric" : " symmetric", " quantization");

  const int64_t phys_rows = transA ? K : M;
  const int64_t phys_cols = transA ? M : K;
  const int64_t batch_A = batch_params.Batch_A;
  const bool is_batched = (batch_A > 1);

  void *scale_buff = const_cast<void *>(params.quant_params.src_scale.buff);
  if (!scale_buff) {
    int64_t n = 1;
    for (auto d : params.quant_params.src_scale.dims) n *= d;
    buffers.scale_buf = static_cast<uint8_t *>(
        malloc(static_cast<size_t>(n) *
               size_of(params.quant_params.src_scale.dt)));
    if (!buffers.scale_buf) {
      log_error("Reorder quantization: failed to allocate scale buffer");
      return status_t::failure;
    }
    scale_buff = buffers.scale_buf;
  }

  zendnnl::lowoha::reorder::reorder_params_t rp;
  rp.src_dtype = orig_src_dtype;
  rp.dst_dtype = quant_dtype;
  rp.dynamic_quant = is_dynamic;
  rp.num_threads = num_threads;
  rp.algo = reorder_algo_t::native;

  if (is_batched) {
    rp.src_shape = {batch_A, phys_rows, phys_cols};
    rp.dst_shape = rp.src_shape;
    const int64_t batch_stride =
        (batch_params.batch_stride_src != static_cast<size_t>(-1))
            ? static_cast<int64_t>(batch_params.batch_stride_src)
            : phys_rows * lda;
    if (lda != static_cast<int>(phys_cols) ||
        batch_stride != phys_rows * phys_cols) {
      rp.src_strides = {batch_stride, static_cast<int64_t>(lda), 1};
    }
  } else {
    rp.src_shape = {phys_rows, phys_cols};
    rp.dst_shape = rp.src_shape;
    if (lda != static_cast<int>(phys_cols)) {
      rp.src_strides = {static_cast<int64_t>(lda), 1};
    }
  }

  rp.quant_params.scale.buff = scale_buff;
  rp.quant_params.scale.dt = params.quant_params.src_scale.dt;
  rp.quant_params.scale.dims = params.quant_params.src_scale.dims;
  if (transA && rp.quant_params.scale.dims.size() == 2) {
    std::swap(rp.quant_params.scale.dims[0], rp.quant_params.scale.dims[1]);
  }

  // Per-token dynamic quant: see ZENDNNL_LOWOHA_DQ_BF16S8 at top of file.
  // Classify granularity against the *logical* shape {M, K} and the original
  // (pre-swap) src_scale.dims so transA=1 per-token cases are recognized:
  // physical dims/shape (post-swap) would mis-classify them as per_channel.
#if ZENDNNL_LOWOHA_DQ_BF16S8
  const std::vector<int64_t> logical_shape_for_gran = {
      static_cast<int64_t>(M), static_cast<int64_t>(K)};
  const bool try_native_bf16_f32_s8 =
      is_dynamic && !needs_zp && quant_dtype == data_type_t::s8 &&
      !is_batched &&
      (orig_src_dtype == data_type_t::bf16 ||
       orig_src_dtype == data_type_t::f32) &&
      get_single_granularity(params.quant_params.src_scale.dims,
                             logical_shape_for_gran) ==
          granularity_type_t::per_token;
#else
  const bool try_native_bf16_f32_s8 = false;
#endif

  if (try_native_bf16_f32_s8) {
    const status_t compute_status =
        zendnnl::lowoha::reorder::reorder_direct(src, nullptr, rp);
    if (compute_status == status_t::success) {
      if (transA && params.quant_params.src_scale.dims.size() == 2) {
        const int64_t M_dim = params.quant_params.src_scale.dims[0];
        const int64_t G_dim = params.quant_params.src_scale.dims[1];
        if (M_dim != G_dim && M_dim > 1 && G_dim > 1) {
          const size_t nelems = static_cast<size_t>(M_dim * G_dim);
          const size_t elem_size = size_of(params.quant_params.src_scale.dt);
          std::vector<uint8_t> tmp(nelems * elem_size);
          uint8_t *buf = static_cast<uint8_t *>(scale_buff);
          std::memcpy(tmp.data(), buf, nelems * elem_size);
          #pragma omp parallel for collapse(2)
          for (int64_t m = 0; m < M_dim; ++m) {
            for (int64_t g = 0; g < G_dim; ++g) {
              std::memcpy(buf + static_cast<size_t>(m * G_dim + g) * elem_size,
                          tmp.data() + static_cast<size_t>(g * M_dim + m) * elem_size,
                          elem_size);
            }
          }
        }
      }
      params.quant_params.src_scale.buff = scale_buff;
      apilog_info("Reorder quantization: per-token dynamic scales (compute-only); "
                  "source left as ",
                  dtype_info(orig_src_dtype),
                  " for native bf16s8/f32s8 GEMM");
      return status_t::success;
    }
    log_error("Reorder quantization: compute-only scale failed, "
              "falling back to s8 source quantization");
  }

  void *zp_buff = nullptr;
  if (needs_zp) {
    zp_buff = const_cast<void *>(params.quant_params.src_zp.buff);
    if (!zp_buff) {
      int64_t n = 1;
      for (auto d : params.quant_params.src_zp.dims) n *= d;
      buffers.zp_buf = static_cast<uint8_t *>(
          malloc(static_cast<size_t>(n) *
                 size_of(params.quant_params.src_zp.dt)));
      if (!buffers.zp_buf) {
        log_error("Reorder quantization: failed to allocate zero-point buffer");
        return status_t::failure;
      }
      zp_buff = buffers.zp_buf;
    }
  }

  if (needs_zp) {
    rp.quant_params.zero_point.buff = zp_buff;
    rp.quant_params.zero_point.dt = params.quant_params.src_zp.dt;
    rp.quant_params.zero_point.dims = params.quant_params.src_zp.dims;
  }

  buffers.src_buf = static_cast<uint8_t *>(
      malloc(static_cast<size_t>(batch_A * phys_rows * phys_cols)));
  if (!buffers.src_buf) {
    log_error("Reorder quantization: failed to allocate quantized source buffer");
    return status_t::failure;
  }

  const status_t reorder_status = zendnnl::lowoha::reorder::reorder_direct(
      src, buffers.src_buf, rp);

  if (reorder_status == status_t::success) {
    src = buffers.src_buf;
    reordered_lda = static_cast<int>(phys_cols);
    params.dtypes.src = quant_dtype;
    src_type_size = size_of(quant_dtype);

    if (is_batched) {
      batch_params.batch_stride_src =
          static_cast<size_t>(phys_rows * phys_cols);
    }

    if (transA && params.quant_params.src_scale.dims.size() == 2) {
      const int64_t M_dim = params.quant_params.src_scale.dims[0];
      const int64_t G_dim = params.quant_params.src_scale.dims[1];
      if (M_dim > 1 && G_dim > 1) {
        const size_t nelems = static_cast<size_t>(M_dim * G_dim);
        const size_t elem_size = size_of(params.quant_params.src_scale.dt);
        std::vector<uint8_t> tmp(nelems * elem_size);
        uint8_t *buf = static_cast<uint8_t *>(scale_buff);
        std::memcpy(tmp.data(), buf, nelems * elem_size);
        #pragma omp parallel for collapse(2)
        for (int64_t m = 0; m < M_dim; ++m) {
          for (int64_t g = 0; g < G_dim; ++g) {
            std::memcpy(buf + static_cast<size_t>(m * G_dim + g) * elem_size,
                        tmp.data() + static_cast<size_t>(g * M_dim + m) * elem_size,
                        elem_size);
          }
        }
      }
    }

    params.quant_params.src_scale.buff = scale_buff;
    if (needs_zp) {
      params.quant_params.src_zp.buff = zp_buff;
    }

    if (apilog_info_enabled()) {
      int64_t ns = 1;
      for (auto d : params.quant_params.src_scale.dims) ns *= d;
      int64_t nz = 0;
      if (needs_zp) {
        nz = 1;
        for (auto d : params.quant_params.src_zp.dims) nz *= d;
      }
      apilog_info(is_dynamic ? "Dynamic" : "Static",
                  needs_zp ? " asymmetric" : " symmetric",
                  " quantization: source converted from ",
                  dtype_info(orig_src_dtype), " to ",
                  dtype_info(quant_dtype),
                  " (scale_elems=", ns, ", zp_elems=", nz, ")");
    }
  } else {
    log_error("Reorder quantization of source failed, "
              "falling back to original path");
  }

  return status_t::success;
}

status_t group_reorder_quantization_wrapper(
    const std::vector<const void *> &src,
    const std::vector<int> &lda,
    const std::vector<bool> &transA,
    const std::vector<int> &M,
    const std::vector<int> &K,
    const int num_threads,
    std::vector<matmul_params> &params,
    std::vector<const void *> &quantized_src,
    std::vector<int> &quantized_lda,
    group_reorder_quant_buffers_t &buffers,
    bool &quantized) {
  quantized = false;

  const size_t num_ops = M.size();
  if (num_ops == 0) return status_t::success;
  if (src.size() < num_ops || lda.size() < num_ops ||
      transA.size() < num_ops || K.size() < num_ops ||
      params.size() < num_ops) {
    return status_t::success;
  }
  // No trimmed `active_src` / `active_K` copies: every internal use is
  // indexed in [0, num_ops) and `src` / `K` are guaranteed at least that
  // long by the guard above, so we read them directly.  Trimmed views
  // are materialised lazily ONLY for the `group_dynamic_quant` call
  // (which requires exact size == num_ops) and ONLY when the caller
  // passed oversized vectors.  `active_src_strides` stays empty unless
  // some expert is strided (lda != K); group_dynamic_quant treats an
  // empty strides vector as "all rows contiguous".
  std::vector<std::vector<int64_t>> active_src_strides;

  const data_type_t src_dtype = params[0].dtypes.src;
  const data_type_t compute_dtype = params[0].dtypes.compute;
  const data_type_t scale_dtype = params[0].quant_params.src_scale.dt;
  const bool requires_dynamic_quant =
      group_reorder_quantization_required(params, num_ops);
  auto fallback_per_expert = [&]() -> status_t {
    // Release any grouped-path arenas a PRIOR call left on a reused
    // `buffers` object and drop their per-expert views.  The fallback
    // path owns its memory via `fallback_buf` and never reads
    // `src_arena` / `scale_arena` / `src_buf` / `scale_buf`, so leaving
    // them would retain large allocations until destruction and keep
    // stale pointers that no longer describe the current state.
    free(buffers.src_arena);
    free(buffers.scale_arena);
    buffers.src_arena   = nullptr;
    buffers.scale_arena = nullptr;
    buffers.src_buf.clear();
    buffers.scale_buf.clear();
    buffers.fallback_buf.resize(num_ops);
    quantized_src.resize(num_ops);
    quantized_lda.resize(num_ops);
    for (size_t i = 0; i < num_ops; ++i) {
      // Inactive experts (M==0): skip reorder — a zero row count makes
      // `reorder_params_t::is_shaped()` fail and would break MoE decode
      // batches where most experts are unrouted.  Still publish pointers
      // and clear `dynamic_quant` so downstream sees a uniform post-pass.
      if (M[i] == 0) {
        quantized_src[i] = src[i];
        quantized_lda[i] = lda[i];
        params[i].dynamic_quant = false;
        continue;
      }
      const void *src_i = src[i];
      int reordered_lda = lda[i];
      size_t src_type_size = size_of(params[i].dtypes.src);
      matmul_batch_params_t bp;
      bp.Batch_A = 1;
      bp.Batch_B = 1;
      const status_t st = reorder_quantization_wrapper(
          src_i, lda[i], reordered_lda, src_type_size, params[i], bp,
          transA[i], M[i], K[i], num_threads,
          buffers.fallback_buf[i]);
      if (st != status_t::success) return st;
      quantized_src[i] = src_i;
      quantized_lda[i] = reordered_lda;
      if (params[i].dtypes.src == params[i].dtypes.compute) {
        params[i].dynamic_quant = false;
      }
    }
    quantized = true;
    return status_t::success;
  };

  if (!requires_dynamic_quant) {
    return status_t::success;
  }
  if (src_dtype != data_type_t::bf16 && src_dtype != data_type_t::f32) {
    return status_t::success;
  }
  if (compute_dtype != data_type_t::s8) {
    return fallback_per_expert();
  }
  if (scale_dtype != data_type_t::f32 && scale_dtype != data_type_t::bf16) {
    return fallback_per_expert();
  }

  for (size_t i = 0; i < num_ops; ++i) {
    // Basic shape/stride validity applies to EVERY expert — active OR
    // inactive.  `group_dynamic_quant` validates `K>0` and
    // `row_stride >= K` for ALL ops BEFORE its own `if (M[i]==0)
    // continue;` (see lowoha_reorder.cpp), so a degenerate/stale `K<=0`
    // or `lda<K` on an UNROUTED expert would still make the grouped call
    // hard-fail.  Gate it here first and route the whole group to the
    // (shape-tolerant) per-expert fallback instead of erroring out.
    //
    // NEGATIVE dims are NOT a fallback case: `reorder_quantization_wrapper`
    // sizes its allocation as `batch * rows * cols` and a negative M/K casts
    // to a huge `size_t`, risking an oversized allocation / OOM.  Treat them
    // as a hard error before they can reach either path.
    if (M[i] < 0 || K[i] < 0) {
      log_error("Group reorder quantization: negative dimension at op ", i,
                " (M=", M[i], ", K=", K[i], ")");
      return status_t::failure;
    }
    if (K[i] == 0 || lda[i] < K[i]) {
      return fallback_per_expert();
    }
    if (lda[i] != K[i]) {
      if (active_src_strides.empty()) active_src_strides.resize(num_ops);
      active_src_strides[i] = {static_cast<int64_t>(lda[i]), 1};
    }
    // Inactive experts (M==0, routed no tokens): `group_dynamic_quant`
    // skips them right after the shape check above, so their per-expert
    // QUANT metadata (scale dims/dtype, zp, granularity) is never read.
    // Skip ONLY those quant-config gates for them — otherwise a single
    // unrouted expert carrying stale/degenerate scale dims (e.g. a
    // granularity that is neither per_token nor M<=1 per_tensor) would
    // needlessly force the WHOLE group onto the slow per-expert fallback.
    if (M[i] == 0) continue;
    if (!params[i].dynamic_quant ||
        params[i].dtypes.wei != data_type_t::s8 ||
        params[i].dtypes.src != src_dtype ||
        params[i].dtypes.compute != compute_dtype ||
        params[i].quant_params.src_scale.dt != scale_dtype ||
        transA[i]) {
      return fallback_per_expert();
    }
    if (params[i].quant_params.src_zp.buff != nullptr ||
        params[i].quant_params.src_zp.dt != data_type_t::none) {
      return fallback_per_expert();
    }
    // Per-token granularity gate.  CRITICAL decode case: when an expert
    // routes a single token (M == 1) the per-token scale buffer is
    // {1, 1}, which `get_single_granularity` classifies as `per_tensor`
    // (one row ⇒ one scale ⇒ per-token and per-tensor are identical).
    // Accept that degenerate form so a single 1-token expert does not
    // force the WHOLE group onto the slow per-expert fallback — in MoE
    // decode most experts route 1 token, so a strict per_token check
    // here disabled the grouped fast path on essentially every call.
    // per_channel / per_group stay rejected (genuinely K-varying scales
    // the grouped per-token kernel cannot serve).
    const std::vector<int64_t> logical_shape = {
        static_cast<int64_t>(M[i]), static_cast<int64_t>(K[i])};
    const granularity_type_t gran = get_single_granularity(
        params[i].quant_params.src_scale.dims, logical_shape);
    const bool per_token_ok =
        gran == granularity_type_t::per_token
        || (gran == granularity_type_t::per_tensor && M[i] <= 1);
    if (!per_token_ok) {
      return fallback_per_expert();
    }
  }

  buffers.src_buf.assign(num_ops, nullptr);
  buffers.scale_buf.assign(num_ops, nullptr);
  std::vector<void *> dst(num_ops, nullptr);
  std::vector<void *> scale(num_ops, nullptr);

  // Pass 1: size the two arenas.  The s8 dst is 1 byte/elem (M*K); the
  // scale arena only covers experts whose caller supplied no scale
  // buffer (per-token => M elements of `scale_dtype`).
  // Sizes use checked arithmetic (`__builtin_*_overflow`, mirroring the
  // fused-MoE arena sizing in group_matmul_fused_moe.cpp): an unchecked
  // M*K or running sum could wrap on a pathological shape and yield an
  // undersized malloc that the Pass 2 slices then overrun.  Fail early on
  // overflow instead.
  const size_t scale_elem = size_of(scale_dtype);
  size_t src_total = 0, scale_total = 0;
  for (size_t i = 0; i < num_ops; ++i) {
    const size_t m_sz = static_cast<size_t>(std::max(0, M[i]));
    const size_t k_sz = static_cast<size_t>(K[i]);
    size_t src_bytes = 0;
    if (__builtin_mul_overflow(m_sz, k_sz, &src_bytes) ||
        __builtin_add_overflow(src_total, src_bytes, &src_total)) {
      log_error("Group reorder quantization: source arena size overflow");
      return status_t::failure;
    }
    if (params[i].quant_params.src_scale.buff == nullptr) {
      size_t scale_bytes = 0;
      if (__builtin_mul_overflow(m_sz, scale_elem, &scale_bytes) ||
          __builtin_add_overflow(scale_total, scale_bytes, &scale_total)) {
        log_error("Group reorder quantization: scale arena size overflow");
        return status_t::failure;
      }
    }
  }
  // Allocate into locals and commit to `buffers` only once BOTH arenas
  // succeed.  This (a) avoids leaking the src arena if the scale arena
  // fails, and (b) avoids leaking any arenas a prior call left on a
  // REUSED `buffers` object (we free the stale ones at commit time).
  uint8_t *new_src_arena = nullptr;
  uint8_t *new_scale_arena = nullptr;
  if (src_total > 0) {
    new_src_arena = static_cast<uint8_t *>(malloc(src_total));
    if (!new_src_arena) {
      log_error("Group reorder quantization: failed to allocate source arena");
      return status_t::failure;
    }
  }
  if (scale_total > 0) {
    new_scale_arena = static_cast<uint8_t *>(malloc(scale_total));
    if (!new_scale_arena) {
      log_error("Group reorder quantization: failed to allocate scale arena");
      free(new_src_arena);  // release the src arena taken just above
      return status_t::failure;
    }
  }
  // Commit: drop any stale arenas from a prior reuse (free(nullptr) is a
  // no-op), then adopt the freshly-sized ones.  The per-expert
  // `src_buf` / `scale_buf` views were already reset to null above, so
  // they cannot dangle into the old arenas.
  free(buffers.src_arena);
  free(buffers.scale_arena);
  buffers.src_arena   = new_src_arena;
  buffers.scale_arena = new_scale_arena;

  // Release any per-expert fallback buffers a PRIOR fallback call left on
  // this reused `buffers` object.  The grouped fast path never reads
  // `fallback_buf`, so retaining it would keep the previous per-expert
  // allocations alive until destruction and grow peak memory across
  // alternating fallback/grouped call patterns (mirror of the arena
  // cleanup `fallback_per_expert` does for the inverse transition).
  buffers.fallback_buf.clear();

  // Pass 2: hand each expert a slice of the arenas (no per-expert malloc).
  // Byte counts recompute the SAME products Pass 1 validated, so they
  // cannot overflow here; the `<= total` bounds checks are defensive
  // guards ensuring no slice steps past the committed arena even if the
  // two passes ever drift out of sync.
  size_t src_off = 0, scale_off = 0;
  for (size_t i = 0; i < num_ops; ++i) {
    const size_t src_bytes =
        static_cast<size_t>(std::max(0, M[i])) *
        static_cast<size_t>(K[i]);
    if (src_bytes > 0) {
      // Subtraction-based bound (no `src_off + src_bytes` which could wrap
      // size_t on a pathological total) — keeps the guard overflow-safe.
      if (src_off > src_total || src_bytes > src_total - src_off) {
        log_error("Group reorder quantization: source slice exceeds arena");
        return status_t::failure;
      }
      buffers.src_buf[i] = buffers.src_arena + src_off;
      src_off += src_bytes;
    }
    dst[i] = buffers.src_buf[i];

    void *scale_buff =
        const_cast<void *>(params[i].quant_params.src_scale.buff);
    if (!scale_buff) {
      const size_t scale_bytes =
          static_cast<size_t>(std::max(0, M[i])) * scale_elem;
      if (scale_bytes > 0) {
        // Subtraction-based bound (see the src_buf guard above) so the
        // check cannot wrap size_t on a pathological scale total.
        if (scale_off > scale_total || scale_bytes > scale_total - scale_off) {
          log_error("Group reorder quantization: scale slice exceeds arena");
          return status_t::failure;
        }
        buffers.scale_buf[i] = buffers.scale_arena + scale_off;
        scale_off += scale_bytes;
        scale_buff = buffers.scale_buf[i];
      }
    }
    scale[i] = scale_buff;
  }

  zendnnl::lowoha::reorder::group_dynamic_quant_params_t gparams;
  gparams.src_dtype = src_dtype;
  gparams.dst_dtype = compute_dtype;
  gparams.scale_dtype = scale_dtype;
  gparams.num_threads = num_threads;

  // group_dynamic_quant requires src/K vectors of EXACTLY num_ops; pass
  // the caller's vectors directly when already that size (the common
  // fused-MoE case), else materialise trimmed copies.  dst is contiguous
  // (lda == K), so dst_strides stays empty.
  const std::vector<const void *> *src_arg = &src;
  const std::vector<int> *K_arg = &K;
  std::vector<const void *> src_trimmed;
  std::vector<int> K_trimmed;
  if (src.size() != num_ops) {
    src_trimmed.assign(
        src.begin(),
        src.begin() +
            static_cast<std::vector<const void *>::difference_type>(num_ops));
    src_arg = &src_trimmed;
  }
  if (K.size() != num_ops) {
    K_trimmed.assign(
        K.begin(),
        K.begin() + static_cast<std::vector<int>::difference_type>(num_ops));
    K_arg = &K_trimmed;
  }
  const std::vector<std::vector<int64_t>> dst_strides;

  const status_t st = zendnnl::lowoha::reorder::group_dynamic_quant(
      *src_arg, M, *K_arg, active_src_strides, dst, dst_strides,
      scale, gparams);
  if (st != status_t::success) return st;

  quantized_src.resize(num_ops);
  quantized_lda.resize(num_ops);
  for (size_t i = 0; i < num_ops; ++i) {
    // Inactive experts (M==0) got no arena slice, so `dst[i]` is null and
    // `group_dynamic_quant` produced no s8 rows / scale for them.  A null
    // `quantized_src` can trip pointer validation in downstream kernels
    // that loop over ALL ops without an M>0 guard (e.g. ALGO 1).  Publish
    // the ORIGINAL src/lda (non-null, matches the fallback path) and leave
    // the source dtype unchanged — only clear `dynamic_quant` for uniform
    // downstream state.  The 0-row expert does no real compute either way.
    if (M[i] == 0) {
      quantized_src[i] = src[i];
      quantized_lda[i] = lda[i];
      params[i].dynamic_quant = false;
      continue;
    }
    quantized_src[i] = dst[i];
    quantized_lda[i] = K[i];
    params[i].dtypes.src = compute_dtype;
    params[i].dynamic_quant = false;
    params[i].quant_params.src_scale.buff = scale[i];
  }

  quantized = true;
  return status_t::success;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
