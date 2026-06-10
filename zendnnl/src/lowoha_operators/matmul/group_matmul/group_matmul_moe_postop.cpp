/*******************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

/// MoE weighted-reduce post-op for parallel group matmul.
///
/// This library performs ONLY the weighted-reduce — not the gather.
/// The caller must pre-build the row_ptrs array (typically during the
/// token-to-expert scatter step on the frontend/Python side) before
/// calling group_matmul_direct with the moe_postop parameter.
///
/// Kernel:
///   For each token t and hidden dim d:
///     output[t, d] = Σ_k  topk_weights[t, k] * row_ptrs[t*topk+k][d]
///
/// Two implementations selected at runtime via ISA detection:
///   - AVX-512 (default on Zen 3/4/5): 16 floats / 16 bf16 per iteration
///   - Scalar fallback: when AVX-512 is unavailable

#include <cstring>
#include <type_traits>
#include <vector>

#include <immintrin.h>
#include <omp.h>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"
#include "common/zendnnl_global.hpp"
#include "group_matmul_direct.hpp"
#include "lowoha_operators/common/operator_instrumentation.hpp"
#include "lowoha_operators/matmul/lowoha_matmul_utils.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {

using namespace zendnnl::ops;
using zendnnl::common::bfloat16_t;
using zendnnl::common::float16_t;
using zendnnl::memory::data_type_t;

// ---------------------------------------------------------------------------
// Weighted-reduce kernel — vectorized (AVX-512) + scalar fallback
// ---------------------------------------------------------------------------

namespace {

// ── Scalar implementation (runtime fallback when AVX-512 is unavailable) ──
//
// Using `data_type_t` directly keeps a single source of truth for dtype
// identity across the library; the dispatcher in `group_matmul_moe_postop_execute`
// already receives `data_type_t dst_elem` from the caller, so no translation is needed.
template <typename Elem, data_type_t Kind>
inline float load_as_f32(const Elem *ptr, int idx) {
  if constexpr(Kind == data_type_t::f32) {
    return ptr[idx];
  }
  else if constexpr(Kind == data_type_t::bf16) {
    // Preserve raw BF16 bits without implementation-defined uint16→int16 cast.
    int16_t raw;
    std::memcpy(&raw, &ptr[idx], sizeof(int16_t));
    return bfloat16_t::bf16_to_f32_val(raw);
  }
  else {
    static_assert(Kind == data_type_t::f16,
                  "load_as_f32: unhandled dtype "
                  "(supported: f32, bf16, f16)");
    return float16_t::f16_to_f32_val(static_cast<uint16_t>(ptr[idx]));
  }
}

template <typename Elem, data_type_t Kind>
inline void store_from_f32(Elem *ptr, int idx, float val) {
  if constexpr(Kind == data_type_t::f32) {
    ptr[idx] = val;
  }
  else if constexpr(Kind == data_type_t::bf16) {
    int16_t raw = bfloat16_t::f32_to_bf16_val(val);
    std::memcpy(&ptr[idx], &raw, sizeof(raw));
  }
  else {
    static_assert(Kind == data_type_t::f16,
                  "store_from_f32: unhandled dtype "
                  "(supported: f32, bf16, f16)");
    ptr[idx] = static_cast<Elem>(float16_t::f32_to_f16_val(val));
  }
}

template <typename Elem, data_type_t Kind>
void moe_weighted_reduce_scalar(
  const group_matmul_moe_postop_params *postop,
  const int D,
  const int num_threads) {

  auto *out_base = static_cast<Elem *>(postop->output);
  const size_t out_stride = static_cast<size_t>(postop->ldc_output);

  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int t = 0; t < postop->num_tokens; ++t) {
    Elem *out_row = out_base + static_cast<size_t>(t) * out_stride;
    for (int d = 0; d < D; ++d) {
      float acc = 0.f;
      for (int k = 0; k < postop->topk; ++k) {
        const int slot = t * postop->topk + k;
        const auto *src_row =
          static_cast<const Elem *>(postop->row_ptrs[static_cast<size_t>(slot)]);
        const float w =
          postop->skip_weighted
          ? 1.f
          : postop->topk_weights[static_cast<size_t>(slot)];
        acc += w * load_as_f32<Elem, Kind>(src_row, d);
      }
      store_from_f32<Elem, Kind>(out_row, d, acc);
    }
  }
}

// ── AVX-512 vectorized implementation ───────────────────────────────────

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
void moe_weighted_reduce_avx512_f32(
  const group_matmul_moe_postop_params *postop,
  const int D,
  const int num_threads) {

  auto *out_base = static_cast<float *>(postop->output);
  const size_t out_stride = static_cast<size_t>(postop->ldc_output);

  const int topk = postop->topk;
  const int D16 = D & ~15;

  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int t = 0; t < postop->num_tokens; ++t) {
    float *out_row = out_base + static_cast<size_t>(t) * out_stride;

    // k=0: initialize output with w0 * src0 (no load of output needed).
    {
      const int slot = t * topk;
      const auto *src_row =
        static_cast<const float *>(postop->row_ptrs[static_cast<size_t>(slot)]);
      const float w = postop->skip_weighted
                      ? 1.f
                      : postop->topk_weights[static_cast<size_t>(slot)];
      const __m512 vw = _mm512_set1_ps(w);
      int d = 0;
      for (; d < D16; d += 16) {
        __m512 vsrc = _mm512_loadu_ps(src_row + d);
        _mm512_storeu_ps(out_row + d, _mm512_mul_ps(vw, vsrc));
      }
      for (; d < D; ++d) {
        out_row[d] = w * src_row[d];
      }
    }

    // k=1..topk-1: accumulate with FMA.
    for (int k = 1; k < topk; ++k) {
      const int slot = t * topk + k;
      const auto *src_row =
        static_cast<const float *>(postop->row_ptrs[static_cast<size_t>(slot)]);
      const float w = postop->skip_weighted
                      ? 1.f
                      : postop->topk_weights[static_cast<size_t>(slot)];
      const __m512 vw = _mm512_set1_ps(w);
      int d = 0;
      for (; d < D16; d += 16) {
        __m512 vacc = _mm512_loadu_ps(out_row + d);
        __m512 vsrc = _mm512_loadu_ps(src_row + d);
        _mm512_storeu_ps(out_row + d, _mm512_fmadd_ps(vw, vsrc, vacc));
      }
      for (; d < D; ++d) {
        out_row[d] += w * src_row[d];
      }
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
void moe_weighted_reduce_avx512_bf16(
  const group_matmul_moe_postop_params *postop,
  const int D,
  const int num_threads) {

  auto *out_base = static_cast<uint16_t *>(postop->output);
  const size_t out_stride = static_cast<size_t>(postop->ldc_output);

  const int topk = postop->topk;
  const int D16 = D & ~15;

  // Accumulate entirely in FP32 to avoid BF16 truncation between topk
  // iterations. Per-thread buffer reused across calls to amortize allocation.
  #pragma omp parallel num_threads(num_threads)
  {
    static thread_local std::vector<float> acc_buf;
    if (acc_buf.size() < static_cast<size_t>(D)) {
      acc_buf.resize(static_cast<size_t>(D));
    }

    #pragma omp for schedule(static)
    for (int t = 0; t < postop->num_tokens; ++t) {
      uint16_t *out_row = out_base + static_cast<size_t>(t) * out_stride;
      float *acc = acc_buf.data();

      // k=0: initialize FP32 accumulator with w0 * src0.
      {
        const int slot = t * topk;
        const auto *src_row = static_cast<const uint16_t *>(
                                postop->row_ptrs[static_cast<size_t>(slot)]);
        const float w = postop->skip_weighted
                        ? 1.f
                        : postop->topk_weights[static_cast<size_t>(slot)];
        const __m512 vw = _mm512_set1_ps(w);
        int d = 0;
        for (; d < D16; d += 16) {
          __m256i src_bf16 = _mm256_loadu_si256(
                               reinterpret_cast<const __m256i *>(src_row + d));
          __m512 vsrc = _mm512_castsi512_ps(
                          _mm512_slli_epi32(_mm512_cvtepu16_epi32(src_bf16), 16));
          _mm512_storeu_ps(acc + d, _mm512_mul_ps(vw, vsrc));
        }
        for (; d < D; ++d)
          acc[d] = w * bfloat16_t::bf16_to_f32_val(
                     static_cast<int16_t>(src_row[d]));
      }

      // k=1..topk-1: FMA into FP32 accumulator (no BF16 truncation).
      for (int k = 1; k < topk; ++k) {
        const int slot = t * topk + k;
        const auto *src_row = static_cast<const uint16_t *>(
                                postop->row_ptrs[static_cast<size_t>(slot)]);
        const float w = postop->skip_weighted
                        ? 1.f
                        : postop->topk_weights[static_cast<size_t>(slot)];
        const __m512 vw = _mm512_set1_ps(w);
        int d = 0;
        for (; d < D16; d += 16) {
          __m256i src_bf16 = _mm256_loadu_si256(
                               reinterpret_cast<const __m256i *>(src_row + d));
          __m512 vsrc = _mm512_castsi512_ps(
                          _mm512_slli_epi32(_mm512_cvtepu16_epi32(src_bf16), 16));
          __m512 vacc = _mm512_loadu_ps(acc + d);
          _mm512_storeu_ps(acc + d, _mm512_fmadd_ps(vw, vsrc, vacc));
        }
        for (; d < D; ++d)
          acc[d] += w * bfloat16_t::bf16_to_f32_val(
                      static_cast<int16_t>(src_row[d]));
      }

      // Final: convert FP32 accumulator → BF16 output.
      // Round-to-nearest-even matching bfloat16_t::f32_to_bf16_avx512:
      //   lsb = (bits >> 16) & 1; bits += 0x7FFF + lsb; result = bits >> 16
      // Narrowing via _mm512_cvtepi32_epi16 (vpmovsdw) — same pattern as
      // bfloat16_t::f32_to_bf16_avx512 in common/bfloat16.cpp.
      {
        const __m512i one = _mm512_set1_epi32(1);
        const __m512i bias = _mm512_set1_epi32(0x7FFF);
        int d = 0;
        for (; d < D16; d += 16) {
          __m512i bits = _mm512_castps_si512(_mm512_loadu_ps(acc + d));
          __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), one);
          bits = _mm512_add_epi32(bits, _mm512_add_epi32(bias, lsb));
          // Arithmetic right shift preserves sign bit, so negative BF16
          // values (sign bit set) produce negative int32 values that
          // _mm512_cvtepi32_epi16 narrows without saturation.
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_row + d),
                              _mm512_cvtepi32_epi16(_mm512_srai_epi32(bits, 16)));
        }
        for (; d < D; ++d)
          out_row[d] = static_cast<uint16_t>(
                         bfloat16_t::f32_to_bf16_val(acc[d]));
      }
    }
  }
}

// ── AVX-512-FP16 specialisation for F16 dst ─────────────────────────────
//
// Mirrors `moe_weighted_reduce_avx512_bf16` line-for-line, with the
// load / store cvt swapped from BF16 (integer-RNE shift+add) to F16
// (the AVX-512F `VCVTPH2PS` / `VCVTPS2PH` intrinsics `_mm512_cvtph_ps`
// / `_mm512_cvtps_ph`).  The FP32 accumulator pattern is unchanged —
// F16 has only 10 mantissa bits, so accumulating in F16 would lose
// ~6 bits of precision per topk iteration; FP32 accumulation matches
// the BF16 specialisation's rationale and avoids that loss.
//
// Load / store stay on `__m256i` (16 × uint16_t) — same storage ABI as
// the BF16 specialisation.  The `__m256i`-typed AVX-512F cvt intrinsics
// consume / produce that type directly (no `__m256h` cast), keeping
// this path on the same widely-available intrinsic set as
// `common/float16.cpp` and off the `avx512fp16`-only `_mm512_cvtxph_ps`
// / `_mm512_cvtxps_ph` forms.
__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
void moe_weighted_reduce_avx512_f16(
  const group_matmul_moe_postop_params *postop,
  const int D,
  const int num_threads) {

  auto *out_base = static_cast<uint16_t *>(postop->output);
  const size_t out_stride = static_cast<size_t>(postop->ldc_output);

  const int topk = postop->topk;
  const int D16 = D & ~15;

  // Per-thread FP32 accumulator — same buffer-reuse pattern as the BF16
  // specialisation.  Sized to D once and reused across calls; vectors
  // monotonically grow, so steady-state calls see zero allocation.
  #pragma omp parallel num_threads(num_threads)
  {
    static thread_local std::vector<float> acc_buf;
    if (acc_buf.size() < static_cast<size_t>(D)) {
      acc_buf.resize(static_cast<size_t>(D));
    }

    #pragma omp for schedule(static)
    for (int t = 0; t < postop->num_tokens; ++t) {
      uint16_t *out_row = out_base + static_cast<size_t>(t) * out_stride;
      float *acc = acc_buf.data();

      // k=0: initialize FP32 accumulator with w0 * src0.
      {
        const int slot = t * topk;
        const auto *src_row = static_cast<const uint16_t *>(
                                postop->row_ptrs[static_cast<size_t>(slot)]);
        const float w = postop->skip_weighted
                        ? 1.f
                        : postop->topk_weights[static_cast<size_t>(slot)];
        const __m512 vw = _mm512_set1_ps(w);
        int d = 0;
        for (; d < D16; d += 16) {
          // Load 16 × F16 (256-bit) and cvt to 16 × F32 (512-bit).
          __m256i src_f16 = _mm256_loadu_si256(
                              reinterpret_cast<const __m256i *>(src_row + d));
          __m512 vsrc = _mm512_cvtph_ps(src_f16);
          _mm512_storeu_ps(acc + d, _mm512_mul_ps(vw, vsrc));
        }
        for (; d < D; ++d) {
          acc[d] = w * float16_t::f16_to_f32_val(src_row[d]);
        }
      }

      // k=1..topk-1: FMA into FP32 accumulator (no F16 truncation between
      // iterations).
      for (int k = 1; k < topk; ++k) {
        const int slot = t * topk + k;
        const auto *src_row = static_cast<const uint16_t *>(
                                postop->row_ptrs[static_cast<size_t>(slot)]);
        const float w = postop->skip_weighted
                        ? 1.f
                        : postop->topk_weights[static_cast<size_t>(slot)];
        const __m512 vw = _mm512_set1_ps(w);
        int d = 0;
        for (; d < D16; d += 16) {
          __m256i src_f16 = _mm256_loadu_si256(
                              reinterpret_cast<const __m256i *>(src_row + d));
          __m512 vsrc = _mm512_cvtph_ps(src_f16);
          __m512 vacc = _mm512_loadu_ps(acc + d);
          _mm512_storeu_ps(acc + d, _mm512_fmadd_ps(vw, vsrc, vacc));
        }
        for (; d < D; ++d) {
          acc[d] += w * float16_t::f16_to_f32_val(src_row[d]);
        }
      }

      // Final: convert FP32 accumulator → F16 output via the AVX-512F
      // `VCVTPS2PH` intrinsic (`_mm512_cvtps_ph`).  The explicit
      // `_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC` immediate pins
      // round-to-nearest-even independent of the runtime MXCSR mode,
      // matching the rounding semantics of `float16_t::f32_to_f16_val`
      // (and `cvt_f32_to_f16_vec`) in common/float16.cpp.
      //
      // Saturation note: `vcvtps2ph` saturates inputs outside F16's
      // ±65504 normal range to ±inf.  This is the documented IEEE 754
      // binary16 round-toward-nearest behaviour and matches the scalar
      // `f32_to_f16_val` path; callers that need finite outputs for
      // very large weighted sums must already bound their inputs.
      {
        int d = 0;
        for (; d < D16; d += 16) {
          __m512 vacc = _mm512_loadu_ps(acc + d);
          __m256i out_f16 = _mm512_cvtps_ph(
                              vacc,
                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm256_storeu_si256(reinterpret_cast<__m256i *>(out_row + d),
                              out_f16);
        }
        for (; d < D; ++d) {
          out_row[d] = float16_t::f32_to_f16_val(acc[d]);
        }
      }
    }
  }
}

// ── Dispatch: AVX-512 when available, scalar fallback otherwise ─────────

template <typename Elem, data_type_t Kind>
void moe_weighted_reduce(
  const group_matmul_moe_postop_params *postop,
  const int D,
  const int num_threads) {

  const auto &plat = zendnnl::common::zendnnl_platform_info();

  if constexpr(Kind == data_type_t::f32) {
    if (plat.get_avx512f_status()) {
      moe_weighted_reduce_avx512_f32(postop, D, num_threads);
    }
    else {
      moe_weighted_reduce_scalar<Elem, Kind>(postop, D, num_threads);
    }
  }
  else if constexpr(Kind == data_type_t::bf16) {
    if (plat.get_avx512f_status()) {
      moe_weighted_reduce_avx512_bf16(postop, D, num_threads);
    }
    else {
      moe_weighted_reduce_scalar<Elem, Kind>(postop, D, num_threads);
    }
  }
  else {
    static_assert(Kind == data_type_t::f16,
                  "moe_weighted_reduce: unhandled dtype "
                  "(supported: f32, bf16, f16)");
    // F16 vector path needs only AVX-512F: the F16<->F32 boundary uses
    // the AVX-512F `VCVTPH2PS` / `VCVTPS2PH` intrinsics (`_mm512_cvtph_ps`
    // / `_mm512_cvtps_ph`) and the accumulator math is FP32, so the gate
    // is `get_avx512f_status()` — identical to the f32 / bf16 siblings
    // above (no AVX-512-FP16 dependency).
    if (plat.get_avx512f_status()) {
      moe_weighted_reduce_avx512_f16(postop, D, num_threads);
    }
    else {
      moe_weighted_reduce_scalar<Elem, Kind>(postop, D, num_threads);
    }
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

status_t validate_group_matmul_moe_postop(
  const group_matmul_moe_postop_params *postop,
  const int D,
  const data_type_t dst_elem) {

  if (postop == nullptr) {
    return status_t::success;
  }

  if (postop->num_tokens <= 0 || postop->topk <= 0) {
    log_error("group_matmul_direct: moe_postop invalid num_tokens or topk");
    return status_t::failure;
  }
  if (postop->output == nullptr || postop->row_ptrs == nullptr) {
    log_error("group_matmul_direct: moe_postop output or row_ptrs is null");
    return status_t::failure;
  }
  if (!postop->skip_weighted && postop->topk_weights == nullptr) {
    log_error("group_matmul_direct: moe_postop requires topk_weights unless skip_weighted");
    return status_t::failure;
  }
  if (postop->ldc_output < D) {
    log_error("group_matmul_direct: moe_postop ldc_output < D");
    return status_t::failure;
  }
  if (dst_elem != data_type_t::f32
      && dst_elem != data_type_t::bf16
      && dst_elem != data_type_t::f16) {
    log_error("group_matmul_direct: moe_postop requires FP32, BF16, or F16 dst");
    return status_t::failure;
  }

  // Per-element null check gated behind diagnostics (O(num_tokens*topk)).
  status_t slot_check = op_instrumentation::validate([&]() {
    const size_t total = static_cast<size_t>(postop->num_tokens) * postop->topk;
    for (size_t i = 0; i < total; ++i) {
      if (postop->row_ptrs[i] == nullptr) {
        log_error("group_matmul_direct: moe_postop row_ptrs[", i, "] is null");
        return status_t::failure;
      }
    }
    return status_t::success;
  });
  if (slot_check != status_t::success) {
    return slot_check;
  }

  return status_t::success;
}

status_t group_matmul_moe_postop_execute(
  const group_matmul_moe_postop_params *postop,
  const int D,
  const int num_threads,
  const data_type_t dst_elem) {

  if (dst_elem == data_type_t::f32) {
    moe_weighted_reduce<float, data_type_t::f32>(postop, D, num_threads);
    return status_t::success;
  }
  if (dst_elem == data_type_t::bf16) {
    moe_weighted_reduce<uint16_t, data_type_t::bf16>(postop, D, num_threads);
    return status_t::success;
  }
  if (dst_elem == data_type_t::f16) {
    // F16 weighted-reduce runs on AVX-512F (F16<->F32 via VCVTPH2PS /
    // VCVTPS2PH + FP32 accumulator); the dispatch template gates on
    // `get_avx512f_status()` and falls back to the scalar template
    // otherwise.  Note: F16 group_matmul as a whole still requires
    // AVX-512-FP16 for the GEMM — the public `group_matmul_direct` F16
    // ISA gate enforces that upstream — but this reduce post-op does not.
    moe_weighted_reduce<uint16_t, data_type_t::f16>(postop, D, num_threads);
    return status_t::success;
  }

  log_error("group_matmul_moe_postop_execute: unsupported dst element type");
  return status_t::failure;
}

} // namespace matmul
} // namespace lowoha
} // namespace zendnnl
