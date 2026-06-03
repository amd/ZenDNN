/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 ******************************************************************************/

//==============================================================================
// FP16 STATIC PER-TENSOR QUANT / DEQUANT — FP16-FMA backend (Strategy A)
//
// Native __m512h kernels. The whole quant / dequant chain stays in FP16
// (32 lanes per vector register) using AVX-512-FP16 ISA. The host CPU MUST
// support `avx512fp16` (verified upstream by the dispatcher via
// zendnnl_platform_info().get_avx512_f16_status()) or these kernels will
// SIGILL. On toolchains older than GCC 12, this TU compiles to empty
// stubs and the dispatcher falls back to the F32-FMA kernels.
//
// Companion F32-FMA kernels live in quant_dequant_per_tensor_avx512_f32_fma_kernel.cpp
// (suffix `_avx512`); these are the matching `_avx512fp16` siblings selected
// at dispatch time by can_use_f16_fma_kernel(). The build-time master switch
// ZENDNNL_NATIVE_F32_ACCUM=ON forces the F32-FMA path.
//
// Forward (quant)  : f16 -> s8 / u8  (VMULPH + VCVTPH2W + VPMOVSWB/VPMOVUSWB)
// Reverse (dequant): s8 / u8 -> f16  (VCVTW2PH + VFMSUB + VMULPH-style chain)
//
// Precision contract:
//   - Quant: NOT bit-exact with the scalar reference (different rounding
//     chain). Tolerate |delta| <= 1 LSB on quantized output in the
//     common regime.
//   - The intermediate quotient round(val/scale) is materialized in s16
//     by VCVTPH2W BEFORE the int32 zero-point add (val * (1/scale_f16)
//     stays in FP16 for throughput). For scales/values where |q| > 32767
//     this saturates at the int16 narrow even though the public contract
//     accepts any int32 zp; a sufficiently large |zp| could then have
//     rescued the saturated quotient back into s8/u8 range, producing
//     an arbitrarily-wrong final output if the vector path ran. To
//     prevent that, the kernel gates the vector loop on
//     common::fp16_zp_safe_for_s16_narrow(zero_point) (|zp| <= 32512)
//     in addition to common::fp16_inv_scale_is_finite(scale); when
//     either guard fails the whole call runs through the scalar tail,
//     which computes val/scale in f32 and the zp add in int32 —
//     bit-equivalent to the F32-FMA / scalar reference.
//   - Dequant: end-to-end round-trip via scale/zp -> f16 is exact up to
//     FP16 representability, which the reference also obeys via
//     float16_t::f32_to_f16_val (rounds to nearest even).
//   - The dequant chain widens (input - zp) to FP16 via VCVTDQ2PH BEFORE
//     the FP16 scale multiply (for throughput parity with the rest of
//     the FP16-FMA kernel set). Two narrow/widen steps need guarding:
//     (a) For |zp| outside the FP16 finite range the s32 widen saturates
//         to +/-Inf and the FP16 multiply propagates the infinity. The
//         scalar / F32-FMA path computes the same chain in f32 and gets
//         a finite small number.
//     (b) For |scale| outside [FP16_MIN_NORMAL, FP16_MAX] the f32->_Float16
//         narrow of the user scale rounds to 0 or +/-Inf. Overflow produces
//         NaN for input == zp lanes (0 * Inf); underflow forces small
//         finite scalar results to 0.
//     The vector loop is therefore gated on BOTH
//     common::fp16_zp_safe_for_dequant_widen(zero_point) (|zp| <= 65000)
//     AND common::fp16_scale_safe_for_dequant_narrow(scale) (scale == 0
//     or |scale| inside the FP16 normal range). When either guard fails
//     the whole call runs through the scalar tail which stays in f32
//     until the final f16 store. The s32 zero_point and finite-scale
//     API contracts are preserved in both regimes.
//==============================================================================

#include "lowoha_operators/reorder/reorder_data_type/static_quant_dequant_impl/static_kernels.hpp"
#include "common/float16.hpp"

#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace zendnnl {
namespace lowoha {
namespace reorder {

#if defined(__GNUC__) && (__GNUC__ >= 12)

//==============================================================================
// Pass-2 building block: PH * (1/scale_f16), round to 32 s16 lanes.
//==============================================================================
__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline __m512i ph_quantize_to_s16(__m512h v, __m512h vinv_scale) {
  __m512h q_ph = _mm512_mul_ph(v, vinv_scale);
  return _mm512_cvt_roundph_epi16(q_ph,
      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

//==============================================================================
// KERNEL: f16 -> int8 (per-tensor, static quantization, FP16-FMA)
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void quantize_f16_to_int8_avx512fp16(const uint16_t *input, int8_t *output,
                                      size_t nelems, float scale, int zero_point) {
  // Zero-point arithmetic is done in int32 to match the reference / F32-FMA
  // kernels: the public contract (see docs/operator/lowoha_reorder_operator.md
  // §"Zero-point validation") accepts ANY int32 zero_point and clamps to the
  // destination dtype's range during quantization. A naive int16 add here
  // would silently change semantics for zp values outside int16 range.
  const __m512i vzp32 = _mm512_set1_epi32(zero_point);

  size_t i = 0;
  // Twin guards: (i) when scale is so small that 1/scale overflows the
  // FP16 normal range, _Float16(1/scale) becomes +Inf and every nonzero
  // lane of the subsequent VMULPH saturates; (ii) when |zero_point| is
  // large enough that the int16-saturating VCVTPH2W narrow can pre-
  // saturate quotients that the int32 zp add would have rescued (the
  // public contract accepts any int32 zp — a user can legitimately pass
  // a zp outside int16 alongside a scale that produces an out-of-int16
  // quotient that lands back inside s8 after the offset). Either
  // failure routes the whole call to the scalar tail, which computes
  // val/scale in f32 and the +zp in int32 — bit-equal to the F32-FMA /
  // scalar reference.
  if (common::fp16_inv_scale_is_finite(scale) &&
      common::fp16_zp_safe_for_s16_narrow(zero_point)) {
    const _Float16 inv_scale_f16 = static_cast<_Float16>(1.0f / scale);
    const __m512h vinv  = _mm512_set1_ph(inv_scale_f16);

    for (; i + 31 < nelems; i += 32) {
      __m512h v   = _mm512_loadu_ph(input + i);
      // PH * (1/scale_f16) -> 32 s16 lanes, signed-saturated by VCVTPH2W.
      __m512i s16 = ph_quantize_to_s16(v, vinv);
      // Widen s16 -> s32 (lo/hi 256-bit halves), add zp in int32 (no saturation),
      // then signed-saturate-narrow s32 -> s8 with VPMOVSDB.
      __m512i s32_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(s16));
      __m512i s32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(s16, 1));
      s32_lo = _mm512_add_epi32(s32_lo, vzp32);
      s32_hi = _mm512_add_epi32(s32_hi, vzp32);
      __m128i s8_lo  = _mm512_cvtsepi32_epi8(s32_lo);
      __m128i s8_hi  = _mm512_cvtsepi32_epi8(s32_hi);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i),      s8_lo);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i + 16), s8_hi);
    }
  }

  for (; i < nelems; ++i) {
    float val = common::float16_t::f16_to_f32_val(input[i]);
    int32_t q = static_cast<int32_t>(std::nearbyint(val / scale)) + zero_point;
    q = std::max(-128, std::min(127, q));
    output[i] = static_cast<int8_t>(q);
  }
}

//==============================================================================
// KERNEL: f16 -> uint8 (per-tensor, static quantization, FP16-FMA)
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void quantize_f16_to_uint8_avx512fp16(const uint16_t *input, uint8_t *output,
                                       size_t nelems, float scale, int zero_point) {
  // Zero-point arithmetic in int32; see quantize_f16_to_int8_avx512fp16 for
  // the rationale (preserves the s32-zp contract across all backends).
  // Tiny-scale guard also matches; see that kernel for the explanation.
  const __m512i vzp32 = _mm512_set1_epi32(zero_point);

  size_t i = 0;
  // Twin guards — see quantize_f16_to_int8_avx512fp16 for the full
  // rationale: (i) tiny scales overflow 1/scale in FP16, (ii) large
  // |zero_point| can let VCVTPH2W pre-saturate quotients that the
  // int32 zp add would have brought back inside [0, 255]. Either
  // failure routes through the scalar tail (f32 quotient + int32 zp).
  if (common::fp16_inv_scale_is_finite(scale) &&
      common::fp16_zp_safe_for_s16_narrow(zero_point)) {
    const _Float16 inv_scale_f16 = static_cast<_Float16>(1.0f / scale);
    const __m512h vinv  = _mm512_set1_ph(inv_scale_f16);

    // Manual clamp BEFORE VPMOVUSDB. _mm512_cvtusepi32_epi8 interprets
    // its input as UNSIGNED (per the Intel Intrinsics Guide), so a
    // negative s32 (e.g. -1) would wrap to 0xFFFFFFFF and saturate to
    // 255 instead of clamping to 0. Clamp to [0, 255] in signed s32
    // first; after that any cvtepi32_epi8-family narrow gives the same
    // result.
    const __m512i vlo32 = _mm512_setzero_si512();
    const __m512i vhi32 = _mm512_set1_epi32(255);

    for (; i + 31 < nelems; i += 32) {
      __m512h v   = _mm512_loadu_ph(input + i);
      __m512i s16 = ph_quantize_to_s16(v, vinv);
      __m512i s32_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(s16));
      __m512i s32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(s16, 1));
      s32_lo = _mm512_add_epi32(s32_lo, vzp32);
      s32_hi = _mm512_add_epi32(s32_hi, vzp32);
      // Signed clamp [0, 255] before narrowing.
      s32_lo = _mm512_max_epi32(vlo32, _mm512_min_epi32(vhi32, s32_lo));
      s32_hi = _mm512_max_epi32(vlo32, _mm512_min_epi32(vhi32, s32_hi));
      // After the clamp every lane is in [0, 255]; narrow via VPMOVUSDB.
      __m128i u8_lo = _mm512_cvtusepi32_epi8(s32_lo);
      __m128i u8_hi = _mm512_cvtusepi32_epi8(s32_hi);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i),      u8_lo);
      _mm_storeu_si128(reinterpret_cast<__m128i *>(output + i + 16), u8_hi);
    }
  }

  for (; i < nelems; ++i) {
    float val = common::float16_t::f16_to_f32_val(input[i]);
    int32_t q = static_cast<int32_t>(std::nearbyint(val / scale)) + zero_point;
    q = std::max(0, std::min(255, q));
    output[i] = static_cast<uint8_t>(q);
  }
}

//==============================================================================
// KERNEL: int8 -> f16 (per-tensor, static dequantization, FP16-FMA)
//
// 32 lanes / iter: load 32x int8 -> sign-extend to int32 (in two 16-lane
// halves) -> subtract zp in int32 (preserves the s32-zp contract; see the
// quantize_f16_to_int8_avx512fp16 comment) -> convert s32 -> PH -> multiply
// by scale_f16 -> store PH.
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void dequantize_int8_to_f16_avx512fp16(const int8_t *input, uint16_t *output,
                                        size_t nelems, float scale, int zero_point) {
  const _Float16 scale_f16 = static_cast<_Float16>(scale);
  const __m512h vscale = _mm512_set1_ph(scale_f16);
  const __m512i vzp32  = _mm512_set1_epi32(zero_point);

  size_t i = 0;
  // Twin guards on the dequant FP16-FMA fast path:
  // (i)  VCVTDQ2PH narrows the int32 difference (input - zp) to FP16
  //      (max finite |x| <= 65504). For |zp| outside the FP16
  //      representable range the s32 difference overflows to +/-Inf
  //      at the widen, then the FP16 scale multiply propagates the
  //      infinity -- the scalar / F32-FMA reference would have
  //      computed (input - zp) * scale entirely in f32 and produced
  //      a finite small number.
  // (ii) The user-supplied f32 scale is narrowed to _Float16 once
  //      before the vector loop; if |scale| > FP16_MAX (65504) the
  //      narrow saturates to +/-Inf, and lanes with input == zp
  //      compute 0 * Inf = NaN (scalar produces 0). If 0 < |scale|
  //      < FP16_MIN_NORMAL (~6.1e-5) the narrow can round to zero,
  //      and lanes whose scalar result is a representable small f16
  //      are forced to 0.
  // Either failure routes the whole call through the scalar tail,
  // which computes the chain in f32 and only narrows at the final
  // f16 store -- bit-equivalent to the F32-FMA / scalar reference.
  // The s32 zero_point and finite-scale API contracts permit values
  // outside the FP16 range, so this guard is required for
  // cross-backend agreement on user-supplied scale / zero_point.
  if (common::fp16_zp_safe_for_dequant_widen(zero_point) &&
      common::fp16_scale_safe_for_dequant_narrow(scale)) {
    for (; i + 31 < nelems; i += 32) {
      __m256i s8_vals    = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
      // Split into two 16-lane s32 halves so the zp subtract runs in int32
      // (preserves the int32 zero_point contract).
      __m512i s32_lo     = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(s8_vals));
      __m512i s32_hi     = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(s8_vals, 1));
      s32_lo             = _mm512_sub_epi32(s32_lo, vzp32);
      s32_hi             = _mm512_sub_epi32(s32_hi, vzp32);
      // VCVTDQ2PH: 16 s32 lanes -> 16 ph lanes. Concatenate the two halves
      // into a 32-lane __m512h via int-domain insert (no actual data movement
      // on AVX-512; intrinsic acts on lane semantics only).
      __m256h ph_lo      = _mm512_cvtepi32_ph(s32_lo);
      __m256h ph_hi      = _mm512_cvtepi32_ph(s32_hi);
      __m512i ph_concat  = _mm512_inserti64x4(
          _mm512_castsi256_si512(_mm256_castph_si256(ph_lo)),
          _mm256_castph_si256(ph_hi), 1);
      __m512h ph_vals    = _mm512_castsi512_ph(ph_concat);
      __m512h out_ph     = _mm512_mul_ph(ph_vals, vscale);   // * scale_f16
      _mm512_storeu_ph(output + i, out_ph);
    }
  }

  for (; i < nelems; ++i) {
    float val = (static_cast<float>(input[i]) - zero_point) * scale;
    output[i] = common::float16_t::f32_to_f16_val(val);
  }
}

//==============================================================================
// KERNEL: uint8 -> f16 (per-tensor, static dequantization, FP16-FMA)
//
// Same as the s8 path but with zero-extend on load. The (x - zp) step is
// performed in int32 to preserve the s32 zero_point contract — see the
// dequantize_int8_to_f16_avx512fp16 comment for the rationale.
//==============================================================================

__attribute__((target("avx512f,avx512vl,avx512bw,avx512fp16")))
void dequantize_uint8_to_f16_avx512fp16(const uint8_t *input, uint16_t *output,
                                         size_t nelems, float scale, int zero_point) {
  const _Float16 scale_f16 = static_cast<_Float16>(scale);
  const __m512h vscale = _mm512_set1_ph(scale_f16);
  const __m512i vzp32  = _mm512_set1_epi32(zero_point);

  size_t i = 0;
  // Twin guards: (i) |zp| outside FP16 -> VCVTDQ2PH widen produces
  // +/-Inf; (ii) |scale| outside FP16 normal range -> (_Float16)scale
  // saturates to +/-Inf (overflow) or rounds to 0 (underflow). Either
  // failure routes through the scalar tail (which keeps the chain in
  // f32). See dequantize_int8_to_f16_avx512fp16 for the full rationale
  // and pointer to the s32 zero_point / finite-scale API contracts.
  if (common::fp16_zp_safe_for_dequant_widen(zero_point) &&
      common::fp16_scale_safe_for_dequant_narrow(scale)) {
    for (; i + 31 < nelems; i += 32) {
      __m256i u8_vals    = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + i));
      __m512i s32_lo     = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(u8_vals));
      __m512i s32_hi     = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(u8_vals, 1));
      s32_lo             = _mm512_sub_epi32(s32_lo, vzp32);
      s32_hi             = _mm512_sub_epi32(s32_hi, vzp32);
      __m256h ph_lo      = _mm512_cvtepi32_ph(s32_lo);
      __m256h ph_hi      = _mm512_cvtepi32_ph(s32_hi);
      __m512i ph_concat  = _mm512_inserti64x4(
          _mm512_castsi256_si512(_mm256_castph_si256(ph_lo)),
          _mm256_castph_si256(ph_hi), 1);
      __m512h ph_vals    = _mm512_castsi512_ph(ph_concat);
      __m512h out_ph     = _mm512_mul_ph(ph_vals, vscale);
      _mm512_storeu_ph(output + i, out_ph);
    }
  }

  for (; i < nelems; ++i) {
    float val = (static_cast<float>(input[i]) - zero_point) * scale;
    output[i] = common::float16_t::f32_to_f16_val(val);
  }
}

#else  // !(GCC >= 12) — Strategy A not buildable on this toolchain
       //
       // The FP16-FMA kernels cannot be compiled (no __m512h intrinsics on
       // toolchains older than GCC 12), but we still emit symbols for the
       // declared functions so the link succeeds. Instead of leaving them as
       // no-op stubs (which would silently corrupt output if the dispatcher
       // ever mis-routed to them), delegate to the always-available F32-FMA
       // siblings in quant_dequant_per_tensor_avx512_f32_fma_kernel.cpp. The
       // can_use_f16_fma_kernel() helper returns false on this toolchain
       // (gated on __GNUC__ >= 12 in lowoha_reorder_common.hpp), so the
       // dispatcher should never select these in practice -- the delegation
       // is defense in depth.

void quantize_f16_to_int8_avx512fp16(const uint16_t *input, int8_t *output,
                                      size_t nelems, float scale, int zp) {
  quantize_f16_to_int8_avx512(input, output, nelems, scale, zp);
}
void quantize_f16_to_uint8_avx512fp16(const uint16_t *input, uint8_t *output,
                                       size_t nelems, float scale, int zp) {
  quantize_f16_to_uint8_avx512(input, output, nelems, scale, zp);
}
void dequantize_int8_to_f16_avx512fp16(const int8_t *input, uint16_t *output,
                                        size_t nelems, float scale, int zp) {
  dequantize_int8_to_f16_avx512(input, output, nelems, scale, zp);
}
void dequantize_uint8_to_f16_avx512fp16(const uint8_t *input, uint16_t *output,
                                         size_t nelems, float scale, int zp) {
  dequantize_uint8_to_f16_avx512(input, output, nelems, scale, zp);
}

#endif  // __GNUC__ >= 12

} // namespace reorder
} // namespace lowoha
} // namespace zendnnl
