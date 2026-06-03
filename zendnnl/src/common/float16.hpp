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
#ifndef _FLOAT16_HPP_
#define _FLOAT16_HPP_

#include <cstdint>
#include <cmath>
#include <initializer_list>
#include <type_traits>
#include <immintrin.h>

namespace zendnnl {
namespace common {

/** @union fp32fp16_t
 *  @brief Bit-level representation for float32 ↔ float16 conversion.
 *
 *  IEEE 754 half-precision (float16) layout:
 *    - 1 bit  sign
 *    - 5 bits exponent (bias 15)
 *    - 10 bits mantissa
 *
 *  IEEE 754 single-precision (float32) layout:
 *    - 1 bit  sign
 *    - 8 bits exponent (bias 127)
 *    - 23 bits mantissa
 */
union fp32fp16_t {
  /** @brief float constructor */
  fp32fp16_t(float ff) : fp32{ff} {}
  /** @brief uint16_t raw-bits constructor */
  fp32fp16_t(uint16_t hf) : u16{hf} {}

  float    fp32;  /**< float32 value */
  uint16_t u16;   /**< float16 raw bits */
  uint32_t u32;   /**< corresponding u32 value */
};

/** @class float16_t
 *  @brief Implements an IEEE 754 half-precision (float16) type.
 *
 *  Float16 is a numeric type not directly supported in C++. This class
 *  provides this numeric type using uint16_t as internal storage.
 *
 *  Conversion from other numeric types like float32 and integer types
 *  are supported.
 *
 *  Unlike bfloat16 (which shares the float32 exponent range), float16
 *  has a smaller exponent range (5 bits, bias 15) and larger mantissa
 *  (10 bits), requiring exponent rebasing during conversion.
 */
class float16_t {
 public:
  /** @name Constructors, Destructors and Assignment
   */
  /**@{*/
  /** @brief Default constructor, initializes to zero. */
  float16_t();

  /** @brief Conversion constructor from float32 to float16.
   * @param f : float32 value.
   */
  float16_t(float f);

  /** @brief Conversion assignment from float32 to float16.
   * @param f : float32 value.
   * @return A reference to converted float16 value.
   */
  float16_t &operator=(float f);

  /** @brief Conversion constructor from an integer type to float16.
   * @param i : an integer type value.
   */
  template<typename integer_type,
           typename SFINAE = std::enable_if_t<std::is_integral_v<integer_type>>>
               float16_t(integer_type i): float16_t{float(i)} {
  }

  /** @brief Conversion assignment from an integer type to float16.
   * @param i : an integer type value.
   * @return A reference to converted float16 value.
   */
  template<typename integer_type,
           typename SFINAE = std::enable_if_t<std::is_integral_v<integer_type>>>
  float16_t &operator=(integer_type i) {
    return (*this) = float16_t(i);
  }
  /**@}*/

  /** @name Conversion Operators
   */
  /**@{*/
  /** @brief Conversion from float16 to float. */
  operator float() const;

  /** @brief Conversion from float16 to int. */
  operator int()   const;
  /**@}*/

  /** @brief Get the raw 16-bit representation.
   * @return The raw uint16_t bits of this float16 value.
   */
  uint16_t raw() const {
    return raw_bits_;
  }

  /** @brief Construct a float16_t directly from its raw IEEE 754
   *         half-precision bit pattern.
   *
   *  This is a zero-cost factory: no arithmetic conversion is performed,
   *  the 16 bits are simply reinterpreted as a half-precision value.
   *  Use this when you already have the FP16 bit pattern in a uint16_t
   *  (e.g. read from a byte buffer via memcpy) and want to avoid the
   *  FP16 → FP32 → FP16 round-trip that the value-based constructor
   *  would incur.
   *
   * @param bits The raw 16-bit IEEE 754 half-precision pattern.
   * @return A float16_t whose raw() equals @p bits.
   */
  static float16_t from_bits(uint16_t bits) {
    float16_t v;
    v.raw_bits_ = bits;
    return v;
  }

  /**
   * @brief Convert float16 raw value to float32 value.
   * @param f16_val The float16 value (as uint16_t) to be converted.
   * @return The converted float32 value.
   */
  static float f16_to_f32_val(uint16_t f16_val);

  /**
   * @brief Convert float32 value to float16 value using rounding to
   *        nearest-even.
   * @param val The float32 value to be converted.
   * @return The converted float16 value as uint16_t.
   */
  static uint16_t f32_to_f16_val(float val);

  /**
   * @brief Convert a float16 buffer to float32 buffer.
   * @param f16_buf Pointer to the float16 buffer.
   * @param f32_buf Pointer to the output float32 buffer.
   * @param size_ Size of the buffer.
   */
  static void f16_to_f32_buf(const uint16_t *f16_buf, float *f32_buf,
                             int64_t size_);

  /**
   * @brief Convert a float32 buffer to float16 buffer.
   * @param f32_buf Pointer to the float32 buffer.
   * @param f16_buf Pointer to the output float16 buffer.
   * @param size_ Size of the buffer.
   */
  static void f32_to_f16(const float *f32_buf, uint16_t *f16_buf,
                         int64_t size_);

  /**
   * @brief Convert a float32 buffer to a float16_t buffer.
   *
   * Type-safe overload that writes directly into a @ref float16_t
   * destination, avoiding the strict-aliasing UB that would result
   * from casting a @c float16_t* to @c uint16_t* at the call site.
   * Each element is constructed via @ref from_bits from the converted
   * raw 16-bit pattern, so the store is a normal class assignment to
   * the member rather than a type-punning write.
   *
   * @param f32_buf Pointer to the float32 buffer.
   * @param f16_buf Pointer to the output float16_t buffer.
   * @param size_   Number of elements to convert.
   */
  static void f32_to_f16(const float *f32_buf, float16_t *f16_buf,
                         int64_t size_);

#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS) || \
    (defined(__GNUC__) && (__GNUC__ >= 12))
  //===--------------------------------------------------------------------===//
  // SIMD vector conversions (AVX-512 + AVX-512-FP16)
  //
  // These are pure register-to-register conversions: the caller is
  // responsible for the F32 load/store (with full or masked memory
  // access). Because the conversion itself is uniform across all 32
  // lanes, a single API serves both the regular (32-lane) and tail
  // (< 32 lanes) paths in vectorized kernels.
  //
  // The host CPU must support AVX-512 + AVX-512-FP16; the caller is
  // responsible for ISA dispatch before invoking these.
  //===--------------------------------------------------------------------===//

  /**
   * @brief Convert 32 float32 lanes (split as two __m512) to a single
   *        __m512h. Round mode is nearest-even, no exception bits raised.
   * @param lo Lanes 0..15 as float32.
   * @param hi Lanes 16..31 as float32.
   * @return The 32 converted float16 lanes as a __m512h.
   */
  static __m512h cvt_f32_to_f16_vec(__m512 lo, __m512 hi);

  /**
   * @brief Convert a __m512h to 32 float32 lanes split as two __m512.
   * @param val The float16 source vector.
   * @param lo  [out] Lanes 0..15 widened to float32.
   * @param hi  [out] Lanes 16..31 widened to float32.
   */
  static void cvt_f16_to_f32_vec(__m512h val, __m512 &lo, __m512 &hi);
#endif

 private:
  uint16_t raw_bits_; /*!< float16 raw bits (IEEE 754 half-precision) */
};

#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS) || \
    (defined(__GNUC__) && (__GNUC__ >= 12))
//===----------------------------------------------------------------------===//
// AVX-512-FP16 masked load/store shims
//
// Some compilers ship AVX-512-FP16 intrinsics but do not expose
// _mm512_maskz_loadu_ph and _mm512_mask_storeu_ph. Both lower to the
// same VMOVDQU16 instruction as their epi16 counterparts, so fall back
// to the epi16 form only when the native PH intrinsics are unavailable.
// Defined inline in this header so they always inline at the call site.
//===----------------------------------------------------------------------===//

__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline __m512h f16_maskz_loadu_vec(__mmask32 k, const void *addr) {
#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS)
  return _mm512_maskz_loadu_ph(k, addr);
#else
  return (__m512h)_mm512_maskz_loadu_epi16(k, addr);
#endif
}

__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline void f16_mask_storeu_vec(void *addr, __mmask32 k, __m512h val) {
#if defined(ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS)
  _mm512_mask_storeu_ph(addr, k, val);
#else
  _mm512_mask_storeu_epi16(addr, k, (__m512i)val);
#endif
}

//===----------------------------------------------------------------------===//
// AVX-512-FP16 32-lane load/store helpers
//
// Building blocks for SIMD kernels that compute in __m512h. The in-memory
// storage type (f16 or f32) is selected at compile time via the template
// parameter; the conversion (when needed) happens once at the load/store
// boundary via the float16_t::cvt_*_vec methods above, so the FMA inner
// loop stays in __m512h regardless of storage. For T = float16_t/uint16_t
// the helpers collapse to a plain PH intrinsic with no conversion
// overhead, so the all-f16 fast path pays nothing for templating.
//
// API surface (every helper is templated on the in-memory storage type):
//
//   f16x32_load_typed<T>(p)                      - 32-lane unmasked load
//   f16x32_load_mask_typed<T>(p, mask)           - 32-lane masked load
//   f16x32_load_tail_typed<T>(p, mask, tail)     - masked load + tail count
//   f16x32_store_typed<T>(p, v)                  - 32-lane unmasked store
//   f16x32_store_mask_typed<T>(p, v, mask)       - 32-lane masked store
//   f16x32_store_tail_typed<T>(p, v, mask, tail) - masked store + tail count
//
// Callers must compute the __mmask32 themselves (typically once per
// masked-tail block, then pass it to every load/store in that block).
// This avoids redundant mask construction when multiple masked operations
// share the same tail (the common case in the normalization kernels).
//
// All helpers are header-only and force-inlined; the per-function target
// attribute makes them callable from any TU without requiring a file-level
// #pragma GCC target.
//===----------------------------------------------------------------------===//

// ---- Typed load (f16 or f32 source) -------------------------------------

template <typename InType>
__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline __m512h f16x32_load_typed(const void *p) {
  if constexpr(std::is_same_v<InType, float16_t> ||
               std::is_same_v<InType, uint16_t>) {
    return _mm512_loadu_ph(p);
  }
  else {
    // InType == float
    const float *fp = static_cast<const float *>(p);
    __m512 lo = _mm512_loadu_ps(fp);
    __m512 hi = _mm512_loadu_ps(fp + 16);
    return float16_t::cvt_f32_to_f16_vec(lo, hi);
  }
}

template <typename InType>
__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline __m512h f16x32_load_tail_typed(const void *p, __mmask32 mask,
    int tail) {
  if constexpr(std::is_same_v<InType, float16_t> ||
               std::is_same_v<InType, uint16_t>) {
    return f16_maskz_loadu_vec(mask, p);
  }
  else {
    // InType == float
    const float *fp = static_cast<const float *>(p);
    __m512 lo, hi;
    if (tail <= 16) {
      __mmask16 lo_mask = (__mmask16)((1u << tail) - 1u);
      lo = _mm512_maskz_loadu_ps(lo_mask, fp);
      hi = _mm512_setzero_ps();
    }
    else {
      lo = _mm512_loadu_ps(fp);
      __mmask16 hi_mask = (__mmask16)((1u << (tail - 16)) - 1u);
      hi = _mm512_maskz_loadu_ps(hi_mask, fp + 16);
    }
    return float16_t::cvt_f32_to_f16_vec(lo, hi);
  }
}

// 32-lane masked typed load. Companion to f16x32_load_tail_typed for
// callers that already have a __mmask32 in hand and don't need an explicit
// tail count (e.g., gamma/beta tail paths in the normalization kernels).
// Splits the 32-lane mask into two 16-lane halves on the f32 path.
template <typename InType>
__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline __m512h f16x32_load_mask_typed(const void *p, __mmask32 mask) {
  if constexpr(std::is_same_v<InType, float16_t> ||
               std::is_same_v<InType, uint16_t>) {
    return f16_maskz_loadu_vec(mask, p);
  }
  else {
    // InType == float
    const float *fp = static_cast<const float *>(p);
    const __mmask16 lo_mask = (__mmask16)(mask & 0xFFFFu);
    const __mmask16 hi_mask = (__mmask16)((mask >> 16) & 0xFFFFu);
    __m512 lo = _mm512_maskz_loadu_ps(lo_mask, fp);
    __m512 hi = _mm512_maskz_loadu_ps(hi_mask, fp + 16);
    return float16_t::cvt_f32_to_f16_vec(lo, hi);
  }
}

// ---- Typed store (f16 or f32 destination) -------------------------------

template <typename OutType>
__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline void f16x32_store_typed(void *p, __m512h v) {
  if constexpr(std::is_same_v<OutType, float16_t> ||
               std::is_same_v<OutType, uint16_t>) {
    _mm512_storeu_ph(p, v);
  }
  else {
    // OutType == float
    float *fp = static_cast<float *>(p);
    __m512 lo, hi;
    float16_t::cvt_f16_to_f32_vec(v, lo, hi);
    _mm512_storeu_ps(fp,      lo);
    _mm512_storeu_ps(fp + 16, hi);
  }
}

// 32-lane masked typed store. Companion to f16x32_load_mask_typed for
// callers that already have a __mmask32 in hand and don't need an explicit
// tail count (e.g., the masked-tail residual store in FusedAddRMSNorm).
// Splits the 32-lane mask into two 16-lane halves on the f32 path.
template <typename OutType>
__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline void f16x32_store_mask_typed(void *p, __m512h v, __mmask32 mask) {
  if constexpr(std::is_same_v<OutType, float16_t> ||
               std::is_same_v<OutType, uint16_t>) {
    f16_mask_storeu_vec(p, mask, v);
  }
  else {
    // OutType == float
    float *fp = static_cast<float *>(p);
    __m512 lo, hi;
    float16_t::cvt_f16_to_f32_vec(v, lo, hi);
    const __mmask16 lo_mask = (__mmask16)(mask & 0xFFFFu);
    const __mmask16 hi_mask = (__mmask16)((mask >> 16) & 0xFFFFu);
    _mm512_mask_storeu_ps(fp,      lo_mask, lo);
    _mm512_mask_storeu_ps(fp + 16, hi_mask, hi);
  }
}

template <typename OutType>
__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline void f16x32_store_tail_typed(void *p, __m512h v, __mmask32 mask,
    int tail) {
  if constexpr(std::is_same_v<OutType, float16_t> ||
               std::is_same_v<OutType, uint16_t>) {
    f16_mask_storeu_vec(p, mask, v);
  }
  else {
    // OutType == float
    float *fp = static_cast<float *>(p);
    __m512 lo, hi;
    float16_t::cvt_f16_to_f32_vec(v, lo, hi);
    if (tail <= 16) {
      __mmask16 lo_mask = (__mmask16)((1u << tail) - 1u);
      _mm512_mask_storeu_ps(fp, lo_mask, lo);
    }
    else {
      _mm512_storeu_ps(fp, lo);
      __mmask16 hi_mask = (__mmask16)((1u << (tail - 16)) - 1u);
      _mm512_mask_storeu_ps(fp + 16, hi_mask, hi);
    }
  }
}

// ---- Horizontal reduce: 32 FP16 lanes -> 1 FP32 scalar ------------------
//
// Widens to FP32 once before the horizontal sum to avoid FP16 precision
// loss across long rows: 32 lanes added in FP16 can already lose meaningful
// bits when the row sum is large compared to per-lane values (common for
// hidden sizes >= 4 K). Costs 2 vcvtph2ps + 1 vaddps + 1 reduce per call,
// negligible against the multi-K-lane main loop that produced v.

__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline float reduce_add_ph_to_fp32(__m512h v) {
  __m256i lo16 = _mm512_castsi512_si256(_mm512_castph_si512(v));
  __m256i hi16 = _mm512_extracti64x4_epi64(_mm512_castph_si512(v), 1);
  __m512  lo32 = _mm512_cvtph_ps(lo16);
  __m512  hi32 = _mm512_cvtph_ps(hi16);
  return _mm512_reduce_add_ps(_mm512_add_ps(lo32, hi32));
}

// ---- FP16 lane classification ------------------------------------------
//
// finite_mask_ph(v) returns a __mmask32 whose bit k is 1 iff lane k of v
// is finite (not NaN, not +/-Inf). Built on _mm512_fpclass_ph_mask which
// flags the non-finite set, then inverted with knot to keep finite lanes.
// Useful for absmax / min / max reductions that must ignore non-finite
// inputs (otherwise a single NaN propagates through the whole reduction).

__attribute__((always_inline, target("avx512f,avx512vl,avx512bw,avx512fp16")))
static inline __mmask32 finite_mask_ph(__m512h v) {
  // fpclass categories: 0x01 QNaN, 0x02 +0, 0x04 -0, 0x08 +Inf,
  // 0x10 -Inf, 0x20 Denormal, 0x40 -Finite, 0x80 SNaN. Match the
  // non-finite set (NaNs + Infs) and invert to keep finite lanes.
  constexpr int kNonFinite = 0x01 | 0x08 | 0x10 | 0x80;
  __mmask32 nonfin = _mm512_fpclass_ph_mask(v, kNonFinite);
  return _knot_mask32(nonfin);
}

// ---- Safe-inv-scale guard for FP16-FMA quantize kernels ----------------
//
// FP16-FMA quantize kernels broadcast (1/scale_f32) as a __m512h and use
// VMULPH for the hot multiply. If scale_f32 is so small that 1/scale_f32
// exceeds the FP16 max normal (~65504), the conversion overflows to +Inf
// and every nonzero lane of the subsequent VMULPH saturates -- the row /
// chunk becomes garbage that bears no relation to the scalar reference.
//
// Callers should query this helper before entering the vector loop. If it
// returns false, fall back to the F32-mul scalar/tail path which keeps the
// reciprocal in f32 and stays bit-equal to the F32-FMA backend's output.
//
// 65504.0f / 2 is a conservative threshold: scale = 1/65504 ≈ 1.527e-5 is
// the smallest scale whose reciprocal still fits in FP16 normal range.
// We add a factor of 2 of headroom so an FP16-rounded inv_scale doesn't
// overflow either.

__attribute__((always_inline))
static inline bool fp16_inv_scale_is_finite(float scale_f32) {
  // 65504 is the FP16 max normal value. 1/scale ≤ 65504/2 keeps both
  // the exact and FP16-rounded reciprocal comfortably representable.
  constexpr float kMaxFiniteInvScale = 65504.0f * 0.5f;
  return scale_f32 > (1.0f / kMaxFiniteInvScale);
}

// Companion guard for FP16-FMA *asymmetric* dynamic-quant paths that
// narrow round(val/scale) to int16 (VCVTPH2W) before applying the int32
// zero point. The quotient ranges over [-zp, 255 - zp] by construction
// of the dynamic asymmetric formula. For the int16 narrow to be a no-op
// (i.e. not silently saturate before the int32 zp add), we need both
// endpoints to fit inside int16:
//   -zp     >= INT16_MIN   ->  zp <= 32767
//   255-zp  <= INT16_MAX   ->  zp >= 255 - 32767 = -32512
// Picking the tighter symmetric bound |zp| <= 32512 keeps the check
// branchless and gives ~255 lanes of headroom either side.
//
// When this returns false the kernel must fall back to the scalar tail
// (which computes the quotient in int32 from the start) for that row /
// group; otherwise the FP16-FMA output can diverge from the scalar /
// F32-FMA result by more than 1 LSB. This regime is structurally
// reachable from FP16 source data clustered far from zero, e.g.
// min=-65000, max=-64500 -> zp ≈ 33150 (outside int16).
__attribute__((always_inline))
static inline bool fp16_zp_safe_for_s16_narrow(int32_t zp) {
  constexpr int32_t kMaxZpInS16Narrow = 32512;
  return (zp >= -kMaxZpInS16Narrow) && (zp <= kMaxZpInS16Narrow);
}

// Companion guard for FP16-FMA *dequant* paths (s8/u8 -> f16) that
// convert the int32 difference (input - zp) to FP16 via VCVTDQ2PH
// before the FP16 scale multiply. The difference ranges over:
//   s8 source: [-128 - zp, 127 - zp]   ->  max |diff| <= 128 + |zp|
//   u8 source: [   0 - zp, 255 - zp]   ->  max |diff| <= 255 + |zp|
// For VCVTDQ2PH to keep the difference inside FP16 finite range
// (|x| <= 65504), the tightest bound across s8/u8 is |zp| <= 65249,
// since the u8 case has the wider input range. We pick a single
// conservative |zp| <= 65000 cap that works for both s8 and u8
// inputs and leaves ~250 of headroom for the s32 difference. When
// this returns false the kernel must fall back to the scalar tail
// (which computes (input - zp) * scale in f32 before narrowing to
// f16); otherwise VCVTDQ2PH saturates the difference to +/-Inf and
// the subsequent FP16 multiply propagates the infinity, producing
// a wrong dequantized value compared to the scalar / F32-FMA
// reference (which computes in f32 throughout). User-supplied
// static dequant is the regime that can hit this; the int32
// zero_point contract explicitly permits any int32, including
// values far outside FP16's range.
__attribute__((always_inline))
static inline bool fp16_zp_safe_for_dequant_widen(int32_t zp) {
  constexpr int32_t kMaxZpInDequantWiden = 65000;
  return (zp >= -kMaxZpInDequantWiden) && (zp <= kMaxZpInDequantWiden);
}

// Companion guard for FP16-FMA *dequant* paths (s8/u8 -> f16) that
// narrow the user-supplied f32 scale to _Float16 once before the
// vector loop. Two failure modes diverge from the scalar / F32-FMA
// reference (which keeps scale in f32 throughout):
//   (i)  |scale| > 65504    -> (_Float16)scale = +/-Inf, and any lane
//                              with (input - zp) == 0 produces
//                              0 * Inf = NaN instead of 0.
//   (ii) 0 < |scale| < FP16_MIN_NORMAL = 2^-14 (~6.1e-5)
//                           -> (_Float16)scale rounds to 0 (or a
//                              subnormal that flushes to 0 on some
//                              configurations), so lanes whose scalar
//                              result is a small representable f16
//                              are stored as 0.
// The guard returns true iff scale is exactly 0 (degenerate but
// arithmetically consistent: 0 * finite_diff = 0 on both paths) OR
// FP16_MIN_NORMAL <= |scale| <= FP16_MAX_NORMAL. When it returns
// false the kernel must fall back to the scalar tail, which computes
// (input - zp) * scale entirely in f32 before the final f16 store.
__attribute__((always_inline))
static inline bool fp16_scale_safe_for_dequant_narrow(float scale_f32) {
  constexpr float kFp16MinNormal = 6.103515625e-5f;  // 2^-14
  constexpr float kFp16MaxNormal = 65504.0f;
  if (scale_f32 == 0.0f) return true;
  const float abs_scale = scale_f32 < 0.0f ? -scale_f32 : scale_f32;
  return (abs_scale >= kFp16MinNormal) && (abs_scale <= kFp16MaxNormal);
}

#endif  // ZENDNNL_HAS_AVX512FP16_MASK_LOAD_STORE_INTRINSICS || __GNUC__ >= 12

// Narrow an f32 dynamic-quant scale to FP16 with an explicit floor at
// FP16's min positive normal (2^-14 ≈ 6.10e-5). Dynamic-quant scale
// computation clamps the f32 scale at 1e-10 to avoid division-by-zero
// in the symmetric / asymmetric formulas; but 1e-10 rounds to FP16
// zero (below even FP16's min positive subnormal 5.96e-8), which would
// break the subsequent static-quant read-back (round(val / 0) = ±Inf).
// Use this helper from any path that writes the dynamic-quant scale
// into an f16 user buffer (compute_dynamic_quant_params fall-through
// and the fused fast paths in dynamic_dispatch). The bf16 path doesn't
// need an analogous helper because BF16's min positive normal (~1.2e-38)
// easily holds 1e-10. The non-negativity of `scale` is asserted by the
// dynamic-quant formulas; the clamp is one-sided.
static inline uint16_t narrow_f32_scale_to_f16(float scale_f32) {
  constexpr float kFp16MinPosNormal = 6.103515625e-5f;  // 2^-14
  const float clamped = (scale_f32 < kFp16MinPosNormal) ? kFp16MinPosNormal
                                                        : scale_f32;
  return float16_t::f32_to_f16_val(clamped);
}

}//namespace common

namespace interface {
using float16_t = zendnnl::common::float16_t;
}//interface

}//namespace zendnnl

#endif