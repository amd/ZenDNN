/*******************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#ifndef MATMUL_NATIVE_COMMON_NATIVE_INT8_SCHEME_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_INT8_SCHEME_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// How the weight scales of an INT8 call divide the weight: per tensor or
/// per output column, or per group of K rows and output column ({G > 1, N}).
enum class native_int8_scheme_t : uint8_t { per_channel, per_group };

/// A set of schemes is a mask of these bits.
inline constexpr uint8_t native_int8_scheme_bit(native_int8_scheme_t s) {
    return static_cast<uint8_t>(1u << static_cast<unsigned>(s));
}
inline constexpr uint8_t kNativeInt8AllSchemes
        = native_int8_scheme_bit(native_int8_scheme_t::per_channel)
        | native_int8_scheme_bit(native_int8_scheme_t::per_group);

/// The scheme's name in the schema and the log.
inline constexpr const char *native_int8_scheme_name(native_int8_scheme_t s) {
    return s == native_int8_scheme_t::per_group ? "per_group" : "per_channel";
}

/// The order of an INT8 call's dequantization.  fused: the kernel subtracts
/// the zero-point term, multiplies by the product of the source and weight
/// scales and adds the bias with one FMA.  reference: one rounding per
/// operation, in the reference algorithm's order.
enum class native_int8_order_t : uint8_t { fused, reference };

/// What an INT8 call does where its scheme and order give a choice.
struct native_int8_decision_t {
    native_int8_scheme_t scheme = native_int8_scheme_t::per_channel;
    int groups = 0; ///< G of a per-group call, 0 otherwise
    /// per_group: the groups are those of an {M, G} source scale, and the
    /// weight scales, per tensor or per output column, apply after the sum
    /// of the groups, with alpha and beta.
    bool scales_after_groups = false;
    native_int8_order_t order = native_int8_order_t::fused;
    /// The bias may move into the dequantization (native_int8_call_t::bias).
    bool bias_in_dequant = false;
    /// The kernel may store the destination itself when the epilogue has
    /// nothing else to do.
    bool direct_store = false;
};

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_INT8_SCHEME_HPP
