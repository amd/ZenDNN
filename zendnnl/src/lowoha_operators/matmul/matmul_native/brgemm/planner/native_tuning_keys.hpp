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

#ifndef MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_TUNING_KEYS_HPP
#define MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_TUNING_KEYS_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "lowoha_operators/matmul/matmul_native/common/native_int8_scheme.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

/// Value type of a native_config_t key.
enum class native_key_type_t : uint8_t {
    int32,
    int64,
    boolean,
    schedule, ///< auto, slice, row_groups, blocks
    traversal, ///< m_inner, n_inner
    kc_split, ///< budget, equal
    cw_list, ///< INT8 widths such as 32/16
};

/// A key only a per-channel INT8 call applies.
inline constexpr uint8_t kNativeKeyPerChannel
        = native_int8_scheme_bit(native_int8_scheme_t::per_channel);
/// A key only a per-group INT8 call (every s4 call) applies.
inline constexpr uint8_t kNativeKeyPerGroup
        = native_int8_scheme_bit(native_int8_scheme_t::per_group);

/// One key of native_config_t, named as in the text and JSON forms.
struct native_config_key_t {
    const char *name;
    native_key_type_t type;
    int64_t min; ///< integer keys: legal values are min .. max
    int64_t max;
    int64_t step; ///< integer keys: legal values are multiples of step
    bool pow2; ///< integer keys: legal values are powers of two
    /// The INT8 schemes the key applies to, a mask of native_int8_scheme_bit;
    /// an INT8 call of another scheme does not apply the key.
    uint8_t int8_schemes = kNativeInt8AllSchemes;

    bool applies_to(native_int8_scheme_t s) const {
        return (int8_schemes & native_int8_scheme_bit(s)) != 0;
    }
};

/// Every key, in the order format_native_config writes.
const native_config_key_t *native_config_keys(size_t &count);

/// Text helpers of the config and override grammars.
std::string_view native_text_trim(std::string_view s);
bool native_text_iequals(std::string_view a, std::string_view b);
/// All of s as a decimal integer.
bool native_text_int64(std::string_view s, int64_t &v);
/// 0 / 1 / true / false / on / off, in any case.
bool native_text_bool(std::string_view s, bool &v);

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_BRGEMM_PLANNER_NATIVE_TUNING_KEYS_HPP
