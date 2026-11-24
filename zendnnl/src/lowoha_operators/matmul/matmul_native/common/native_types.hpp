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

#ifndef MATMUL_NATIVE_COMMON_NATIVE_TYPES_HPP
#define MATMUL_NATIVE_COMMON_NATIVE_TYPES_HPP

#include <cstddef>
#include <cstdint>

#include "common/data_types.hpp"

namespace zendnnl {
namespace lowoha {
namespace matmul {
namespace native {

using zendnnl::common::data_type_t;

// The ids of the weight formats and the scale types, for the kernels'
// headers, which take them without the tables (native_format.hpp).

/// How a weight is stored in the layout (native_format_desc_t).
enum class native_format_t : uint8_t { bf16, s8, s4, f32, f16 };

/// How a weight's scales are stored; the kernels and the tiles read each
/// exactly as f32.  A set of them is a mask of native_scale_bit.  Every
/// dispatch on one is a switch without a default, so a new scale type fails
/// to compile until each reads it; the call checks refuse scales of a type
/// outside their format's contract (native_contract_t::scale_types), which
/// never holds none.
enum class native_scale_t : uint8_t { none, f32, f16, bf16 };

constexpr uint8_t native_scale_bit(native_scale_t s) {
    return static_cast<uint8_t>(1u << static_cast<unsigned>(s));
}

/// Bytes of one scale of type s.
constexpr size_t native_scale_bytes(native_scale_t s) {
    // One case per switch: the tiles' code of two tests, of f32 and of none.
    switch (s) {
        case native_scale_t::f32: return 4;
        case native_scale_t::f16:
        case native_scale_t::bf16:
        case native_scale_t::none: break;
    }
    switch (s) {
        case native_scale_t::none: return 0;
        case native_scale_t::f32:
        case native_scale_t::f16:
        case native_scale_t::bf16: break;
    }
    return 2;
}

/// The scale type of scales the call gives as dt; none for a type that no
/// scale type stores.
constexpr native_scale_t native_scale_of(data_type_t dt) {
    switch (dt) {
        case data_type_t::f32: return native_scale_t::f32;
        case data_type_t::f16: return native_scale_t::f16;
        case data_type_t::bf16: return native_scale_t::bf16;
        default: return native_scale_t::none;
    }
}

/// The data type a call gives scales of type s as, native_scale_of's
/// inverse; none for none.
constexpr data_type_t native_scale_dt(native_scale_t s) {
    switch (s) {
        case native_scale_t::f32: return data_type_t::f32;
        case native_scale_t::f16: return data_type_t::f16;
        case native_scale_t::bf16: return data_type_t::bf16;
        case native_scale_t::none: break;
    }
    return data_type_t::none;
}

} // namespace native
} // namespace matmul
} // namespace lowoha
} // namespace zendnnl

#endif // MATMUL_NATIVE_COMMON_NATIVE_TYPES_HPP
