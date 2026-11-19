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

// Unit tests for the OCP MX element and scale types: f4_e2m1 and f8_e8m0.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "common/data_types.hpp"
#include "memory/tensor.hpp"

namespace {

using zendnnl::common::data_type_t;
using zendnnl::common::dtype_info;
using zendnnl::common::is_4bit;
using zendnnl::common::kE2M1ToInt8x2;
using zendnnl::common::size_of;
using zendnnl::error_handling::exception_t;
using zendnnl::memory::tensor_t;

// The OCP MX v1.0 E2M1 table, decoded from the bit fields rather than copied.
float e2m1_reference(uint8_t code) {
    const int sign = (code >> 3) & 1;
    const int exp = (code >> 1) & 3;
    const int man = code & 1;
    const float mag
            = exp == 0 ? 0.5f * man : std::ldexp(1.0f + 0.5f * man, exp - 1);
    return sign ? -mag : mag;
}

TEST(MxDataTypes, StorageAndNames) {
    EXPECT_EQ(size_of(data_type_t::f4_e2m1), 1u);
    EXPECT_EQ(size_of(data_type_t::f8_e8m0), 1u);
    EXPECT_EQ(dtype_info(data_type_t::f4_e2m1), "f4_e2m1");
    EXPECT_EQ(dtype_info(data_type_t::f8_e8m0), "f8_e8m0");
}

TEST(MxDataTypes, FourBitTypesPackTwoPerByte) {
    EXPECT_TRUE(is_4bit(data_type_t::s4));
    EXPECT_TRUE(is_4bit(data_type_t::u4));
    EXPECT_TRUE(is_4bit(data_type_t::f4_e2m1));
    EXPECT_FALSE(is_4bit(data_type_t::s8));
    EXPECT_FALSE(is_4bit(data_type_t::f8_e8m0));
}

TEST(MxDataTypes, E2M1TableIsTwiceTheOcpValue) {
    for (int code = 0; code < 16; ++code) {
        EXPECT_EQ(static_cast<float>(kE2M1ToInt8x2[code]),
                2.0f * e2m1_reference(static_cast<uint8_t>(code)))
                << "code " << code;
    }
}

// An indexed raw handle of a 4-bit tensor is its pair's byte; the odd
// element of a pair has none of its own.
TEST(MxDataTypes, FourBitIndexedHandlesAddressBytes) {
    for (const data_type_t dt :
            {data_type_t::f4_e2m1, data_type_t::s4, data_type_t::u4}) {
        std::vector<uint8_t> bytes(32);
        tensor_t t;
        t.set_name("four_bit")
                .set_size({4, 16})
                .set_data_type(dt)
                .set_storage(bytes.data(), bytes.size())
                .create();
        ASSERT_TRUE(t.check()) << dtype_info(dt);
        EXPECT_EQ(t.get_raw_handle_const({3, 14}),
                static_cast<const void *>(bytes.data() + 31))
                << dtype_info(dt);
        EXPECT_EQ(t.get_raw_handle_unsafe({1, 2}),
                static_cast<void *>(bytes.data() + 9))
                << dtype_info(dt);
        EXPECT_THROW(t.get_raw_handle_const({3, 15}), exception_t)
                << dtype_info(dt);
        EXPECT_THROW(t.get_raw_handle_unsafe({0, 1}), exception_t)
                << dtype_info(dt);
    }
}

} // namespace
