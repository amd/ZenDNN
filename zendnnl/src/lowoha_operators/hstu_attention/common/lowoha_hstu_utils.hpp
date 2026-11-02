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

/// Helpers shared by the HSTU attention kernels, GEMMs and quantizers.

#ifndef LOWOHA_HSTU_UTILS_HPP
#define LOWOHA_HSTU_UTILS_HPP

#include <cstdint>

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

/// Bits [lo, hi) of a 64-bit mask, 0 <= lo, hi <= 64.
inline uint64_t range_bits(int64_t lo, int64_t hi) {
    if (hi <= lo) return 0;
    const uint64_t upto = hi >= 64 ? ~0ull : (1ull << hi) - 1;
    return upto & ~((1ull << lo) - 1);
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl

#endif // LOWOHA_HSTU_UTILS_HPP
