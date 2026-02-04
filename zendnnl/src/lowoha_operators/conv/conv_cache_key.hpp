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

#ifndef _LOWOHA_CONV_CACHE_KEY_HPP
#define _LOWOHA_CONV_CACHE_KEY_HPP

#include "common/hash_object.hpp"

namespace zendnnl {
namespace lowoha {
namespace conv {

/**
 * @brief Cache key for convolution weight reordering
 *
 * Minimal key that identifies unique weight reordering requirements.
 * Only includes parameters that directly affect OneDNN's weight memory layout.
 *
 * Note: stride/padding/dilation affect reordering indirectly via blocking_hash,
 * which captures OneDNN's algorithm selection and optimal memory format.
 */
struct Key_conv {
    // Filter pointer - identifies the actual weight tensor
    const void *filter_ptr = nullptr;

    // Dimensions that affect blocking layout
    uint64_t in_channels = 0;
    uint64_t out_channels = 0;
    uint64_t filter_height = 0;
    uint64_t filter_width = 0;

    // Data type affects memory layout (FP32 vs BF16)
    uint32_t dtype = 0;

    // OneDNN's chosen blocking format (captures algorithm + layout)
    size_t blocking_hash = 0;

    // Default constructor
    Key_conv() = default;

    // Equality operator for unordered_map
    bool operator==(const Key_conv& other) const {
        return filter_ptr == other.filter_ptr &&
               in_channels == other.in_channels &&
               out_channels == other.out_channels &&
               filter_height == other.filter_height &&
               filter_width == other.filter_width &&
               dtype == other.dtype &&
               blocking_hash == other.blocking_hash;
    }
};

} // namespace conv
} // namespace lowoha
} // namespace zendnnl

// Hash function for Key_conv
namespace std {
    template<>
    struct hash<zendnnl::lowoha::conv::Key_conv> {
        size_t operator()(const zendnnl::lowoha::conv::Key_conv& key) const {
            std::size_t seed = 0;
            seed = zendnnl::common::hash_combine(seed, key.filter_ptr);
            seed = zendnnl::common::hash_combine(seed, key.in_channels);
            seed = zendnnl::common::hash_combine(seed, key.out_channels);
            seed = zendnnl::common::hash_combine(seed, key.filter_height);
            seed = zendnnl::common::hash_combine(seed, key.filter_width);
            seed = zendnnl::common::hash_combine(seed, key.dtype);
            seed = zendnnl::common::hash_combine(seed, key.blocking_hash);
            return seed;
        }
    };
}

#endif // _LOWOHA_CONV_CACHE_KEY_HPP
