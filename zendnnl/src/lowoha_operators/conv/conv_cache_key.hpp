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

#include <cstdint>
#include <cstddef>

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

    // Depthwise parameters affect channel grouping
    bool is_depthwise = false;
    uint32_t groups = 1;
    uint32_t depth_multiplier = 1;

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
               is_depthwise == other.is_depthwise &&
               groups == other.groups &&
               depth_multiplier == other.depth_multiplier &&
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
            size_t h = 0;
            const size_t prime = 31;

            // Hash filter pointer
            h = h * prime + std::hash<const void*>{}(key.filter_ptr);

            // Hash dimensions
            h = h * prime + std::hash<uint64_t>{}(key.in_channels);
            h = h * prime + std::hash<uint64_t>{}(key.out_channels);
            h = h * prime + std::hash<uint64_t>{}(key.filter_height);
            h = h * prime + std::hash<uint64_t>{}(key.filter_width);

            // Hash depthwise parameters
            h = h * prime + std::hash<bool>{}(key.is_depthwise);
            h = h * prime + std::hash<uint32_t>{}(key.groups);
            h = h * prime + std::hash<uint32_t>{}(key.depth_multiplier);

            // Hash dtype and blocking format
            h = h * prime + std::hash<uint32_t>{}(key.dtype);
            h = h * prime + std::hash<size_t>{}(key.blocking_hash);

            return h;
        }
    };
}

#endif // _LOWOHA_CONV_CACHE_KEY_HPP
