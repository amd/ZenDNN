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

#include "lowoha_operators/hstu_attention/tiled/lowoha_hstu_tiled_scratch.hpp"

namespace zendnnl {
namespace lowoha {
namespace hstu_attention {

arena_t &packed_arena() {
    thread_local arena_t arena;
    return arena;
}

arena_t &tile_arena() {
    thread_local arena_t arena;
    return arena;
}

} // namespace hstu_attention
} // namespace lowoha
} // namespace zendnnl
