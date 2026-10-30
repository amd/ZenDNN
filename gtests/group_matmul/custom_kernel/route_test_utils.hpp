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
 *******************************************************************************/

/// @file route_test_utils.hpp
/// @brief `ScalarRouteConfigGuard` — restores the matmul ALGO, weight
///        cache, custom-kernel route and OpenMP dynamic setting a
///        `matmul_direct` route test changes.  Shared by
///        `custom_kernel/test_dispatch_routing.cpp` and
///        `single_expert/test_single_expert_route.cpp`.

#ifndef ZENDNNL_GTESTS_GROUP_MATMUL_CUSTOM_KERNEL_ROUTE_TEST_UTILS_HPP
#define ZENDNNL_GTESTS_GROUP_MATMUL_CUSTOM_KERNEL_ROUTE_TEST_UTILS_HPP

#include <cstdint>

#include <omp.h>

#include "common/op_config.hpp"

namespace route_test_utils {

struct ScalarRouteConfigGuard {
    zendnnl::common::matmul_config_t &config
            = zendnnl::common::matmul_config_t::instance();
    int32_t old_algo = config.get_algo();
    int32_t old_cache = config.get_weight_cache();
    bool old_route = config.get_custom_kernel_route();
    int old_dynamic = omp_get_dynamic();

    ~ScalarRouteConfigGuard() {
        config.set_algo(old_algo);
        config.set_weight_cache(old_cache);
        config.set_custom_kernel_route(old_route);
        omp_set_dynamic(old_dynamic);
    }
};

} // namespace route_test_utils

#endif // ZENDNNL_GTESTS_GROUP_MATMUL_CUSTOM_KERNEL_ROUTE_TEST_UTILS_HPP
