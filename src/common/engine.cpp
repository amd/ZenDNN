/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#include <memory>

#include "zendnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "memory.hpp"
#include "nstl.hpp"
#include "primitive.hpp"
#include "utils.hpp"

#if ZENDNN_CPU_RUNTIME != ZENDNN_RUNTIME_NONE
#include "cpu/cpu_engine.hpp"
#endif
#include "zendnn_logging.hpp"

#if ZENDNN_GPU_RUNTIME == ZENDNN_RUNTIME_OCL
#include "gpu/ocl/ocl_engine.hpp"
#endif

#ifdef ZENDNN_WITH_SYCL
#include "sycl/sycl_engine.hpp"
#endif

namespace zendnn {
namespace impl {

static inline std::unique_ptr<engine_factory_t> get_engine_factory(
        engine_kind_t kind, runtime_kind_t runtime_kind) {

#if ZENDNN_CPU_RUNTIME != ZENDNN_RUNTIME_NONE
    if (kind == engine_kind::cpu && is_native_runtime(runtime_kind)) {
        return std::unique_ptr<engine_factory_t>(
                new cpu::cpu_engine_factory_t());
    }
#endif

#if ZENDNN_GPU_RUNTIME == ZENDNN_RUNTIME_OCL
    if (kind == engine_kind::gpu && runtime_kind == runtime_kind::ocl) {
        return std::unique_ptr<engine_factory_t>(
                new gpu::ocl::ocl_engine_factory_t(kind));
    }
#endif
#ifdef ZENDNN_WITH_SYCL
    if (runtime_kind == runtime_kind::sycl)
        return sycl::get_engine_factory(kind);
#endif
    return nullptr;
}

} // namespace impl
} // namespace zendnn

using namespace zendnn;
using namespace zendnn::impl;
using namespace zendnn::impl::status;
using namespace zendnn::impl::utils;

size_t zendnn_engine_get_count(engine_kind_t kind) {
    auto ef = get_engine_factory(kind, get_default_runtime(kind));
    return ef != nullptr ? ef->count() : 0;
}

status_t zendnn_engine_create(
        engine_t **engine, engine_kind_t kind, size_t index) {
    if (engine == nullptr) return invalid_arguments;

    auto ef = get_engine_factory(kind, get_default_runtime(kind));
    if (ef == nullptr || index >= ef->count()) return invalid_arguments;

    zendnnInfo(ZENDNN_CORELOG, "CPU Engine created [engine]");
    return ef->engine_create(engine, index);
}

status_t zendnn_engine_get_kind(engine_t *engine, engine_kind_t *kind) {
    if (engine == nullptr) return invalid_arguments;
    *kind = engine->kind();
    return success;
}

status_t zendnn_engine_destroy(engine_t *engine) {
#ifdef ZENDNN_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    if (engine != nullptr) engine->release();
#else
    delete engine;
#endif
    zendnnInfo(ZENDNN_CORELOG, "CPU Engine deleted [engine]");
	return success;
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
