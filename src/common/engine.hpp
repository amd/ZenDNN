/*******************************************************************************
* Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
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

#ifndef COMMON_ENGINE_HPP
#define COMMON_ENGINE_HPP

#include "zendnn.h"

#if ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_THREADPOOL
#include "zendnn_threadpool_iface.hpp"
#endif

#ifdef ZENDNN_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
#include "engine_id.hpp"
#endif

#include "c_types_map.hpp"
#include "memory.hpp"
#include "memory_storage.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"
#include "zendnn_logging.hpp"

/** \brief An abstraction of an execution unit with shared resources
 *
 * Responsibilities:
 *   - Provide engine specific memory allocation
 *   - Provide engine specific primitive_desc_t creators
 */
struct zendnn_engine : public zendnn::impl::c_compatible {
    zendnn_engine(zendnn::impl::engine_kind_t kind,
            zendnn::impl::runtime_kind_t runtime_kind, size_t index)
        : kind_(kind)
        , runtime_kind_(runtime_kind)
        , index_(index)
#ifdef ZENDNN_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
        , counter_(1)
#endif
    {
    }

    /** get kind of the current engine */
    zendnn::impl::engine_kind_t kind() const { return kind_; }

    /** get the runtime kind of the current engine */
    zendnn::impl::runtime_kind_t runtime_kind() const { return runtime_kind_; }

    /** get index of the current engine */
    size_t index() const { return index_; }

    virtual zendnn::impl::device_id_t device_id() const = 0;

#ifdef ZENDNN_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    virtual zendnn::impl::engine_id_t engine_id() const = 0;
#endif

    /** create memory storage */
    virtual zendnn::impl::status_t create_memory_storage(
            zendnn::impl::memory_storage_t **storage, unsigned flags, size_t size,
            void *handle)
            = 0;
    zendnn::impl::status_t create_memory_storage(
            zendnn::impl::memory_storage_t **storage, size_t size) {
        return create_memory_storage(
                storage, zendnn::impl::memory_flags_t::alloc, size, nullptr);
    }

    /** create stream */
    virtual zendnn::impl::status_t create_stream(
            zendnn::impl::stream_t **stream, unsigned flags)
            = 0;

#if ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_THREADPOOL
    virtual zendnn::impl::status_t create_stream(zendnn::impl::stream_t **stream,
            zendnn::threadpool_interop::threadpool_iface *threadpool) {
        return zendnn::impl::status::invalid_arguments;
    }
#endif

    virtual zendnn::impl::status_t get_service_stream(
            zendnn::impl::stream_t *&stream) {
        stream = nullptr;
        return zendnn::impl::status::success;
    }

    /* implementation section */

    /** return the list of reorder implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const zendnn::impl::impl_list_item_t *get_reorder_implementation_list(
            const zendnn::impl::memory_desc_t *src_md,
            const zendnn::impl::memory_desc_t *dst_md) const = 0;

    /** return the list of concat implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const zendnn::impl::impl_list_item_t *
    get_concat_implementation_list() const = 0;

    /** return the list of sum implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const zendnn::impl::impl_list_item_t *
    get_sum_implementation_list() const = 0;

    /** return the list of implementations for a given descriptor.
     * engine guarantees to return a NULL-terminated list */

    virtual const zendnn::impl::impl_list_item_t *get_implementation_list(
            const zendnn::impl::op_desc_t *desc) const = 0;

    virtual zendnn::impl::status_t serialize_device(
            zendnn::impl::serialization_stream_t &sstream) const {
        assert(!"unexpected");
        return zendnn::impl::status::runtime_error;
    }

#ifdef ZENDNN_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    void retain() { counter_++; }

    void release() {
        if (--counter_ == 0) { delete this; }
    }
#else
    virtual ~zendnn_engine() = default;
#endif

protected:
    zendnn::impl::engine_kind_t kind_;
    zendnn::impl::runtime_kind_t runtime_kind_;
    size_t index_;

#ifdef ZENDNN_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
    virtual ~zendnn_engine() = default;

private:
    std::atomic<int> counter_;
#endif
};

namespace zendnn {
namespace impl {

inline runtime_kind_t get_default_runtime(engine_kind_t kind) {
#if ZENDNN_GPU_RUNTIME == ZENDNN_RUNTIME_OCL
    if (kind == engine_kind::gpu) return runtime_kind::ocl;
#elif ZENDNN_GPU_RUNTIME == ZENDNN_RUNTIME_SYCL
    if (kind == engine_kind::gpu) return runtime_kind::sycl;
#endif
#if ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_SEQ
    return runtime_kind::seq;
#elif ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_OMP
    return runtime_kind::omp;
#elif ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_TBB
    return runtime_kind::tbb;
#elif ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_THREADPOOL
    return runtime_kind::threadpool;
#elif ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_SYCL
    return runtime_kind::sycl;
#else
    return runtime_kind::none;
#endif
}

inline runtime_kind_t get_cpu_native_runtime() {
#if ZENDNN_CPU_THREADING_RUNTIME == ZENDNN_RUNTIME_SEQ
    return runtime_kind::seq;
#elif ZENDNN_CPU_THREADING_RUNTIME == ZENDNN_RUNTIME_OMP
    return runtime_kind::omp;
#elif ZENDNN_CPU_THREADING_RUNTIME == ZENDNN_RUNTIME_TBB
    return runtime_kind::tbb;
#elif ZENDNN_CPU_THREADING_RUNTIME == ZENDNN_RUNTIME_THREADPOOL
    return runtime_kind::threadpool;
#else
    return runtime_kind::none;
#endif
}

struct engine_factory_t : public c_compatible {
    virtual size_t count() const = 0;
    virtual status_t engine_create(engine_t **engine, size_t index) const = 0;
    virtual ~engine_factory_t() = default;
};

struct engine_deleter_t {
    void operator()(engine_t *e) const {
#ifdef ZENDNN_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE
        e->release();
#else
        delete e;
#endif
    }
};

} // namespace impl
} // namespace zendnn

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
