/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
        : kind_(kind), runtime_kind_(runtime_kind), index_(index) {}
    virtual ~zendnn_engine() = default;

    /** get kind of the current engine */
    zendnn::impl::engine_kind_t kind() const { return kind_; }

    /** get the runtime kind of the current engine */
    zendnn::impl::runtime_kind_t runtime_kind() const { return runtime_kind_; }

    /** get index of the current engine */
    size_t index() const { return index_; }

    virtual zendnn::impl::device_id_t device_id() const = 0;

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
    /** implementation section (typedefs) */

    // TODO: remove engine?
    typedef zendnn::impl::status_t (*reorder_primitive_desc_create_f)(
            zendnn::impl::reorder_pd_t **, zendnn::impl::engine_t *engine,
            const zendnn::impl::primitive_attr_t *attr,
            zendnn::impl::engine_t *src_engine,
            const zendnn::impl::memory_desc_t *src_md,
            zendnn::impl::engine_t *dst_engine,
            const zendnn::impl::memory_desc_t *dst_md);

    typedef zendnn::impl::status_t (*concat_primitive_desc_create_f)(
            zendnn::impl::concat_pd_t **, zendnn::impl::engine_t *engine,
            const zendnn::impl::primitive_attr_t *attr,
            const zendnn::impl::memory_desc_t *dst_md, int n, int concat_dim,
            const zendnn::impl::memory_desc_t *src_mds);

    typedef zendnn::impl::status_t (*sum_primitive_desc_create_f)(
            zendnn::impl::sum_pd_t **, zendnn::impl::engine_t *engine,
            const zendnn::impl::primitive_attr_t *attr,
            const zendnn::impl::memory_desc_t *dst_md, int n, const float *scales,
            const zendnn::impl::memory_desc_t *src_mds);

    typedef zendnn::impl::status_t (*primitive_desc_create_f)(
            zendnn::impl::primitive_desc_t **, const zendnn::impl::op_desc_t *,
            const zendnn::impl::primitive_attr_t *attr, zendnn::impl::engine_t *,
            const zendnn::impl::primitive_desc_t *);

    /* implementation section */

    /** return the list of reorder implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const reorder_primitive_desc_create_f *
    get_reorder_implementation_list(const zendnn::impl::memory_desc_t *src_md,
            const zendnn::impl::memory_desc_t *dst_md) const = 0;

    /** return the list of concat implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const concat_primitive_desc_create_f *
    get_concat_implementation_list() const = 0;

    /** return the list of sum implementations. engine guarantees to return
     * a NULL-terminated list */
    virtual const sum_primitive_desc_create_f *
    get_sum_implementation_list() const = 0;

    /** return the list of implementations for a given descriptor.
     * engine guarantees to return a NULL-terminated list */
    virtual const primitive_desc_create_f *get_implementation_list(
            const zendnn::impl::op_desc_t *desc) const = 0;

protected:
    zendnn::impl::engine_kind_t kind_;
    zendnn::impl::runtime_kind_t runtime_kind_;
    size_t index_;
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

inline bool is_native_runtime(runtime_kind_t kind) {
    return utils::one_of(kind, runtime_kind::seq, runtime_kind::omp,
            runtime_kind::tbb, runtime_kind::threadpool);
}

struct engine_factory_t : public c_compatible {
    virtual size_t count() const = 0;
    virtual status_t engine_create(engine_t **engine, size_t index) const = 0;
    virtual ~engine_factory_t() = default;
};

} // namespace impl
} // namespace zendnn

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
