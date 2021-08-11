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

#ifndef COMMON_STREAM_HPP
#define COMMON_STREAM_HPP

#include <assert.h>
#include "zendnn.h"
#include "zendnn_threadpool_iface.hpp"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "utils.hpp"

struct zendnn_stream : public zendnn::impl::c_compatible {
    zendnn_stream(zendnn::impl::engine_t *engine, unsigned flags)
        : engine_(engine), flags_(flags) {}
    virtual ~zendnn_stream() {}

    /** returns stream's engine */
    zendnn::impl::engine_t *engine() const { return engine_; }
    template <typename tgt_engine_t>
    tgt_engine_t *engine() const {
        return zendnn::impl::utils::downcast<tgt_engine_t *>(engine_);
    }

    /** returns stream's kind */
    unsigned flags() const { return flags_; }

    virtual zendnn::impl::status_t enqueue_primitive(
            const primitive_iface_t *primitive_iface,
            zendnn::impl::exec_ctx_t &ctx);

    /** blocks until all submitted primitives to the stream are completed */
    virtual zendnn::impl::status_t wait() = 0;

    virtual void before_exec_hook() {}
    virtual void after_exec_hook() {}

    virtual zendnn::impl::status_t zero_pad(const zendnn::impl::memory_t *memory,
            const zendnn::impl::exec_ctx_t &ctx);

#if ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_THREADPOOL
    zendnn_stream(zendnn::impl::engine_t *engine,
            zendnn::threadpool_interop::threadpool_iface *threadpool)
        : zendnn_stream(engine, zendnn::impl::stream_flags::in_order) {
        assert(engine->kind() == zendnn::impl::engine_kind::cpu);
        threadpool_ = threadpool;
    }

    zendnn::impl::status_t get_threadpool(
            zendnn::threadpool_interop::threadpool_iface **threadpool) const {
        using namespace zendnn::impl;
        if (engine_->kind() != engine_kind::cpu)
            return status::invalid_arguments;
        *threadpool = threadpool_;
        return status::success;
    }
#endif

protected:
    zendnn::impl::engine_t *engine_;
    unsigned flags_;
#if ZENDNN_CPU_RUNTIME == ZENDNN_RUNTIME_THREADPOOL
    zendnn::threadpool_interop::threadpool_iface *threadpool_ = nullptr;
#endif
};

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
