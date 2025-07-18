/*******************************************************************************
* Modifications Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <string>

#include <assert.h>

#include "c_types_map.hpp"
#include "engine.hpp"

#if defined(ZENDNN_ENABLE_ITT_TASKS)
    #include "ittnotify.hpp"
#endif

#include "primitive.hpp"
#include "primitive_desc.hpp"
#include "primitive_exec_types.hpp"
#include "reorder_pd.hpp"
#include "scratchpad_debug.hpp"
#include "stack_checker.hpp"
#include "stream.hpp"
#include "utils.hpp"
#include "zendnn_logging.hpp"
#include "common/zendnn_private.hpp"

#ifndef _WIN32
    #include "zendnn_perf.hpp"
    #if UPROF_ENABLE
        #include <AMDProfileController.h>
    #endif
#endif

using namespace zendnn;
using namespace zendnn::impl;
using namespace zendnn::impl::status;
using namespace zendnn::impl::primitive_kind;

namespace {
// XXX: this is a huge hammer. This disables all and any msan checks on
// primitives outputs.
//
// A proper approach would be an implementation-specific unpoisoning.
void unpoison_outputs(const exec_args_t &args) {
    for (const auto &arg : args) {
        if (arg.second.is_const) {
            continue;
        }
        auto *mem = arg.second.mem;
        void *p;
        mem->get_data_handle(&p);
        size_t s = memory_desc_wrapper(*mem->md()).size();
        msan_unpoison(p, s);
    }
}
} // namespace

namespace zendnn {
namespace impl {

nested_scratchpad_t::nested_scratchpad_t(const exec_ctx_t &master_ctx, int key,
        const std::shared_ptr<primitive_t> &nested_p) {
    auto scratchpad = master_ctx.get_scratchpad_grantor();
    scratchpad_mem_storage_ = scratchpad.get_memory_storage(key);
    grantor_ = utils::make_unique<memory_tracking::grantor_t>(
                   nested_p->pd()->scratchpad_registry().grantor(
                       scratchpad_mem_storage_.get(), master_ctx));
#ifdef ZENDNN_ENABLE_MEM_DEBUG
    if (scratchpad_debug::is_protect_scratchpad()) {
        scratchpad_debug::protect_scratchpad_buffer(
            grantor_->get_base_storage(), grantor_->get_registry());
    }
#endif
}

#ifdef ZENDNN_ENABLE_MEM_DEBUG
nested_scratchpad_t::~nested_scratchpad_t() {
    if (scratchpad_debug::is_protect_scratchpad()) {
        scratchpad_debug::unprotect_scratchpad_buffer(
            grantor_->get_base_storage(), grantor_->get_registry());
    }
}
#else
nested_scratchpad_t::~nested_scratchpad_t() = default;
#endif

status_t primitive_create(primitive_iface_t **primitive_iface,
                          const primitive_desc_iface_t *primitive_desc_iface,
                          const cache_blob_t &cache_blob = cache_blob_t()) {

    std::pair<primitive_iface_t *, bool> p_iface;

    // Default enabling of log and corresponding duration_ms calculation
    // leads to penaly for latency measurements. Environment variable
    // ZENDNN_PRIMITIVE_LOG_ENABLE can be used to optionally enable below
    // primitive log.
    if (zendnn_getenv_int("ZENDNN_PRIMITIVE_LOG_ENABLE") == 1 ||
            zendnn_getenv_int("ZENDNN_PRIMITIVE_LOG_ENABLE") == 2) {
        auto start_ms = std::chrono::high_resolution_clock::now();
        CHECK(primitive_desc_iface->create_primitive_iface(
                  p_iface, cache_blob));
        auto end_ms = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end_ms - start_ms).count();

        const char *str = p_iface.second ? "cache_hit," : "cache_miss,";
        if (cache_blob) {
            str = "from_cache_blob";
        }
        if (zendnn_getenv_int("ZENDNN_PRIMITIVE_LOG_ENABLE") == 1)
            zendnnInfo(ZENDNN_PROFLOG, "zendnn_primitive_create,",
                       str, p_iface.first->pd()->info(), ",", duration_ms,  ",ms");
        else
            zendnnInfo(ZENDNN_PERFLOG, "zendnn_primitive_create,",
                       str, p_iface.first->pd()->info(), ",", duration_ms,  ",ms");
    }

    else {
        CHECK(primitive_desc_iface->create_primitive_iface(
                  p_iface, cache_blob));
    }
    return safe_ptr_assign((*primitive_iface), p_iface.first);
}

status_t primitive_execute(
    const primitive_iface_t *primitive_iface, exec_ctx_t &ctx) {
    auto stream = ctx.stream();
    status_t status = success;

    stream->before_exec_hook();

#if defined(ZENDNN_ENABLE_ITT_TASKS)
    const bool enable_itt = itt::get_itt(itt::__itt_task_level_low);
    if (enable_itt) {
        itt::primitive_task_start(primitive_iface->pd()->impl()->kind());
    }
#endif

    // Default enabling of log and corresponding duration_ms calculation
    // amd stream wait leads to penaly for latency measurements. Environment
    // variable ZENDNN_PRIMITIVE_LOG_ENABLE can be used to optionally enable
    // below primitive execute log.
    if (zendnn_getenv_int("ZENDNN_PRIMITIVE_LOG_ENABLE") == 1) {
        stream->wait();
        auto start_ms = std::chrono::high_resolution_clock::now();
        status = stream->enqueue_primitive(primitive_iface, ctx);
        stream->wait();
        auto end_ms = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end_ms - start_ms).count();
        zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
        if (obj.is_log) {
            zendnnInfo(ZENDNN_PROFLOG, "zendnn_primitive_execute,",
                       primitive_iface->pd()->info(), ",", duration_ms, ",ms");
        }
    }
#ifndef _WIN32
    else if (zendnn_getenv_int("ZENDNN_PRIMITIVE_LOG_ENABLE") == 2) {
        stream->wait();

        //PERF PROFILE
        single_event ipc;
        single_event llc;
        ipc.open_event(event_type::IPC);
        llc.open_event(event_type::LLC_MISS_RATE);
        ipc.start_event("zendnn_primitive_execute");
        llc.start_event("zendnn_primitive_execute");

        double start_ms = get_msec();
        status = stream->enqueue_primitive(primitive_iface, ctx);
        stream->wait();
        double duration_ms = get_msec() - start_ms;

        //PERF PROFILE
        ipc.stop_event();
        llc.stop_event();
        double ipc_val=ipc.read_event();
        double llc_val=llc.read_event();
        ipc.close_event();
        llc.close_event();

        std::string stamp;
        if (get_verbose_timestamp()) {
            stamp = "," + std::to_string(start_ms);
        }
        //PERF PROFILE
        zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
        if (obj.is_log) {
            zendnnInfo(ZENDNN_PERFLOG, "zendnn_primitive_execute,",
                       primitive_iface->pd()->info(),",",duration_ms, ",ms",";",ipc.event_in,":",
                       ipc_val,";",llc.event_in,":",llc_val);
        }
    }
#if UPROF_ENABLE
    else if (zendnn_getenv_int("ZENDNN_PRIMITIVE_LOG_ENABLE") == 3) {
        stream->wait();
        auto start_ms = std::chrono::high_resolution_clock::now();
        amdProfileResume();
        status = stream->enqueue_primitive(primitive_iface, ctx);
        stream->wait();
        amdProfilePause();
        auto end_ms = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration<double, std::milli>(end_ms - start_ms).count();
        zendnnOpInfo &obj = zendnnOpInfo::ZenDNNOpInfo();
        if (obj.is_log) {
            zendnnInfo(ZENDNN_PROFLOG, "zendnn_primitive_execute,",
                       primitive_iface->pd()->info(), ",", duration_ms, ",ms");
        }
    }
#endif
#endif
    else {
        status = stream->enqueue_primitive(primitive_iface, ctx);
    }

#if defined(ZENDNN_ENABLE_ITT_TASKS)
    if (enable_itt) {
        itt::primitive_task_end();
    }
#endif

    stream->after_exec_hook();

    if (msan_enabled) {
        unpoison_outputs(ctx.args());
    }

    return status;
}

} // namespace impl
} // namespace zendnn

// API
status_t zendnn_primitive_desc_destroy(
    primitive_desc_iface_t *primitive_desc_iface) {
    delete primitive_desc_iface;
    return success;
}

status_t zendnn_primitive_create(primitive_iface_t **primitive_iface,
                                 const primitive_desc_iface_t *primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface)) {
        return invalid_arguments;
    }

#ifdef ZENDNN_ENABLE_STACK_CHECKER
    stack_checker::stack_checker_t sc("zendnn_primitive_create");
    bool is_wino = std::string(primitive_desc_iface->info()).find("wino")
                   != std::string::npos;

    if (!is_wino) {
        const cache_blob_t dummy;
        return sc.check(zendnn::impl::primitive_create, primitive_iface,
                        primitive_desc_iface, std::ref(dummy));
    }
#endif
    return zendnn::impl::primitive_create(primitive_iface, primitive_desc_iface);
}

status_t zendnn_primitive_create_from_cache_blob(
    primitive_iface_t **primitive_iface,
    const primitive_desc_iface_t *primitive_desc_iface, size_t size,
    const uint8_t *cache_blob) {
    if (utils::any_null(primitive_iface, primitive_desc_iface, cache_blob)
            || size == 0) {
        return invalid_arguments;
    }
    const auto ekind = primitive_desc_iface->engine()->kind();
    const auto runtime_kind = primitive_desc_iface->engine()->runtime_kind();
    if (ekind != engine_kind::gpu
            || (ekind == engine_kind::gpu
                && runtime_kind != runtime_kind::ocl)) {
        return status::unimplemented;
    }

    cache_blob_t cb(const_cast<uint8_t *>(cache_blob), size);
    return zendnn::impl::primitive_create(
               primitive_iface, primitive_desc_iface, cb);
}

status_t zendnn_primitive_execute(const primitive_iface_t *primitive_iface,
                                  stream_t *stream, int nargs, const zendnn_exec_arg_t *c_args) {
    bool ok = true && !utils::any_null(primitive_iface, stream)
              && primitive_iface->engine() == stream->engine()
              && IMPLICATION(nargs > 0, c_args != nullptr);
    if (!ok) {
        return invalid_arguments;
    }

    exec_args_t args;
    status_t status = cvt_primitive_args(
                          primitive_iface->pd()->impl().get(), nargs, c_args, args);
    if (status != status::success) {
        return status;
    }

    exec_ctx_t ctx(stream, std::move(args));
#ifdef ZENDNN_ENABLE_STACK_CHECKER
    stack_checker::stack_checker_t sc("zendnn_primitive_execute");
    const auto *pd_iface = primitive_iface->pd();
    bool is_wino
        = std::string(pd_iface->info()).find("wino") != std::string::npos;
    if (!is_wino) {
        return sc.check(
                   zendnn::impl::primitive_execute, primitive_iface, std::ref(ctx));
    }
#endif
    return zendnn::impl::primitive_execute(primitive_iface, ctx);
}

status_t zendnn_primitive_get_primitive_desc(
    const primitive_iface_t *primitive_iface,
    const primitive_desc_iface_t **primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface)) {
        return invalid_arguments;
    }
    return safe_ptr_assign(*primitive_desc_iface, primitive_iface->pd());
}

status_t zendnn_primitive_get_cache_blob(const primitive_iface_t
        *primitive_iface,
        size_t *size, uint8_t *cache_blob) {
    if (utils::any_null(primitive_iface, size)) {
        return status::invalid_arguments;
    }

    const auto ekind = primitive_iface->engine()->kind();
    const auto runtime_kind = primitive_iface->engine()->runtime_kind();
    if (ekind != engine_kind::gpu
            || (ekind == engine_kind::gpu
                && runtime_kind != runtime_kind::ocl)) {
        return status::unimplemented;
    }

    if (!cache_blob) {
        size_t sz = 0;
        CHECK(primitive_iface->get_cache_blob_size(&sz));
        (*size) = sz;
        return status::success;
    }

    cache_blob_t cb(cache_blob, *size);
    return primitive_iface->get_cache_blob(cb);
}

status_t zendnn_primitive_destroy(primitive_iface_t *primitive_iface) {
    if (primitive_iface != nullptr) {
        primitive_iface->release();
    }
    return success;
}

// primitive_iface_t implementation
zendnn_primitive::zendnn_primitive(
    const std::shared_ptr<primitive_t> &primitive, engine_t *engine)
    : counter_(1)
    , primitive_(primitive)
    , pd_(utils::make_unique<primitive_desc_iface_t>(
              primitive_->pd(), engine)) {}

// reorder specialization
zendnn_primitive::zendnn_primitive(const std::shared_ptr<primitive_t>
                                   &primitive,
                                   engine_t *engine, engine_t *src_engine, engine_t *dst_engine)
    : counter_(1)
    , primitive_(primitive)
    , pd_(utils::make_unique<reorder_primitive_desc_iface_t>(
              primitive_->pd(), engine, src_engine, dst_engine)) {}

zendnn_primitive::~zendnn_primitive() {
    if (scratchpad_debug::is_protect_scratchpad() && scratchpad_ != nullptr
            && scratchpad_->get_memory_storage() != nullptr) {
        const memory_tracking::registry_t &registry
            = primitive_->pd()->scratchpad_registry();
        scratchpad_debug::unprotect_scratchpad_buffer(
            scratchpad_->get_memory_storage(), registry);
    }
}

status_t zendnn_primitive::init() {
    const size_t scratchpad_size
        = primitive_->pd()->scratchpad_size(scratchpad_mode::library);

    if (scratchpad_size) {
        const memory_tracking::registry_t &registry
            = primitive_->pd()->scratchpad_registry();
        bool use_global_scratchpad = scratchpad_debug::is_protect_scratchpad()
                                     ? false
                                     : primitive_->use_global_scratchpad();
        auto *scratchpad_ptr = create_scratchpad(
                                   pd_->engine(), scratchpad_size, use_global_scratchpad);
        if (scratchpad_ptr == nullptr) {
            return out_of_memory;
        }
        if (scratchpad_ptr->get_memory_storage() == nullptr) {
            delete scratchpad_ptr;
            return out_of_memory;
        }

        if (scratchpad_debug::is_protect_scratchpad()) {
            scratchpad_debug::protect_scratchpad_buffer(
                scratchpad_ptr->get_memory_storage(), registry);
        }
        scratchpad_.reset(scratchpad_ptr);
        if (scratchpad_ptr->size() < scratchpad_size) {
            return out_of_memory;
        }
    }
    return primitive_->create_resource(pd()->engine(), resource_mapper_);
}

engine_t *zendnn_primitive::engine() const {
    return pd_->engine();
}

const primitive_desc_iface_t *zendnn_primitive::pd() const {
    return pd_.get();
}

status_t zendnn_primitive::execute(exec_ctx_t &ctx) const {
    const memory_storage_t *mem_storage = nullptr;
    if (primitive_->pd()->attr()->scratchpad_mode_ == scratchpad_mode::user) {
        memory_t *scratchpad_memory = ctx.output(ZENDNN_ARG_SCRATCHPAD);
        mem_storage = scratchpad_memory ? scratchpad_memory->memory_storage()
                      : nullptr;
    }
    else if (scratchpad_) {
        mem_storage = scratchpad_->get_memory_storage();
    }

    auto scratchpad_grantor
        = primitive_->pd()->scratchpad_registry().grantor(mem_storage, ctx);
    ctx.set_scratchpad_grantor(&scratchpad_grantor);
    ctx.set_resource_mapper(&resource_mapper_);

    auto status = primitive_->execute(ctx);
    ctx.set_scratchpad_grantor(nullptr);
    return status;
}

status_t zendnn_primitive::get_cache_blob_size(size_t *size) const {
    (*size) = 0;
    return primitive_->get_cache_blob_size(size);
}

status_t zendnn_primitive::get_cache_blob(cache_blob_t cache_blob) const {
    return primitive_->get_cache_blob(engine(), cache_blob);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
