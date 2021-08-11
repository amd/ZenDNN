/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

#ifndef COMMON_MEMORY_HPP
#define COMMON_MEMORY_HPP

#include <assert.h>
#include <memory>

#include "zendnn.h"

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "memory_storage.hpp"
#include "nstl.hpp"
#include "utils.hpp"

namespace zendnn {
namespace impl {

struct exec_ctx_t;

enum memory_flags_t { alloc = 0x1, use_runtime_ptr = 0x2 };
} // namespace impl
} // namespace zendnn

struct zendnn_memory : public zendnn::impl::c_compatible {
    /** XXX: Parameter flags must contain either alloc or use_runtime_ptr from
     * memory_flags_t. */
    zendnn_memory(zendnn::impl::engine_t *engine,
            const zendnn::impl::memory_desc_t *md, unsigned flags, void *handle);
    zendnn_memory(zendnn::impl::engine_t *engine,
            const zendnn::impl::memory_desc_t *md,
            std::unique_ptr<zendnn::impl::memory_storage_t> &&memory_storage);
    virtual ~zendnn_memory() = default;

    /** returns memory's engine */
    zendnn::impl::engine_t *engine() const { return engine_; }
    /** returns memory's description */
    const zendnn::impl::memory_desc_t *md() const { return &md_; }
    /** returns the underlying memory storage */
    zendnn::impl::memory_storage_t *memory_storage() const {
        return memory_storage_.get();
    }
    /** returns the underlying memory storage */
    zendnn::impl::memory_storage_t *memory_storage_clean(
            const zendnn::impl::exec_ctx_t &ctx,
            zendnn::impl::status_t &status) const {
        status = zero_pad(ctx);
        return memory_storage_.get();
    }
    /** returns the underlying memory storage */
    zendnn::impl::memory_storage_t *memory_storage_clean(
            const zendnn::impl::exec_ctx_t &ctx) const {
        zero_pad(ctx);
        return memory_storage_.get();
    }
    /** returns data handle */
    zendnn::impl::status_t get_data_handle(void **handle) const {
        return memory_storage()->get_data_handle(handle);
    }

    /** sets data handle */
    zendnn::impl::status_t set_data_handle(void *handle, zendnn_stream *stream);

    /** zeros padding */
    zendnn::impl::status_t zero_pad(const zendnn::impl::exec_ctx_t &ctx) const;

    zendnn::impl::status_t reset_memory_storage(
            std::unique_ptr<zendnn::impl::memory_storage_t> &&memory_storage);

protected:
    zendnn::impl::engine_t *engine_;
    const zendnn::impl::memory_desc_t md_;

private:
    zendnn_memory() = delete;
    ZENDNN_DISALLOW_COPY_AND_ASSIGN(zendnn_memory);

    std::unique_ptr<zendnn::impl::memory_storage_t> memory_storage_;
};

#endif
