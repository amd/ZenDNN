/*******************************************************************************
* Modifications Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "primitive_exec_types.hpp"
#include "engine.hpp"
#include "memory.hpp"
#include "memory_storage.hpp"
#include "primitive.hpp"
#include "primitive_desc.hpp"

namespace zendnn {
namespace impl {

status_t cvt_primitive_args(const primitive_desc_t *pd, int nargs,
        const zendnn_exec_arg_t *c_args, exec_args_t &args) {
    using namespace status;

    if (!IMPLICATION(nargs > 0, c_args != nullptr)) return invalid_arguments;

    // TODO: better put extra_* in primitive_desc
    int n_inputs = 0, extra_inputs = 0;
    int n_outputs = 0, extra_outputs = 0;

    for (int i = 0; i < nargs; ++i) {
        int arg = c_args[i].arg;
        auto *mem = c_args[i].memory;

        // allows dummy arguments
        if (mem == nullptr) continue;

        switch (pd->arg_usage(arg)) {
            case primitive_desc_t::arg_usage_t::input:
                if (args.count(arg) != 0) return invalid_arguments;
                args[arg] = {mem, true};
                n_inputs++;
                extra_inputs += (arg == ZENDNN_ARG_ATTR_OUTPUT_SCALES)
                        || (arg & ZENDNN_ARG_ATTR_ZERO_POINTS)
                        || (arg & ZENDNN_ARG_ATTR_SCALES)
                        || (arg == ZENDNN_ARG_ATTR_WOQ_SCALES);
                extra_inputs += (arg & ZENDNN_ARG_ATTR_INPUT_SCALES) != 0;
                break;
            case primitive_desc_t::arg_usage_t::output:
                if (args.count(arg) != 0) return invalid_arguments;
                args[arg] = {mem, false};
                n_outputs++;
                extra_outputs += (arg == ZENDNN_ARG_SCRATCHPAD);
                break;
            case primitive_desc_t::arg_usage_t::unused: break;
        }
    }

    if (n_inputs != pd->n_inputs() + extra_inputs) return invalid_arguments;
    if (n_outputs != pd->n_outputs() + extra_outputs) return invalid_arguments;

    return success;
}

memory_t *exec_ctx_t::input(int arg) const {
    if (args_.count(arg) != 1) return nullptr;
    const auto ma = args_.at(arg);
    assert(ma.is_const);
    return ma.mem;
}

memory_t *exec_ctx_t::output(int arg) const {
    if (args_.count(arg) != 1) return nullptr;
    const auto ma = args_.at(arg);
    assert(!ma.is_const);
    return ma.mem;
}

status_t exec_ctx_t::zero_pad_output(int arg) const {
    memory_t *mem = this->output(arg);
    if (mem == nullptr) return status::success;

    return mem->zero_pad(*this);
}

memory_t *exec_ctx_t::memory(int arg) const {
    assert(args_.count(arg) == 1);
    const auto ma = args_.at(arg);
    assert(!ma.is_const);
    return ma.mem;
}

void exec_ctx_t::register_memory_mapping(void *handle, void *host_ptr) {
    assert(memory_mapping_.count(handle) == 0);
    memory_mapping_.insert({handle, host_ptr});
}

void *exec_ctx_t::host_ptr(int arg, bool do_zeropad, status_t *status_) const {
    status_t status = status::success;
    if (status_) *status_ = status;

    if (args_.count(arg) != 1) return nullptr;

    auto *mem = args_.at(arg).mem;
    if (do_zeropad) status = mem->zero_pad(*this);
    if (status_) *status_ = status;

    auto *mem_storage = mem->memory_storage();
    return host_ptr(mem_storage);
}

void *exec_ctx_t::host_ptr(const memory_storage_t *mem_storage) const {
    if (!mem_storage || mem_storage->is_null()) return nullptr;

    void *handle = mem_storage->data_handle();
    void *base_ptr = nullptr;
    if (memory_mapping_.count(handle) > 0) {
        base_ptr = memory_mapping_.at(handle);
    } else {
        assert(mem_storage->is_host_accessible());
        base_ptr = handle;
    }
    return base_ptr;
}

void *exec_ctx_t::map_memory_storage(
        const memory_storage_t *storage, stream_t *stream, size_t size) const {
    if (!storage || storage->is_null()) return nullptr;

    if (memory_mapping_.count(storage->data_handle()) > 0) {
        return host_ptr(storage);
    }

    void *mapped_ptr;
    status_t status = storage->map_data(&mapped_ptr, stream, size);
    assert(status == status::success);
    MAYBE_UNUSED(status);
    return mapped_ptr;
}

void exec_ctx_t::unmap_memory_storage(const memory_storage_t *storage,
        void *mapped_ptr, stream_t *stream) const {
    if (!storage || storage->is_null()
            || memory_mapping_.count(storage->data_handle()) > 0)
        return;

    status_t status = storage->unmap_data(mapped_ptr, stream);
    assert(status == status::success);
    MAYBE_UNUSED(status);
}

memory_desc_wrapper exec_ctx_t::memory_mdw(
        int arg, const memory_desc_t *md_from_primitive_desc) const {
    if (md_from_primitive_desc) {
        memory_desc_wrapper mdw_from_primitive_desc(md_from_primitive_desc);
        if (!mdw_from_primitive_desc.has_runtime_dims_or_strides())
            return mdw_from_primitive_desc;
    }
    if (args_.count(arg) != 1) return memory_desc_wrapper(&glob_zero_md);
    return memory_desc_wrapper(args_.at(arg).mem->md());
}

const resource_mapper_t *exec_ctx_t::get_resource_mapper() const {
    assert(resource_mapper_);
    return resource_mapper_;
}

void exec_ctx_t::set_resource_mapper(const resource_mapper_t *resource_mapper) {
    resource_mapper_ = resource_mapper;
}

} // namespace impl
} // namespace zendnn
