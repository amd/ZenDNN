/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_ITERATOR_HPP
#define COMMON_PRIMITIVE_ITERATOR_HPP

#include "zendnn.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive_attr.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"

struct zendnn_primitive_desc_iterator : public zendnn::impl::c_compatible {
    using pd_create_f = zendnn::impl::engine_t::primitive_desc_create_f;

    zendnn_primitive_desc_iterator(zendnn::impl::engine_t *engine,
            const zendnn::impl::op_desc_t *op_desc,
            const zendnn::impl::primitive_attr_t *attr,
            const zendnn::impl::primitive_desc_t *hint_fwd_pd)
        : idx_(-1)
        , engine_(engine)
        , op_desc_(op_desc)
        , attr_(attr ? *attr : zendnn::impl::primitive_attr_t())
        , hint_fwd_pd_(hint_fwd_pd)
        , impl_list_(engine_->get_implementation_list(op_desc_))
        , last_idx_(0) {
        while (impl_list_[last_idx_] != nullptr)
            ++last_idx_;
        is_initialized_ = is_initialized_ && attr_.is_initialized();
    }

    zendnn::impl::engine_t *engine() const { return engine_; }

    bool operator==(const zendnn::impl::primitive_desc_iterator_t &rhs) const {
        return idx_ == rhs.idx_ && engine_ == rhs.engine_;
    }
    bool operator!=(const zendnn::impl::primitive_desc_iterator_t &rhs) const {
        return !operator==(rhs);
    }

    zendnn::impl::primitive_desc_iterator_t end() const {
        return zendnn_primitive_desc_iterator(engine_, last_idx_);
    }

    zendnn::impl::primitive_desc_iterator_t &operator++() {
        pd_.reset();
        while (++idx_ != last_idx_) {
            zendnn::impl::primitive_desc_t *candidate_pd = nullptr;
            auto s = impl_list_[idx_](
                    &candidate_pd, op_desc_, &attr_, engine_, hint_fwd_pd_);
            if (s == zendnn::impl::status::success) {
                pd_.reset(candidate_pd);
                break;
            }
        }
        return *this;
    }

    zendnn::impl::primitive_desc_t *operator*() const {
        if (*this == end() || pd_ == nullptr) return nullptr;
        return pd_->clone();
    }

    // Unlike `operator*()`, fetch_once() returns the current primitive
    // descriptor and invalidates the underlying primitive descriptor (but
    // saves the current position in the implementation list). Hence, the
    // subsequent calls to `operator*()` and / or `fetch_once()` will return
    // nullptr, until the iterator moves to the next implementation.
    zendnn::impl::primitive_desc_t *fetch_once() {
        if (*this == end() || pd_ == nullptr) return nullptr;
        zendnn::impl::primitive_desc_t *return_pd = pd_.release();
        return return_pd;
    }

    const zendnn::impl::primitive_attr_t &attr() const { return attr_; }

    bool is_initialized() const { return is_initialized_; }

protected:
    int idx_;
    zendnn::impl::engine_t *engine_;
    std::unique_ptr<zendnn::impl::primitive_desc_t> pd_;
    const zendnn::impl::op_desc_t *op_desc_;
    const zendnn::impl::primitive_attr_t attr_;
    const zendnn::impl::primitive_desc_t *hint_fwd_pd_;
    const pd_create_f *impl_list_;
    int last_idx_;

private:
    zendnn_primitive_desc_iterator(zendnn::impl::engine_t *engine, int last_idx)
        : idx_(last_idx)
        , engine_(engine)
        , op_desc_(nullptr)
        , hint_fwd_pd_(nullptr)
        , impl_list_(nullptr)
        , last_idx_(last_idx) {}

    zendnn_primitive_desc_iterator(zendnn_primitive_desc_iterator &&other)
        : idx_(other.idx_)
        , engine_(other.engine_)
        , pd_(std::move(other.pd_))
        , op_desc_(other.op_desc_)
        , attr_(other.attr_)
        , hint_fwd_pd_(other.hint_fwd_pd_)
        , impl_list_(other.impl_list_) {}

    ZENDNN_DISALLOW_COPY_AND_ASSIGN(zendnn_primitive_desc_iterator);
};

#endif
