/*******************************************************************************
* Modifications Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_ATTR_HPP
#define COMMON_PRIMITIVE_ATTR_HPP

#include <map>
#include <initializer_list>

#include "zendnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace zendnn {
namespace impl {

const primitive_attr_t &default_attr();
struct runtime_scales_t;
const runtime_scales_t &default_runtime_scale();

struct rnn_data_qparams_t : public c_compatible {
    rnn_data_qparams_t() : scale_(1.), shift_(0.) {}
    bool has_default_values() const {
        return (scale_ == 1. && shift_ == 0.);
    }
    bool defined() const {
        return !is_runtime_value(scale_) && !is_runtime_value(shift_);
    }

    status_t set(float scale, float shift) {
        scale_ = scale;
        shift_ = shift;
        return status::success;
    }

    bool operator==(const rnn_data_qparams_t &rhs) const {
        using namespace utils;
        return equal_with_nan(scale_, rhs.scale_)
               && equal_with_nan(shift_, rhs.shift_);
    }

    float scale_;
    float shift_;
};

struct rnn_tparams_t : public c_compatible {
    rnn_tparams_t()
        : test_mode_(false), scales_(nullptr), ngates_(0), cscale_(0.0f) {}

    ~rnn_tparams_t() {
        test_mode_ = false;
        if (scales_ != nullptr) {
            impl::free(scales_);
        }
        scales_ = nullptr;
        ngates_ = 0;
        cscale_ = 0.0f;
    }

    bool operator==(const rnn_tparams_t &rhs) const {
        using namespace utils;

        bool ret = test_mode_ == rhs.test_mode_ && ngates_ == rhs.ngates_
                   && equal_with_nan(cscale_, rhs.cscale_);

        if (!ret) {
            return ret;
        }

        if (scales_) {
            if (std::memcmp(scales_, rhs.scales_, sizeof(float) * ngates_)) {
                return false;
            }
        }
        return true;
    }

    bool has_default_values() const {
        return (test_mode_ == false && scales_ == nullptr && ngates_ == 0
                && cscale_ == 0.0f);
    }

    status_t set(bool mode, dim_t ngates, const float *scales, float cscale) {
        test_mode_ = mode;
        ngates_ = ngates;
        scales_ = nullptr;
        if (scales != nullptr) {
            scales_ = (float *)impl::malloc(ngates_ * sizeof(*scales_), 64);
            if (scales_ == nullptr) {
                return status::out_of_memory;
            }
            utils::array_copy(scales_, scales, ngates_);
        }

        cscale_ = cscale;

        return status::success;
    }

    // copy_from() functions are used for each attribute member instead of
    // operator= in order to return a status.
    // TODO: consider replacing copy_from() functions with copy-constructors and
    // std::move, since there are only a few places in the library that actually
    // use them.
    status_t copy_from(const rnn_tparams_t &other) {
        return set(
                   other.test_mode_, other.ngates_, other.scales_, other.cscale_);
    }

    bool test_mode_; /* we could also use scale_ == nullptr as a test to check test_mode*/
    float *scales_;
    dim_t ngates_; /* ngates is equel to the number of scales */
    float cscale_; /* =0.0f if no c state */

  private:
    ZENDNN_DISALLOW_COPY_AND_ASSIGN(rnn_tparams_t);
};

struct runtime_scales_t : public c_compatible {
    runtime_scales_t() {}

    runtime_scales_t &operator=(const runtime_scales_t &rhs) {
        mask_ = rhs.mask_;
        is_set_ = rhs.is_set_;
        ndims_ = rhs.ndims_;
        if (ndims_ > 0) utils::array_copy(group_dims_, rhs.group_dims_, ndims_);
        data_type_ = rhs.data_type_;
        return *this;
    }

    status_t set(int mask) { return set(0, mask, {}, data_type::f32); }

    status_t set(int ndims, int mask, const dims_t group_dims,
            data_type_t data_type = data_type::f32) {
        mask_ = mask;
        is_set_ = true;
        ndims_ = ndims;
        if (ndims > 0) utils::array_copy(group_dims_, group_dims, ndims);
        data_type_ = data_type;
        return status::success;
    }

    bool operator==(const runtime_scales_t &rhs) const {
        return mask_ == rhs.mask_ && is_set_ == rhs.is_set_
                && ndims_ == rhs.ndims_
                && IMPLICATION(ndims_ > 0,
                        utils::array_cmp(group_dims_, rhs.group_dims_, ndims_))
                && data_type_ == rhs.data_type_;
    }

    bool has_default_values() const { return *this == default_runtime_scale(); }

    bool has_default_groups() const { return 0 == ndims_; }
    bool has_default_data_type() const { return data_type_ == data_type::f32; }

    bool defined() const { return has_default_values(); }

    void reset() { *this = default_runtime_scale(); }

    // TODO: replace with `-1` to remove `is_set_`.
    int mask_ = 0;
    bool is_set_ = false;
    int ndims_ = 0;
    dims_t group_dims_ = {};
    data_type_t data_type_ = data_type::f32;
};

struct static_scales_t : public c_compatible {
    static_scales_t() = default;

    const runtime_scales_t &get(int arg) const {
        static const runtime_scales_t default_scales;
        auto it = scales_.find(arg);
        if (it == scales_.end()) {
            return default_scales;
        }
        return it->second;
    }

    bool operator==(const static_scales_t &rhs) const {
        return scales_ == rhs.scales_;
    }

    bool has_default_values(const std::vector<int> &skip_args = {}) const {
        auto predicate = [](const runtime_scales_t &s) {
            return s.has_default_values();
        };
        return has_default_property(skip_args, predicate);
    }

    bool has_default_data_type(const std::vector<int> &skip_args = {}) const {
        auto predicate = [](const runtime_scales_t &s) {
            return s.has_default_data_type();
        };
        return has_default_property(skip_args, predicate);
    }

    bool has_default_groups(const std::vector<int> &skip_args = {}) const {
        auto predicate = [](const runtime_scales_t &s) {
            return s.has_default_groups();
        };
        return has_default_property(skip_args, predicate);
    }

    status_t set(int arg, int mask, int ndims, const dims_t group_dims,
            data_type_t data_type) {
        if (!check_arg(arg)) return status::invalid_arguments;
        return scales_[arg].set(ndims, mask, group_dims, data_type);
    }

    status_t get(int arg, int *mask, bool *is_set, int *ndims = nullptr,
            dims_t group_dims = nullptr,
            data_type_t *data_type = nullptr) const {
        if (!check_arg(arg)) return status::invalid_arguments;
        const auto &s = get(arg);
        if (mask) *mask = s.mask_;
        if (is_set) *is_set = s.is_set_;
        if (ndims) *ndims = s.ndims_;
        if (group_dims && s.ndims_ > 0)
            utils::array_copy(group_dims, s.group_dims_, s.ndims_);
        if (data_type) *data_type = s.data_type_;
        return status::success;
    }

    std::map<int, runtime_scales_t> scales_;

private:
    bool check_arg(int arg) const {
        for (const auto &sa : {ZENDNN_ARG_SRC, ZENDNN_ARG_WEIGHTS, ZENDNN_ARG_DST}) {
            if (arg == sa) return true;
        }
        return false;
    }

    bool has_default_property(const std::vector<int> &skip_args,
            bool (*predicate)(const runtime_scales_t &)) const {
        for (const auto &s : scales_) {
            if (!predicate(s.second)) {
                bool skip = false;
                for (const auto &skip_a : skip_args)
                    if (s.first == skip_a) {
                        skip = true;
                        break;
                    }
                if (skip) continue;
                return false;
            }
        }
        return true;
    }
};

struct scales_t : public c_compatible {
    scales_t() : count_(1), mask_(0), scales_(scales_buf_) {
        set(1.);
    }
    scales_t(dim_t count, int mask, const float *scales)
        : scales_(scales_buf_) {
        set(count, mask, scales);
    }

    ~scales_t() {
        cleanup();
    }

    bool operator==(const scales_t &rhs) const {
        bool ret = count_ == rhs.count_ && mask_ == rhs.mask_
                   && !utils::any_null(scales_, rhs.scales_)
                   && defined() == rhs.defined()
                   && IMPLICATION(defined(),
                                  !std::memcmp(
                                      scales_, rhs.scales_, sizeof(float) * count_));
        return ret;
    }

    bool has_default_values() const {
        for (dim_t c = 0; c < count_; ++c) {
            if (scales_[c] != 1.) {
                return false;
            }
        }
        return true;
    }

    bool defined() const {
        return !is_runtime_value(scales_[0]);
    }

    status_t set(dim_t count, int mask, const float *scales);
    status_t set(float single_scale) {
        return this->set(1, 0, &single_scale);
    }

    status_t copy_from(const scales_t &other) {
        return set(other.count_, other.mask_, other.scales_);
    }

    dim_t count_;
    int mask_;
    float *scales_;

  private:
    enum { scales_buf_size = 16 };
    float scales_buf_[scales_buf_size];

    void cleanup() {
        if (scales_ != scales_buf_ && scales_ != nullptr) {
            impl::free(scales_);
        }

        count_ = 1;
        mask_ = 0;
        scales_ = scales_buf_;
    }

    ZENDNN_DISALLOW_COPY_AND_ASSIGN(scales_t);
};

struct arg_scales_t : public c_compatible {
    arg_scales_t() {
        for (const auto &sa : {
                    ZENDNN_ARG_SRC_0, ZENDNN_ARG_SRC_1
                }) {
            set(sa, 1.f);
        }
    }

    const scales_t &get(int arg) const {
        static const scales_t default_scales;
        const auto it = scales_.find(arg);
        if (it == scales_.end()) {
            return default_scales;
        }
        return it->second;
    }

    bool operator==(const arg_scales_t &rhs) const {
        return scales_ == rhs.scales_;
    }

    bool has_default_values() const {
        for (const auto &s : scales_) {
            if (!s.second.has_default_values()) {
                return false;
            }
        }
        return true;
    }

    bool defined() const {
        for (const auto &s : scales_) {
            if (!s.second.defined()) {
                return false;
            }
        }
        return true;
    }

    status_t get(int arg, dim_t *count, int *mask, const float **scales) const;
    status_t set(int arg, dim_t count, int mask, const float *scales);
    status_t set(int arg, float single_scale) {
        return set(arg, 1, 0, &single_scale);
    }

    status_t copy_from(const arg_scales_t &other) {
        for (auto it = other.scales_.begin(); it != other.scales_.end(); ++it) {
            // Find an entry that can match the arguments without constructing a
            // new object.
            if (scales_.count(it->first) == 1) {
                auto &entry = scales_[it->first];
                bool exists = entry.count_ == it->second.count_
                              && entry.mask_ == it->second.mask_
                              && !utils::any_null(entry.scales_, it->second.scales_)
                              && !is_runtime_value(entry.scales_[0])
                              == !is_runtime_value(it->second.scales_[0])
                              && IMPLICATION(!is_runtime_value(entry.scales_[0]),
                                             utils::array_cmp(entry.scales_,
                                                     it->second.scales_, it->second.count_));

                if (exists) {
                    continue;
                }
            }

            CHECK(set(it->first, it->second.count_, it->second.mask_,
                      it->second.scales_));
        }
        return status::success;
    }

    int get_index_val(int arg) const {
        switch (arg) {
        case ZENDNN_ARG_SRC_0:
            return 0;
        case ZENDNN_ARG_SRC_1:
            return 1;
        default:
            assert(!"unsupported arg");
        }
        return -1;
    }

    std::map<int, scales_t> scales_;

  private:
    bool check_arg(int arg) const {
        for (const auto &sa : {
                    ZENDNN_ARG_SRC_0, ZENDNN_ARG_SRC_1
                }) {
            if (arg == sa) {
                return true;
            }
        }
        return false;
    }
};

struct zero_points_t : public c_compatible {
    bool operator==(const zero_points_t &rhs) const {
        auto eq = [](int a, int b) {
            return a == b || (is_runtime_value(a) && is_runtime_value(b));
        };
        return eq(zero_point_src, rhs.zero_point_src)
               && eq(mask_src, rhs.mask_src)
               && eq(zero_point_wei, rhs.zero_point_wei)
               && eq(mask_wei, rhs.mask_wei)
               && eq(zero_point_dst, rhs.zero_point_dst)
               && eq(mask_dst, rhs.mask_dst);
    }

    // arg-specific checks
    bool common(int arg) const {
        return get_mask(arg) == 0;
    }
    bool defined(int arg) const {
        return !is_runtime_value(*get(arg));
    }
    bool has_default_values(int arg) const {
        return *get(arg) == 0 && get_mask(arg) == 0;
    }

    // same checks but for all supported arguments at once
    bool common() const {
        return check_all(&zero_points_t::common);
    }
    bool defined() const {
        return check_all(&zero_points_t::defined);
    }
    bool has_default_values() const {
        return check_all(&zero_points_t::has_default_values);
    }

    const int *get(int arg) const {
        arg &= ~ZENDNN_ARG_ATTR_ZERO_POINTS;
        switch (arg) {
        case ZENDNN_ARG_SRC:
            return &zero_point_src;
        case ZENDNN_ARG_WEIGHTS:
            return &zero_point_wei;
        case ZENDNN_ARG_DST:
            return &zero_point_dst;
        }
        static int zero = 0;
        return &zero;
    }

    status_t get(
        int arg, dim_t *count, int *mask, const int **zero_points) const;

    status_t set(int arg, dim_t count, int mask, const int *zero_points);
    status_t set(int arg, int single_zero_points) {
        return set(arg, 1, 0, &single_zero_points);
    }

  private:
    // TODO: support count and mask
    int zero_point_src = 0, zero_point_wei = 0, zero_point_dst = 0;
    int mask_src = 0, mask_wei = 0, mask_dst = 0;

    int get_mask(int arg) const {
        int mask = 0;
        switch (arg) {
        case ZENDNN_ARG_SRC:
            mask = mask_src;
            break;
        case ZENDNN_ARG_WEIGHTS:
            mask = mask_wei;
            break;
        case ZENDNN_ARG_DST:
            mask = mask_dst;
            break;
        default:
            mask = 0;
        }
        return mask;
    }

    bool check_all(bool (zero_points_t::*f)(int) const) const {
        for (int arg : {
                    ZENDNN_ARG_SRC, ZENDNN_ARG_WEIGHTS, ZENDNN_ARG_DST
                })
            if (!(this->*f)(arg)) {
                return false;
            }
        return true;
    }
};

} // namespace impl
} // namespace zendnn

struct zendnn_post_ops : public zendnn::impl::c_compatible {
    struct entry_t {
        entry_t() : kind(zendnn::impl::primitive_kind::undefined) {}
        entry_t(const entry_t &other) {
            copy_from(other);
        }

        zendnn::impl::status_t copy_from(const entry_t &other) {
            clear();
            return set(other);
        }

        // TODO: This operator has to be deleted, and its usage has to be
        // replaced with copy_from() or copy/move constructors in order to
        // extract a status.
        entry_t &operator=(const entry_t &other) {
            ZENDNN_SHORT_CIRCUIT_SELF_ASSIGN(other);
            clear();
            set(other);
            return *this;
        }

        struct eltwise_t {
            zendnn::impl::alg_kind_t alg;
            float scale, alpha, beta;
        };

        struct depthwise_conv_t {
            zendnn::impl::dim_t kernel;
            zendnn::impl::dim_t stride;
            zendnn::impl::dim_t padding;
            zendnn::impl::data_type_t wei_dt;
            zendnn::impl::data_type_t bias_dt;
            zendnn::impl::data_type_t dst_dt;
            zendnn::impl::dim_t count;
            int mask;
            float *scales;
        };

        struct binary_t {
            zendnn::impl::alg_kind_t alg;
            // This is an unmodifiable user copy of attributes which is used in
            // caching mechanism. Not to be used internally.
            zendnn::impl::memory_desc_t user_src1_desc;
            // This is a modifiable copy of memory desc. It changes format kind
            // and tag of md in case user passed format_kind::any. To be used
            // everywhere internally.
            zendnn::impl::memory_desc_t src1_desc;
        };

        struct prelu_t {
            int mask;
        };

        zendnn::impl::primitive_kind_t kind
            = zendnn::impl::primitive_kind::undefined;
        union {
            struct {
                float scale;
                int32_t zero_point;
                zendnn::impl::data_type_t dt;
            } sum;
            eltwise_t eltwise;
            depthwise_conv_t depthwise_conv;
            binary_t binary;
            prelu_t prelu;
        };

        bool is_eltwise(bool require_scale_one = false) const {
            using namespace zendnn::impl;
            return kind == primitive_kind::eltwise
                   && IMPLICATION(require_scale_one, eltwise.scale == 1.f);
        }

        bool is_relu(bool require_scale_one = true,
                     bool require_nslope_zero = true) const {
            using namespace zendnn::impl;
            return is_eltwise(require_scale_one)
                   && eltwise.alg == alg_kind::eltwise_relu
                   && IMPLICATION(require_nslope_zero, eltwise.alpha == 0.f);
        }

        bool is_sum(bool require_scale_one = true,
                    bool require_zp_zero = true) const {
            using namespace zendnn::impl;
            return kind == primitive_kind::sum
                   && IMPLICATION(require_scale_one, sum.scale == 1.f)
                   && IMPLICATION(require_zp_zero, sum.zero_point == 0);
        }

        bool is_convolution() const {
            using namespace zendnn::impl;
            return kind == primitive_kind::convolution;
        }

        bool is_binary() const {
            return kind == zendnn::impl::primitive_kind::binary;
        }

        bool is_prelu() const {
            return kind == zendnn::impl::primitive_kind::prelu;
        }

        zendnn::impl::status_t set_depthwise_scales(const float *scales);

        bool operator==(const entry_t &rhs) const {
            using namespace zendnn::impl;
            using namespace zendnn::impl::utils;
            if (kind != rhs.kind) {
                return false;
            }
            bool ret = true;
            switch (kind) {
            case primitive_kind::eltwise:
                ret = eltwise.alg == rhs.eltwise.alg
                      && equal_with_nan(eltwise.scale, rhs.eltwise.scale)
                      && equal_with_nan(eltwise.alpha, rhs.eltwise.alpha)
                      && equal_with_nan(eltwise.beta, rhs.eltwise.beta);
                break;
            case primitive_kind::sum:
                ret = equal_with_nan(sum.scale, rhs.sum.scale)
                      && sum.zero_point == rhs.sum.zero_point
                      && sum.dt == rhs.sum.dt;
                break;
            case primitive_kind::convolution:
                // Depthwise Only
                ret = depthwise_conv.kernel == rhs.depthwise_conv.kernel
                      && depthwise_conv.stride
                      == rhs.depthwise_conv.stride
                      && depthwise_conv.padding
                      == rhs.depthwise_conv.padding
                      && depthwise_conv.wei_dt
                      == rhs.depthwise_conv.wei_dt
                      && depthwise_conv.bias_dt
                      == rhs.depthwise_conv.bias_dt
                      && depthwise_conv.dst_dt
                      == rhs.depthwise_conv.dst_dt
                      && depthwise_conv.count == rhs.depthwise_conv.count
                      && depthwise_conv.mask == rhs.depthwise_conv.mask;
                if (!ret) {
                    break;
                }

                // only call memcmp with valid pointers
                if (depthwise_conv.count == 0) {
                    break;
                }
                ret = !utils::any_null(depthwise_conv.scales,
                                       rhs.depthwise_conv.scales)
                      && !std::memcmp(depthwise_conv.scales,
                                      rhs.depthwise_conv.scales,
                                      sizeof(float) * depthwise_conv.count);
                break;
            case primitive_kind::binary:
                ret = binary.alg == rhs.binary.alg
                      && binary.user_src1_desc
                      == rhs.binary.user_src1_desc;
                break;
            case primitive_kind::prelu:
                ret = prelu.mask == rhs.prelu.mask;
                break;
            default:
                assert(!"unsupported post_op");
            }
            return ret;
        }

        bool operator!=(const entry_t &rhs) const {
            return !this->operator==(rhs);
        }

        ~entry_t() {
            clear();
        }

      private:
        void clear() {
            if (is_convolution() && depthwise_conv.count
                    && depthwise_conv.scales) {
                zendnn::impl::free(depthwise_conv.scales);
            }
            depthwise_conv.scales = nullptr;
            return;
        }

        zendnn::impl::status_t set(const entry_t &other) {

            // Copying by if (is_convolution()) {} else if(is_sum()) {}
            // else if(is_relu()) {} seems to be unreliable. memcpying for now.
            zendnn::impl::utils::array_copy(
                (char *)this, (char *)&other, sizeof(*this));
            if (other.is_convolution()) {
                return set_depthwise_scales(other.depthwise_conv.scales);
            }
            return zendnn::impl::status::success;
        }
    };

    zendnn_post_ops() : entry_() {}

    zendnn::impl::status_t append_sum(float scale, int32_t zero_point = 0,
                                      zendnn::impl::data_type_t dt = zendnn_data_type_undef);
    zendnn::impl::status_t append_eltwise(
        float scale, zendnn::impl::alg_kind_t alg, float alpha, float beta);
    zendnn::impl::status_t append_dw(zendnn::impl::data_type_t wei_dt,
                                     zendnn::impl::data_type_t bias_dt, zendnn::impl::data_type_t dst_dt,
                                     zendnn::impl::dim_t kernel_size, zendnn::impl::dim_t stride_size,
                                     zendnn::impl::dim_t padding_l_size, zendnn::impl::dim_t count, int mask,
                                     const float *scales);
    zendnn::impl::status_t append_binary(zendnn::impl::alg_kind_t alg,
                                         const zendnn::impl::memory_desc_t *user_src1_desc);
    zendnn::impl::status_t append_prelu(int mask);

    int find(zendnn::impl::primitive_kind_t kind, int start = 0,
             int stop = -1) const {
        if (stop == -1) {
            stop = len();
        }
        stop = zendnn::impl::nstl::min(stop, len());
        for (int idx = start; idx < stop; ++idx)
            if (entry_[idx].kind == kind) {
                return idx;
            }
        return -1;
    }

    zendnn::impl::data_type_t get_sum_dt(
        const zendnn::impl::data_type_t dst_dt) const {
        const int sum_ind = find(zendnn::impl::primitive_kind::sum);
        if (sum_ind == -1) {
            return dst_dt;
        }
        const auto sum_dt = entry_[sum_ind].sum.dt;
        if (sum_dt != zendnn::impl::data_type::undef) {
            return sum_dt;
        }
        return dst_dt;
    }

    bool defined() const;
    int len() const {
        return (int)entry_.size();
    }
    bool has_default_values() const {
        return len() == 0;
    }

    zendnn::impl::status_t set_default_formats(
        const zendnn::impl::memory_desc_t *dst_md);

    bool check_sum_consistent_dt(const zendnn::impl::data_type_t dst_dt,
                                 const bool diverse_sum_dt_allowed = false) const;

    bool sum_with_default_dt(
        zendnn::impl::data_type_t dst_dt = zendnn_data_type_undef) const {
        int sum_ind = find(zendnn::impl::primitive_kind::sum);
        return sum_ind == -1 || entry_[sum_ind].sum.dt == zendnn_data_type_undef
               || entry_[sum_ind].sum.dt == dst_dt;
    }

    bool contain(zendnn::impl::primitive_kind_t kind, int index) const {
        return find(kind, index, index + 1) == index;
    }

    bool operator==(const zendnn_post_ops &rhs) const {
        bool ret = len() == rhs.len();
        for (int i = 0; i < len(); ++i) {
            ret = ret && entry_[i] == rhs.entry_[i];
        }
        return ret;
    }

    zendnn::impl::status_t copy_from(const zendnn_post_ops &other) {
        using namespace zendnn::impl;

        for (int idx = 0; idx < other.len(); ++idx) {
            if (len() > idx) {
                if (entry_[idx] == other.entry_[idx]) {
                    continue;
                }
            }
            else {
                entry_.emplace_back();
            }
            CHECK(entry_[idx].copy_from(other.entry_[idx]));
        }

        return zendnn::impl::status::success;
    }

    std::vector<entry_t> entry_;

    // Since binary post op accepts no more than 32 memory arguments by
    // design, we limit the amount of post-ops to 32.
    static constexpr int post_ops_limit = 32;
};

struct zendnn_primitive_attr : public zendnn::impl::c_compatible {
    zendnn_primitive_attr()
        : scratchpad_mode_(zendnn::impl::scratchpad_mode::library)
        , fpmath_mode_(zendnn::impl::get_fpmath_mode())
        , autoTunerEnable(false)
        , plugin_op(""){}

    zendnn_primitive_attr *clone() const {
        return new zendnn_primitive_attr(*this);
    }

    zendnn_primitive_attr(const zendnn_primitive_attr &other) {
        if (copy_from(other) != zendnn::impl::status::success) {
            is_initialized_ = false;
        }
    }

    zendnn::impl::status_t copy_from(const zendnn_primitive_attr &other) {
        using namespace zendnn::impl;

        CHECK(output_scales_.copy_from(other.output_scales_));
        CHECK(scales_.copy_from(other.scales_));
        zero_points_ = other.zero_points_;
        scratchpad_mode_ = other.scratchpad_mode_;
        fpmath_mode_ = other.fpmath_mode_;
        CHECK(post_ops_.copy_from(other.post_ops_));
        rnn_data_qparams_ = other.rnn_data_qparams_;
        CHECK(rnn_weights_qparams_.copy_from(other.rnn_weights_qparams_));
        CHECK(rnn_weights_projection_qparams_.copy_from(
                  other.rnn_weights_projection_qparams_));
        CHECK(rnn_tparams_.copy_from(other.rnn_tparams_));
        autoTunerEnable = other.autoTunerEnable;
        computeSrcDtype = other.computeSrcDtype;
        woqScales_ = other.woqScales_;
        static_scales_ = other.static_scales_;
        plugin_op = other.plugin_op;
        return zendnn::impl::status::success;
    }

    bool is_initialized() const {
        return is_initialized_;
    }

    enum class skip_mask_t : unsigned {
        none = 0,
        oscale = 1u << 0,
        oscale_runtime = (unsigned)oscale | (1u << 1),
        scales = 1u << 2,
        scales_runtime = (unsigned)scales | (1u << 3),
        zero_points = 1u << 4,
        zero_points_runtime = (unsigned)zero_points | (1u << 5),
        post_ops = 1u << 6,
        rnn_data_qparams = 1u << 7,
        rnn_weights_qparams = 1u << 8,
        rnn_tparams = 1u << 9,
        sum_dt = 1u << 10,
        rnn_weights_projection_qparams = 1u << 11
    };

    /** Returns true if the attributes have default values.
     *
     * @note The scratchpad_mode_ is not take into account */
    bool has_default_values(skip_mask_t mask = skip_mask_t::none,
                            zendnn::impl::data_type_t dst_dt = zendnn_data_type_undef) const;

    /** Returns true if the attributes are fully defined. */
    bool defined(skip_mask_t mask = skip_mask_t::none) const;

    bool operator==(const zendnn_primitive_attr &rhs) const {
        bool ret = scratchpad_mode_ == rhs.scratchpad_mode_
                   && fpmath_mode_ == rhs.fpmath_mode_
                   && output_scales_ == rhs.output_scales_
                   && scales_ == rhs.scales_ && zero_points_ == rhs.zero_points_
                   && post_ops_ == rhs.post_ops_
                   && rnn_data_qparams_ == rhs.rnn_data_qparams_
                   && rnn_weights_qparams_ == rhs.rnn_weights_qparams_
                   && rnn_weights_projection_qparams_
                   == rhs.rnn_weights_projection_qparams_
                   && rnn_tparams_ == rhs.rnn_tparams_
                   && autoTunerEnable == rhs.autoTunerEnable
                   && computeSrcDtype == rhs.computeSrcDtype
                   && plugin_op == rhs.plugin_op
                   && woqScales_ == rhs.woqScales_
                   && static_scales_ == rhs.static_scales_;
        return ret;
    }

    zendnn::impl::status_t set_autoTunerEnable(bool autoTunerFlag);
    zendnn::impl::status_t set_computeSrcDType(zendnn::impl::data_type_t data_type);
    zendnn::impl::status_t set_plugin_op_name(const std::string plugin_op_name);
    zendnn::impl::status_t set_fpmath_mode(zendnn::impl::fpmath_mode_t fpmath_mode);
    zendnn::impl::status_t set_scratchpad_mode(
        zendnn::impl::scratchpad_mode_t scratchpad_mode);
    zendnn::impl::status_t set_post_ops(const zendnn::impl::post_ops_t &post_ops);
    zendnn::impl::status_t set_default_formats(
        const zendnn::impl::memory_desc_t *dst_md);

    /* Auxiliary functions */
    bool mayidownconvert(zendnn::impl::data_type_t dt_from,
                         zendnn::impl::data_type_t dt_to) const {
        using namespace zendnn::impl;

        bool is_compat = is_fpsubtype(dt_to, dt_from);
        auto can_downconvert = [&]() {
            switch (fpmath_mode_) {
            case zendnn::impl::fpmath_mode::strict:
                return dt_from == dt_to;
            case zendnn::impl::fpmath_mode::any:
                return true;
            case zendnn::impl::fpmath_mode::bf16:
                return is_fpsubtype(data_type::bf16, dt_to);
            case zendnn::impl::fpmath_mode::f16:
                return is_fpsubtype(data_type::f16, dt_to);
            default:
                return false;
            }
        };
        return is_compat && can_downconvert();
    }

    // NOTE: make sure that the types below have overloaded comparison operator
    zendnn::impl::scales_t output_scales_;
    zendnn::impl::arg_scales_t scales_;
    zendnn::impl::zero_points_t zero_points_;
    zendnn::impl::scratchpad_mode_t scratchpad_mode_;
    zendnn::impl::fpmath_mode_t fpmath_mode_;
    zendnn::impl::post_ops_t post_ops_;
    zendnn::impl::rnn_data_qparams_t rnn_data_qparams_;
    zendnn::impl::scales_t rnn_weights_qparams_;
    zendnn::impl::scales_t rnn_weights_projection_qparams_;
    zendnn::impl::rnn_tparams_t rnn_tparams_;
    bool autoTunerEnable;
    zendnn::impl::data_type_t computeSrcDtype;
    std::string plugin_op;
    zendnn::impl::runtime_scales_t woqScales_;
    zendnn::impl::static_scales_t static_scales_;

    zendnn_primitive_attr &operator=(const zendnn_primitive_attr &other) = delete;
};

inline zendnn_primitive_attr::skip_mask_t operator|(
    zendnn_primitive_attr::skip_mask_t lhs,
    zendnn_primitive_attr::skip_mask_t rhs) {
    return static_cast<zendnn_primitive_attr::skip_mask_t>(
               static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}
inline zendnn_primitive_attr::skip_mask_t operator&(
    zendnn_primitive_attr::skip_mask_t lhs,
    zendnn_primitive_attr::skip_mask_t rhs) {
    return static_cast<zendnn_primitive_attr::skip_mask_t>(
               static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}
inline zendnn_primitive_attr::skip_mask_t &operator|=(
    zendnn_primitive_attr::skip_mask_t &lhs,
    zendnn_primitive_attr::skip_mask_t rhs) {
    lhs = lhs | rhs;
    return lhs;
}
inline zendnn_primitive_attr::skip_mask_t &operator&=(
    zendnn_primitive_attr::skip_mask_t &lhs,
    zendnn_primitive_attr::skip_mask_t rhs) {
    lhs = lhs & rhs;
    return lhs;
}
inline bool operator!=(zendnn_primitive_attr::skip_mask_t lhs,
                       zendnn_primitive_attr::skip_mask_t rhs) {
    return (static_cast<unsigned>(lhs) != static_cast<unsigned>(rhs));
}
inline zendnn_primitive_attr::skip_mask_t operator~(
    zendnn_primitive_attr::skip_mask_t rhs) {
    return static_cast<zendnn_primitive_attr::skip_mask_t>(
               ~static_cast<unsigned>(rhs));
}

#endif
