/********************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/
#ifndef _EMBAG_CONTEXT_HPP_
#define _EMBAG_CONTEXT_HPP_

#include <cstdint>
#include "operator_context.hpp"

namespace zendnnl {
namespace ops {

using namespace zendnnl::memory;

enum class embag_algo_t : uint8_t {
  sum = 1, mean = 2, max = 3
};

class embag_context_t final : public op_context_t<embag_context_t> {
public:
  embag_context_t();
  ~embag_context_t() = default;

  embag_context_t& set_algo(embag_algo_t algo_);
  embag_algo_t     get_algo() const;

  embag_context_t& set_padding_index(int32_t padding_index_);
  int32_t          get_padding_index() const;

  embag_context_t& set_scatter_stride(uint32_t scatter_stride_);
  uint32_t         get_scatter_stride() const;

  embag_context_t& set_scatter_offset(uint32_t scatter_offset_);
  uint32_t         get_scatter_offset() const;

  embag_context_t& set_include_last_offset(bool include_last_offset_);
  bool             get_include_last_offset() const;

private:
  embag_algo_t  algo;
  int32_t       padding_index;
  uint32_t      scatter_stride;
  uint32_t      scatter_offset;
  bool          include_last_offset;

};

//implementation
embag_context_t::embag_context_t():
  op_context_t(),
  algo{embag_algo_t::sum},
  scatter_stride{1},
  scatter_offset{0},
  padding_index{-1},
  include_last_offset{false} {
}

embag_context_t& embag_context_t::set_algo(embag_algo_t algo_) {
    LOG_DEBUG_INFO("Setting algo for embag_context_t");
    algo = algo_;
    return *this;
}

embag_algo_t embag_context_t::get_algo() const {
    LOG_DEBUG_INFO("Getting algo for embag_context_t");
    return algo;
}

embag_context_t& embag_context_t::set_padding_index(int32_t padding_index_) {
    LOG_DEBUG_INFO("Setting padding index for embag_context_t");
    padding_index = padding_index_;
    return *this;
}

int32_t embag_context_t::get_padding_index() const {
    LOG_DEBUG_INFO("Getting padding index for embag_context_t");
    return padding_index;
}

embag_context_t& embag_context_t::set_scatter_stride(uint32_t scatter_stride_) {
    LOG_DEBUG_INFO("Setting scatter_stride for embag_context_t");
    scatter_stride = scatter_stride_;
    return *this;
}

uint32_t embag_context_t::get_scatter_stride() const {
    LOG_DEBUG_INFO("Getting scatter_stride for embag_context_t");
    return scatter_stride;
}

embag_context_t& embag_context_t::set_scatter_offset(uint32_t scatter_offset_) {
    LOG_DEBUG_INFO("Setting scatter_offset for embag_context_t");
    scatter_offset = scatter_offset_;
    return *this;
}

uint32_t embag_context_t::get_scatter_offset() const {
    LOG_DEBUG_INFO("Getting scatter_offset for embag_context_t");
    return scatter_offset;
}

embag_context_t& embag_context_t::set_include_last_offset(bool include_last_offset_) {
  LOG_DEBUG_INFO("Setting include_last_offset parameter for embag_context_t");
  include_last_offset = include_last_offset_;
    return *this;
}

bool embag_context_t::get_include_last_offset() const {
  LOG_DEBUG_INFO("Getting include_last_offset parameter for embag_context_t");
    return include_last_offset;
}

} //namespace ops
} //namespace zendnnl
#endif
