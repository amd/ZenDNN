/********************************************************************************
# * Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "example_utils.hpp"

namespace zendnnl {
namespace examples {

using namespace zendnnl::memory;
using namespace zendnnl::error_handling;
using namespace zendnnl::common;

tensor_t tensor_factory_t::zero_tensor(const std::vector<index_type> size_,
                                       data_type dtype_) {

  auto ztensor = tensor_t()
    .set_name("zero tensor")
    .set_size(size_)
    .set_data_type(dtype_)
    .set_storage()
    .create();

  if (! ztensor.check()) {
    log_warning("tensor creation of ", ztensor.get_name(), " failed.");
  } else {
    auto  buf_size = ztensor.get_buffer_sz_bytes();
    void* buf_ptr  = ztensor.get_raw_handle_unsafe();
    std::memset(buf_ptr, 0, buf_size);
  }
  return ztensor;
}

tensor_t tensor_factory_t::uniform_tensor(const std::vector<index_type> size_,
                                          data_type dtype_,
                                          float val_) {

  auto utensor = tensor_t()
    .set_name("uniform tensor")
    .set_size(size_)
    .set_data_type(dtype_)
    .set_storage()
    .create();

  if (! utensor.check()) {
    log_warning("tensor creation of ", utensor.get_name(), " failed.");
  }
  else {
    auto  buf_nelem  = utensor.get_nelem();
    void* buf_vptr   = utensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float* buf_ptr = static_cast<float*>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = val_;
      }
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t* buf_ptr = static_cast<bfloat16_t*>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = bfloat16_t(val_);
      }
    }
    else if(dtype_ == data_type::s8) {
      int8_t* buf_ptr = static_cast<int8_t*>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = static_cast<int8_t>(val_);
      }
    }
    else {
      log_warning("tensor ", utensor.get_name(), " unsupported data type.");
    }
  }
  return utensor;
}

tensor_t tensor_factory_t::uniform_dist_tensor(const std::vector<index_type> size_,
                                               data_type dtype_, float range_) {

  auto udtensor = tensor_t()
    .set_name("uniform dist tensor")
    .set_size(size_)
    .set_data_type(dtype_)
    .set_storage()
    .create();

  if (! udtensor.check()) {
    log_warning("tensor creation of ", udtensor.get_name(), " failed.");
  }
  else {
    std::mt19937 gen(100);
    std::uniform_real_distribution<float> dist(-1.0 * range_, 1.0 * range_);

    auto  buf_nelem  = udtensor.get_nelem();
    void* buf_vptr   = udtensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float* buf_ptr = static_cast<float*>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&]{return dist(gen);});
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t* buf_ptr = static_cast<bfloat16_t*>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&]{return bfloat16_t(dist(gen));});
    }
    else if(dtype_ == data_type::s8) {
      int8_t* buf_ptr = static_cast<int8_t*>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&]{return int8_t(dist(gen));});
    }
    else {
      log_warning("tensor ", udtensor.get_name(), " unsupported data type.");
    }
  }
  return udtensor;
}

tensor_t tensor_factory_t::blocked_tensor(const std::vector<index_type> size_, data_type dtype_,
                                          size_t size, void* reord_buff) {

  auto btensor = tensor_t()
    .set_name("blocked tensor")
    .set_size(size_)
    .set_data_type(dtype_)
    .set_storage(reord_buff, size)
    .set_layout(tensor_layout_t::blocked)
    .create();

  if (! btensor.check()) {
    log_warning("tensor creation of ", btensor.get_name(), " failed.");
  }

  return btensor;
}

} //examples
} //zendnnl

