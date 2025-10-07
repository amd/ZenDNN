/********************************************************************************
# * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

using namespace zendnnl::interface;

tensor_t tensor_factory_t::uniform_dist_strided_tensor(const
    std::vector<index_type> size_, const std::vector<index_type> aligned_size_,
    data_type dtype_, float range_, std::string tensor_name_,
    tensor_t scale, tensor_t zp) {
  auto udstensor = tensor_t()
                   .set_name(tensor_name_)
                   .set_size(size_)
                   .set_data_type(dtype_)
                   .set_aligned_size(aligned_size_)
                   .set_storage();

  if (scale.get_nelem() != 0) {
    udstensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    udstensor.set_quant_zero_point(zp);
  }

  udstensor.create();

  if (! udstensor.check()) {
    log_warning("tensor creation of ", udstensor.get_name(), " failed.");
  }
  else {
    std::mt19937 gen(100);
    std::uniform_real_distribution<float> dist(-1.0 * range_, 1.0 * range_);

    auto  buf_nelem   = aligned_size_[0];
    for (size_t i = 1; i < aligned_size_.size(); i++) {
      buf_nelem *= aligned_size_[i];
    }
    void *buf_vptr    = udstensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return dist(gen);});
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return bfloat16_t(dist(gen));});
    }
    else if (dtype_ == data_type::s8) {
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return int8_t(dist(gen));});
    }
    else {
      log_warning("tensor ", udstensor.get_name(), " unsupported data type.");
    }
  }
  return udstensor;
}

tensor_t tensor_factory_t::zero_tensor(const std::vector<index_type> size_,
                                       data_type dtype_, std::string tensor_name_,
                                       tensor_t scale, tensor_t zp) {

  auto ztensor = tensor_t()
                 .set_name(tensor_name_)
                 .set_size(size_)
                 .set_data_type(dtype_)
                 .set_storage();

  if (scale.get_nelem() != 0) {
    ztensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    ztensor.set_quant_zero_point(zp);
  }

  ztensor.create();

  if (! ztensor.check()) {
    log_warning("tensor creation of ", ztensor.get_name(), " failed.");
  }
  else {
    auto  buf_size = ztensor.get_buffer_sz_bytes();
    void *buf_ptr  = ztensor.get_raw_handle_unsafe();
    std::memset(buf_ptr, 0, buf_size);
  }
  return ztensor;
}

tensor_t tensor_factory_t::uniform_tensor(const std::vector<index_type> size_,
    data_type dtype_, float val_,
    std::string tensor_name_, tensor_t scale,
    tensor_t zp) {

  auto utensor = tensor_t()
                 .set_name(tensor_name_)
                 .set_size(size_)
                 .set_data_type(dtype_)
                 .set_storage();

  if (scale.get_nelem() != 0) {
    utensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    utensor.set_quant_zero_point(zp);
  }

  utensor.create();
  if (! utensor.check()) {
    log_warning("tensor creation of ", utensor.get_name(), " failed.");
  }
  else {
    auto  buf_nelem  = utensor.get_nelem();
    void *buf_vptr   = utensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = val_;
      }
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = bfloat16_t(val_);
      }
    }
    else if (dtype_ == data_type::s8) {
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
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

tensor_t tensor_factory_t::broadcast_uniform_tensor(const
    std::vector<index_type> size_,
    const std::vector<index_type> stride_, data_type dtype_, float val_,
    std::string tensor_name_, tensor_t scale, tensor_t zp) {

  auto utensor = tensor_t()
                 .set_name(tensor_name_)
                 .set_size(size_)
                 .set_stride(stride_)
                 .set_data_type(dtype_)
                 .set_storage();

  if (scale.get_nelem() != 0) {
    utensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    utensor.set_quant_zero_point(zp);
  }

  utensor.create();

  if (! utensor.check()) {
    log_warning("tensor creation of ", utensor.get_name(), " failed.");
  }
  else {
    auto  buf_nelem  = utensor.get_nelem();
    void *buf_vptr   = utensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = val_;
      }
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = bfloat16_t(val_);
      }
    }
    else if (dtype_ == data_type::s8) {
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
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

tensor_t tensor_factory_t::non_uniform_tensor(const std::vector<index_type>
    size_,
    data_type dtype_, std::vector<int64_t> val_,
    std::string tensor_name_, tensor_t scale, tensor_t zp) {

  auto utensor = tensor_t()
                 .set_name(tensor_name_)
                 .set_size(size_)
                 .set_data_type(dtype_)
                 .set_storage();

  if (scale.get_nelem() != 0) {
    utensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    utensor.set_quant_zero_point(zp);
  }

  utensor.create();

  if (! utensor.check()) {
    log_warning("tensor creation of ", utensor.get_name(), " failed.");
  }
  else {
    auto  buf_nelem  = utensor.get_nelem();
    void *buf_vptr   = utensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::s64) {
      int64_t *buf_ptr = static_cast<int64_t *>(buf_vptr);
      for (index_type i = 0; i < buf_nelem; ++i) {
        buf_ptr[i] = val_[i];
      }
    }
    else {
      log_warning("tensor ", utensor.get_name(), " unsupported data type.");
    }
  }
  return utensor;
}

tensor_t tensor_factory_t::uniform_dist_tensor(const std::vector<index_type>
    size_, data_type dtype_, float range_, std::string tensor_name_, bool trans,
    tensor_t scale, tensor_t zp) {
  auto udtensor = tensor_t()
                  .set_name(tensor_name_)
                  .set_size(size_)
                  .set_data_type(dtype_)
                  .set_storage();

  if (scale.get_nelem() != 0) {
    udtensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    udtensor.set_quant_zero_point(zp);
  }

  auto tensor_dim = udtensor.get_dim();
  if (trans && tensor_dim >=2) {
    std::string tag;
    for (size_t i=0; i<tensor_dim; ++i) {
      tag += 'a' + i;
    }
    std::swap(tag[size_.size() - 2], tag[size_.size() - 1]);
    udtensor.set_order(tag);
  }
  udtensor.create();

  if (! udtensor.check()) {
    log_warning("tensor creation of ", udtensor.get_name(), " failed.");
  }
  else {
    std::mt19937 gen(100);
    std::uniform_real_distribution<float> dist(-1.0 * range_, 1.0 * range_);

    auto  buf_nelem  = udtensor.get_nelem();
    void *buf_vptr   = udtensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return dist(gen);});
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return bfloat16_t(dist(gen));});
    }
    else if (dtype_ == data_type::s8) {
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return int8_t(dist(gen));});
    }
    else {
      log_warning("tensor ", udtensor.get_name(), " unsupported data type.");
    }
  }
  return udtensor;
}

tensor_t tensor_factory_t::blocked_tensor(const std::vector<index_type> size_,
    data_type dtype_, float range_, std::string tensor_name_,
    tensor_t scale, tensor_t zp) {

  auto btensor = tensor_t()
                 .set_name(tensor_name_)
                 .set_size(size_)
                 .set_data_type(dtype_)
                 .set_layout(tensor_layout_t::blocked);

  if (scale.get_nelem() != 0) {
    btensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    btensor.set_quant_zero_point(zp);
  }

  btensor.set_storage().create();

  if (! btensor.check()) {
    log_warning("tensor creation of ", btensor.get_name(), " failed.");
  }
  else {
    std::mt19937 gen(100);
    std::uniform_real_distribution<float> dist(-1.0 * range_, 1.0 * range_);

    auto  buf_nelem  = btensor.get_nelem();
    void *buf_vptr   = btensor.get_raw_handle_unsafe();

    if (dtype_ == data_type::f32) {
      float *buf_ptr = static_cast<float *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return dist(gen);});
    }
    else if (dtype_ == data_type::bf16) {
      bfloat16_t *buf_ptr = static_cast<bfloat16_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return bfloat16_t(dist(gen));});
    }
    else if (dtype_ == data_type::s8) {
      int8_t *buf_ptr = static_cast<int8_t *>(buf_vptr);
      std::generate(buf_ptr, buf_ptr+buf_nelem, [&] {return int8_t(dist(gen));});
    }
    else {
      log_warning("tensor ", btensor.get_name(), " unsupported data type.");
    }
  }
  return btensor;
}

tensor_t tensor_factory_t::copy_tensor(const std::vector<index_type> size_,
                                       data_type dtype_, StorageParam param, bool trans,
                                       bool is_blocked, std::string tensor_name_,
                                       tensor_t scale, tensor_t zp) {

  auto ctensor = tensor_t()
                 .set_name(tensor_name_)
                 .set_size(size_)
                 .set_data_type(dtype_);

  auto tensor_dim = ctensor.get_dim();
  if (trans && tensor_dim >=2) {
    std::string tag;
    for (size_t i=0; i<tensor_dim; ++i) {
      tag += 'a' + i;
    }
    std::swap(tag[size_.size() - 2], tag[size_.size() - 1]);
    ctensor.set_order(tag);
  }

  if (scale.get_nelem() != 0) {
    ctensor.set_quant_scale(scale);
  }
  if (zp.get_nelem() != 0) {
    ctensor.set_quant_zero_point(zp);
  }

  if (is_blocked) {
    ctensor.set_layout(tensor_layout_t::blocked);
  }

  if (std::holds_alternative<std::pair<size_t, void *>>(param)) {
    auto [reorder_size, reorder_buff] = std::get<std::pair<size_t, void *>>(param);
    ctensor.set_storage(reorder_buff, reorder_size);
  }
  else if (std::holds_alternative<tensor_t>(param)) {
    tensor_t input_tensor = std::get<tensor_t>(param);
    ctensor.set_storage(input_tensor);
  }
  ctensor.create();

  if (! ctensor.check()) {
    log_warning("tensor creation of ", ctensor.get_name(), " failed.");
  }
  return ctensor;
}

tensor_t tensor_factory_t::random_indices_tensor(const std::vector<index_type>
    size_,
    uint64_t num_embeddings) {
  auto indices_tensor = tensor_t()
                        .set_name("indices_tensor")
                        .set_size(size_)
                        .set_data_type(data_type_t::s32)
                        .set_storage()
                        .create();
  void *data = indices_tensor.get_raw_handle_unsafe();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dist(0,
      static_cast<uint32_t>(num_embeddings - 1));

  int num_indices = size_[0];

  for (int i = 0; i < num_indices; ++i) {
    static_cast<uint32_t *>(data)[i] = dist(gen);
  }

  return indices_tensor;
}

tensor_t tensor_factory_t::random_offsets_tensor(const std::vector<index_type>
    size_,
    uint64_t num_indices,
    bool include_last_offset) {
  auto tensor = tensor_t()
                .set_name("offsets_tensor")
                .set_size(size_)
                .set_data_type(data_type_t::s32)
                .set_storage()
                .create();
  void *data = tensor.get_raw_handle_unsafe();

  int num_offsets = size_[0];
  if (include_last_offset) {
    num_offsets--;
  }

  for (int i = 0; i < num_offsets; ++i) {
    static_cast<uint32_t *>(data)[i] = (i * num_indices) / num_offsets;
  }

  if (include_last_offset) {
    static_cast<uint32_t *>(data)[num_offsets] = num_indices;
  }

  return tensor;
}

void tensor_functions_t::tensor_pretty_print(const tensor_t &tensor_) {
  //works only for 3D as of now
  auto tensor_size = tensor_.get_size();

  auto depths = tensor_size[0];
  auto rows   = tensor_size[1];
  auto cols   = tensor_size[2];

  for (uint64_t d = 0; d < depths; ++d) {
    std::cout << "depth = " << d << std::endl;
    for (uint64_t r = 0; r < rows; ++r) {
      std::cout << "r" << r << " : ";
      for (uint64_t c = 0; c < cols; ++c) {
        std::cout << tensor_.at({d,r,c}) << ", ";
      }
      std::cout << std::endl;
    }
  }
}

size_t get_aligned_size(size_t alignment, size_t size_) {
  return ((size_ + alignment - 1) & ~(alignment - 1));
}

} //examples
} //zendnnl
