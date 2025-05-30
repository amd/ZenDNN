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

#include "tensor.hpp"

namespace zendnnl {
namespace memory {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

//constructors and destructor
tensor_t::tensor_t():
  option{}, quant{},
  storage{std::make_shared<tensor_storage_t>()},
  allocate{false},
  name{"unknown tensor"} {
}

tensor_t::tensor_t(tensor_t &&other_):
  parent_type(std::move(other_)) {
  option   = other_.option;
  quant    = other_.quant;
  storage  = other_.storage;
  allocate = other_.allocate;
  name     = other_.name;

  other_.reset();
}

tensor_t &tensor_t::operator=(tensor_t &&other_) {
  if (*this != other_) {
    parent_type::operator=(std::move(other_));
    option   = other_.option;
    quant    = other_.quant;
    storage  = other_.storage;
    allocate = other_.allocate;
    name     = other_.name;

    other_.reset();
  }
  return *this;
}

tensor_t &tensor_t::set_size(std::vector<uint64_t> size_) {
  if (status != status_t::success) {
    option.size = size_;
  }

  return (*this);
}

std::vector<uint64_t> tensor_t::get_size() const {
  return option.size;
}

uint64_t tensor_t::get_size(uint32_t index_) const {
  return option.size.at(index_);
}

uint32_t tensor_t::get_dim() const {
  return option.size.size();
}

tensor_t &tensor_t::set_stride_size(std::vector<uint64_t> stride_size_) {
  if (status != status_t::success) {
    option.stride_size = stride_size_;
  }
  return (*this);
}

std::vector<uint64_t>  tensor_t::get_stride_size() const {
  return option.stride_size;
};

uint64_t tensor_t::get_stride_size(uint32_t index_) const {
  return option.stride_size.at(index_);
}

tensor_t &tensor_t::set_base_index(std::vector<uint64_t> base_) {
  if (status != status_t::success) {
    option.base = base_;
  }
  return (*this);
}

std::vector<uint64_t>  tensor_t::get_base_index() const {
  return option.base;
}

tensor_t &tensor_t::set_data_type(data_type_t data_type_) {
  if (status != status_t::success) {
    option.data_type = data_type_;
  }
  return (*this);
}

data_type_t tensor_t::get_data_type() const {
  return option.data_type;
}

tensor_t &tensor_t::set_layout(tensor_layout_t layout_) {
  if (status != status_t::success) {
    option.layout = layout_;
  }
  return (*this);
}

tensor_layout_t  tensor_t::get_layout() const {
  return option.layout;
};

tensor_t &tensor_t::set_order(std::string order_) {
  if (status != status_t::success) {
    option.order = order_;
  }
  return (*this);
}

std::string  tensor_t::get_order() const {
  return option.order;
};

tensor_t &tensor_t::set_tensor_option(const tensor_option_t &option_) {
  if (status != status_t::success) {
    option = option_;
  }
  return (*this);
}

tensor_option_t &tensor_t::get_tensor_option() {
  return option;
}

tensor_t &tensor_t::set_const(bool constness_) {
  option.is_const = constness_;
  return (*this);
}

bool  tensor_t::get_const() const {
  return option.is_const;
};

tensor_t &tensor_t::set_name(std::string name_) {
  //if (status != status_t::success) {
  name = name_;
  //}
  return (*this);
}

std::string  tensor_t::get_name() const {
  return name;
};

float tensor_t::at(const std::vector<index_type> &index_) const {
  LOG_DEBUG_INFO("Getting tensor element");
  if ((option.layout != tensor_layout_t::contiguous) &&
      (option.layout != tensor_layout_t::strided)) {
    std::string message  = "attempt to get an element of a non-contiguous";
    message += " or non-strided tensor.";
    EXCEPTION_WITH_LOC(message);
  }

  try {
    const void *raw_handle = get_raw_handle_const();
    auto  offset     = compute_offset(index_);

    switch (option.data_type) {
    case data_type_t::f32 : {
      using cpptype   = prec_traits<data_type_t::f32>::type;
      const cpptype *handle = static_cast<const cpptype *>(raw_handle);
      return handle[offset];
      break;
    }
    case data_type_t::bf16 : {
      using cpptype   = prec_traits<data_type_t::bf16>::type;
      const cpptype *handle = static_cast<const cpptype *>(raw_handle);
      return float(handle[offset]);
      break;
    }
    default :
      std::string message  = "getting element with this data type is unimplemented";
      EXCEPTION_WITH_LOC(message);
    }
  }
  catch (const exception_t &ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }

  return 0.0;
}

uint64_t tensor_t::get_nelem() const {
  return option.nelem;
}

uint64_t tensor_t::get_buffer_sz_bytes() const {
  return storage->size;
}

uint32_t tensor_t::get_storage_count() const {
  return storage.use_count();
}

void *tensor_t::get_raw_handle_unsafe() const {
  if (status != status_t::success) {
    std::string message  = "attempt to get raw handle of an invalid tensor.";
    EXCEPTION_WITH_LOC(message);
  }

  if (option.is_const) {
    std::string message  = "attempt to get raw handle of a const tensor.";
    EXCEPTION_WITH_LOC(message);
  }

  return (void *)((uint8_t *)storage->get_raw_handle() + option.base_offset);
}

const void *tensor_t::get_raw_handle_const() const {
  if (status != status_t::success) {
    std::string message  = "attempt to get raw handle of an invalid tensor.";
    EXCEPTION_WITH_LOC(message);
  }

  return (const void *)((uint8_t *)storage->get_raw_handle() +
                        option.base_offset);
}

tensor_t &tensor_t::set_storage() {
  if (status != status_t::success) {
    allocate = true;
  }
  return (*this);
}

tensor_t &tensor_t::set_storage(uint32_t aligned_to_) {
  if (status != status_t::success) {
    allocate              = true;
    storage->aligned_to   = aligned_to_;
  }
  return (*this);
}

tensor_t &tensor_t::set_storage(void *raw_ptr_, uint64_t sz_bytes_) {
  if (status != status_t::success) {
    storage->set_raw_handle(raw_ptr_, sz_bytes_);
  }
  return (*this);
}

tensor_t &tensor_t::set_storage(const tensor_t &other_) {
  if (other_.status != status_t::success) {
    return *this;
  }
  storage = other_.storage;
  return (*this);
}

void tensor_t::reset() {
  parent_type::reset();

  option.reset();
  quant.reset();
  storage.reset();

  allocate = false;
  storage  = std::make_shared<tensor_storage_t>();
}

tensor_t &tensor_t::create() {
  LOG_DEBUG_INFO("Creating tensor object");

  if (status != status_t::success) {
    validate_meta_info();
    if (status != status_t::success) {
      return *this;
    }

    uint64_t buffer_size = option.strided_nelem * size_of(option.data_type);
    if (allocate) {
      // allocate new storage
      if (buffer_size) {
        try {
          storage->allocate(buffer_size);
          storage->allocated = true;
        }
        catch (const exception_t &ex) {
          std::string message = get_name() + "-" + ex.what();
          EXCEPTION_WITH_LOC(message);
        }
      }
      else {
        status = status_t::bad_hash_object;
        return *this;
      }
    }
    else {
      if (storage->get_raw_handle() == nullptr) {
        status = status_t::bad_hash_object;
        return *this;
      }

      //check if allocated raw pointer is proper
      if (storage->size < buffer_size) {
        // throw exception
        std::string message = get_name();
        message += "-imporper memory size ";
        message += std::to_string(storage->size);
        message += "against expected size ";
        message += std::to_string(buffer_size);
        EXCEPTION_WITH_LOC(message);
      }
    }

    //compute hash
    status = status_t::success;
    hash();
  }
  apilog_info("Tensor create - ",tensor_info());
  return *this;
}

std::size_t tensor_t::hash() {
  LOG_DEBUG_INFO("Generating tensor hash");

  if (status == status_t::success) {
    if (hash_key) {
      return hash_key;
    }

    hash_key = hash_combine(hash_key, (*storage));
    hash_key = hash_combine(hash_key, option);
  }

  return hash_key;
}

uint64_t tensor_t::compute_offset(const std::vector<index_type> index_) const {
  LOG_DEBUG_INFO("Computing offset corresponding to the index");
  uint64_t offset = 0;
  for (int i = index_.size() -1; i >= 0; i--) {
    offset += option.stride[i]*index_[i];
  }

  return offset;
}

void tensor_t::set_default_stride() {
  LOG_DEBUG_INFO("Setting default stride");
  option.stride.resize(option.stride_size.size());
  option.strided_nelem = option.nelem = 1;
  for (int i = option.stride_size.size() -1; i >= 0; i--) {
    option.stride[i]       = option.strided_nelem;
    option.strided_nelem  *= option.stride_size[i];
    option.nelem          *= option.size[i];
  }
}

void tensor_t::set_default_base() {
  LOG_DEBUG_INFO("Setting default base index");
  option.base.resize(option.size.size());
  for (size_t i = 0; i < option.size.size(); ++i) {
    option.base[i] = 0;
  }
  option.base_offset = 0;
}

void tensor_t::stride_sanity_check() {
  LOG_DEBUG_INFO("Stride sanity check");
  if (option.size.size() != option.stride_size.size()) {
    status = status_t::bad_hash_object;
    return;
  }

  for (size_t i = 0; i < option.size.size(); ++i) {
    //if stride_size less than size flag error
    if (option.stride_size[i] < option.size[i]) {
      status = status_t::bad_hash_object;
      return;
    }
    else if (option.stride_size[i] > option.size[i]) {
      //set the tensor layout as strided
      option.layout = tensor_layout_t::strided;
    }
  }

  status = status_t::success;
}

void tensor_t::base_sanity_check() {
  LOG_DEBUG_INFO("Base sanity check");
  if (option.base.size() != option.stride_size.size()) {
    status = status_t::bad_hash_object;
    return;
  }

  for (size_t i = 0; i < option.stride_size.size(); ++i) {
    if (option.base[i] > option.stride_size[i]) {
      status = status_t::bad_hash_object;
      return;
    }
  }

  option.base_offset = compute_offset(option.base)*size_of(option.data_type);

  status = status_t::success;
}

void tensor_t::validate_meta_info() {
  LOG_DEBUG_INFO("Validating meta data");
  if (option.size.empty()) {
    status = status_t::bad_hash_object;
    return;
  }

  if (option.stride_size.empty()) {
    option.stride_size = option.size;
  }
  else {
    stride_sanity_check();
    if (status != status_t::success) {
      return;
    }
  }

  set_default_stride();

  if (option.base.empty()) {
    set_default_base();
  }
  else {
    base_sanity_check();
    if (status != status_t::success) {
      return;
    }
  }

  status = status_t::success;
}

std::string tensor_t::tensor_info() {
  std::stringstream ss;
  auto layout = get_layout();
  auto dtype  = get_data_type();

  ss << get_name() << "[";
  uint32_t dim = get_dim();
  if (dim == 1) {
    ss << "1,";
  }
  for (uint32_t i = 0; i < dim; ++i) {
    ss << get_size(i);
    if (i < dim - 1) {
      ss << ",";
    }
  }
  ss << "]:"
     << dtype_info(dtype) << ":";

  switch (layout) {
  case tensor_layout_t::contiguous:
    ss << "contiguous";
    break;
  case tensor_layout_t::strided:
    ss << "strided";
    break;
  case tensor_layout_t::blocked:
    ss << "blocked";
    break;
  case tensor_layout_t::oblique:
    ss << "oblique";
    break;
  default:
    ss << "";
  }

  return ss.str();
}

} //memory
} //zendnnl
