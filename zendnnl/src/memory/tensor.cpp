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

tensor_t &tensor_t::set_aligned_size(std::vector<uint64_t> aligned_size_) {
  if (status != status_t::success) {
    option.aligned_size = aligned_size_;
  }
  return (*this);
}

std::vector<uint64_t>  tensor_t::get_aligned_size() const {
  return option.aligned_size;
};

uint64_t tensor_t::get_aligned_size(uint32_t index_) const {
  return option.aligned_size.at(index_);
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

tensor_t &tensor_t::set_stride(std::vector<uint64_t> stride_) {
  if (status != status_t::success) {
    option.stride = stride_;
  }
  return (*this);
}

std::vector<uint64_t>  tensor_t::get_stride() const {
  return option.stride;
};

uint64_t tensor_t::get_stride(uint32_t index_) const {
  return option.stride.at(index_);
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

uint64_t tensor_t::compute_offset(const std::vector<index_type> index_) const {
  LOG_DEBUG_INFO("Computing offset corresponding to the index");
  uint64_t offset = 0;
  for (int i = index_.size() -1; i >= 0; i--) {
    offset += option.stride[i]*index_[i];
  }

  //round to aligned_nelem as with tensor repeating across some
  //axes will generate offset > aligned_nelem
  return (offset % option.aligned_nelem);
}

float tensor_t::at(const std::vector<index_type> &index_) const {
  LOG_DEBUG_INFO("Getting tensor element");
  if ((option.layout != tensor_layout_t::contiguous) &&
      (option.layout != tensor_layout_t::aligned)) {
    std::string message  = "attempt to get an element of a non-contiguous";
    message += " or non-aligned tensor.";
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

void *tensor_t::get_raw_handle_unsafe(const index_vec_type& index_) const {
  if (status != status_t::success) {
    std::string message  = "attempt to get raw handle of an invalid tensor.";
    EXCEPTION_WITH_LOC(message);
  }

  if (option.is_const) {
    std::string message  = "attempt to get raw handle of a const tensor.";
    EXCEPTION_WITH_LOC(message);
  }

  if (index_sanity_check(index_) != status_t::success) {
    std::string message  = "attempt to get raw handle with invalid index.";
    EXCEPTION_WITH_LOC(message);
  }

  auto index_offset = compute_offset(index_)*size_of(option.data_type);

  return (void *)((uint8_t *)storage->get_raw_handle() +
                  option.base_offset +
                  index_offset);
}

const void *tensor_t::get_raw_handle_const() const {
  if (status != status_t::success) {
    std::string message  = "attempt to get raw handle of an invalid tensor.";
    EXCEPTION_WITH_LOC(message);
  }

  return (const void *)((uint8_t *)storage->get_raw_handle() +
                        option.base_offset);
}

const void *tensor_t::get_raw_handle_const(const index_vec_type& index_) const {
  if (status != status_t::success) {
    std::string message  = "attempt to get raw handle of an invalid tensor.";
    EXCEPTION_WITH_LOC(message);
  }

  if (index_sanity_check(index_) != status_t::success) {
    std::string message  = "attempt to get raw handle with invalid index.";
    EXCEPTION_WITH_LOC(message);
  }

  auto index_offset = compute_offset(index_)*size_of(option.data_type);

  return (const void *)((uint8_t *)storage->get_raw_handle() +
                        option.base_offset +
                        index_offset);
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

tensor_t &tensor_t::create() {
  LOG_DEBUG_INFO("Creating tensor object");

  //if already formed object, return the object
  if (status == status_t::success)
    return (*this);

  //validate meta info
  auto l_status = validate_meta_info();
  if (l_status != status_t::success) {
    status = l_status;
    return (*this);
  }

  uint64_t buffer_size = option.aligned_nelem * size_of(option.data_type);
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
      status = status_t::memory_bad_storage;
      return *this;
    }
  }
  else {
    if (storage->get_raw_handle() == nullptr) {
      status = status_t::memory_bad_storage;
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

  apilog_info("Tensor create - ",tensor_info());

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

std::string tensor_t::tensor_info() {
  std::stringstream ss;
  auto layout = get_layout();
  auto dtype  = get_data_type();
  auto order  = get_order();

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
     << dtype_info(dtype) << ":" << order << ":";

  switch (layout) {
  case tensor_layout_t::contiguous:
    ss << "contiguous";
    break;
  case tensor_layout_t::aligned:
    ss << "aligned";
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

status_t tensor_t::size_sanity_check() const {

  LOG_DEBUG_INFO("size sanity check");

  //size can not be empty
  if (option.size.empty())
    return status_t::memory_bad_size;

  //size can not be zero
  if (std::find(option.size.begin(), option.size.end(), 0)
      != option.size.end())
    return status_t::memory_bad_size;

  return status_t::success;
};

status_t tensor_t::aligned_size_sanity_check() {

  LOG_DEBUG_INFO("Stride sanity check");

  //aligned size length should be equal to size
  if (option.size.size() != option.aligned_size.size())
    return status_t::memory_bad_aligned_size;

  for (size_t i = 0; i < option.size.size(); ++i) {
    //if aligned_size less than size flag error
    if (option.aligned_size[i] < option.size[i]) {
      return status_t::memory_bad_aligned_size;
    }
    else if (option.aligned_size[i] > option.size[i]) {
      //set the tensor layout as strided
      option.layout = tensor_layout_t::aligned;
    }
  }

  return status_t::success;
}

void tensor_t::set_default_order(bool is_stride_) {

  LOG_DEBUG_INFO("Set default order");

  option.order.clear();
  option.order.resize(option.size.size());

  if (is_stride_) {
    auto stride = option.stride;

    std::vector<uint32_t> ivec;
    //scan stride for zeros
    for (uint32_t i = 0; i < stride.size(); ++i) {
      if (option.stride[i] == 0)
        ivec.push_back(i);
    }

    //scan stride from max to min
    while (ivec.size() < stride.size()) {
      uint32_t max_index = 0;
      for(uint32_t i = 1; i < stride.size(); ++i) {
        if (stride[i] > stride[max_index])
          max_index = i;
      }

      ivec.push_back(max_index);
      stride[max_index] = 0;
    }

    //allocate order
    char ch = 'a';
    for (uint8_t i = 0; i < uint8_t(ivec.size()); ++i) {
      auto idx = ivec[i];
      option.order[i] = char(ch + idx);
    }

  }
  else {
    for (int8_t i = 0; i < int8_t(option.size.size()); ++i)
      option.order[i] = char('a' + i);
  }
}

status_t tensor_t::order_sanity_check() {

  LOG_DEBUG_INFO("order sanity check");

  if (option.order.size() != option.size.size())
    return status_t::memory_bad_order;

  for (int8_t i = 0; i < int8_t(option.size.size()); ++i) {
    auto count = std::count(option.order.cbegin(),
                            option.order.cend(),
                            char('a' + i));
    if (count != 1)
      return status_t::memory_bad_order;
  }

  return status_t::success;
}

void tensor_t::set_default_stride() {
  LOG_DEBUG_INFO("Setting default stride");

  //get default order size
  auto default_order_size =
    permute_axes_order(option.size, true);

  //get default order aligned size
  auto default_order_aligned_size =
    permute_axes_order(option.aligned_size, true);

  //compute default order stride
  index_vec_type default_order_stride;
  default_order_stride.resize(option.aligned_size.size());

  option.aligned_nelem = option.nelem = 1;
  for (int i = option.aligned_size.size() -1; i >= 0; i--) {
    default_order_stride[i]  = option.aligned_nelem;
    option.aligned_nelem    *= default_order_aligned_size[i];
    option.nelem            *= default_order_size[i];
  }

  //permute stride to given order
  option.stride = permute_axes_order(default_order_stride, false);
}

status_t tensor_t::stride_sanity_check() {

  LOG_DEBUG_INFO("stride sanity check");

  //stride size should be at least size of tensor size
  if (option.stride.size() != option.aligned_size.size())
    return status_t::memory_bad_stride;

  //permute size and stride if order is set
  auto default_order_size =
    permute_axes_order(option.size, true);

  auto default_order_aligned_size =
    permute_axes_order(option.aligned_size, true);

  auto default_order_stride =
    permute_axes_order(option.stride, true);

  //ignore trailing zeros in stride
  int32_t stride_rightmost_nz = int32_t(default_order_stride.size() - 1);
  while (default_order_stride[stride_rightmost_nz] == 0) {
    stride_rightmost_nz--;

    //if strides are all zero, flag error
    if (stride_rightmost_nz < 0)
      return status_t::memory_bad_stride;
  }

  //scan strides for consistency with tensor size
  int32_t stride_right = stride_rightmost_nz;
  int32_t stride_left  = 0;

  option.nelem         = 1;
  option.aligned_nelem = 1;

  for (int32_t i = stride_right; i >= stride_left; --i ) {
    auto& stride = default_order_stride[i];

    //ignore zero strides
    if (stride == 0)
      continue;

    //match stride with alignd_nelem as it is product of aligned_size
    //so far. stride should match this.
    if (stride != option.aligned_nelem)
      return status_t::memory_bad_stride;

    option.nelem         *= default_order_size[i];
    option.aligned_nelem *= default_order_aligned_size[i];
  }

  return status_t::success;
}

void tensor_t::set_default_base() {

  LOG_DEBUG_INFO("Setting default base index");

  option.base.resize(option.size.size(), 0);
  option.base_offset = 0;
}


status_t tensor_t::index_sanity_check(const index_vec_type& index_) const {

  LOG_DEBUG_INFO("Index sanity check");

  if (index_.size() != option.size.size()) {
    return status_t::memory_bad_index;
  }

  for (size_t i = 0; i < option.size.size(); ++i) {
    if (index_[i] > option.size[i]) {
      return status_t::memory_bad_index;
    }
  }

  return status_t::success;
}

tensor_t::index_vec_type
tensor_t::permute_axes_order(const index_vec_type& in_vec_,
                             bool order_to_default) {
  //clear and fill with zeros
  index_vec_type out_vec(in_vec_.size(),0);

  //permute
  int32_t sz = int32_t(option.order.size());
  for(auto i = 0; i < sz; ++i) {
    int32_t index = int32_t(option.order[i] - 'a');

    //direction true:order to default, false:default to order
    // if (order_to_default)
    //   out_vec[sz -1 -i] = in_vec_[sz -1 -index];
    // else
    //   out_vec[sz -1 -index] = in_vec_[sz -1 -i];

    if (order_to_default)
      out_vec[i] = in_vec_[index];
    else
      out_vec[index] = in_vec_[i];

  }

  return out_vec;
}

status_t tensor_t::validate_meta_info() {

  LOG_DEBUG_INFO("Validating meta data");

  if (size_sanity_check() != status_t::success) {
    return status_t::memory_bad_size;
  }

  if (option.aligned_size.empty()) {
    option.aligned_size = option.size;
  }
  else if (aligned_size_sanity_check() != status_t::success) {
    apilog_error("tensor ", name, " bad aligned size.");
    return status_t::memory_bad_aligned_size;
  }

  if (option.base.empty()) {
    set_default_base();
  }
  else if (index_sanity_check(option.base) != status_t::success) {
    apilog_error("tensor ", name, " bad base index.");
    return status_t::memory_bad_base;
  }

  bool is_order  = !option.order.empty();
  bool is_stride = !option.stride.empty();

  if (is_order) {
    if (order_sanity_check() != status_t::success) {
      apilog_error("tensor ", name, " bad axes order.");
      return status_t::memory_bad_order;
    }
    if (is_stride) {
      if (stride_sanity_check() != status_t::success) {
        apilog_error("tensor ", name, " order and stride mismatch.");
        return status_t::memory_bad_stride;
      }
    }
    else {
      set_default_stride();
    }
  }
  else {
    set_default_order(is_stride);

    if (is_stride) {
      if (stride_sanity_check() != status_t::success) {
        apilog_error("tensor ", name, " order and stride mismatch.");
        return status_t::memory_bad_stride;
      }
    }
    else {
      set_default_stride();
    }
  }

  return status_t::success;
}


} //memory
} //zendnnl
