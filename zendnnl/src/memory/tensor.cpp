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
#include <sstream>

namespace zendnnl {
namespace memory {

using namespace zendnnl::common;
using namespace zendnnl::error_handling;

// Helper to stringify vectors for logging
namespace {
template <typename T>
std::string vec_to_string(const std::vector<T> &v) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) {
      oss << ",";
    }
    oss << v[i];
  }
  oss << "]";
  return oss.str();
}
}

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

tensor_t &tensor_t::set_size(tensor_t::index_vec_type size_) {
  if (status != status_t::success) {
    option.size = size_;
  }

  return (*this);
}

tensor_t::index_vec_type tensor_t::get_size() const {
  return option.size;
}

tensor_t::index_type tensor_t::get_size(uint32_t index_) const {
  return option.size.at(index_);
}

uint32_t tensor_t::get_dim() const {
  return option.size.size();
}

tensor_t &tensor_t::set_aligned_size(tensor_t::index_vec_type aligned_size_) {
  if (status != status_t::success) {
    option.aligned_size = aligned_size_;
  }
  return (*this);
}

tensor_t::index_vec_type  tensor_t::get_aligned_size() const {
  return option.aligned_size;
};

tensor_t::index_type tensor_t::get_aligned_size(uint32_t index_) const {
  return option.aligned_size.at(index_);
}

tensor_t &tensor_t::set_base_index(tensor_t::index_vec_type base_) {
  if (status != status_t::success) {
    option.base = base_;
  }
  return (*this);
}

tensor_t::index_vec_type  tensor_t::get_base_index() const {
  return option.base;
}

tensor_t &tensor_t::set_stride(tensor_t::index_vec_type stride_) {
  if (status != status_t::success) {
    option.stride = stride_;
  }
  return (*this);
}

tensor_t::index_vec_type  tensor_t::get_stride() const {
  return option.stride;
};

tensor_t::index_type tensor_t::get_stride(uint32_t index_) const {
  return option.stride.at(index_);
}

tensor_t  &tensor_t::set_quant_scale(const tensor_t &quant_scale_) {
  //return if tensor is created
  if (status == status_t::success) {
    return (*this);
  }

  //check if quant is iniatilized
  if (! quant) {
    quant.emplace();
  }

  //set the values
  quant->scale_size      = quant_scale_.option.size;
  quant->scale_data_type = quant_scale_.option.data_type;
  quant->scales          = quant_scale_.storage;

  return (*this);
}

tensor_t  &tensor_t::set_quant_zero_point(const tensor_t &quant_zero_) {
  //return if tensor is created
  if (status == status_t::success) {
    return (*this);
  }

  //check if quant is iniatilized
  if (! quant) {
    quant.emplace();
  }

  //set the values
  quant->zero_size      = quant_zero_.option.size;
  quant->zero_data_type = quant_zero_.option.data_type;
  quant->zeros          = quant_zero_.storage;

  return (*this);
}

bool tensor_t::is_quantized() const {
  if (quant) {
    return true;
  }

  return false;
}

quant_type_t tensor_t::get_quant_type() const {

  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return quant_type_t::none;
  }

  return quant->type;
}

quant_subtype_t tensor_t::get_quant_subtype() const {

  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return quant_subtype_t::none;
  }

  return quant->subtype;
}

tensor_t::index_vec_type tensor_t::get_quant_scale_size() const {

  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return {};
  }

  return quant->scale_size;
}

tensor_t::index_vec_type tensor_t::get_quant_scale_stride() const {

  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return {};
  }

  return quant->scale_stride;
}

tensor_t::index_vec_type tensor_t::get_quant_scale_block_size() const {

  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return {};
  }

  return quant->scale_block_size;
}

uint64_t tensor_t::compute_quant_scale_offset(const index_vec_type &index_)
const {

  //sanity check
  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return 0;
  }

  //compute the block index to which the element belongs
  const index_vec_type &scale_block_size = quant->scale_block_size;
  index_vec_type  scale_index(scale_block_size.size(),0);

  for (uint32_t i = 0; i < scale_block_size.size(); ++i) {
    scale_index[i] = index_[i]/scale_block_size[i];
  }

  //compute offset
  const index_vec_type &scale_stride = quant->scale_stride;

  uint64_t offset = 0;
  for (int i = scale_index.size() -1; i >= 0; i--) {
    offset += scale_stride[i]*scale_index[i];
  }

  return offset;
}

data_type_t tensor_t::get_quant_scale_data_type() const {

  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return data_type_t::none;
  }

  return quant->scale_data_type;
}

const void *tensor_t::get_quant_scale_raw_handle_const() const {

  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return nullptr;
  }

  return (const void *)(quant->scales->get_raw_handle());
}

const void *tensor_t::get_quant_scale_raw_handle_const(const index_vec_type
    &index_) const {
  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return nullptr;
  }

  auto     scale_offset  =  compute_quant_scale_offset(index_);
  scale_offset          *= size_of(quant->scale_data_type);

  return (const void *)((uint8_t *)quant->scales->get_raw_handle() +
                        scale_offset);
}

tensor_t::index_vec_type tensor_t::get_quant_zero_size() const {
  if (quant) {
    return quant->zero_size;
  }

  return {};
}

tensor_t::index_vec_type tensor_t::get_quant_zero_stride() const {
  if (quant) {
    return quant->zero_stride;
  }

  return {};
}

tensor_t::index_vec_type tensor_t::get_quant_zero_block_size() const {
  if (quant) {
    return quant->zero_block_size;
  }

  return {};
}

uint64_t tensor_t::compute_quant_zero_offset(const index_vec_type &index_)
const {

  //sanity check
  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return 0;
  }

  //compute the block index to which the element belongs
  const index_vec_type &zero_block_size = quant->zero_block_size;
  index_vec_type  zero_index(zero_block_size.size(),0);

  for (uint32_t i = 0; i < zero_block_size.size(); ++i) {
    zero_index[i] = index_[i]/zero_block_size[i];
  }

  //compute offset
  const index_vec_type &zero_stride = quant->zero_stride;

  uint64_t offset = 0;
  for (int i = zero_index.size() -1; i >= 0; i--) {
    offset += zero_stride[i]*zero_index[i];
  }

  return offset;
}

data_type_t tensor_t::get_quant_zero_data_type() const {
  if (quant) {
    return quant->zero_data_type;
  }

  return data_type_t::none;
}

const void *tensor_t::get_quant_zero_raw_handle_const() const {
  if (quant) {
    return (const void *)(quant->zeros->get_raw_handle());
  }

  return nullptr;
}

const void *tensor_t::get_quant_zero_raw_handle_const(const index_vec_type
    &index_) const {
  if (! quant) {
    apilog_error(name, " invoked a quantization api on non-quantized tensor.");
    return nullptr;
  }

  auto     zero_offset  =  compute_quant_zero_offset(index_);
  zero_offset          *= size_of(quant->zero_data_type);

  return (const void *)((uint8_t *)quant->zeros->get_raw_handle() + zero_offset);
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
    option.layout |= uint8_t(layout_);
  }
  return (*this);
}

uint8_t  tensor_t::get_layout() const {
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

uint64_t tensor_t::compute_offset(const index_vec_type &index_) const {
  LOG_DEBUG_INFO("Computing offset corresponding to the index");
  uint64_t offset = 0;
  for (int i = index_.size() -1; i >= 0; i--) {
    offset += option.stride[i]*index_[i];
  }

  //round to aligned_nelem as with tensor repeating across some
  //axes will generate offset > aligned_nelem
  return (offset % option.aligned_nelem);
}

// TODO: Update the return type to accept different types
float tensor_t::at(const index_vec_type &index_) const {
  LOG_DEBUG_INFO("Getting tensor element");

  //check if a tensor is blocked or oblique
  if ((option.layout & uint8_t(tensor_layout_t::blocked)) ||
      (option.layout & uint8_t(tensor_layout_t::oblique))) {
    std::string message  = "attempt to get an element of a blocked";
    message += " or oblique tensor.";
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
    case data_type_t::s8 : {
      using cpptype = prec_traits<data_type_t::s8>::type;
      const cpptype *handle = static_cast<const cpptype *>(raw_handle);
      return static_cast<float>(handle[offset]);
      break;
    }
    case data_type_t::u8 : {
        using cpptype = prec_traits<data_type_t::u8>::type;
        const cpptype *handle = static_cast<const cpptype *>(raw_handle);
        return static_cast<float>(handle[offset]);
        break;
    }
    case data_type_t::s32 : {
        using cpptype = prec_traits<data_type_t::s32>::type;
        const cpptype *handle = static_cast<const cpptype *>(raw_handle);
        return static_cast<float>(handle[offset]);
        break;
    }
    default :
      std::string message  = "getting element with this data type is unimplemented";
      EXCEPTION_WITH_LOC(std::move(message));
    }
  }
  catch (const exception_t &ex) {
    EXCEPTION_WITH_LOC(ex.what());
  }
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

void *tensor_t::get_raw_handle_unsafe(const index_vec_type &index_) const {
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

const void *tensor_t::get_raw_handle_const(const index_vec_type &index_) const {
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
  if (status == status_t::success) {
    return (*this);
  }

  //validate meta info
  auto l_status = validate_meta_info();
  if (l_status != status_t::success) {
    status = l_status;
    return (*this);
  }

  //validate quant info
  l_status = validate_quant_info();
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

  if (apilog_verbose_enabled()) {
    apilog_verbose("Tensor create - ", tensor_info());
  }
  return (*this);
}

void tensor_t::reset() {
  parent_type::reset();

  option.reset();
  storage.reset();

  if (quant) {
    quant->reset();
  }

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
  //auto layout = get_layout();
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

  if (is_quantized()) {
    ss << "scale(" << dtype_info(get_quant_scale_data_type()) << ":"
       << vec_to_string(get_quant_scale_size()) << "):";
    if (get_quant_subtype() == quant_subtype_t::asymmetric) {
      ss << "zp(" << dtype_info(get_quant_zero_data_type()) << ":"
         << vec_to_string(get_quant_zero_size()) << "):";
    }
  }

  // switch (layout) {
  // case tensor_layout_t::contiguous:
  //   ss << "contiguous";
  //   break;
  // case tensor_layout_t::aligned:
  //   ss << "aligned";
  //   break;
  // case tensor_layout_t::blocked:
  //   ss << "blocked";
  //   break;
  // case tensor_layout_t::oblique:
  //   ss << "oblique";
  //   break;
  // default:
  //   ss << "";
  // }

  return ss.str();
}

status_t tensor_t::size_sanity_check() const {

  LOG_DEBUG_INFO("size sanity check");

  //size can not be empty
  if (option.size.empty()) {
    return status_t::memory_bad_size;
  }

  //size can not be zero
  if (std::find(option.size.begin(), option.size.end(), 0)
      != option.size.end()) {
    return status_t::memory_bad_size;
  }

  return status_t::success;
};

status_t tensor_t::aligned_size_sanity_check() {

  LOG_DEBUG_INFO("Stride sanity check");

  //aligned size length should be equal to size
  if (option.size.size() != option.aligned_size.size()) {
    return status_t::memory_bad_aligned_size;
  }

  for (size_t i = 0; i < option.size.size(); ++i) {
    //if aligned_size less than size flag error
    if (option.aligned_size[i] < option.size[i]) {
      return status_t::memory_bad_aligned_size;
    }
    else if (option.aligned_size[i] > option.size[i]) {
      //set the tensor layout as strided
      option.layout |= uint8_t(tensor_layout_t::aligned);
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
      if (option.stride[i] == 0) {
        ivec.push_back(i);
      }
    }

    //scan stride from max to min
    while (ivec.size() < stride.size()) {
      uint32_t max_index = 0;
      for (uint32_t i = 1; i < stride.size(); ++i) {
        if (stride[i] > stride[max_index]) {
          max_index = i;
        }
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
    for (int8_t i = 0; i < int8_t(option.size.size()); ++i) {
      option.order[i] = char('a' + i);
    }
  }
}

status_t tensor_t::order_sanity_check() {

  LOG_DEBUG_INFO("order sanity check");

  if (option.order.size() != option.size.size()) {
    return status_t::memory_bad_order;
  }

  for (int8_t i = 0; i < int8_t(option.size.size()); ++i) {
    auto count = std::count(option.order.cbegin(),
                            option.order.cend(),
                            char('a' + i));
    if (count != 1) {
      return status_t::memory_bad_order;
    }
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
  if (option.stride.size() != option.aligned_size.size()) {
    apilog_error("tensor ", name, " bad stride size: got stride.size()=",
                 option.stride.size(),
                 " expected=", option.aligned_size.size());
    return status_t::memory_bad_stride;
  }

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
    if (stride_rightmost_nz < 0) {
      apilog_error("tensor ", name, " all strides are zero: stride=",
                   vec_to_string(default_order_stride),
                   " (must have at least one non-zero stride)");
      return status_t::memory_bad_stride;
    }
  }

  //scan strides for consistency with tensor size
  int32_t stride_right = stride_rightmost_nz;
  int32_t stride_left  = 0;

  option.nelem         = 1;
  option.aligned_nelem = 1;

  for (int32_t i = stride_right; i >= stride_left; --i) {
    auto &stride = default_order_stride[i];

    //ignore zero strides
    if (stride == 0) {
      continue;
    }

    //match stride with alignd_nelem as it is product of aligned_size
    //so far. stride should match this.
    if (stride != option.aligned_nelem) {
      apilog_error("tensor ", name, " invalid stride at dimension ", i,
                   ": got stride=", stride,
                   " expected=", option.aligned_nelem);
      return status_t::memory_bad_stride;
    }

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

status_t tensor_t::index_sanity_check(const index_vec_type &index_) const {

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
tensor_t::permute_axes_order(const index_vec_type &in_vec_,
                             bool order_to_default) {
  //clear and fill with zeros
  index_vec_type out_vec(in_vec_.size(),0);

  //permute
  int32_t sz = int32_t(option.order.size());
  for (auto i = 0; i < sz; ++i) {
    int32_t index = int32_t(option.order[i] - 'a');

    //direction true:order to default, false:default to order
    // if (order_to_default)
    //   out_vec[sz -1 -i] = in_vec_[sz -1 -index];
    // else
    //   out_vec[sz -1 -index] = in_vec_[sz -1 -i];

    if (order_to_default) {
      out_vec[i] = in_vec_[index];
    }
    else {
      out_vec[index] = in_vec_[i];
    }

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
  else {
    if (aligned_size_sanity_check() != status_t::success) {
      apilog_error("tensor ", name, " bad aligned size.");
      return status_t::memory_bad_aligned_size;
    }
    else {
      option.layout |= uint8_t(tensor_layout_t::aligned);
    }
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

tensor_t::index_vec_type
tensor_t::compute_quant_block_size(const tensor_t::index_vec_type
                                   &quant_size_) {

  index_vec_type block_size{};
  //validate dim and compute block size
  if (quant_size_.size() != option.size.size()) {
    apilog_error(name, "quant dimension mismatch with tensor dimension");
    return {};
  }

  for (uint32_t i = 0; i < option.size.size(); ++i) {
    auto  tensor_size = option.size[i];
    auto  quant_size  = quant->scale_size[i];

    if (tensor_size % quant_size) {
      apilog_error(name, " tensor size and quant size mismatch.");
      return {};
    }

    block_size.push_back(tensor_size/quant_size);
  }

  return block_size;
}

tensor_t::index_vec_type
tensor_t::compute_quant_stride(const tensor_t::index_vec_type &size_) {
  //get default order size
  auto default_order_size = permute_axes_order(size_, true);

  //compute default order stride
  index_vec_type default_order_stride(size_.size(), 0);

  uint64_t nelem = 1;
  for (int i = size_.size() -1; i >= 0; i--) {
    default_order_stride[i]  = nelem;
    nelem                   *= size_[i];
  }

  //compute permuted stride
  auto permuted_stride = permute_axes_order(default_order_stride, false);

  //make strides zero for broadcast axes
  for (uint32_t i = 0; i < option.stride.size(); ++i) {
    if (option.stride[i] == 0) {
      permuted_stride[i] = 0;
    }
  }

  return permuted_stride;
}

status_t tensor_t::validate_quant_scale() {

  //compute scale block size
  quant->scale_block_size = compute_quant_block_size(quant->scale_size);
  if ((quant->scale_block_size).empty()) {
    return status_t::memory_bad_quant;
  }

  //compute scale stride
  quant->scale_stride = compute_quant_stride(quant->scale_size);

  //check scale data type
  if (quant->scale_data_type != data_type_t::f32 &&
      quant->scale_data_type != data_type_t::bf16) {
    apilog_error(name, " unsupported scale data type.");
    return status_t::memory_bad_quant;
  }
  return status_t::success;
}

status_t tensor_t::validate_quant_zero() {

  //compute zero block size
  quant->zero_block_size = compute_quant_block_size(quant->zero_size);
  if ((quant->zero_block_size).empty()) {
    return status_t::memory_bad_quant;
  }

  //compute zero stride
  quant->zero_stride = compute_quant_stride(quant->zero_size);

  //check zero data type
  if (quant->zero_data_type != data_type_t::s32 &&
      quant->zero_data_type != data_type_t::s8 &&
      quant->zero_data_type != data_type_t::u8) {
    apilog_error(name, " unsupported zero data type.");
    return status_t::memory_bad_quant;
  }
  return status_t::success;
}

status_t tensor_t::validate_quant_info() {

  //if tensor is not quantized return success
  if (! quant) {
    return status_t::success;
  }

  quant->type = quant_type_t::uniform;

  if (quant->scales != nullptr) {
    if (validate_quant_scale() != status_t::success) {
      return status_t::memory_bad_quant;
    }

    quant->subtype = quant_subtype_t::symmetric;
  }
  else {
    apilog_error(name, " quant scales not provided for a quantized tensor.");
    return status_t::memory_bad_quant;
  }

  if (quant->zeros != nullptr) {
    if (validate_quant_zero() != status_t::success) {
      return status_t::memory_bad_quant;
    }

    quant->subtype = quant_subtype_t::asymmetric;
  }

  return status_t::success;
}

} //memory
} //zendnnl
