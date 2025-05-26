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

#include "tensor_storage.hpp"

namespace zendnnl {
namespace memory {

using namespace zendnnl::error_handling;

//implementation
tensor_storage_t::tensor_storage_t():
  allocated{false}, aligned_to{0}, size{0}, raw_ptr{nullptr} {
}

tensor_storage_t::~tensor_storage_t() {
    reset();
}

void tensor_storage_t::allocate(std::size_t size_) {
  LOG_DEBUG_INFO("Allocating memory buffer for tensor data");

  //allocate storage
  size      = size_;
  if (aligned_to) {
    //adjust size to be a multiple of aligned_to
    if (size % aligned_to) {
      size = (1 + size/aligned_to)*aligned_to;
    }
    raw_ptr = std::aligned_alloc(aligned_to, size);
  }
  else {
    raw_ptr = std::malloc(size);
  }
  if (! raw_ptr) {
    std::string message  = "memory allocation failed for requested ";
    message             += std::to_string(size);
    message             += " bytes,";
    message             += " aligned to ";
    message             += std::to_string(aligned_to);
    message             += " boundary.";
    EXCEPTION_WITH_LOC(message);
  }

  //compute hash again
  status    = status_t::success;
  hash();
}

void tensor_storage_t::reset() {
  LOG_DEBUG_INFO("Resetting tensor object");

  parent_type::reset();

  if (allocated && raw_ptr)
    std::free(raw_ptr);

  allocated  = false;
  aligned_to = 0;
  size       = 0;
  raw_ptr    = nullptr;
}

void* tensor_storage_t::get_raw_handle() {
  LOG_DEBUG_INFO("Getting raw pointer to memory");
  return raw_ptr;
}

void tensor_storage_t::set_raw_handle(void* ptr_, std::size_t size_) {
  LOG_DEBUG_INFO("Setting raw pointer to memory");
  //set the raw handle
  raw_ptr = ptr_;
  size    = size_;

  //compute hash again
  status  = status_t::success;
  hash();
}

std::size_t tensor_storage_t::hash() {
  LOG_DEBUG_INFO("Generating tensor hash");

  if (status == status_t::success) {
    if (hash_key)
      return hash_key;

    hash_key = hash_combine(hash_key, raw_ptr);
    hash_key = hash_combine(hash_key, size);
  }

  return hash_key;
}

} //memory
} //zendnnl

