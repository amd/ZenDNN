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

#include <memory>
#include <iostream>
#include <optional>
#include <cstdint>

#include "tensor_example.hpp"

namespace zendnnl {
namespace examples {

using namespace zendnnl::interface;

int tensor_unaligned_allocation_example() {
  testlog_info("Tensor unaligned memory allocation example");
  auto tensor =  tensor_t()
                 .set_name("contigeous_f32_tensor")
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_storage()
                 .create();

  if (tensor.check()) {
    testlog_info("Tensor creation of ", tensor.get_name(), " successful.");
    testlog_verbose(tensor.get_name(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("Tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  return OK;
}

int tensor_aligned_allocation_example() {
  testlog_info("Tensor aligned memory allocation example.");
  auto tensor =  tensor_t()
                 .set_name("aligned_f32_tensor")
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_storage(ALIGNMENT_BOUNDARY)
                 .create();

  if (tensor.check()) {
    testlog_info("tensor creation of ", tensor.get_name(), " successful.");
    testlog_verbose(tensor.get_name(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  return OK;
}

int tensor_strided_aligned_allocation_example() {
  testlog_info("Tensor strided aligned memory allocation example");
  auto tensor =  tensor_t()
                 .set_name("aligned_strided_f32_tensor")
                 .set_stride_size({MATMUL_ROWS, MATMUL_STRIDE_COLS})
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_storage(ALIGNMENT_BOUNDARY)
                 .create();

  if (tensor.check()) {
    testlog_info("tensor creation of ", tensor.get_name(), " successful.");
    testlog_verbose(tensor.get_name(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  return OK;
}

int tensor_copy_and_compare_example() {
  testlog_info("Tensor copy and compare example.");
  auto tensor =  tensor_t()
                 .set_name("contigeous_bf16_tensor")
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_data_type(data_type_t::bf16)
                 .set_storage()
                 .create();

  if (tensor.check()) {
    testlog_verbose("Tensor creation of ", tensor.get_name(), " successful.");
    testlog_verbose(tensor.get_name(),
                " hash :", tensor.hash(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  auto copy_tensor = tensor;
  copy_tensor.set_name("copied_bf16_tensor");
  testlog_info("copied ", tensor.get_name(), " to ", copy_tensor.get_name());

  testlog_info("comparing ", tensor.get_name(), " with ", copy_tensor.get_name());
  if (copy_tensor == tensor) {
    testlog_info("tensor copy of ", copy_tensor.get_name(),
             " from ", tensor.get_name(), " is successful");
    testlog_verbose(tensor.get_name(), " storage count : ", tensor.get_storage_count());
  }
  else {
    testlog_error("tensor copy failed with original hash:", tensor.hash(),
              " copied hash:", copy_tensor.hash(), " mismatch.");

    return NOT_OK;
  }

  return OK;
}

int tensor_move_and_refcount_example() {
  testlog_info("Tensor move and refcount example.");
  auto tensor =  tensor_t()
                 .set_name("contigeous_s8_tensor")
                 .set_size({MATMUL_ROWS, MATMUL_COLS})
                 .set_data_type(data_type_t::s8)
                 .set_storage()
                 .create();

  if (tensor.check()) {
    testlog_verbose("tensor creation of ", tensor.get_name(), " successful.");
    testlog_verbose(tensor.get_name(),
                " hash :", tensor.hash(),
                " elements :", tensor.get_nelem(),
                " buffer size :", tensor.get_buffer_sz_bytes(),
                " raw ptr :", reinterpret_cast<std::uintptr_t>(tensor.get_raw_handle_unsafe()));
  }
  else {
    testlog_error("tensor creation of ", tensor.get_name(), " failed!");
    return NOT_OK;
  }

  auto move_tensor = std::move(tensor);
  move_tensor.set_name("move_s8_tensor");
  testlog_info("moved ", tensor.get_name(), " to ", move_tensor.get_name());

  if (move_tensor != tensor) {
    testlog_info("tensor move of ", tensor.get_name(),
             " to ", move_tensor.get_name(), " is successful");
    testlog_info(move_tensor.get_name(),
             " hash :", move_tensor.hash(),
             " storage count : ", move_tensor.get_storage_count());
    testlog_info(tensor.get_name(),
             " hash :", tensor.hash(),
             " storage count : ", tensor.get_storage_count());
  }
  else {
    log_error("move tensor failed.");
    return NOT_OK;
  }

  return OK;
}

int tensor_constness_example() {
  testlog_info("**tensor constness example.");

  //use tensor factor to create a uniform tensor
  tensor_factory_t tensor_factory;
  auto utensor = tensor_factory.uniform_tensor({MATMUL_DEPTH, MATMUL_ROWS, MATMUL_COLS},
                 data_type_t::f32,
                 2.5);

  //make the tensor const
  utensor.set_const(true);

  //try to grab its raw pointer
  try {
    void *ptr = utensor.get_raw_handle_unsafe();
    testlog_info("raw pointer of const tensor", ptr);
  }
  catch (const exception_t &ex) {
    testlog_info("caught exception of attempt to get raw pointer of a const tensor.");
    testlog_verbose(ex.what());
  }

  //try to get const handle
  try {
    const float *const_ptr = static_cast<const float *>
                             (utensor.get_raw_handle_const());

    //try to modify data
    //const_ptr[2] = 3.0;

    //read the data
    testlog_info("tensor ", utensor.get_name(), "flat index = 2 : value = ",
             const_ptr[2]);

    //at() works
    testlog_info(utensor.get_name(), "[2,2,2] = ", utensor.at({2,2,2}));
  }
  catch (const exception_t &ex) {
    log_error(ex.what());
    return NOT_OK;
  }

  return OK;
}

int tensor_create_alike_example() {
  testlog_info("**tensor create alike example.");

  //use tensor factor to create a uniform tensor
  tensor_factory_t tensor_factory;
  auto utensor = tensor_factory.uniform_tensor({MATMUL_DEPTH, MATMUL_ROWS, MATMUL_COLS},
                 data_type_t::f32,
                 2.5);

  //make the tensor const
  auto tensor_option = utensor.get_tensor_option();

  try {
    tensor_t atensor = tensor_t()
                       .set_tensor_option(tensor_option)
                       .set_storage()
                       .create();

    //check few options
    if ((utensor.get_size() == atensor.get_size()) &&
        (utensor.get_data_type() == atensor.get_data_type()) &&
        (utensor.get_layout() == atensor.get_layout())) {
      testlog_info("created a tensor with same options");
    }
    else {
      log_error("failed to create tensor with same options");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    log_error(ex.what());
    return NOT_OK;
  }

  return OK;
}

}//examples
}//zendnnl
