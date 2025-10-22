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

#ifndef _ZENDNNL_KEY_HPP
#define _ZENDNNL_KEY_HPP

#include "common/hash_object.hpp"
#include "operators/matmul/matmul_config.hpp"

using matmul_algo_t = zendnnl::ops::matmul_algo_t;
//structure to make key
struct Key_matmul {
  bool transpose_weights;
  unsigned int k;
  unsigned int n;
  unsigned int ldb;
  const void *weights;
  uint32_t algo;

  // Default constructor
  Key_matmul() : transpose_weights(false), k(1), n(1),
    ldb(1), weights(nullptr), algo(static_cast<uint32_t>(matmul_algo_t::none)) {}

  // Constructor to initialize all member variables
  Key_matmul(bool TransB, unsigned int K,
             unsigned int N,
             unsigned int ldb, const void *B_Array,
             uint32_t algo)
    : transpose_weights(TransB), k(K), n(N),
      ldb(ldb), weights(B_Array), algo(algo) {
  }

  bool operator==(const Key_matmul &other) const {
    return (k == other.k
            && n == other.n
            && ldb == other.ldb
            && weights == other.weights
            && transpose_weights == other.transpose_weights
            && algo == other.algo
           );
  }
};

namespace std {

template <>
struct hash<Key_matmul> {
  std::size_t operator()(const Key_matmul &k) const {
    std::size_t seed = 0;
    seed = zendnnl::common::hash_combine(seed, (k.transpose_weights));
    seed = zendnnl::common::hash_combine(seed, (k.k));
    seed = zendnnl::common::hash_combine(seed, (k.n));
    seed = zendnnl::common::hash_combine(seed, (k.ldb));
    seed = zendnnl::common::hash_combine(seed, (k.weights));
    seed = zendnnl::common::hash_combine(seed, (k.algo));
    return seed;
  }
};
}

#endif // _ZENDNNL_KEY_HPP