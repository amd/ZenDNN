/*****************************************************************************
* Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
* All rights reserved.
* Notified per clause 4(b) of the license.
******************************************************************************/
/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

#include "xsmm_functors.h"

#if (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#else
#define _mm_pause()
#endif

#include <atomic>

namespace zendnn {
namespace tpp {

static inline void atomic_add_float(double* dst, double fvalue) {
  typedef union {
    unsigned long long intV;
    double floatV;
  } uf64_t;

  uf64_t new_value, old_value;
  std::atomic<unsigned long long>* dst_intV =
      (std::atomic<unsigned long long>*)(dst);

  old_value.floatV = *dst;
  new_value.floatV = old_value.floatV + fvalue;

  unsigned long long* old_intV = (unsigned long long*)(&old_value.intV);
  while (!std::atomic_compare_exchange_strong(
      dst_intV, old_intV, new_value.intV)) {
    _mm_pause();
    old_value.floatV = *dst;
    new_value.floatV = old_value.floatV + fvalue;
  }
}

}
}
