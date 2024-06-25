/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/
#ifndef _ZENTPP_TYPES_HPP_
#define _ZENTPP_TYPES_HPP_

#include <cstdint>

namespace zendnn {
namespace tpp {

// typedef struct bfloat16 {
//     uint16_t data;
// } bfloat16;

using bfloat16 = uint16_t;

typedef struct half {
    uint16_t data;
} half;

typedef struct bfloat8 {
  uint8_t data;
} bfloat8;

}
}
#endif
