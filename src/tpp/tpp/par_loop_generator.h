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

#ifndef _PAR_LOOP_GENERATOR_H_
#define _PAR_LOOP_GENERATOR_H_
namespace zendnn {
namespace tpp {
std::string loop_generator(const char* _loop_nest_desc_extended);
} // namespace tpp
} // namespace zendnn
#endif // _PAR_LOOP_GENERATOR_H_
