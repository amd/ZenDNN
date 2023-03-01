/*******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef COMMON_CBLAS_WITH_BLIS_API_HPP
#define COMMON_CBLAS_WITH_BLIS_API_HPP

/* AOCL BLIS is recommended BLAS library for ZenDNN.
If an application wants to use an another CBLAS library for non ZenDNN calls,
BLIS and another BLAS library wil be linked to same application.
To avoid symbol clash in such scenarios, use ZENDNN_USE_AOCL_BLIS_API path.
ZENDNN_USE_AOCL_BLIS_API makes use of BLIS API instead of CBLAS API.
This should be done in combination with BLIS build without exporting CBLAS API.
ZENDNN_USE_AOCL_BLIS_API can be enabled by adding BLIS_API=1 option to make file */

#include "blis_wrapper.hpp"

/* Find and replace cblas calls with corresponding wrapper functions.

If any new cblas call is used inside ZenDNN in future, please implement declare
and replacement macro for corresponding wrapper function in blis_wrapper.cpp,
blis_wrapper.hpp and bypass_blis.hpp respectively. */
#define cblas_sgemm cblas_sgemm_aocl
#define cblas_sgemv cblas_sgemv_aocl
#define cblas_sdot cblas_sdot_aocl

#endif // COMMON_CBLAS_WITH_BLIS_API_HPP