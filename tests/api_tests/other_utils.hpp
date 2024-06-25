/*******************************************************************************
* Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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
*
*
*******************************************************************************/
#ifndef  _OTHER_UTILS_HPP_
#define  _OTHER_UTILS_HPP_

#include <cstdint>
#include "zendnn.hpp"

class fdiff_t {
public:
    fdiff_t(float tol = 1e-05):tolerance{tol} {}

    bool operator()(float a, float b) {
        if(isnan(a) || isnan(b))
            return true;

        a = a > 0 ? a : -a;

        if((a -b)/a > tolerance)
            return true;
        if((b -a)/a > tolerance)
            return true;

        return false;
    }

private:
    float tolerance;
};

int diff_mem(memory& first, memory& second, float tol = 1e-05) {
    fdiff_t  diff(tol);

    // sanity check on input and output
    auto fdims  = first.get_desc().dims();
    auto sdims  = second.get_desc().dims();

    // number of elements in the tensor
    uint32_t felem = 1;
    for (auto dim : fdims)
	felem *= dim;

    uint32_t selem = 1;
    for (auto dim : sdims)
	selem *= dim;

    auto celem = felem < selem ? felem : selem;

    float*    fptr    = reinterpret_cast<float *>(first.get_data_handle());
    float*    sptr    = reinterpret_cast<float *>(second.get_data_handle());

    uint32_t count = 0;
    for (auto i = 0; i < celem; ++i) {
	if (diff(fptr[i], sptr[i]))
	    count++;
    }

    return count;
}
#endif
