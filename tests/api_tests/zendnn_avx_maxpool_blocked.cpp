/*******************************************************************************
* Modifications Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
* Notified per clause 4(b) of the license.
*******************************************************************************/

/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "zendnn.hpp"
using namespace zendnn;
using tag = memory::format_tag;
using dt = memory::data_type;



int main(int argc, char **argv) {

    zendnnInfo(ZENDNN_TESTLOG, "zendnn_avx_maxpool_blocked API test starts");
    const memory::dim N = 10, // batch size
                      IC = 64, // input channels
                      IH = 112, // input tensor height
                      IW = 112, // input tensor width
                      KH = 2, // kernel height
                      KW = 2, // kernel width
                      PH_L = 0, // height padding: left
                      PH_R = 0, // height padding: right
                      PW_L = 0, // width padding: left
                      PW_R = 0, // width padding: right
                      SH = 2, // height-wise stride
                      SW = 2; // width-wise stride
    //const memory::dim OH = std::ceil((float)(IH - KH + PH_L + PH_R) / SH) + 1;
    //const memory::dim OW = std::ceil((float)(IW - KW + PW_L + PW_R) / SW) + 1;
    const memory::dim OH = ((IH - KH + PH_L + PH_R) / SH) + 1;
    const memory::dim OW = ((IW - KW + PW_L + PW_R) / SW) + 1;
    // Source (src) and destination (dst) tensors dimensions.
    memory::dims src_dims = {N, IC, IH, IW};
    memory::dims dst_dims = {N, IC, OH, OW};
    // Kernel dimensions.
    memory::dims kernel_dims = {KH, KW};
    // Strides, padding dimensions.
    memory::dims strides_dims = {SH, SW};
    memory::dims padding_dims_l = {PH_L, PW_L};
    memory::dims padding_dims_r = {PH_R, PW_R};
    // Allocate buffers.
    std::vector<float> src_data(N*IC*IH*IW);
    std::vector<float> dst_data(N*IC*OH*OW);
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    engine eng(zendnn::engine::kind::cpu, 0);
    stream s(eng);
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    memory::dims pool_src_tz = {N, IC,
                                IH, IW
                               };
    memory::dims pool_dst_tz = {N, IC,
                                OH, OW
                               };

    bool reorder_before = true;
    bool reorder_after = true;
    zendnn::memory pool_src_memory;
    if (reorder_before)
        pool_src_memory = memory({{pool_src_tz}, dt::f32, tag::nhwc},eng,
    (float *)src_data.data());
    else
        pool_src_memory = memory({{pool_src_tz}, dt::f32, tag::aBcd8b},eng,
    (float *)src_data.data());


    zendnn::memory pool_dst_memory, pool_dst_memory_new;
    if (reorder_after) {
        pool_dst_memory = memory({{pool_dst_tz}, dt::f32, tag::aBcd8b},eng);
        pool_dst_memory_new = memory({{pool_dst_tz}, dt::f32, tag::nhwc},eng,
        (float *)dst_data.data());
    }
    else {
        pool_dst_memory = memory({{pool_dst_tz}, dt::f32, tag::aBcd8b},eng,
        (float *)dst_data.data());
        pool_dst_memory_new = memory({{pool_dst_tz}, dt::f32, tag::aBcd8b},eng,
        (float *)dst_data.data());
    }

    memory::desc pool_src_md = memory::desc({pool_src_tz}, dt::f32, tag::aBcd8b);
    memory::desc pool_dst_md = memory::desc({pool_dst_tz}, dt::f32, tag::aBcd8b);
    //memory::desc pool_dst_md = memory::desc({pool_dst_tz}, dt::f32, tag::any);

    //[Create pooling primitive]
    pooling_forward::desc pool_desc = pooling_forward::desc(
                                          prop_kind::forward_inference,
                                          algorithm::pooling_max, pool_src_md, pool_dst_md,
                                          strides_dims, kernel_dims, padding_dims_l, padding_dims_r);
    pooling_forward::primitive_desc pool_pd = pooling_forward::primitive_desc(
                pool_desc, eng);


    zendnn::memory pool1_src_memory = pool_src_memory;
    if (pool_pd.src_desc() != pool_src_memory.get_desc()) {
        pool1_src_memory = memory(pool_pd.src_desc(), eng);
        if (reorder_before) {
            net.push_back(reorder(pool_src_memory, pool1_src_memory));
            net_args.push_back({{ZENDNN_ARG_SRC, pool_src_memory},
                        {ZENDNN_ARG_DST, pool1_src_memory}
            });
        }
    }

    net.push_back(pooling_forward(pool_pd));
    if (reorder_before) {
        net_args.push_back({{ZENDNN_ARG_SRC, pool1_src_memory},
                        {ZENDNN_ARG_DST, pool_dst_memory}
            });
    }
    else {
        net_args.push_back({{ZENDNN_ARG_SRC, pool_src_memory},
            {ZENDNN_ARG_DST, pool_dst_memory}});
    }
    //[Create pooling primitive]

    //[Execute model]
    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(s, net_args.at(i));
    }
    s.wait();
    if (reorder_after) {
        reorder(pool_dst_memory, pool_dst_memory_new).execute(s, pool_dst_memory,
                pool_dst_memory_new);
    }
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_avx_maxpool_blocked API test ends");
}
