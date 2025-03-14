# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package(OpenMP REQUIRED)

execute_process(
    COMMAND bash -c "rm -rf _out/tests && mkdir -p _out/tests"
    WORKING_DIRECTORY ${ZENDNN_DIR}
    RESULT_VARIABLE TEST_DIR_RESULT)

if(NOT TEST_DIR_RESULT EQUAL 0)
    message(FATAL_ERROR "ZENDNN Tests, _out/tests failed")
endif()

set(test_target_list zendnn_conv_test zendnn_avx_conv zendnn_avx_conv_param zendnn_avx_conv_maxpool zendnn_inference_f32 zendnn_avx_conv_param_fusion zendnn_avx_conv_param_direct ref_avx_conv_param zendnn_avx_conv_primitive_cache_test zendnn_matmul_int zendnn_matmul_generic_int8 zendnn_matmul_int4 zendnn_matmul_test zendnn_grp_matmul_test zendnn_matmulFusions_test zendnn_matmul_gelu_test zendnn_matmul_weight_cache_test zendnn_matmul_bf16_test zendnn_avx_maxpool_blocked embedding_bag_test grp_embedding_bag_test grp_embedding_mlp_test zendnn_block_size zendnn_matmul_inplace embedding_bag_benchmark zendnn_attention_multihead_bf16 zendnn_attention_multihead_f32 zendnn_sdpa_test)

foreach(test_target IN LISTS test_target_list)
    if(${test_target} STREQUAL "zendnn_matmul_int")
        set(src_test_file zendnn_matmul_int8_test)
    elseif(${test_target} STREQUAL "zendnn_matmul_int4")
        set(src_test_file zendnn_matmul_int4_test)
    elseif(${test_target} STREQUAL "zendnn_matmul_generic_int8")
        set(src_test_file zendnn_matmul_generic_int8_test)
    elseif(${test_target} STREQUAL "embedding_bag_test")
        set(src_test_file zendnn_embedding_bag_test)
    elseif(${test_target} STREQUAL "grp_embedding_bag_test")
        set(src_test_file zendnn_grp_embedding_bag_test)
    elseif(${test_target} STREQUAL "grp_embedding_mlp_test")
        set(src_test_file zendnn_grp_embedding_mlp_test)
    elseif(${test_target} STREQUAL "zendnn_matmul_inplace")
        set(src_test_file zendnn_matmul_inplace_custom_op_test)
    elseif(${test_target} STREQUAL "embedding_bag_benchmark")
        set(src_test_file zendnn_embedding_bag_benchmark)
    elseif(${test_target} STREQUAL "zendnn_attention_multihead_bf16")
        set(src_test_file zendnn_multihead_attention_bf16)
    elseif(${test_target} STREQUAL "zendnn_attention_multihead_f32")
        set(src_test_file zendnn_multihead_attention_f32)
    else()
        set(src_test_file ${test_target})
    endif()

    add_executable(${test_target} api_tests/${src_test_file}.cpp)
    target_compile_options(${test_target} PRIVATE -fopenmp)
    target_compile_options(${test_target} PRIVATE -Wno-narrowing)

    target_link_libraries(${test_target} PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${test_target} PRIVATE amdZenDNN)
endforeach()
