#*******************************************************************************
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
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
#
#*******************************************************************************

$ZENDNN_GIT_ROOT/_out/tests/zendnn_conv_test cpu
$ZENDNN_GIT_ROOT/_out/tests/zendnn_avx_conv cpu
$ZENDNN_GIT_ROOT/_out/tests/zendnn_avx_conv_param cpu
$ZENDNN_GIT_ROOT/_out/tests/zendnn_avx_conv_maxpool cpu
$ZENDNN_GIT_ROOT/_out/tests/zendnn_inference_f32 cpu
#$ZENDNN_GIT_ROOT/_out/tests/zendnn_training_f32 cpu
#Below is backward compatibility test for current dnnl wrt to original dnnl sha1
#$ZENDNN_GIT_ROOT/_out/tests/ref_avx_conv_param cpu
$ZENDNN_GIT_ROOT/_out/tests/zendnn_avx_conv_param_fusion cpu
