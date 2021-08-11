#*******************************************************************************
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
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
