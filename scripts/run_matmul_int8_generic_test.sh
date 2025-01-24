#!/bash/bin
#*******************************************************************************
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
#
#*******************************************************************************

#uint8

_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #no ops
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-relu --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #relu
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #mul-add
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #sigmoid
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-relu --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #all ops
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-relu --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #relu, mul-add
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #mul-add, sig
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-relu --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #relu,sig

#uint8 with bf16 add,mul

_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #mul-add
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-relu --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #all ops
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-relu --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #relu, mul-add
_out/tests/zendnn_matmul_generic2 --enable-dst-u8 --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #mul-add, sig

#dst-s8

_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #no ops
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-relu --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #relu
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #mul-add
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #sigmoid
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-relu --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #all ops
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-relu --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #relu, mul-add
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #mul-add, sig
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-relu --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #relu,sig

#dst-s8 with bf16 add,mul

_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #mul-add
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-relu --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #all ops
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-relu --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #relu, mul-add
_out/tests/zendnn_matmul_generic2 --enable-dst-s8 --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #mul-add, sig

#bf16

_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-src-wei-scales --enable-src-zp --enable-dst-zp #no ops
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-relu --enable-src-wei-scales --enable-src-zp --enable-dst-zp #relu
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-mul-add --enable-src-wei-scales --enable-src-zp --enable-dst-zp #mul-add
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-sigmoid --enable-src-wei-scales --enable-src-zp --enable-dst-zp #sigmoid
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-relu --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-src-zp --enable-dst-zp #all ops
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-relu --enable-mul-add --enable-src-wei-scales --enable-src-zp --enable-dst-zp #relu, mul-add
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-src-zp --enable-dst-zp #mul-add, sig
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-relu --enable-sigmoid --enable-src-wei-scales --enable-src-zp --enable-dst-zp #relu,sig

#bf16 with bf16 add,mul

_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-mul-add --enable-src-wei-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #mul-add
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-relu --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #all ops
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-relu --enable-mul-add --enable-src-wei-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #relu, mul-add
_out/tests/zendnn_matmul_generic2 --enable-bf16 --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-src-zp --enable-dst-zp --enable-bf16-mul-add #mul-add, sig

#fp32

_out/tests/zendnn_matmul_generic2 --enable-fp32 --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #no ops
_out/tests/zendnn_matmul_generic2 --enable-fp32 --enable-relu --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #relu
_out/tests/zendnn_matmul_generic2 --enable-fp32 --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #mul-add
_out/tests/zendnn_matmul_generic2 --enable-fp32 --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #sigmoid
_out/tests/zendnn_matmul_generic2 --enable-fp32 --enable-relu --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #all ops
_out/tests/zendnn_matmul_generic2 --enable-fp32 --enable-relu --enable-mul-add --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #relu, mul-add
_out/tests/zendnn_matmul_generic2 --enable-fp32 --enable-mul-add --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #mul-add, sig
_out/tests/zendnn_matmul_generic2 --enable-fp32 --enable-relu --enable-sigmoid --enable-src-wei-scales --enable-dst-scales --enable-src-zp --enable-dst-zp #relu,sig
