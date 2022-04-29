#*******************************************************************************
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************

#!/bin/bash

chmod u+x ./scripts/zendnn_ONNXRT_env_setup.sh
echo "source scripts/zendnn_ONNXRT_env_setup.sh"
source scripts/zendnn_ONNXRT_env_setup.sh

#check again if ZENDNN_AOCC_COMP_PATH is defined, otherwise return
if [ -z "$ZENDNN_AOCC_COMP_PATH" ];
then
    echo "Error: Environment variable ZENDNN_AOCC_COMP_PATH needs to be set"
    return
fi

#check again if ZENDNN_BLIS_PATH is defined, otherwise return
if [ -z "$ZENDNN_BLIS_PATH" ];
then
    echo "Error: Environment variable ZENDNN_BLIS_PATH needs to be set"
    return
fi

#check again if ZENDNN_LIBM_PATH is defined, otherwise return
if [ "$ZENDNN_ENABLE_LIBM" = "1" ];
then
    if [ -z "$ZENDNN_LIBM_PATH" ];
    then
        echo "Error: Environment variable ZENDNN_LIBM_PATH needs to be set"
        return
    fi
fi

#echo "make clean"
#make clean

echo "make -j AOCC=1"
make -j AOCC=1

echo "make test AOCC=1"
make test AOCC=1