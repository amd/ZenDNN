#*******************************************************************************
# Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************

#!/bin/bash

if [[ "$1" == "aocc" ]]; then
    chmod u+x ./scripts/zendnn_aocc_env_setup.sh
    echo "source scripts/zendnn_aocc_env_setup.sh"
    source scripts/zendnn_aocc_env_setup.sh
    make_args="AOCC=1"
elif [[ "$1" == "gcc" ]]; then
    chmod u+x ./scripts/zendnn_gcc_env_setup.sh
    echo "source scripts/zendnn_gcc_env_setup.sh"
    source scripts/zendnn_gcc_env_setup.sh
    make_args="AOCC=0"
else
    echo "Input not recognised.  First argument should be one of"
    echo "aocc / gcc"
    return 1;
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

echo "make -j $make_args"
make -j $make_args

echo "make test $make_args"
make test $make_args
