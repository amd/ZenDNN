#*******************************************************************************
# Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#!/bin/bash

if [[ "$1" == "aocc" ]];
then
chmod u+x ./scripts/zendnn_aocc_env_setup.sh
echo "source scripts/zendnn_aocc_env_setup.sh"
source scripts/zendnn_aocc_env_setup.sh
make_args="AOCC=1"
          elif [[ "$1" == "gcc" ]];
then
chmod u+x ./scripts/zendnn_gcc_env_setup.sh
echo "source scripts/zendnn_gcc_env_setup.sh"
source scripts/zendnn_gcc_env_setup.sh
make_args="AOCC=0"
          else
              echo "Input not recognised.  First argument should be one of"
              echo "aocc / gcc"
              return 1;
fi

make_lpgemm_args=""
                 if [[ "$2" == "lpgemm" ]];
then
make_lpgemm_args="LPGEMM=1"
                 fi
                 if [[ "$2" == "lpgemm_v5_0" ]];
then
make_lpgemm_args="LPGEMM=1 LPGEMM_V4_2=1 LPGEMM_V5_0=1"
                 fi

                 if [[ "$2" == "lpgemm_v4_2" ]];
then
make_lpgemm_args="LPGEMM=1 LPGEMM_V4_2=1"
                 fi

#check again if ZENDNN_BLIS_PATH is defined, otherwise return
                 if [ -z "$ZENDNN_BLIS_PATH" ];
then
echo "Error: Environment variable ZENDNN_BLIS_PATH needs to be set"
return
    fi

#echo "make clean"
#make clean

    echo "make -j $make_args $make_lpgemm_args"
    make -j $make_args $make_lpgemm_args

    echo "make test $make_args $make_lpgemm_args"
    make test $make_args $make_lpgemm_args
