#*******************************************************************************
# Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

CURRENT_FOLDER=$(pwd)
#-----------------------------------------------------------------------------
# omp env

unset OMP_PLACES
unset OMP_PROC_BIND
unset OMP_WAIT_POLICY
unset GOMP_CPU_AFFINITY
unset OMP_MAX_ACTIVE_LEVELS
unset OMP_NUM_THREADS

export OMP_PLACES=cores
export OMP_PROC_BIND=FALSE
export OMP_WAIT_POLICY=PASSIVE
export OMP_MAX_ACTIVE_LEVELS=10
export OMP_NUM_THREADS=64
#export GOMP_CPU_AFFINITY="0-31"

export ZENDNN_PRIMITIVE_LOG_ENABLE=1
export ZENDNN_LOG_OPTS=ALL:-1,PROF:2,PERF:0

#-----------------------------------------------------------------------------
#experimental settings
export EMB_ROWS=10
export EMB_NTHREADS=2
export ITERATIONS=100

cd /home/ishita/ZenDNN/_out/tests/
./embedding_bag_benchmark cpu $EMB_ROWS $EMB_NTHREADS $ITERATIONS
cd $CURRENT_FOLDER
