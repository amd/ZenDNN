#!/bin/bash
# *******************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

curr_dir="$(pwd)"
parent_dir="$(dirname "$curr_dir")"
last_dir="$(basename $curr_dir)"

if [ ${last_dir} != "scripts" ];then
    echo "error: <${last_dir}> does not seem to be <scripts> folder."
    return;
fi

echo "switching to <${parent_dir}>."
cd ${parent_dir}

if [ ! -d "build" ];then
    echo "creating <build> directory."
    mkdir -p build
else
    echo "<build> directory exists."
fi

if [ ! -z "$(ls -A "./build")" ];then
    echo "<build> is not empty. please empty it manually for fresh build."
fi

echo "building zendnnl..."
cd build
cmake ..
cmake --build .

cd ${curr_dir}
