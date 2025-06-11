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

# sanity check
curr_dir="$(pwd)"
parent_dir="$(dirname "$curr_dir")"
last_dir="$(basename $curr_dir)"

if [ ${last_dir} != "scripts" ];then
    echo "error: <${last_dir}> does not seem to be <scripts> folder."
    return 1;
fi

# set the user log file
zendnnl_build_dir="${parent_dir}/build/install/zendnnl"
export ZENDNNL_CONFIG_FILE="${zendnnl_build_dir}/config/zendnnl_user_config.json"
# sets log level of various logs
# levels are disabled:0, error:1, warning:2, info:3, verbose:4

export ZENDNNL_COMMON_LOG_LEVEL=0
export ZENDNNL_API_LOG_LEVEL=0
export ZENDNNL_TEST_LOG_LEVEL=0
export ZENDNNL_PROFILE_LOG_LEVEL=0
export ZENDNNL_DEBUG_LOG_LEVEL=0

