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

# sets log level of various logs
# levels are disabled:0, error:1, warning:2, info:3, verbose:4

export ZENDNNL_COMMON_LOG_LEVEL=4
export ZENDNNL_API_LOG_LEVEL=4
export ZENDNNL_TEST_LOG_LEVEL=4
export ZENDNNL_PROFILE_LOG_LEVEL=4
export ZENDNNL_DEBUG_LOG_LEVEL=4

