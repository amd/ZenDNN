# *******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# *******************************************************************************

from zenml.config.profiler_config import log_info
from zenml.parsers.positional_parser import positional_log_parser
from zenml.parsers.kv_parser import kv_log_parser


def is_kv_format(backend):
    """Check if a backend uses key-value log format."""
    return log_info.get(backend, {}).get("format") == "kv"


def parse_log(log, backend, safe_mode, log_type, inference_script, t_time=False, special=False):
    """
    Unified log parser that dispatches to the correct backend parser.

    Parameters:
    log (str)              : Log content to parse.
    backend (str)          : Backend name (e.g., "ZenDNN_5.1", "ZenDNN_5.2").
    safe_mode (bool)       : Only parse safe_mode_ops if True.
    log_type (str)         : "execute", "create", or "prof".
    inference_script (str) : "custom" or "hugging_face".
    t_time (bool)          : If True, also return the total time.
    special (bool)         : If True, parse only non-safe-mode ops.

    Returns:
    op_info_dict, or (op_info_dict, total_time_str) if t_time is True.
    """
    if is_kv_format(backend):
        return kv_log_parser(log, backend, safe_mode, log_type, inference_script, t_time, special)
    return positional_log_parser(log, backend, safe_mode, log_type, inference_script, t_time, special)
