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

import math
from zenml.config.profiler_config import log_info, SAFE_MODE_OPS
from zenml.parsers.dispatcher import is_kv_format
from zenml.parsers.kv_parser import parse_kv_line, kv_build_ind

# Fraction of the log to scan when searching for a repeating pattern.
# 75% ensures we detect patterns even if the last partial iteration is incomplete.
_PATTERN_SCAN_RATIO = 0.75


def kv_custom_log_generator(log, backend):
    """
    Generate custom log representation for KV-format logs, used for
    pattern analysis / iteration detection.

    Parameters:
    - log (string): The entire log.
    - backend (string): The backend name.

    Returns:
    - custom_log (list): List of 'op,dim' strings in sequential order.
    """
    custom_log = []
    # Pattern detection only operates on execute-type log lines
    for j in log.strip().split("\n"):
        for typ_name in log_info[backend]["execute"]:
            typ_cfg = log_info[backend]["execute"][typ_name]
            kv = parse_kv_line(j, typ_cfg)
            if kv is None:
                continue
            ind, _ = kv_build_ind(kv, backend, typ_cfg)
            if ind is None:
                continue
            op = ind.split(",")[0]
            dim = ind.split(",")[2]
            if op in SAFE_MODE_OPS:
                custom_log.append(op + "," + dim)
            break
    return custom_log


def custom_log_generator(log, backend):
    """
    Traverses the log sequentially and creates a custom representation
    of the same log, which will be used for recognizing the pattern.

    Parameters:
    - log (string): The entire log content to process.
    - backend (string): The backend used to create the log (e.g., ZenDNN).

    Returns:
    - custom_log (list): A list of strings representing a custom format of the log.
                         Each entry contains the operation type and dimensions.
    """
    if is_kv_format(backend):
        return kv_custom_log_generator(log, backend)

    custom_log = []  # List to store the custom representation of the log.

    # Iterate through each line in the log.
    for _lineno, j in enumerate(log.strip().split("\n")):
        # Iterate through each type configuration for the "execute" log type.
        for typ_name in log_info[backend]["execute"]:
            typ = log_info[backend]["execute"][typ_name]
            start = typ["start"]

            # Check if the log line matches the expected format.
            if (
                start in j
                and len(j[j.find(start) :].split(typ["sep"])) == typ["num_args"]
                and j[j.find(start) :].split(typ["sep"])[typ["op"]] in SAFE_MODE_OPS
            ):
                # Extract the operation type and dimensions.
                dim = j[j.find(start) :].split(typ["sep"])[typ["dim"]]
                op = j[j.find(start) :].split(typ["sep"])[typ["op"]]

                # Make a custom representation of the log with just operation and dimensions
                # And append the new custom representation of the current line to the custom log.
                custom_log.append(op + "," + dim)
    return custom_log


def pattern_analyzer(custom_log):
    """
    Analyzes the log to detect repeating patterns. It iterates through the log
    and splits miniature versions of the log from the beginning to check if
    they form a repeating pattern.

    Parameters:
    - custom_log (list): A custom representation of the log (output of `custom_log_generator`).

    Returns:
    - iteration (int): The number of total iterations calculated based on the repeating pattern.
    """
    result = []  # List to store the detected repeating pattern.

    for i in range(round(len(custom_log) * _PATTERN_SCAN_RATIO)):
        # Check if the current segment of the log repeats in the remaining log.
        if repeat_checker(custom_log[: i + 1], custom_log[i + 1 :]):
            result.extend(custom_log[: i + 1])  # Store the repeating pattern.
            break

    # Calculate the total number of iterations based on the repeating pattern.
    if len(result) > 0:
        iteration = math.ceil(len(custom_log) / len(result))
    else:
        iteration = 1  # If no pattern is found, assume a single iteration.

    return iteration


def repeat_checker(mini_log, remaining_log):
    """
    Checks whether the mini log (a segment of the log) repeats consistently
    throughout the remaining log without breaking.

    Parameters:
    - mini_log (list): A segment of the log to check for repetition.
    - remaining_log (list): The remaining part of the log to check against.

    Returns:
    - flag (bool): True if the mini log repeats consistently, False otherwise.
    """
    if not mini_log:
        return False

    flag = True

    for i in range(0, len(remaining_log), len(mini_log)):
        if i + len(mini_log) < len(remaining_log):
            # Compare the current chunk with the mini log.
            if "+".join(mini_log) != "+".join(remaining_log[i : i + len(mini_log)]):
                flag = False  # If they don't match, set the flag to False.
                break
        else:
            # Handle the last chunk of the remaining log.
            ent_log_len = len(remaining_log[i:])
            end_limit = round(ent_log_len * _PATTERN_SCAN_RATIO)
            if "+".join(mini_log[:end_limit]) != "+".join(
                remaining_log[i : i + end_limit]
            ):
                flag = False  # If they don't match, set the flag to False.
                break

    return flag
