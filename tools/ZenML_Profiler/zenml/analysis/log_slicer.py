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


def log_slicer(n_iter, total_iter, entire_log, index_list, export, file_name):
    """
    Slices the entire log and provides the log for any particular
    iteration the user wants.

    Parameters:
    - n_iter (int): The iteration on which analysis is to be done.
    - total_iter (int): Total number of iterations in the log.
    - entire_log (string): The entire log content as a string.
    - index_list (list): Contains boundary indices of each iteration.
    - export (bool): Flag to enable exporting the sliced log.
    - file_name (string): Name of the file to save the sliced log.

    Returns:
    - If analysis is done for the last iteration:
        - sl (string): Contains the log of the nth iteration.
        - sl1 (string): Contains the log of the (n-1)th iteration.
    - Otherwise:
        - sl (string): Contains the log of the nth iteration.
    """
    log_lines = entire_log.strip().split("\n")

    if n_iter == total_iter and total_iter != 1 and len(index_list) >= 2:
        sl = "\n".join(log_lines[index_list[-1] + 1 :])
        sl1 = "\n".join(log_lines[index_list[-2] + 1 : index_list[-1] + 1])
    else:
        if total_iter == 1:
            sl = entire_log
        elif n_iter < len(index_list):
            sl = "\n".join(log_lines[index_list[n_iter - 1] + 1 : index_list[n_iter] + 1])
        else:
            sl = entire_log

    # Export the sliced log to a file if the export flag is enabled.
    if export:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(sl)

    if n_iter == total_iter and total_iter != 1 and len(index_list) >= 2:
        return sl, sl1
    else:
        return sl, None
