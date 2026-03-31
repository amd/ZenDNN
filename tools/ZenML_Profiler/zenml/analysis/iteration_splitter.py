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
import time
import sys
import logging
from zenml.config.profiler_config import log_info, SAFE_MODE_OPS
from zenml.parsers.dispatcher import is_kv_format
from zenml.parsers.positional_parser import log_validator
from zenml.parsers.kv_parser import parse_kv_line, kv_build_ind

# Abort iteration splitting if it takes longer than this (seconds).
_ITER_SPLIT_TIMEOUT_S = 60
# Warn the user if iteration splitting exceeds this threshold (seconds).
_ITER_SPLIT_WARN_S = 10
# Prefix string stripped from post-ops fields in ZenDNN logs.
_ATTR_POST_OPS_PREFIX = "attr-post-ops:"


def kv_iteration_splitter(ent_log_op_info, backend, total_iter, entire_log, safe_mode):
    """
    Iteration splitter for KV-format logs.
    Identifies iteration boundaries and anomalies.

    Parameters:
    - ent_log_op_info (dict): Operation info of the entire log.
    - backend (string): Backend name.
    - total_iter (int): Total number of iterations.
    - entire_log (string): The entire log content.
    - safe_mode (bool): Safe mode flag.

    Returns:
    - index_list (list): Boundary indices of each iteration.
    - anomaly (list): Anomaly indices (execute).
    - create_anomaly (list): Anomaly indices (create).
    """
    if total_iter <= 0:
        return [0], [], []

    ref_dict = {}
    for i in ent_log_op_info:
        ref_dict[i] = math.ceil(len(ent_log_op_info[i]) / total_iter)

    ini_time = time.time()

    while True:
        d = {}
        index_list = []
        in_flag = True

        # Iteration splitting only operates on execute-type log lines
        for i, j in enumerate(entire_log.strip().split("\n")):
            matched = False
            for typ_name in log_info[backend]["execute"]:
                typ_cfg = log_info[backend]["execute"][typ_name]
                kv = parse_kv_line(j, typ_cfg)
                if kv is None:
                    continue
                ind, op_time = kv_build_ind(kv, backend, typ_cfg)
                if ind is None or op_time is None:
                    continue

                op = ind.split(",")[0]
                if safe_mode and op not in SAFE_MODE_OPS:
                    continue

                matched = True
                if ind not in d:
                    d[ind] = 1
                else:
                    d[ind] += 1

                if ind in ref_dict and d[ind] > ref_dict[ind]:
                    ref_dict[ind] += 1
                    in_flag = False
                    break

                flag = True
                for r in ref_dict:
                    if r not in d:
                        flag = False
                        break
                    if ref_dict[r] != d[r]:
                        flag = False
                        break
                if flag:
                    d = {}
                    index_list.append(i)
                break  # Only one config should match per line

            if not in_flag:
                break

        if in_flag:
            break

        temp_time = time.time()
        if temp_time - ini_time > _ITER_SPLIT_TIMEOUT_S:
            print("Script Terminated")
            logging.critical(
                "The execution is taking an abnormal amount of time, therefore the program is auto-aborting."
            )
            print("Please check if the logs and the iter value are correct.")
            sys.exit(1)

    temp = [0]
    temp.extend(index_list)
    if len(index_list) == total_iter:
        index_list = temp[:-1]
    else:
        index_list = temp

    return index_list, [], []


def iteration_splitter(ent_log_op_info, backend, total_iter, entire_log, safe_mode):
    """
    Traverses the log sequentially and identifies the boundary
    of each iteration. Also identifies anomalies in the log.
    If safe mode is enabled, only major operations are considered for
    iteration splitting.

    Parameters:
    - ent_log_op_info (dict): Contains operation info of the entire log.
    - backend (string): Backend used to create the log (e.g., ZenDNN).
    - total_iter (int): Total number of iterations in the log.
    - entire_log (string): The entire log content as a string.
    - safe_mode (bool): Flag to enable safe mode (only major operations are considered).

    Returns:
    - index_list (list): Contains boundary indices of each iteration.
    - execute_anomaly (list): Contains indices of anomalies in "execute" operations.
    - create_anomaly (list): Contains indices of anomalies in "create" operations.
    """
    if is_kv_format(backend):
        return kv_iteration_splitter(
            ent_log_op_info, backend, total_iter, entire_log, safe_mode
        )

    if total_iter <= 0:
        return [0], [], []

    ref_dict = (
        {}
    )  # Reference dictionary to track expected counts of operations per iteration.
    anomaly = []  # List to store anomalies in "execute" operations.
    create_anomaly = []  # List to store anomalies in "create" operations.
    new_dict = {}  # Temporary dictionary for auto-tuner adjustments.
    auto_tuner_flag = False  # Flag to check if auto-tuner is enabled.

    # Check if the backend is ZenDNN and if "auto_tuner=" is present in the log.
    if "ZenDNN" in backend and "auto_tuner=" in entire_log:
        auto_tuner_list = []
        # Extract all occurrences of "auto_tuner=" from the log.
        for i in entire_log.strip().split("\n"):
            if "auto_tuner=" in i:
                # Extract the value of "auto_tuner=" from the log line.
                auto_tuner_list.append(i.split("auto_tuner=")[1].split()[0])
        if len(set(auto_tuner_list)) == 1:
            # If all auto-tuner values are consistent, enable the auto-tuner flag.
            if auto_tuner_list[0] == "True":
                auto_tuner_flag = True
        else:
            # Warn the user if inconsistent auto-tuner values are detected.
            logging.warning(
                "There are some custom print statements that print 'auto_tuner=', which is not recommended."
            )
            if max(set(auto_tuner_list), key=auto_tuner_list.count) == "1":
                auto_tuner_flag = True

    # Adjust reference dictionary if auto-tuner is enabled.
    if auto_tuner_flag:
        for i in ent_log_op_info:
            # Calculate the expected count of each operation per iteration.
            ref_dict[i] = math.ceil(len(ent_log_op_info[i]) / total_iter)
            # Simplify the operation identifier for auto-tuner mode.
            temp_ind = i.split(",")[0] + "," + i.split(",")[2]
            new_dict[temp_ind] = ref_dict[i]
        ref_dict = new_dict
    else:
        # Populate the reference dictionary with expected counts for each operation.
        for i in ent_log_op_info:
            ref_dict[i] = math.ceil(len(ent_log_op_info[i]) / total_iter)

    ini_time = time.time()  # Record the start time for execution.

    while True:
        d = {}  # Dictionary to track counts of operations in the current iteration.
        index_list = []  # List to store the indices of iteration boundaries.

        # Iterate through each line in the log.
        for i, j in enumerate(entire_log.strip().split("\n")):
            # Check for anomalies in "create" operations.
            for typ_name in log_info[backend]["create"]:
                typ = log_info[backend]["create"][typ_name]
                in_flag = True
                start = typ["start"]

                # Validate the log line format and arguments.
                if (
                    start in j
                    and len(j[j.find(start) :].split(typ["sep"])) == typ["num_args"]
                ):
                    check_ = log_validator(backend, typ, start, j)
                    if not (
                        (j[j.find(start) :].split(typ["sep"])[typ["time"]])
                        .replace(".", "")
                        .isnumeric()
                    ):
                        # Add to the create_anomaly list if the time field is invalid.
                        if i not in create_anomaly:
                            create_anomaly.append(i)
                    elif check_ == 0:
                        # Add to the create_anomaly list if the log line fails validation.
                        if i not in create_anomaly:
                            create_anomaly.append(i)
                elif (
                    start in j
                    and len(j[j.find(start) :].split(typ["sep"])) != typ["num_args"]
                ):
                    # Add to the create_anomaly list if the number of arguments is incorrect.
                    if i not in create_anomaly:
                        create_anomaly.append(i)

            # Check for anomalies in "execute" operations.
            for typ_name in log_info[backend]["execute"]:
                typ = log_info[backend]["execute"][typ_name]
                in_flag = True
                start = typ["start"]
                check_ = 0

                # Imitation log check
                if (
                    start in j
                    and len(j[j.find(start) :].split(typ["sep"])) == typ["num_args"]
                ):
                    check_ = log_validator(backend, typ, start, j)
                if (
                    start in j
                    and len(j[j.find(start) :].split(typ["sep"])) == typ["num_args"]
                    and (
                        (j[j.find(start) :].split(typ["sep"])[typ["time"]])
                        .replace(".", "")
                        .isnumeric()
                    )
                    and check_ == 1
                    and (
                        j[j.find(start) :].split(typ["sep"])[typ["op"]] in SAFE_MODE_OPS
                        if safe_mode
                        else True
                    )
                ):
                    # Extract operation details.
                    dim = j[j.find(start) :].split(typ["sep"])[typ["dim"]]
                    op = j[j.find(start) :].split(typ["sep"])[typ["op"]]
                    if op == "matmul":
                        # Check if the operation is a batch matmul based on dimensions.
                        if (
                            all(len(ele.split("x")) > 2 for ele in dim.split(":"))
                            and dim.strip().split(":")[0].split("x")[0]
                            == dim.strip().split(":")[1].split("x")[0]
                        ):
                            op = "batch_matmul"
                    ker = j[j.find(start) :].split(typ["sep"])[typ["ker"]]
                    post = j[j.find(start) :].split(typ["sep"])[typ["post"]]
                    data_format = j[j.find(start) :].split(typ["sep"])[typ["format"]]
                    if post == "":
                        post = "NA"
                    elif _ATTR_POST_OPS_PREFIX in post:
                        post = post[len(_ATTR_POST_OPS_PREFIX):]
                    if auto_tuner_flag:
                        ind = op + "," + dim
                    else:
                        ind = (
                            op
                            + ","
                            + ker
                            + ","
                            + dim
                            + ","
                            + post
                            + ","
                            + backend
                            + ","
                            + data_format
                        )
                        if "plugin_op" in typ:
                            fwk = j[j.find(start) :].split(typ["sep"])[typ["plugin_op"]]
                            ind = ind + "," + fwk[fwk.find(":") + 1 :]
                    if ind not in d:
                        # Add the operation to the dictionary if not already present.
                        d[ind] = 1
                    else:
                        # Increment the count if the operation is already present.
                        d[ind] += 1
                    if d[ind] > ref_dict[ind]:
                        # If the count exceeds the reference, adjust the reference and break.
                        ref_dict[ind] += 1
                        in_flag = False
                        break
                elif (
                    start in j
                    and len(j[j.find(start) :].split(typ["sep"])) != typ["num_args"]
                ):
                    # Add to the anomaly list if the number of arguments is incorrect.
                    if i not in anomaly:
                        anomaly.append(i)
                elif len(j[j.find(start) :].split(typ["sep"])) == typ["num_args"]:
                    if not (
                        (j[j.find(start) :].split(typ["sep"])[typ["time"]])
                        .replace(".", "")
                        .isnumeric()
                    ):
                        # Add to the anomaly list if the time field is invalid.
                        if i not in anomaly:
                            anomaly.append(i)
                    elif check_ == 0:
                        # Add to the anomaly list if the log line fails validation.
                        if i not in anomaly:
                            anomaly.append(i)

                # Check if all operations match the reference counts.
                flag = True
                for r in ref_dict:
                    if r not in d:
                        flag = False
                        break
                    if ref_dict[r] != d[r]:
                        flag = False
                        break
                if flag:
                    # If all operations match, reset the dictionary and add the index.
                    d = {}
                    index_list.append(i)

        # Break the loop if all iterations are processed.
        if in_flag:
            break

        temp_time = time.time()
        if temp_time - ini_time > _ITER_SPLIT_TIMEOUT_S:
            print("Script Terminated")
            logging.critical(
                "The execution is taking an abnormal amount of time, therefore the program is auto-aborting."
            )
            print("Please check if the logs and the iter value are correct.")
            if len(anomaly) > 0:
                logging.warning("Some anomalies have been detected in the logs.")
                print(
                    "Please check the following line numbers of the log, and verify if they are correct"
                )
                for i, j in enumerate(anomaly):
                    if i != len(anomaly) - 1:
                        print(str(j + 1), end=" , ")
                    else:
                        print(str(j + 1))
            sys.exit(1)

    check_time = time.time()
    if check_time - ini_time > _ITER_SPLIT_WARN_S:
        logging.warning(
            "The current execution took more time than the ideal time expected to perform the analysis."
        )
        logging.warning(
            "Please double-check the logs and iter value before finalizing the data.\n"
        )

    # Adjust the index list to include the start of the log.
    temp = [0]
    temp.extend(index_list)
    if len(index_list) == total_iter:
        # If the number of boundaries matches the total iterations, exclude the last boundary.
        index_list = temp[:-1]
    else:
        # Otherwise, include all boundaries.
        index_list = temp

    return index_list, anomaly, create_anomaly
