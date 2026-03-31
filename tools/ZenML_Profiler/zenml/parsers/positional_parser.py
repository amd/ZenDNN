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

import logging
from zenml.config.profiler_config import log_info, SAFE_MODE_OPS

_ATTR_POST_OPS_PREFIX = "attr-post-ops:"


def log_validator(backend, typ, start, log_line):
    """
    Validating the logs. Check if logs are in required format.

    Parameters:
    - backend (string): Backend used to create the log (e.g., ZenDNN).
    - typ (dict): Type configuration for the log (e.g., execute/create/prof).
    - start (string): Starting keyword of the log line.
    - log_line (string): The log line to validate.

    Returns:
    - check_ (int): 1 if the log line passes validation, 0 otherwise.
    """
    j = log_line  # The log line being validated.
    check_ = 0  # Initialize the validation flag to 0 (invalid).

    # Extract the dimensions, separator, and operation type from the log line.
    dims = j[j.find(start) :].split(typ["sep"])[typ["dim"]]
    sep = typ["dim_sep"]
    op = j[j.find(start) :].split(typ["sep"])[typ["op"]]

    try:
        # Iterate through each separator defined in the log type configuration.
        for s in sep:
            if s == ":":
                # Check if the separator ":" exists in the dimensions string.
                if s in dims:
                    dims_split = dims.split(s)
                    for di in range(len(dims_split)):
                        if "x" in dims_split[di]:
                            # If "x" is found in the dimension, mark it as valid.
                            check_ = 1
                            # For ZenDNN logs, ensure the log line ends with "ms".
                            if backend == "ZenDNN_5.1":
                                if (j[j.find(start) :].split(typ["sep"])[-1]) != "ms":
                                    check_ = 0
                        else:
                            # If "x" is not found, mark it as invalid.
                            check_ = 0
            elif s == "x":
                # Check if the separator "x" exists in the dimensions string.
                if s in dims:
                    dims_split = dims.split(s)
                    for ele in dims_split:
                        # Ensure each part of the dimension is numeric (ignoring dots).
                        ele_check = ele.replace(".", "").isnumeric()
                        if ele_check:
                            check_ = 1
                        else:
                            check_ = 0
            elif s in ("mb", "oc", "ic"):
                # Special case for "mb", "oc", and "ic" separators.
                if "x" not in dims:
                    # Extract and validate the "mb", "ic", and "oc" parts of the dimension.
                    m = (
                        (dims.split("mb")[1].split("ic")[0])
                        .replace(".", "")
                        .isnumeric()
                    )
                    k = (
                        (dims.split("ic")[1].split("oc")[0])
                        .replace(".", "")
                        .isnumeric()
                    )
                    n = (dims.split("oc")[1]).replace(".", "").isnumeric()
                    if m and k and n:
                        check_ = 1
                    else:
                        check_ = 0
            else:
                # For other separators, check if the separator exists in the dimensions string.
                if s in dims:
                    check_ = 1
                else:
                    check_ = 0

        # Additional validation for "convolution" operations.
        if op == "convolution":
            # Ensure that "mb", "oc", and "ic" exist in the dimensions string.
            for d in ["mb", "oc", "ic"]:
                if d in dims:
                    check_ = 1
                else:
                    check_ = 0
                    break

        return check_
    except Exception as e:
        # Log any exceptions for debugging purposes.
        logging.debug(e)
        check_ = 0
        return check_


def positional_log_parser(log, backend, safe_mode, log_type, in_sc, t_time=False, special=False):
    """
    Parse the log iteratively based on the backend, to extract necessary
    information and store it in a dictionary, paired with corresponding
    timings. Also grep the total time from the log if it is present.

    Parameters:
    - log (string) : string containing the log that needs to be parsed
    - backend (string) : backend used to create the log
    - safe_mode (bool) : flag for enabling safe mode
    - log_type (string) : type of log (execute/create/prof)
    - in_sc (string) : inference script used for log creation

    Optional parameters:
    - t_time (bool) : bool value to return the total time
    - special (bool) : to analyse only the faulty ops

    Returns:
    If t_time is True:
        - op_info_dict (Dictionary) : Contains op level details as key and it
        timing as values
        - t_t (string) : total time grepped from the log (if present)
    If t_time is False:
        - op_info_dict (Dictionary) : Contains op level details as key and it
        timing as values
    """
    op_info_dict = {}  # Dictionary to store operation details and timings.
    t_t = ""  # Total time extracted from the log.
    log_len = 1  # Flag to track log line length consistency.

    # Iterate through each line in the log.
    for i, j in enumerate(log.strip().split("\n")):
        # Iterate through each type configuration for the given backend and log type.
        for typ_name in log_info[backend][log_type]:
            try:
                typ = log_info[backend][log_type][typ_name]
                start = typ["start"]

                # Special handling for Hugging Face logs.
                if in_sc == "hugging_face":
                    if (
                        "Model Name             Batch Size     Seq Length     Time in s"
                        in j
                    ):
                        # Extract total time from the log if the specific pattern is found.
                        t_t = log.strip().split("\n")[i + 2].split()[-1]
                    if "Latency for step " in j:
                        # Convert latency from milliseconds to seconds.
                        t_t = str(float(j[16:].strip().split(" ")[1]) / 1000)

                # Check if the log line starts with the expected keyword and has the correct number of arguments.
                if (
                    start in j
                    and len(j[j.find(start) :].split(typ["sep"])) == typ["num_args"]
                ):
                    check_ = log_validator(backend, typ, start, j)

                # Perform additional checks if the log line passes initial validation.
                if (
                    start in j
                    and len(j[j.find(start) :].split(typ["sep"])) == typ["num_args"]
                    and (j[j.find(start) :].split(typ["sep"])[typ["time"]])
                    .replace(".", "")
                    .isnumeric()  # Ensure the time field is numeric.
                    and check_ == 1  # Ensure the log line passes validation.
                    and (
                        (
                            j[j.find(start) :].split(typ["sep"])[typ["op"]]
                            in SAFE_MODE_OPS  # Check if the operation is in the safe mode list.
                            if safe_mode
                            else True
                        )
                        if not special
                        else j[j.find(start) :].split(typ["sep"])[typ["op"]]
                        not in SAFE_MODE_OPS  # Exclude non safe mode operations if `special` is True.
                    )
                ):
                    # Extract operation details from the log line.
                    dim = j[j.find(start) :].split(typ["sep"])[typ["dim"]]
                    op = j[j.find(start) :].split(typ["sep"])[typ["op"]]
                    if "plugin_op" in typ:
                        fwk = j[j.find(start) :].split(typ["sep"])[typ["plugin_op"]]
                    if op == "matmul":
                        # Check if the operation is a batch matmul based on the dimensions.
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
                    # Create a unique identifier for the operation.
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
                        ind = ind + "," + fwk[fwk.find(":") + 1 :]
                    if op == "embedding_bag":
                        # Handle special case for "embedding_bag" operation.
                        if log_type != "prof":
                            op_time = float(
                                j[j.find(start) :].split(typ["sep"])[typ["time"] + 1]
                            )
                        else:
                            op_time = (
                                j[j.find(start) :].split(typ["sep"])[typ["time"] + 1]
                                if ";"
                                in j[j.find(start) :].split(typ["sep"])[typ["time"] + 1]
                                else ""
                            )
                    else:
                        # Extract the operation time for other operations.
                        if log_type != "prof":
                            op_time = float(
                                j[j.find(start) :].split(typ["sep"])[typ["time"]]
                            )
                        else:
                            op_time = (
                                j[j.find(start) :].split(typ["sep"])[typ["time"]]
                                if ";"
                                in j[j.find(start) :].split(typ["sep"])[typ["time"]]
                                else ""
                            )

                    # Add the operation details and timing to the dictionary.
                    if ind not in op_info_dict:
                        op_info_dict[ind] = [op_time]
                    else:
                        op_info_dict[ind].append(op_time)

                # If the log line length does not match the expected number of arguments, set the flag to 0.
                if (
                    start in j
                    and "execute" in start
                    and len(j[j.find(start) :].split(typ["sep"])) != typ["num_args"]
                ):
                    log_len = 0
            except Exception as e:
                # Log any exceptions for debugging purposes.
                logging.debug(
                    f"Error occurred while parsing the following line (Line num: {i}) :"
                )
                logging.debug(j)

    # Log a warning if any log line length inconsistencies were detected.
    if log_len == 0:
        logging.warning(
            "The number of parameters in the log doesn't match the expected number of params."
        )

    # If `t_time` is True, return the total time along with the operation details.
    if t_time:
        try:
            float(t_t.strip())
            return op_info_dict, t_t
        except (ValueError, AttributeError):
            return op_info_dict, ""

    # Return the operation details dictionary.
    return op_info_dict
