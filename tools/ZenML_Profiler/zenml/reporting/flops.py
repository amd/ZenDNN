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
import json
import logging
import os
import subprocess
import sys
from zenml.reporting.utils import non_zero_round

# Default roofline directory (legacy fallback).
_DEFAULT_ROOFLINE_DIR = "../../Roofline/"


def flops_dict_maker(op_info_dict, t_time, args, log_index):
    """
    Creates flops table dictionary

    Parameters:
    op_info_dict (dictionary) : contains op details of nth iteration
    t_time (string) : total time for each iteration
    args (ArgParser) : Argument parser object
    log_index (int) : Index of log for which result is being generated

    Returns:
    flops_dict (dictionary) : contains flops table information
    """
    flops_dict = {
        "Backend": [],
        "Op Type": [],
        "Kernel": [],
        "Post ops": [],
        "Data Format": [],
        "Count": [],
        "Batches": [],
        "Dimension": [],
    }
    extra = {
        "Flag": False,
        "Frequency": -1,
        "DoublePumpMultiplier": -1,
        "Threads": -1,
        "Dtype": {},
    }
    flops_dict["% Impact"] = []
    flops_dict["Kernel Efficiency"] = []

    flops_dict["  FLOPS details  "] = []
    total_lib_time = sum(sum(op_info_dict[k]) for k in op_info_dict)
    if t_time != "":
        impact_denominator = float(t_time) * 1000
    else:
        impact_denominator = total_lib_time if total_lib_time > 0 else 1
    roofline_dir = getattr(args, "roofline_path", "") or _DEFAULT_ROOFLINE_DIR
    machines_config = os.path.join(roofline_dir, "machines_config.json")
    if args.flops:
        extra["Flag"] = True
        try:
            with open(machines_config, "r", encoding="utf-8") as f:
                flops_data = json.load(f)
            machine_key = args.machine
            machine_cfg = flops_data[machine_key]
            frequency = float(machine_cfg["frequency"] / 1000000000)
            threads = int(machine_cfg["coreCount"])
            alt_key = args.machine.replace("_", " ")
            DoublePumpMultiplier = float(flops_data.get(alt_key, machine_cfg)["avxDoublePumpMultiplier"])
            extra["Frequency"] = frequency
            extra["DoublePumpMultiplier"] = DoublePumpMultiplier
            extra["Threads"] = threads
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            logging.warning(f"Could not load machine config for '{args.machine}': {e}")
            args.flops = False
            extra["Flag"] = False
    for i in op_info_dict:
        if i.split(",")[0] in ["matmul", "batch_matmul", "inner_product", "gemm_api"]:
            if i.split(",")[0] in ["matmul", "batch_matmul", "gemm_api"]:
                m = int(i.split(",")[2].split(":")[0].split("x")[-2])
                n = int(i.split(",")[2].split(":")[1].split("x")[-1])
                k = int(i.split(",")[2].split(":")[0].split("x")[-1])
                dim = (
                    str(m)
                    + "x"
                    + str(k)
                    + ":"
                    + str(k)
                    + "x"
                    + str(n)
                    + ":"
                    + str(m)
                    + "x"
                    + str(n)
                )
                if i.split(",")[0] == "batch_matmul":
                    batch_parts = i.split(",")[2].split(":")[0].split("x")[:-2]
                    batches = 1
                    for bp in batch_parts:
                        batches *= int(bp)
                else:
                    batches = 1
            else:
                m = int(i.split(",")[2].split("mb")[1].split("ic")[0])
                k = int(i.split(",")[2].split("ic")[1].split("oc")[0])
                n = int(i.split(",")[2].split("oc")[1])
                dim = (
                    str(m)
                    + "x"
                    + str(k)
                    + ":"
                    + str(k)
                    + "x"
                    + str(n)
                    + ":"
                    + str(m)
                    + "x"
                    + str(n)
                )
                batches = 1
            gops = 0.000000001 * 2 * m * n * k * batches
            count = len(op_info_dict[i])
            time = non_zero_round(sum(op_info_dict[i]) / count, 3) if count > 0 else 0
            fl = (gops / time) * 1000 if time > 0 else 0
            flops_dict["Backend"].append(i.split(",")[4])
            flops_dict["Op Type"].append(i.split(",")[0])
            flops_dict["Kernel"].append(i.split(",")[1])
            flops_dict["Post ops"].append(i.split(",")[3])
            flops_dict["Data Format"].append(i.split(",")[5])
            flops_dict["Count"].append(len(op_info_dict[i]))
            flops_dict["Batches"].append(batches)
            flops_dict["Dimension"].append(dim)
            prc = None
            ap = None
            wp = None
            for pr in i.split(",")[5].split("::"):
                if "dst_" in pr and "bf16" in pr[pr.find("dst_") + 4 :]:
                    prc = 16
                    ap = "bf16"
                elif "dst_" in pr and "f32" in pr[pr.find("dst_") + 4 :]:
                    prc = 32
                    ap = "fp32"
                elif "dst_" in pr and "s4" in pr[pr.find("dst_") + 4 :]:
                    prc = 4
                    ap = "int4"
                if "wei_" in pr and "bf16" in pr[pr.find("wei_") + 4 :]:
                    wp = "bf16"
                elif "wei_" in pr and "f32" in pr[pr.find("wei_") + 4 :]:
                    wp = "fp32"
                elif "wei_" in pr and "s4" in pr[pr.find("wei_") + 4 :]:
                    wp = "int4"
            if args.flops:
                if prc is None or ap is None or wp is None:
                    logging.warning(
                        "Unrecognized data format for roofline: %s. Skipping efficiency calculation.",
                        i.split(",")[5],
                    )
                    flops_dict["Kernel Efficiency"].append("-")
                else:
                    try:
                        act_fl = 512 / prc * threads * 2 * 2 * DoublePumpMultiplier * frequency
                        cmd = [sys.executable, "mmRoofline.py", "-m", str(m), "-n", str(n), "-k", str(k), "-mc", str(args.machine), "-ap", str(ap), "-wp", str(wp)]
                        p = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            cwd=roofline_dir,
                        )
                        out, err = p.communicate()
                        result = out.strip()
                        tokens = result.split()
                        if len(tokens) < 7:
                            raise ValueError(f"Unexpected roofline output: {result!r}")
                        roofline_time = max(
                            float(tokens[2]) / 1000, float(tokens[6]) / 1000
                        )
                        rf_fl = (gops / roofline_time) * 1000
                        rf_eff = non_zero_round((rf_fl / act_fl) * 100, 3)
                        eff = non_zero_round((fl / act_fl) * 100, 3)
                        tmp_eff = (
                            "Achieved:"
                            + str(eff)
                            + "% Roofline:"
                            + str(rf_eff)
                            + "% Offset:"
                            + str(non_zero_round(rf_eff - eff, 3))
                        )
                        flops_dict["Kernel Efficiency"].append(tmp_eff)
                        if ap not in extra["Dtype"]:
                            extra["Dtype"][ap] = act_fl
                    except Exception as roofline_err:
                        logging.warning(
                            "Roofline calculation failed for %s: %s. Skipping.",
                            dim, roofline_err,
                        )
                        flops_dict["Kernel Efficiency"].append("-")
            else:
                flops_dict["Kernel Efficiency"].append("-")
            # flops_dict["Time"].append(time)
            # flops_dict["GFlops"].append(non_zero_round(fl, 3))
            tmp_fl = (
                "GOPS:"
                + str(non_zero_round(gops, 5))
                + " Time:"
                + str(time)
                + " GFlops:"
                + str(non_zero_round(fl, 3))
            )
            flops_dict["  FLOPS details  "].append(tmp_fl)
            flops_dict["% Impact"].append(
                non_zero_round(
                    (sum(op_info_dict[i]) / impact_denominator) * 100, 2
                )
            )
    return flops_dict, extra
