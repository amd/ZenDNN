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
from zenml.reporting.utils import non_zero_round


def algo_extractor(log, log_info, backend):
    """
    Traverses the log sequentially and creates a custom representation
    of the same log, which will be used for recognizing the pattern.

    Parameters:
    log (string) : contains entire log
    backend (string) : backend used to create the log

    Returns:
    print table with dimensions and algo type used
    """
    algos = []
    algo_details = []
    algo_mapper_bf16 = {
        "0": "Auto",
        "1": "aocl_b16",
        "2": "blocked brgemm",
        "3": "brgemm",
    }
    algo_mapper_f32 = {
        "0": "Auto",
        "1": "cblas",
        "2": "cblas + zen parallel",
        "3": "blocked brg",
        "4": "brg",
        "5": "aocl",
    }
    algo_mapper_woq = {
        "0": "Auto",
        "1": "aocl_s4",
        "2": "simulated blocked brg",
        "3": "simulated aocl_bf16",
    }
    dtype_map = {"bfloat16": "bf16", "float32": "fp32"}
    algo_print = ""
    if "ZenDNN_5.1" == backend:
        if "Data Type" in log:
            for i in log.strip().split("\n"):
                if "Data Type" in i:
                    d_type = (i.split(":")[1]).strip()
                    break
            d_rep = dtype_map.get(d_type, d_type)
        else:
            d_type = "bfloat16"
            d_rep = "bfloat16"
        if "ZENDNN_MATMUL_ALGO" in log:
            for i in log.strip().split("\n"):
                if "ZENDNN_MATMUL_ALGO" in i:
                    zen_mat = (i.split("=")[1]).strip()
                    algo_print = "\nNOTE : The ZENDNN_MATMUL_ALGO is set as : " + str(
                        zen_mat
                    )
                    break
        for i, j in enumerate(log.strip().split("\n")):
            for typ_name in log_info[backend]["algo"]:
                typ = log_info[backend]["algo"][typ_name]
                start = typ["start"]
                start = start.replace("*", d_rep)
                if (
                    start in j
                    and len(j[j.find(start) :].split(typ["sep"])) == typ["num_args"]
                ):
                    # autotuner = j[j.find(start) :].split(typ["sep"])[typ["autotuner"]]
                    algo = j[j.find(start) :].split(typ["sep"])[typ["algotype"]]
                    M = j[j.find(start) :].split(typ["sep"])[typ["M"]].split("=")[1]
                    N = j[j.find(start) :].split(typ["sep"])[typ["N"]].split("=")[1]
                    K = j[j.find(start) :].split(typ["sep"])[typ["K"]].split("=")[1]
                    algo_type = algo.split("=")[1]
                    algos.append(algo_type)
                    if d_type == "bfloat16":
                        algo_name = algo_mapper_bf16.get(algo_type, "Unknown")
                    elif d_type == "float32":
                        algo_name = algo_mapper_f32.get(algo_type, "Unknown")
                    else:
                        algo_name = "Unknown"
                    algo_det = []
                    algo_det.append(M)
                    algo_det.append(N)
                    algo_det.append(K)
                    algo_det.append(algo_type)
                    algo_det.append(algo_name)
                    algo_details.append(algo_det)
        unique_data = [list(x) for x in set(tuple(x) for x in algo_details)]
        if len(set(algos)) > 1:
            # print("\nAlgo used for each dimensions:\n")
            # t = PrettyTable(["M", "N", "K", "ALGO_TYPE", "ALGO USED"])
            # for ud in unique_data:
            #     t.add_row(ud)
            # print(t)
            return 1, unique_data, algo_print
        if len(unique_data) == 0:
            return 3, 0, algo_print
        else:
            return 2, unique_data, algo_print
    else:
        return 0, 0, algo_print


def benchdnn_support(ifile, df, cfile):
    """
    Parameters:
    df (list of dictionary) : contains op details of nth iteration
    ifile (string) : text file to store dimensions
    cfile (string) : text file to store count of each dimensions
    """
    if not df:
        logging.warning("No data available for BenchDNN support.")
        return
    df_details = df[0]
    dims = []
    with open(ifile, "w", encoding="utf-8") as ip_file, open(cfile, "w", encoding="utf-8") as c_file:
        for index, row in df_details.iterrows():
            if row["Op Type"] == "matmul":
                dims.append(row["Dimension"])
                dimstr = str(row["Dimension"])
                m1 = dimstr.split(",")
                m_str = m1[1]
                m = m_str.split(":")[1]
                n_str = m1[2]
                n = n_str.split(":")[1]
                k_str = m1[3]
                k = k_str.split(":")[1]
                app_string = "m" + str(m) + "n" + str(n) + "k" + str(k) + "\n"
                ip_file.write(app_string)
                c_file.write(str((row["Count"])) + "\n")
            elif row["Op Type"] == "inner_product":
                dims.append(row["Dimension"])
                dimstr = str(row["Dimension"])
                m1 = dimstr.split(",")
                m_str = m1[1]
                m = m_str.split(":")[1]
                n_str = m1[2]
                n = n_str.split(":")[1]
                k_str = m1[3]
                k = k_str.split(":")[1]
                app_string = str(m) + "x" + str(k) + ":" + str(k) + "x" + str(n) + "\n"
                ip_file.write(app_string)
                c_file.write(str((row["Count"])) + "\n")
