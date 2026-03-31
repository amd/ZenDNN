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
import re
from prettytable import PrettyTable, ALL
from zenml.reporting.utils import non_zero_round, print_impact_footnote


def compare_order_finder(lists):
    """
    Finds the display order of ops in the compare table.

    Parameters:
    lists (list of lists) : contains the list of all op type and
                            dimension of all logs

    Returns:
    sorted_common_strings (list) : contains sequential order of ops
                                   in sorted order for compare table
    """
    # No logs to compare -- return empty order
    if not lists:
        return []
    common_strings = set(lists[0])
    for lst in lists[1:]:
        common_strings &= set(lst)
    counts = {}
    for lst in lists:
        for string in lst:
            if string in counts:
                counts[string] += 1
            else:
                counts[string] = 1
    sorted_common_strings = sorted(
        common_strings, key=lambda x: counts[x], reverse=True
    )
    sub_sorted_common_strings = []
    for string in lists[0]:
        if string not in sorted_common_strings:
            sub_sorted_common_strings.append(string)
    sub_sorted_common_strings = sorted(
        sub_sorted_common_strings, key=lambda x: counts[x], reverse=True
    )
    sorted_common_strings.extend(sub_sorted_common_strings)
    for i in range(1, len(lists)):
        for string in lists[i]:
            if string not in sorted_common_strings:
                sorted_common_strings.append(string)
    return sorted_common_strings


def _build_matmul_compare_table(
    compare_order, new_key_list, new_dict_list, new_key_kernel_list,
    log_name_list, impact_denominators, n_logs
):
    """
    Build a matmul/batch_matmul focused comparison table with unified
    B, M, N, K columns instead of per-log raw dimension strings.

    Parameters:
    compare_order (list)        : Ordered list of normalized op keys across all logs.
    new_key_list (list of lists): Per-log lists of normalized op keys (e.g., "matmul,384M384N64K").
    new_dict_list (list of dicts): Per-log dicts mapping "op_key,kernel" -> (old_key, [times]).
    new_key_kernel_list (list of lists): Per-log lists of kernel names corresponding to new_key_list.
    log_name_list (list)        : Display names for each log.
    impact_denominators (list)  : Per-log denominators for % Impact calculation.
    n_logs (int)                : Number of logs being compared.

    Returns:
    None if no matmul/batch_matmul ops found, otherwise a tuple of:
        mm_table (PrettyTable)  : The formatted comparison table.
        mm_columns (list)       : Column header names.
        mm_rows (list of lists) : Raw row data for Excel export.
    """
    matmul_ops = [op for op in compare_order if op.split(",")[0] in ["matmul", "batch_matmul"]]
    if not matmul_ops:
        return None

    # Column layout groups similar fields together (matching the normal
    # comparison table):  B,M,N,K | Kernels... | Times... | Counts... |
    # %Impacts... | Ratios...
    mm_columns = ["B", "M", "N", "K"]
    for ln in log_name_list:
        mm_columns.append(f"{ln} Kernel")
    for ln in log_name_list:
        mm_columns.append(f"{ln} Time (ms)")
    for ln in log_name_list:
        mm_columns.append(f"{ln} Count")
    for ln in log_name_list:
        mm_columns.append(f"{ln} % Impact")
    for ln in log_name_list[1:]:
        mm_columns.append(f"{log_name_list[0]}/{ln} Time")

    mm_table = PrettyTable(align="c")
    mm_table.hrules = ALL
    mm_table.field_names = mm_columns

    mm_rows = []
    for op in matmul_ops:
        dim_str = op.split(",")[1]
        b_match = re.search(r"(\d+)B", dim_str)
        m_match = re.search(r"(\d+)M", dim_str)
        n_match = re.search(r"(\d+)N", dim_str)
        k_match = re.search(r"(\d+)K", dim_str)
        B = int(b_match.group(1)) if b_match else 1
        M = int(m_match.group(1)) if m_match else 0
        N = int(n_match.group(1)) if n_match else 0
        K = int(k_match.group(1)) if k_match else 0

        kernels = []
        times = []
        counts = []
        impacts = []
        for ind in range(n_logs):
            if op in new_key_list[ind]:
                temp_ker = "," + new_key_kernel_list[ind][new_key_list[ind].index(op)]
                ker = new_dict_list[ind][op + temp_ker][0].split(",")[1]
                op_time = sum(new_dict_list[ind][op + temp_ker][1])
                count = len(new_dict_list[ind][op + temp_ker][1])
                impact = non_zero_round((op_time / impact_denominators[ind]) * 100, 2)
                kernels.append(ker)
                times.append(non_zero_round(op_time, 2))
                counts.append(count)
                impacts.append(impact)
            else:
                kernels.append("-")
                times.append("-")
                counts.append("-")
                impacts.append("-")

        ratios = []
        for ind in range(1, n_logs):
            if op in new_key_list[ind] and op in new_key_list[0]:
                temp_ker_0 = "," + new_key_kernel_list[0][new_key_list[0].index(op)]
                temp_ker_i = "," + new_key_kernel_list[ind][new_key_list[ind].index(op)]
                denom = sum(new_dict_list[ind][op + temp_ker_i][1])
                if denom != 0:
                    ratio = non_zero_round(
                        sum(new_dict_list[0][op + temp_ker_0][1]) / denom, 2
                    )
                else:
                    ratio = "-"
                ratios.append(ratio)
            else:
                ratios.append("-")

        row = [B, M, N, K] + kernels + times + counts + impacts + ratios
        mm_rows.append(row)

    # Sorting indices: kernels start at col 4, times at 4+n_logs, etc.
    ker_offset = 4
    time_offset = ker_offset + n_logs
    impact_offset = time_offset + 2 * n_logs  # skip times + counts

    def mm_sort_key(r):
        all_present = all(r[ker_offset + li] != "-" for li in range(n_logs))
        has_all = 0 if all_present else 1
        first_impact = r[impact_offset]
        impact_num = float(first_impact) if first_impact != "-" else 0
        max_time = 0
        for li in range(n_logs):
            t = r[time_offset + li]
            if t != "-":
                max_time = max(max_time, float(t))
        return (has_all, -impact_num, -max_time)

    mm_rows.sort(key=mm_sort_key)
    for r in mm_rows:
        mm_table.add_row(r)

    return mm_table, mm_columns, mm_rows


def compare_logs(op_dicts, backends, args, t_times=None):
    """
    Generate comparitive views between all logs

    Parameters:
    op_dicts (list of dictionies) : List of dictionaries containing parsed
                                    parser info of both logs
    backends (list) : contains both backend names
    args (ArgParser) : Argument parser object
    t_times (list or None) : List of total execution times (strings) per log,
                             "" if not available for a given log

    Returns:
    compare_table (PrettyTable object) : contains compare table in the form of
                                         PrettyTable
    """
    if t_times is None:
        t_times = [""] * len(op_dicts)
    log_op_dict_list = []
    new_dict_list = []
    new_key_list = []
    new_key_kernel_list = []
    equivalent = {"matmul": ["inner_product", "gemm_api"], "softmax": ["softmax_v2"]}

    for ind, op_dict in enumerate(op_dicts):
        log_op_dict_list.append(op_dict)
        new_dict_list.append({})
        new_key_list.append([])
        new_key_kernel_list.append([])
        for old_key in op_dict:
            new_key_list[ind].append(
                ",".join([old_key.split(",")[0], old_key.split(",")[2]])
            )
            new_key_kernel_list[ind].append(old_key.split(",")[1])
            if (
                new_key_list[ind][-1] + "," + new_key_kernel_list[ind][-1]
                in new_dict_list[-1]
            ):
                new_dict_list[-1][
                    new_key_list[ind][-1] + "," + new_key_kernel_list[ind][-1]
                ][1].extend(log_op_dict_list[-1][old_key])
                del new_key_kernel_list[ind][-1]
                del new_key_list[ind][-1]
            else:
                new_dict_list[-1][
                    new_key_list[ind][-1] + "," + new_key_kernel_list[ind][-1]
                ] = (old_key, log_op_dict_list[-1][old_key])
        for i, key in enumerate(new_key_list[ind]):
            if key.split(",")[0] in ["matmul", "batch_matmul", "gemm_api"]:
                m = key.split(",")[1].split(":")[0].split("x")[-2]
                n = key.split(",")[1].split(":")[1].split("x")[-1]
                k = key.split(",")[1].split(":")[0].split("x")[-1]
                if key.split(",")[0] in ["matmul", "gemm_api"]:
                    dim = m + "M" + n + "N" + k + "K"
                else:
                    batch_parts = key.split(",")[1].split(":")[0].split("x")[:-2]
                    batches = 1
                    for bp in batch_parts:
                        batches *= int(bp)
                    dim = str(batches) + "B" + m + "M" + n + "N" + k + "K"
            elif key.split(",")[0] in ["inner_product"]:
                m = key.split(",")[1].split("mb")[1].split("ic")[0]
                k = key.split(",")[1].split("ic")[1].split("oc")[0]
                n = key.split(",")[1].split("oc")[1]
                dim = m + "M" + n + "N" + k + "K"
            elif key.split(",")[0] == "convolution":
                conv_dim = key.split(",")[1]
                conv_m = re.match(
                    r"mb(\d+)(?:_?)ic(\d+)oc(\d+)_(.+)_(.+)", conv_dim
                )
                if conv_m:
                    m, k, n = conv_m.group(1), conv_m.group(2), conv_m.group(3)
                    h, w = conv_m.group(4), conv_m.group(5)
                else:
                    m, k, n, h, w = "?", "?", "?", "?", "?"
                dim = (
                    "M:"
                    + m
                    + ",N:"
                    + n
                    + ",K:"
                    + k
                    + ",height dimension:"
                    + h
                    + ",width dimension:"
                    + w
                )
            else:
                continue
            temp_key = key.split(",")[0] + "," + dim
            new_key_list[ind][i] = temp_key
            new_dict_list[ind][
                temp_key + "," + new_key_kernel_list[ind][i]
            ] = new_dict_list[ind][key + "," + new_key_kernel_list[ind][i]]
            del new_dict_list[ind][key + "," + new_key_kernel_list[ind][i]]

        for i, key in enumerate(new_key_list[ind]):
            for j in equivalent:
                if key.split(",")[0] in equivalent[j]:
                    new_key_list[ind][i] = j + "," + key.split(",")[1]
                    new_dict_list[ind][
                        new_key_list[ind][i] + "," + new_key_kernel_list[ind][i]
                    ] = new_dict_list[ind][key + "," + new_key_kernel_list[ind][i]]
                    del new_dict_list[ind][key + "," + new_key_kernel_list[ind][i]]
    compare_order = compare_order_finder(new_key_list)
    log_name_list = []
    for i in range(1, args.log_count + 1):
        if isinstance(getattr(args, f"log_name_{i}"), bool):
            log_name_list.append(f"Log {i}")
        else:
            log_name_list.append(getattr(args, f"log_name_{i}"))

    if len(set(backends)) == 1:
        print(f"\nAll logs have the same backend : {backends[0]}")

    total_lib_times = []
    for ind in range(len(op_dicts)):
        total_lib_times.append(sum(sum(op_dicts[ind][k]) for k in op_dicts[ind]))

    all_have_e2e = all(t != "" for t in t_times)
    impact_denominators = []
    for ind in range(len(op_dicts)):
        if all_have_e2e:
            impact_denominators.append(float(t_times[ind]) * 1000)
        else:
            impact_denominators.append(total_lib_times[ind] if total_lib_times[ind] > 0 else 1)
    use_e2e_impact = all_have_e2e

    category = ["Op", "Kernel", "Dimension", "Time (ms)", "Count", "% Impact"]
    comparison_columns = []
    for i in category:
        for j in log_name_list:
            comparison_columns.append(f"{j}\n{i}")
    for i in log_name_list[1:]:
        comparison_columns.append(f"{log_name_list[0]}/\n{i} Time")
    compare_table = PrettyTable(align="c", linebreaks=True, autowrap=True)
    compare_table.hrules = ALL
    category.append("Compare")
    compare_table.add_row(comparison_columns)
    n_logs = len(new_key_list)
    impact_col_idx = 5 * n_logs
    time_col_idx = 3 * n_logs
    all_rows = []
    for op in compare_order:
        temp = []
        temp_ker = ""
        for c in category:
            for ind in range(n_logs):
                if op in new_key_list[ind]:
                    temp_ker = (
                        "," + new_key_kernel_list[ind][new_key_list[ind].index(op)]
                    )
                if c == "Op":
                    if op in new_key_list[ind]:
                        temp.append(new_dict_list[ind][op + temp_ker][0].split(",")[0])
                    else:
                        temp.append("-")
                elif c == "Kernel":
                    if op in new_key_list[ind]:
                        ker = new_dict_list[ind][op + temp_ker][0].split(",")[1]
                        kern = ""
                        for ik, k in enumerate(ker.split(":")):
                            kern += k
                            if ik != len(ker.split(":")) - 1:
                                kern += ":\n"
                        temp.append(kern)
                    else:
                        temp.append("-")
                elif c == "Dimension":
                    if op in new_key_list[ind]:
                        if "convolution" in op:
                            full_dim = new_dict_list[ind][op + temp_ker][0].split(",")[
                                2
                            ]
                            conv_m = re.match(
                                r"mb(\d+)(?:_?)ic(\d+)oc(\d+)_(.+)_(.+)", full_dim
                            )
                            if conv_m:
                                m = conv_m.group(1)
                                k = conv_m.group(2)
                                n = conv_m.group(3)
                                h = conv_m.group(4)
                                w = conv_m.group(5)
                            else:
                                m, k, n, h, w = "?", "?", "?", "?", "?"
                            dims = "M{}N{}K{}:H-{}:W-{}".format(m, n, k, h, w)
                        else:
                            dims = new_dict_list[ind][op + temp_ker][0].split(",")[2]
                        dimn = ""
                        for idx, d in enumerate(dims.split(":")):
                            dimn += d
                            if idx != len(dims.split(":")) - 1:
                                dimn += ":\n"
                        temp.append(dimn)
                    else:
                        temp.append("-")
                elif c == "Time (ms)":
                    if op in new_key_list[ind]:
                        temp.append(
                            non_zero_round(sum(new_dict_list[ind][op + temp_ker][1]), 2)
                        )
                    else:
                        temp.append("-")
                elif c == "Count":
                    if op in new_key_list[ind]:
                        temp.append(len(new_dict_list[ind][op + temp_ker][1]))
                    else:
                        temp.append("-")
                elif c == "% Impact":
                    if op in new_key_list[ind]:
                        op_time = sum(new_dict_list[ind][op + temp_ker][1])
                        temp.append(
                            non_zero_round((op_time / impact_denominators[ind]) * 100, 2)
                        )
                    else:
                        temp.append("-")
                elif c == "Compare" and ind != 0:
                    if op in new_key_list[ind] and op in new_key_list[0]:
                        temp_ker_0 = (
                            "," + new_key_kernel_list[0][new_key_list[0].index(op)]
                        )
                        denom = sum(new_dict_list[ind][op + temp_ker][1])
                        if denom != 0:
                            temp.append(
                                non_zero_round(
                                    sum(new_dict_list[0][op + temp_ker_0][1])
                                    / denom,
                                    2,
                                )
                            )
                        else:
                            temp.append("-")
                    else:
                        temp.append("-")
        all_rows.append(temp)

    def compare_sort_key(row):
        all_present = all(row[li] != "-" for li in range(n_logs))
        has_all = 0 if all_present else 1
        impact_val = row[impact_col_idx]
        impact_num = float(impact_val) if impact_val != "-" else 0
        max_time = 0
        for li in range(n_logs):
            t = row[time_col_idx + li]
            if t != "-":
                max_time = max(max_time, float(t))
        return (has_all, -impact_num, -max_time)

    all_rows.sort(key=compare_sort_key)
    for row in all_rows:
        compare_table.add_row(row)
    print("\nComparison Table:")
    print(compare_table.get_string(header=False))
    print_impact_footnote(use_e2e_impact)

    mm_result = None
    if getattr(args, "compare_mm", False):
        mm_result = _build_matmul_compare_table(
            compare_order, new_key_list, new_dict_list, new_key_kernel_list,
            log_name_list, impact_denominators, n_logs
        )
        if mm_result is not None:
            mm_table, mm_columns, mm_rows = mm_result
            print("\nMatmul/BMM Comparison Table:")
            print(mm_table)
            print_impact_footnote(use_e2e_impact)

    return compare_table, comparison_columns, all_rows, mm_result
