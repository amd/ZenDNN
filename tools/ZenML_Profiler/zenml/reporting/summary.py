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
import statistics
from zenml.reporting.utils import non_zero_round
from zenml.config.profiler_config import AUXILIARY_CREATE_OPS


def group_dict_maker(op_info_dict, t_time, summary_type):
    """
    Creates group table dictionary

    Parameters:
    op_info_dict (dictionary) : contains op details of nth iteration
    t_time (string) : total time for each iteration
    summary_type (string) : type of group summary (execute/create)

    Returns:
    (tuple)
        group_dict (dictionary) : contains group analysis table information
        total_time (float) : total time in milli seconds
        total_count (int) : count of all the ops present
    """
    op_per = {}
    op_count = {}
    total_time = 0
    total_count = 0
    for i in op_info_dict:
        op_name = i.split(",")[0]
        if op_name in AUXILIARY_CREATE_OPS:
            continue
        total_count += len(op_info_dict[i])
        total_time += sum(op_info_dict[i])
        group_key = op_name + "," + i.split(",")[4]
        if group_key in op_per:
            op_per[group_key] += sum(op_info_dict[i])
            op_count[group_key] += len(op_info_dict[i])
        else:
            op_per[group_key] = sum(op_info_dict[i])
            op_count[group_key] = len(op_info_dict[i])
    group_dict = {"Backend": [], "Op Type": [], "Op Time (ms)": [], "Count": []}
    if summary_type == "execute":
        group_dict["% Primitive Exec Impact"] = []
    else:
        group_dict["% Primitive Create Impact"] = []
    if t_time != "":
        group_dict["% E2E Impact"] = []

    for i in op_per:
        group_dict["Backend"].append(i.split(",")[1])
        group_dict["Op Type"].append(i.split(",")[0])
        group_dict["Op Time (ms)"].append(non_zero_round(op_per[i], 3))
        group_dict["Count"].append(op_count[i])
        if summary_type == "execute":
            group_dict["% Primitive Exec Impact"].append(
                non_zero_round((op_per[i] / total_time) * 100, 2) if total_time > 0 else 0
            )
        else:
            group_dict["% Primitive Create Impact"].append(
                non_zero_round((op_per[i] / total_time) * 100, 2) if total_time > 0 else 0
            )

        if t_time != "":
            group_dict["% E2E Impact"].append(
                non_zero_round((op_per[i] / (float(t_time) * 1000)) * 100, 2)
            )
    if t_time != "" and summary_type == "execute":
        group_dict["Backend"].append("Other")
        group_dict["Op Type"].append("-")
        other_time = max(float(t_time) * 1000 - total_time, 0)
        group_dict["Op Time (ms)"].append(non_zero_round(other_time, 3))
        group_dict["Count"].append(-1)
        group_dict["% Primitive Exec Impact"].append(-1)
        e2e_ms = float(t_time) * 1000
        group_dict["% E2E Impact"].append(
            non_zero_round((other_time / e2e_ms) * 100, 2) if e2e_ms > 0 else 0
        )

    return group_dict, total_time, total_count


def detailed_dict_maker(
    op_info_dict, op_info_dict1, t_time, args, prof, iter, num, fl, algo_t
):
    """
    Creates detailed table dictionary

    Parameters:
    op_info_dict (dictionary) : contains op details of nth iteration
    op_info_dict1 (dictionary) : contains op details of n-1 th iteration
    t_time (string) : total time for each iteration
    args (ArgParser) : Argument parser object
    prof (dictionary) : contains details of prof counter
    iter (int) : Total number of iterations
    num (int) : Number of iteration, on which analysis has to be done
    algo_t (List) : List of list containing algo info

    Returns:
    (tuple)
        detailed_dict (dictionary) : contains detailed table information
        mis (int) : number of missing ops
    """
    detailed_dict = {"Backend": []}
    has_plugin_op = any(len(k.split(",")) > 6 for k in op_info_dict)
    if has_plugin_op:
        detailed_dict["Plugin Op"] = []
    detailed_dict["Op Type"] = []
    detailed_dict["Kernel"] = []
    detailed_dict["Dimension"] = []
    detailed_dict["Data Format"] = []
    detailed_dict["Post ops"] = []
    detailed_dict["Total Time (ms)"] = []
    if fl == 2:
        detailed_dict["Algo Type"] = []
        detailed_dict["Algo Name"] = []
    if args.verbose:
        detailed_dict["Avg Time (ms)"] = []
        detailed_dict["Median Time (ms)"] = []
        detailed_dict["Min Time (ms)"] = []
        detailed_dict["Max Time (ms)"] = []
        detailed_dict["Std. Dev"] = []
    detailed_dict["Count"] = []
    if num == iter and iter != 1:
        detailed_dict["Missing"] = []
    detailed_dict["% Impact"] = []
    prof_len = []
    if len(prof) > 0:
        prof_col = []
        for i in prof:
            prof_len = prof[i][0].split(";")[1:]
            break
        if len(prof_len) > 0:
            for i in prof_len:
                prof_col.append(i.split(":")[0])
            for i in prof_col:
                detailed_dict[i] = []
            prof_dict = {}
            for i in prof:
                temp = prof[i]
                temp_dict = {}
                for j in prof[i]:
                    for idx, k in enumerate(j.split(";")[1:]):
                        if idx in temp_dict:
                            temp_dict[idx].append(float(k.split(":")[1]))
                        else:
                            temp_dict[idx] = [float(k.split(":")[1])]
                mini_prof_dict = {}
                for idx, col in enumerate(prof_col):
                    mini_prof_dict[col] = non_zero_round(
                        sum(temp_dict[idx]) / len(temp_dict[idx]), 3
                    )
                prof_dict[i] = mini_prof_dict
    total_lib_time = sum(
        sum(op_info_dict[k]) for k in op_info_dict
        if k.split(",")[0] not in AUXILIARY_CREATE_OPS
    )
    if t_time != "":
        impact_denominator = float(t_time) * 1000
    else:
        impact_denominator = total_lib_time if total_lib_time > 0 else 1
    mis = 0
    for i in op_info_dict1 if num == iter and iter != 1 else op_info_dict:
        if i.split(",")[0] in AUXILIARY_CREATE_OPS:
            continue
        if i in op_info_dict:
            avg = non_zero_round(sum(op_info_dict[i]) / len(op_info_dict[i]), 3)
        else:
            avg = 0
        if len(i.split(",")) > 6:
            detailed_dict["Plugin Op"].append(i.split(",")[6])
        elif "Plugin Op" in detailed_dict:
            detailed_dict["Plugin Op"].append("")
        detailed_dict["Backend"].append(i.split(",")[4])
        detailed_dict["Op Type"].append(i.split(",")[0])
        detailed_dict["Kernel"].append(i.split(",")[1])
        if i.split(",")[0] == "convolution":
            full_dim = i.split(",")[2]
            conv_match = re.match(
                r"mb(\d+)(?:_?)ic(\d+)oc(\d+)_(.+)_(.+)", full_dim
            )
            if conv_match:
                m = conv_match.group(1)
                k = conv_match.group(2)
                n = conv_match.group(3)
                h = conv_match.group(4)
                w = conv_match.group(5)
            else:
                m, k, n, h, w = "?", "?", "?", "?", "?"
            new_dim_conv = (
                "M:{} N:{} K:{}; height dimension:{}; width dimension:{}".format(
                    m, n, k, h, w
                )
            )
            detailed_dict["Dimension"].append(new_dim_conv)
        else:
            detailed_dict["Dimension"].append(i.split(",")[2])
        if fl == 2:
            detailed_dict["Algo Type"].append("-")
            detailed_dict["Algo Name"].append("-")
        detailed_dict["Data Format"].append(i.split(",")[5])
        detailed_dict["Post ops"].append(i.split(",")[3])
        detailed_dict["Total Time (ms)"].append(
            non_zero_round(sum(op_info_dict[i]) if i in op_info_dict else 0, 3)
        )
        if args.verbose:
            detailed_dict["Avg Time (ms)"].append(avg)
            detailed_dict["Median Time (ms)"].append(
                non_zero_round(
                    statistics.median(op_info_dict[i]) if i in op_info_dict else 0, 3
                )
            )
            detailed_dict["Min Time (ms)"].append(
                non_zero_round(min(op_info_dict[i]) if i in op_info_dict else 0, 3)
            )
            detailed_dict["Max Time (ms)"].append(
                non_zero_round(max(op_info_dict[i]) if i in op_info_dict else 0, 3)
            )
            detailed_dict["Std. Dev"].append(
                (
                    non_zero_round(
                        (
                            statistics.stdev(op_info_dict[i])
                            if len(op_info_dict[i]) > 1
                            else 0
                        ),
                        3,
                    )
                    if i in op_info_dict
                    else 0
                )
            )
        detailed_dict["Count"].append(len(op_info_dict[i]) if i in op_info_dict else 0)
        if len(prof_len) > 0:
            ind = ",".join(i.split(",")[:6])
            if ind in prof_dict:
                for j in prof_dict[ind]:
                    detailed_dict[j].append(prof_dict[ind][j])
            else:
                for j in prof_dict.get(list(prof_dict.keys())[0], {}) if prof_dict else {}:
                    detailed_dict[j].append("-")
        if num == iter and iter != 1:
            if i in op_info_dict:
                detailed_dict["Missing"].append(
                    len(op_info_dict1[i]) - len(op_info_dict[i])
                )
                mis += len(op_info_dict1[i]) - len(op_info_dict[i])
            else:
                detailed_dict["Missing"].append(len(op_info_dict1[i]))
                mis += len(op_info_dict1[i])
        detailed_dict["% Impact"].append(
            non_zero_round(
                (detailed_dict["Total Time (ms)"][-1] / impact_denominator) * 100, 2
            )
        )
    if fl == 2:
        algo_lookup = {}
        for al in algo_t:
            m_al, n_al, k_al, al_typ, al_nm = al[0], al[1], al[2], al[3], al[4]
            check_dim = f"{m_al}x{k_al}:{k_al}x{n_al}:{m_al}x{n_al}"
            algo_lookup[check_dim] = (al_typ, al_nm)
        for dim_idx, dim in enumerate(detailed_dict["Dimension"]):
            if str(dim) in algo_lookup:
                detailed_dict["Algo Type"][dim_idx] = algo_lookup[str(dim)][0]
                detailed_dict["Algo Name"][dim_idx] = algo_lookup[str(dim)][1]

    return detailed_dict, mis


def embedding_summary_maker(op_info_dict, t_time):
    """Build a dedicated embedding summary grouped by (plugin_op, table_dim).

    Correlates embedding_context_create, embedding_op_create and embedding
    (execute) entries from op_info_dict.  Create ops carry the table dimension
    (vocab x hidden); execute ops carry the output dimension.  They are linked
    by plugin_op.

    Returns:
        (emb_dict, has_data) -- dict suitable for table_maker, and a bool
        indicating whether any embedding ops were found.
    """
    # Collect per-phase data keyed on (plugin_op, table_dim)
    ctx_create = {}   # table_dim -> {times, count}
    op_create = {}    # (plugin_op, table_dim) -> {times, count}
    execute = {}      # plugin_op -> {total_time, count}

    for key, times in op_info_dict.items():
        parts = key.split(",")
        op = parts[0]

        if op == "embedding_context_create":
            table_dim = parts[2]
            dtype = parts[5] if len(parts) > 5 else "NA"
            if table_dim not in ctx_create:
                ctx_create[table_dim] = {"time": 0.0, "count": 0, "dtype": dtype}
            ctx_create[table_dim]["time"] += sum(times)
            ctx_create[table_dim]["count"] += len(times)

        elif op == "embedding_op_create":
            table_dim = parts[2]
            plugin_op = parts[6] if len(parts) > 6 else "NA"
            gk = (plugin_op, table_dim)
            if gk not in op_create:
                op_create[gk] = {"time": 0.0, "count": 0}
            op_create[gk]["time"] += sum(times)
            op_create[gk]["count"] += len(times)

        elif op == "embedding":
            plugin_op = parts[6] if len(parts) > 6 else "NA"
            if plugin_op not in execute:
                execute[plugin_op] = {"time": 0.0, "count": 0}
            execute[plugin_op]["time"] += sum(times)
            execute[plugin_op]["count"] += len(times)

    # Build rows -- one per (plugin_op, table_dim) from op_create.
    # Fall back to ctx_create keys if no op_create entries exist.
    row_keys = set(op_create.keys())
    if not row_keys and ctx_create:
        for td in ctx_create:
            for po in execute:
                row_keys.add((po, td))
    if not row_keys and execute:
        for po in execute:
            row_keys.add((po, "NA"))

    if not row_keys:
        return {}, False

    total_lib_time = sum(
        sum(v) for k, v in op_info_dict.items()
        if k.split(",")[0] not in AUXILIARY_CREATE_OPS
    )
    if t_time != "":
        impact_denom = float(t_time) * 1000
    else:
        impact_denom = total_lib_time if total_lib_time > 0 else 1

    emb_dict = {
        "Plugin Op": [],
        "Table Dim": [],
        "Dtype": [],
        "Ctx Create (ms)": [],
        "Ctx Cnt": [],
        "Op Create (ms)": [],
        "Op Cnt": [],
        "Execute (ms)": [],
        "Exec Cnt": [],
        "Total (ms)": [],
        "% Impact": [],
    }

    for plugin_op, table_dim in sorted(row_keys):
        cc = ctx_create.get(table_dim, {"time": 0.0, "count": 0, "dtype": "NA"})
        oc = op_create.get((plugin_op, table_dim), {"time": 0.0, "count": 0})
        ex = execute.get(plugin_op, {"time": 0.0, "count": 0})

        total = cc["time"] + oc["time"] + ex["time"]

        emb_dict["Plugin Op"].append(plugin_op)
        emb_dict["Table Dim"].append(table_dim)
        emb_dict["Dtype"].append(cc["dtype"])
        emb_dict["Ctx Create (ms)"].append(non_zero_round(cc["time"], 3))
        emb_dict["Ctx Cnt"].append(cc["count"])
        emb_dict["Op Create (ms)"].append(non_zero_round(oc["time"], 3))
        emb_dict["Op Cnt"].append(oc["count"])
        emb_dict["Execute (ms)"].append(non_zero_round(ex["time"], 3))
        emb_dict["Exec Cnt"].append(ex["count"])
        emb_dict["Total (ms)"].append(non_zero_round(total, 3))
        emb_dict["% Impact"].append(
            non_zero_round((total / impact_denom) * 100, 2)
        )

    return emb_dict, True


def reorder_summary_maker(op_info_dict, t_time):
    """Build a dedicated reorder summary grouped by (plugin_op, algo_format, dim, data_format).

    Correlates reorder_context_create, reorder_op_create and reorder
    (execute) entries from op_info_dict.  Create-phase times are
    apportioned proportionally across dimension rows by execute count.

    Returns:
        (reo_dict, has_data) -- dict suitable for table_maker, and a bool
        indicating whether any reorder ops were found.
    """
    ctx_create = {}   # algo_fmt -> {time, count}
    op_create = {}    # (plugin_op, algo_fmt) -> {time, count}
    execute = {}      # (plugin_op, algo_fmt, dim, data_format) -> {time, count}

    for key, times in op_info_dict.items():
        parts = key.split(",")
        op = parts[0]

        if op == "reorder_context_create":
            algo_fmt = parts[1]
            if algo_fmt not in ctx_create:
                ctx_create[algo_fmt] = {"time": 0.0, "count": 0}
            ctx_create[algo_fmt]["time"] += sum(times)
            ctx_create[algo_fmt]["count"] += len(times)

        elif op == "reorder_op_create":
            algo_fmt = parts[1]
            plugin_op = parts[6] if len(parts) > 6 else "NA"
            gk = (plugin_op, algo_fmt)
            if gk not in op_create:
                op_create[gk] = {"time": 0.0, "count": 0}
            op_create[gk]["time"] += sum(times)
            op_create[gk]["count"] += len(times)

        elif op == "reorder":
            algo_fmt = parts[1]
            plugin_op = parts[6] if len(parts) > 6 else "NA"
            dim = parts[2]
            data_fmt = parts[5] if len(parts) > 5 else "NA"
            gk = (plugin_op, algo_fmt, dim, data_fmt)
            if gk not in execute:
                execute[gk] = {"time": 0.0, "count": 0}
            execute[gk]["time"] += sum(times)
            execute[gk]["count"] += len(times)

    if not execute:
        return {}, False

    # Denominator excludes auxiliary create ops for consistency with all
    # other summary tables.  Row totals include apportioned create time,
    # which is negligible relative to execute time in practice.
    total_lib_time = sum(
        sum(v) for k, v in op_info_dict.items()
        if k.split(",")[0] not in AUXILIARY_CREATE_OPS
    )
    if t_time != "":
        impact_denom = float(t_time) * 1000
    else:
        impact_denom = total_lib_time if total_lib_time > 0 else 1

    # Compute total execute counts to apportion create times proportionally.
    # ctx_create is keyed by algo_fmt only, so apportion across ALL rows
    # sharing that algo_fmt (regardless of plugin_op).
    exec_count_by_af = {}
    exec_count_by_pa = {}
    for (po, af, _, _), ex in execute.items():
        exec_count_by_af[af] = exec_count_by_af.get(af, 0) + ex["count"]
        pa = (po, af)
        exec_count_by_pa[pa] = exec_count_by_pa.get(pa, 0) + ex["count"]

    reo_dict = {
        "Plugin Op": [],
        "Algo Format": [],
        "Dimension": [],
        "Data Format": [],
        "Ctx Create (ms)": [],
        "Ctx Cnt": [],
        "Op Create (ms)": [],
        "Op Cnt": [],
        "Execute (ms)": [],
        "Exec Cnt": [],
        "Total (ms)": [],
        "% Impact": [],
    }

    for (plugin_op, algo_fmt, dim, data_fmt), ex in sorted(execute.items()):
        pa = (plugin_op, algo_fmt)
        af_total = exec_count_by_af.get(algo_fmt, 1)
        pa_total = exec_count_by_pa.get(pa, 1)
        cc_frac = ex["count"] / af_total if af_total > 0 else 0
        oc_frac = ex["count"] / pa_total if pa_total > 0 else 0

        cc = ctx_create.get(algo_fmt, {"time": 0.0, "count": 0})
        oc = op_create.get(pa, {"time": 0.0, "count": 0})

        cc_share = cc["time"] * cc_frac
        oc_share = oc["time"] * oc_frac

        total = cc_share + oc_share + ex["time"]
        reo_dict["Plugin Op"].append(plugin_op)
        reo_dict["Algo Format"].append(algo_fmt)
        reo_dict["Dimension"].append(dim)
        reo_dict["Data Format"].append(data_fmt)
        reo_dict["Ctx Create (ms)"].append(non_zero_round(cc_share, 3))
        reo_dict["Ctx Cnt"].append(round(cc["count"] * cc_frac))
        reo_dict["Op Create (ms)"].append(non_zero_round(oc_share, 3))
        reo_dict["Op Cnt"].append(round(oc["count"] * oc_frac))
        reo_dict["Execute (ms)"].append(non_zero_round(ex["time"], 3))
        reo_dict["Exec Cnt"].append(ex["count"])
        reo_dict["Total (ms)"].append(non_zero_round(total, 3))
        reo_dict["% Impact"].append(
            non_zero_round((total / impact_denom) * 100, 2)
        )

    return reo_dict, True


def grp_matmul_summary_maker(op_info_dict, t_time, verbose=False):
    """Build a summary for group_matmul (MoE fused matmul) ops.

    Each unique (kernel, dimension, post_ops, data_format) combination
    gets its own row.

    Returns:
        (gm_dict, has_data) -- dict suitable for table_maker, and a bool.
    """
    gm_entries = {}  # ind_key -> times list (only group_matmul entries)
    for key, times in op_info_dict.items():
        if key.split(",")[0] == "group_matmul":
            gm_entries[key] = times

    if not gm_entries:
        return {}, False

    total_lib_time = sum(
        sum(v) for k, v in op_info_dict.items()
        if k.split(",")[0] not in AUXILIARY_CREATE_OPS
    )
    if t_time != "":
        impact_denom = float(t_time) * 1000
    else:
        impact_denom = total_lib_time if total_lib_time > 0 else 1

    gm_dict = {
        "Kernel": [],
        "Dimension": [],
        "Post Ops": [],
        "Data Format": [],
        "Total Time (ms)": [],
    }
    if verbose:
        gm_dict["Avg Time (ms)"] = []
        gm_dict["Min Time (ms)"] = []
        gm_dict["Max Time (ms)"] = []
    gm_dict["Count"] = []
    gm_dict["% Impact"] = []

    for key in sorted(gm_entries, key=lambda k: sum(gm_entries[k]), reverse=True):
        parts = key.split(",")
        times = gm_entries[key]
        total = sum(times)
        gm_dict["Kernel"].append(parts[1])
        gm_dict["Dimension"].append(parts[2])
        gm_dict["Post Ops"].append(parts[3])
        gm_dict["Data Format"].append(parts[5])
        gm_dict["Total Time (ms)"].append(non_zero_round(total, 3))
        if verbose:
            gm_dict["Avg Time (ms)"].append(non_zero_round(total / len(times), 3))
            gm_dict["Min Time (ms)"].append(non_zero_round(min(times), 3))
            gm_dict["Max Time (ms)"].append(non_zero_round(max(times), 3))
        gm_dict["Count"].append(len(times))
        gm_dict["% Impact"].append(
            non_zero_round((total / impact_denom) * 100, 2)
        )

    return gm_dict, True
