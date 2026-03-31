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
from zenml.reporting.utils import non_zero_round, table_maker


def create_function_events(trace_data):
    """
    Create a function event to store required data from chrome trace

    Parameters:
    trace_dict (dictionary) : contains trace events from the pytorch
                              profiler json
    """
    function_events = []
    for event in trace_data:
        if (
            "name" in event
            and "ts" in event
            and "dur" in event
            and "tid" in event
            and event["ph"] == "X"
        ):
            try:
                node_id = event["pid"]
                thread_id = int(event["tid"])
                function_event = {
                    "name": event["name"],
                    "time_range": {
                        "start": event["ts"],
                        "end": event["ts"] + event["dur"],
                    },
                    "thread_id": thread_id,
                    "node_id": node_id,
                    "dur": event["dur"],
                    "children": [],
                    "cpu_parent": None,
                }
            except ValueError:
                # Handle the case where thread_id is not a valid integer
                continue
            function_events.append(function_event)
    return function_events


def populate_cpu_children(function_events):
    """
    Update function events with child processes

    Parameters:
    function_events (list) : a list of event dictionaries from the trace data
    """
    # Filter events for sync CPU events
    sync_events = [evt for evt in function_events if evt["node_id"] is not None]
    # Sort events by thread_id
    sync_events.sort(
        key=lambda event: (
            event["thread_id"],
            event["time_range"]["start"],
            -event["time_range"]["end"],
        )
    )

    # Group by both thread_id and node_id
    threads = {}
    for event in sync_events:
        thread_node_key = (event["thread_id"], event["node_id"])
        if thread_node_key not in threads:
            threads[thread_node_key] = []
        threads[thread_node_key].append(event)

    # For each thread, process the events
    for thread_events in threads.values():
        current_events = []
        for event in thread_events:
            while current_events and (
                event["time_range"]["start"] >= current_events[-1]["time_range"]["end"]
                or event["time_range"]["end"] > current_events[-1]["time_range"]["end"]
            ):
                current_events.pop()
            if current_events:
                parent = current_events[-1]
                parent["children"].append(event)
                event["cpu_parent"] = parent
            current_events.append(event)
    return function_events


def calc_self_cpu(event_list):
    """
    Calculate self CPU

    Self-CPU time = Total duration of the operation - Sum of durations of all child operations

    Parameters:
    event_list (list) : a list of event dictionaries from the trace data with parent-child relationships
    """
    self_cpu_dict = {}
    total_self_cpu = 1
    # Access the populated FunctionEvent objects and their relationships
    for event in event_list:
        total_dur = event["dur"]
        name = event["name"]
        child_dur = 0
        for child in event["children"]:
            child_dur += child["dur"]
        self_cpu = total_dur - child_dur
        if name in self_cpu_dict.keys():
            cur_time = self_cpu_dict[name]
            self_cpu += cur_time
            self_cpu_dict[name] = self_cpu
        else:
            self_cpu_dict[name] = self_cpu
    total_self_cpu = sum(self_cpu_dict.values())
    return self_cpu_dict, total_self_cpu


def time_converter(list_of_time):
    new_list = []
    for i in list_of_time:
        if len(i) < 2:
            logging.warning("Skipping unparseable time value: %r", i)
            continue
        if "ms" == i[-2:]:
            new_val = float(i[:-2])
        elif "us" == i[-2:]:
            new_val = float(i[:-2]) / 1000.0
        elif i[-2].isnumeric() and i[-1] == "s":
            new_val = float(i[:-1]) * 1000
        else:
            logging.warning("Unrecognized time unit: %r", i[-2:])
            continue
        new_list.append(non_zero_round(new_val, 4))
    return new_list


def pt_profiler_dict_maker(trace_dict, ts, group=False, log_mode=False):
    """
    Creates pytorch profiler table dictionary

    Parameters:
    trace_dict (dictionary) : contains trace events from the pytorch
                              profiler json

    Optional parameters:
    group (bool) : bool value to enable grouping of ops

    Returns:
    flops_dict (dictionary) : contains flops table information
    """
    if not log_mode:
        function_events = create_function_events(trace_dict)
        updated_events = populate_cpu_children(function_events)
        self_cpu, total_self_cpu = calc_self_cpu(updated_events)
    not_req = [
        "Torch-Compiled Region",
        "TorchDynamo Cache Lookup",
        "PyTorch Profiler (0)",
    ]

    pt_profiler_dict = {
        "Op name": [],
        "Self CPU %": [],
        "Self CPU Time(ms)": [],
        "CPU Total %": [],
        "CPU Total Time(ms)": [],
        "CPU Avg Time(ms)": [],
        "Count": [],
    }
    # Pytorch Profiler JSON Analysis
    if not log_mode:
        profile_dict = {}
        profile_dict_self = {}
        profile_dict_percent = {}
        total_time = total_self_cpu
        trace_dict = trace_dict[1:]
        threshold = ts

        for i in trace_dict:
            if "name" in i and "dur" in i:
                name = i["name"] if not group else i["name"].split(":")[0]
                # if name not in not_req:
                if name in profile_dict:
                    profile_dict[name].append(i["dur"])
                else:
                    profile_dict[name] = [i["dur"]]
        for key, val in self_cpu.items():
            name = key if not group else key.split(":")[0]
            if name in profile_dict:
                if name in profile_dict_self:
                    profile_dict_self[name].append(val)
                else:
                    profile_dict_self[name] = [val]
                if name in profile_dict_percent:
                    profile_dict_percent[name].append(
                        non_zero_round((val / total_self_cpu) * 100, 3)
                    )
                else:
                    profile_dict_percent[name] = [
                        non_zero_round((val / total_self_cpu) * 100, 3)
                    ]
        pt_profiler_dict_other = {
            "Op name": [],
            "Self CPU %": [],
            "Self CPU Time(ms)": [],
            "CPU Total %": [],
            "CPU Total Time(ms)": [],
            "CPU Avg Time(ms)": [],
            "Count": [],
        }
        for i in profile_dict:
            if i not in not_req:
                if (
                    (
                        (non_zero_round(sum(profile_dict[i]) / total_time, 3)) * 100
                        > threshold
                    )
                    if not group
                    else True
                ):
                    pt_profiler_dict["Op name"].append(i)
                    pt_profiler_dict["CPU Total Time(ms)"].append(
                        non_zero_round(sum(profile_dict[i]) / 1000, 3)
                    )
                    pt_profiler_dict["CPU Avg Time(ms)"].append(
                        non_zero_round(
                            (sum(profile_dict[i]) / 1000) / len(profile_dict[i]), 3
                        )
                    )
                    pt_profiler_dict["CPU Total %"].append(
                        non_zero_round((sum(profile_dict[i]) / total_time) * 100, 3)
                    )
                    pt_profiler_dict["Count"].append(len(profile_dict[i]))
                    for j in profile_dict_self:
                        if i == j:
                            pt_profiler_dict["Self CPU Time(ms)"].append(
                                (non_zero_round(sum(profile_dict_self[j]) / 1000, 3))
                            )
                    if i not in list(profile_dict_self.keys()):
                        pt_profiler_dict["Self CPU Time(ms)"].append("-")
                    for j in profile_dict_percent:
                        if i == j:
                            pt_profiler_dict["Self CPU %"].append(
                                (non_zero_round(sum(profile_dict_percent[j]), 3))
                            )
                    if i not in list(profile_dict_percent.keys()):
                        pt_profiler_dict["Self CPU %"].append("-")
            else:
                if (
                    (
                        (non_zero_round(sum(profile_dict[i]) / total_time, 3)) * 100
                        > threshold
                    )
                    if not group
                    else True
                ):
                    pt_profiler_dict_other["Op name"].append(i)
                    pt_profiler_dict_other["CPU Total Time(ms)"].append(
                        non_zero_round(sum(profile_dict[i]) / 1000, 3)
                    )
                    pt_profiler_dict_other["CPU Avg Time(ms)"].append(
                        non_zero_round(
                            (sum(profile_dict[i]) / 1000) / len(profile_dict[i]), 3
                        )
                    )
                    pt_profiler_dict_other["CPU Total %"].append(
                        non_zero_round((sum(profile_dict[i]) / total_time) * 100, 3)
                    )
                    pt_profiler_dict_other["Count"].append(len(profile_dict[i]))
                    for j in profile_dict_self:
                        if i == j:
                            pt_profiler_dict_other["Self CPU Time(ms)"].append(
                                (non_zero_round(sum(profile_dict_self[j]) / 1000, 3))
                            )
                    if i not in list(profile_dict_self.keys()):
                        pt_profiler_dict_other["Self CPU Time(ms)"].append("-")
                    for j in profile_dict_percent:
                        if i == j:
                            pt_profiler_dict_other["Self CPU %"].append(
                                (non_zero_round(sum(profile_dict_percent[j]), 3))
                            )
                    if i not in list(profile_dict_percent.keys()):
                        pt_profiler_dict_other["Self CPU %"].append("-")
        extras = {}
        if group:
            profile_dict = {}
            for i in trace_dict:
                if "name" in i and "dur" in i:
                    name = i["name"]
                    if name in profile_dict:
                        profile_dict[name].append(i["dur"])
                    else:
                        profile_dict[name] = [i["dur"]]
            for i, j in enumerate(pt_profiler_dict["Op name"]):
                if (
                    pt_profiler_dict["CPU Total %"][i]
                    > 5
                    # and not pt_profiler_dict["CPU Total %"][i] >= 100
                ):
                    mini_dict = {
                        "Op name": [],
                        "Self CPU %": [],
                        "Self CPU Time(ms)": [],
                        "CPU Total %": [],
                        "CPU Total Time(ms)": [],
                        "CPU Avg Time(ms)": [],
                        "Count": [],
                    }
                    for ind in profile_dict:
                        if j == ind.split(":")[0]:
                            if (
                                non_zero_round(sum(profile_dict[ind]) / total_time, 3)
                            ) * 100 > threshold:
                                mini_dict["Op name"].append(ind)
                                mini_dict["CPU Total Time(ms)"].append(
                                    non_zero_round(sum(profile_dict[ind]) / 1000, 3)
                                )
                                mini_dict["CPU Avg Time(ms)"].append(
                                    non_zero_round(
                                        (sum(profile_dict[ind]) / 1000)
                                        / len(profile_dict[ind]),
                                        3,
                                    )
                                )
                                mini_dict["CPU Total %"].append(
                                    non_zero_round(
                                        (sum(profile_dict[ind]) / total_time) * 100, 3
                                    )
                                )
                                mini_dict["Count"].append(len(profile_dict[ind]))
                    for k1 in mini_dict["Op name"]:
                        if k1 in list(self_cpu.keys()):
                            for key, val in self_cpu.items():
                                if key == k1:
                                    mini_dict["Self CPU Time(ms)"].append(
                                        non_zero_round(val / 1000, 3)
                                    )
                                    mini_dict["Self CPU %"].append(
                                        non_zero_round(
                                            ((val / total_self_cpu) * 100), 3
                                        )
                                    )
                        else:
                            mini_dict["Self CPU Time(ms)"].append("-")
                            mini_dict["Self CPU %"].append("-")
                    extras[j] = mini_dict
        return pt_profiler_dict, extras, pt_profiler_dict_other, total_self_cpu
    # Pytorch Profiler Log Analysis
    else:
        if not trace_dict:
            logging.warning("PyTorch profiler trace is empty.")
            return {}, {}, {}, 0
        try:
            total_time = trace_dict[-1].split("Self CPU time total: ")[1].strip()
        except (IndexError, AttributeError):
            logging.warning("Could not parse Self CPU time total from PyTorch profiler trace.")
            return {}, {}, {}, 0
        if "ms" == total_time[-2:]:
            total_time = str(float(total_time[:-2]) * 0.001)
        elif "us" == total_time[-2:]:
            total_time = str(float(total_time[:-2]) * 0.000001)
        elif total_time[-2].isnumeric() and total_time[-1] == "s":
            total_time = str(float(total_time[:-1]))
        else:
            logging.warning("Could not identify metric of Self CPU total time")
        for i in trace_dict:
            if "%" in i:
                name = " ".join(i[: i.find("%")].strip().split()[:-1])
                pt_profiler_dict["Op name"].append(name)
                pt_profiler_dict["Self CPU %"].append(
                    float(i[: i.find("%")].strip().split()[-1][:-1])
                )
                rest = i[i.find("%") + 1 :].strip().split()
                pt_profiler_dict["Self CPU Time(ms)"].append(rest[0])
                pt_profiler_dict["CPU Total %"].append(float(rest[1][:-1]))
                pt_profiler_dict["CPU Avg Time(ms)"].append(rest[3])
                pt_profiler_dict["CPU Total Time(ms)"].append(rest[2])
                pt_profiler_dict["Count"].append(float(rest[4]))
        pt_profiler_dict["Self CPU Time(ms)"] = time_converter(
            pt_profiler_dict["Self CPU Time(ms)"]
        )
        pt_profiler_dict["CPU Avg Time(ms)"] = time_converter(
            pt_profiler_dict["CPU Avg Time(ms)"]
        )
        pt_profiler_dict["CPU Total Time(ms)"] = time_converter(
            pt_profiler_dict["CPU Total Time(ms)"]
        )
        extras = {}
        threshold = ts
        group_dict = {}
        group_detail = {
            "Class name": [],
            "Self CPU %": [],
            "Self CPU Time(ms)": [],
            "CPU Total %": [],
            "CPU Avg Time(ms)": [],
            "CPU Total Time(ms)": [],
            "Count": [],
        }

        def round_off(org, val):
            lis = []
            for i in org:
                if val == -1:
                    lis.append(non_zero_round(i))
                else:
                    lis.append(non_zero_round(i, val))
            return lis

        flags = [False, False]
        for ind, name in enumerate(pt_profiler_dict["Op name"]):
            if ":" in name:
                mini_name = name.split(":")[0]
            else:
                mini_name = name.split("_")[0]
            if mini_name in group_dict:
                group_dict[mini_name] += pt_profiler_dict["Self CPU %"][ind]
                mini_ind = group_detail["Class name"].index(mini_name)
                group_detail["Self CPU %"][mini_ind] += pt_profiler_dict["Self CPU %"][
                    ind
                ]
                group_detail["Self CPU Time(ms)"][mini_ind] += pt_profiler_dict[
                    "Self CPU Time(ms)"
                ][ind]
                group_detail["CPU Total %"][mini_ind] += pt_profiler_dict[
                    "CPU Total %"
                ][ind]
                group_detail["CPU Avg Time(ms)"][mini_ind] += pt_profiler_dict[
                    "CPU Avg Time(ms)"
                ][ind]
                group_detail["CPU Total Time(ms)"][mini_ind] += pt_profiler_dict[
                    "CPU Total Time(ms)"
                ][ind]
                group_detail["Count"][mini_ind] += pt_profiler_dict["Count"][ind]
            else:
                group_dict[mini_name] = pt_profiler_dict["Self CPU %"][ind]
                group_detail["Class name"].append(mini_name)
                group_detail["Self CPU %"].append(pt_profiler_dict["Self CPU %"][ind])
                group_detail["Self CPU Time(ms)"].append(
                    pt_profiler_dict["Self CPU Time(ms)"][ind]
                )
                group_detail["CPU Total %"].append(pt_profiler_dict["CPU Total %"][ind])
                group_detail["CPU Avg Time(ms)"].append(
                    pt_profiler_dict["CPU Avg Time(ms)"][ind]
                )
                group_detail["CPU Total Time(ms)"].append(
                    pt_profiler_dict["CPU Total Time(ms)"][ind]
                )
                group_detail["Count"].append(pt_profiler_dict["Count"][ind])
        group_detail["Self CPU %"] = round_off(group_detail["Self CPU %"], 2)
        group_detail["Self CPU Time(ms)"] = round_off(
            group_detail["Self CPU Time(ms)"], 4
        )
        group_detail["CPU Total %"] = round_off(group_detail["CPU Total %"], 2)
        group_detail["CPU Avg Time(ms)"] = round_off(
            group_detail["CPU Avg Time(ms)"], 4
        )
        group_detail["CPU Total Time(ms)"] = round_off(
            group_detail["CPU Total Time(ms)"], 4
        )
        group_detail["Count"] = round_off(group_detail["Count"], -1)

        _MIN_GROUP_DISPLAY = 5
        _GROUP_SELF_CPU_THRESHOLD = 5.0
        n_groups = len(group_detail["Class name"])
        qualifying = [
            idx for idx in range(n_groups)
            if group_detail["Self CPU %"][idx] > _GROUP_SELF_CPU_THRESHOLD
        ]
        if len(qualifying) >= _MIN_GROUP_DISPLAY:
            keep = qualifying
        else:
            ranked = sorted(range(n_groups),
                            key=lambda x: group_detail["Self CPU %"][x],
                            reverse=True)
            keep = ranked[:_MIN_GROUP_DISPLAY]
        keep_set = set(keep)
        for key in group_detail:
            group_detail[key] = [group_detail[key][idx] for idx in range(n_groups) if idx in keep_set]

        for grp in group_dict:
            if group_dict[grp] > 5:
                mini_dict = {
                    "Op name": [],
                    "Self CPU %": [],
                    "Self CPU Time(ms)": [],
                    "CPU Total %": [],
                    "CPU Avg Time(ms)": [],
                    "CPU Total Time(ms)": [],
                    "Count": [],
                }
                for ind, name in enumerate(pt_profiler_dict["Op name"]):
                    if grp in name and pt_profiler_dict["Self CPU %"][ind] > threshold:
                        mini_dict["Op name"].append(name)
                        mini_dict["Self CPU %"].append(
                            pt_profiler_dict["Self CPU %"][ind]
                        )
                        mini_dict["Self CPU Time(ms)"].append(
                            pt_profiler_dict["Self CPU Time(ms)"][ind]
                        )
                        mini_dict["CPU Total %"].append(
                            pt_profiler_dict["CPU Total %"][ind]
                        )
                        mini_dict["CPU Avg Time(ms)"].append(
                            pt_profiler_dict["CPU Avg Time(ms)"][ind]
                        )
                        mini_dict["CPU Total Time(ms)"].append(
                            pt_profiler_dict["CPU Total Time(ms)"][ind]
                        )
                        mini_dict["Count"].append(pt_profiler_dict["Count"][ind])
                if len(mini_dict["Op name"]) > 0:
                    mini_dict["Self CPU %"] = round_off(mini_dict["Self CPU %"], 2)
                    mini_dict["Self CPU Time(ms)"] = round_off(
                        mini_dict["Self CPU Time(ms)"], 4
                    )
                    mini_dict["CPU Total %"] = round_off(mini_dict["CPU Total %"], 2)
                    mini_dict["CPU Avg Time(ms)"] = round_off(
                        mini_dict["CPU Avg Time(ms)"], 4
                    )
                    mini_dict["CPU Total Time(ms)"] = round_off(
                        mini_dict["CPU Total Time(ms)"], 4
                    )
                    mini_dict["Count"] = round_off(mini_dict["Count"], -1)
                    if len(mini_dict["Count"]) > 10:
                        flags[1] = True
                        for _ in mini_dict:
                            mini_dict[_] = mini_dict[_][:10]
                    extras[grp] = mini_dict
        if len(pt_profiler_dict["Count"]) > 15:
            flags[0] = True
            for i in pt_profiler_dict:
                pt_profiler_dict[i] = pt_profiler_dict[i][:15]
        return pt_profiler_dict, group_detail, extras, flags, total_time


def PyTorch_Profiler_view(extras, args):
    ths = args.threshold
    if "pt_profiler" in extras:
        pt_prof_dict, _, _, _ = pt_profiler_dict_maker(extras["pt_profiler"], ths)
        (
            pt_prof_dict_group,
            pt_extras,
            pt_others,
            total_self_cpu,
        ) = pt_profiler_dict_maker(extras["pt_profiler"], ths, group=True)
        pt_prof_table, pt_prof_df = table_maker(
            pt_prof_dict, "Self CPU Time(ms)", False, hr_val=True
        )

        pt_prof_group_table, pt_prof_group_df = table_maker(
            pt_prof_dict_group, "Self CPU Time(ms)", False, hr_val=True
        )
    if "pt_profiler_log" in extras:
        (
            pt_prof_dict,
            pt_prof_dict_group,
            pt_extras,
            pt_flags,
            total_self_cpu,
        ) = pt_profiler_dict_maker(
            extras["pt_profiler_log"], ths, group=True, log_mode=True
        )
        pt_prof_table, pt_prof_df = table_maker(
            pt_prof_dict, "Self CPU Time(ms)", False, hr_val=True
        )
        pt_prof_group_table, pt_prof_group_df = table_maker(
            pt_prof_dict_group, "Self CPU Time(ms)", False, hr_val=True
        )
    if "pt_profiler" in extras or "pt_profiler_log" in extras:
        print("PyTorch Profiler Analysis:")
        if args.info:
            print(
                "\nSelf CPU total %: The percentage of the total CPU time that was spent inside this operation, excluding time in its children (sub-operations)."
            )
            print(
                "Self CPU total time: The total time spent on the CPU for this operation, excluding time in its children."
            )
            print(
                "CPU total %: The percentage of the total CPU time that was spent on this operation,including time in its children."
            )
            print(
                "CPU total time: The total time spent on the CPU for this operation, including time in its children."
            )
            print(
                "CPU time avg: The average time spent on the CPU per call of this operation.\n"
            )
    if "pt_profiler" in extras:
        if args.verbose:
            print(pt_prof_table)
            print()
            # print("\n Self CPU Details of all ops:")
            # self_cpu, _ = calc_self_cpu(updated_events)
            # self_cpu_time = {"Op name" : [], "Self CPU Time(ms)" : []}
            # for key, val in self_cpu.items():
            #     self_cpu_time["Op name"].append(key)
            #     self_cpu_time["Self CPU Time(ms)"].append(non_zero_round(val/1000, 4))
            # self_table = PrettyTable()
            # self_table.add_column("Op name",self_cpu_time["Op name"])
            # self_table.add_column("Self CPU Time(ms)",self_cpu_time["Self CPU Time(ms)"])
            # print(self_table)
            # print()
        print("\nGrouped PyTorch Profiler Analysis:")
        print(pt_prof_group_table)
        print()
        if len(pt_others["Op name"]) != 0:
            pt_prof_other, _ = table_maker(
                pt_others, "CPU Total Time(ms)", False, hr_val=True
            )
            print("\nGrouped op analysis for Torch Region:")
            print(pt_prof_other)
            print()
        if len(pt_extras) > 0:
            for pt_ind in pt_extras:
                if len(pt_extras[pt_ind]["Op name"]) != 0:
                    print("Grouped op analysis for", pt_ind)
                    pt_prof_extra, _ = table_maker(
                        pt_extras[pt_ind], "CPU Total Time(ms)", False, hr_val=True
                    )
                    print(pt_prof_extra)
                    print()
        print(f"Self CPU time total: {non_zero_round(total_self_cpu/1000000,3)}s")
        print()
    elif "pt_profiler_log" in extras:
        if args.verbose:
            print()
            print("Detailed PyTorch Profiler Analysis:")
            if pt_flags[0]:
                print("Note!! Only the top 15 entries are being displayed.")
            print(pt_prof_table)
            print()

        print("\nGrouped PyTorch Profiler Analysis:")
        print(pt_prof_group_table)
        print()
        if pt_flags[1]:
            print("Note!! Only the top 10 entries of the sub group will be displayed.")
        if len(pt_extras) > 0:
            for pt_ind in pt_extras:
                print("Grouped op analysis for", pt_ind)
                pt_prof_extra, _ = table_maker(
                    pt_extras[pt_ind], "Self CPU %", False, hr_val=True
                )
                print(pt_prof_extra)
                print()
        print(f"Self CPU time total: {total_self_cpu}s")
        print()
