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
import csv as cs
import time as time1
from datetime import datetime
from prettytable import PrettyTable
from zenml.reporting.utils import non_zero_round, table_maker, print_impact_footnote
from zenml.reporting.summary import group_dict_maker, detailed_dict_maker, embedding_summary_maker
from zenml.reporting.flops import flops_dict_maker
from zenml.integrations.pytorch_profiler import PyTorch_Profiler_view


def result_generation(
    t_time, op_info_dict, op_info_dict1, args, extras, iter, num, log_index, fl, algo_t
):
    """
    Generate different views for the data parsed from the logs

    Parameters:
    t_time (string) : total time for each iteration
    op_info_dict (dictionary) : contains op details of nth iteration
    op_info_dict1 (dictionary) : contains op details of n-1 th iteration
    args (ArgParser) : Argument parser object
    extras (dictionary) : contains additional information such as prof
                          counter, pytorch profiler info, etc.
    iter (int) : Total number of iterations
    num (int) : Number of iteration, on which analysis has to be done
    log_index (int) : Index of log for which result is being generated
    algo_t (list) : List of list containing algo details

    Returns:
    table_df_list (list of dataframes) : List containing dataframe of group,
                                         detailed and flop analysis
    """
    flops = args.flops
    export_csv = args.csv
    group_dict, total_time, total_count = group_dict_maker(
        op_info_dict, t_time, "execute"
    )
    group_table, group_df = table_maker(
        group_dict, "% Primitive Exec Impact", False, hr_val=True
    )
    prof = {}
    if "prof" in extras:
        prof = extras["prof"]["kwargs"]["op_info_dict"]
    detailed_dict, mis = detailed_dict_maker(
        op_info_dict, op_info_dict1, t_time, args, prof, iter, num, fl, algo_t
    )
    detailed_table, detailed_df = table_maker(
        detailed_dict, "Total Time (ms)", False, {"Data Format": "max"}, hr_val=True
    )
    if "create" in extras:
        group_create_dict, total_time_create, total_count_create = group_dict_maker(
            extras["create"]["kwargs"]["op_info_dict"], t_time, "create"
        )
        group_create_table, group_create_df = table_maker(
            group_create_dict, "% Primitive Create Impact", False, hr_val=True
        )
    PyTorch_Profiler_view(extras, args)
    for pr in extras["print_statements"]:
        print(pr)
    if iter == num and num == 1:
        print("\nAnalysis provided for entire log")
    else:
        print("\nAnalysis provided for run", num)
    print("Please note that all timings are given in ms (milliseconds)")
    if mis > 0:
        print(
            "Please note that analysis is done on final iteration, and some ops appear to be missing."
        )
    print("\nTotal execute ops :", total_count)
    if "create" in extras:
        print("Total create ops :", total_count_create)
    if num == iter and iter != 1:
        print("Total Missing ops :", mis)
    print("Total time of primivite_execute (ms): ", non_zero_round(total_time, 3))
    if "create" in extras:
        print(
            "Total time of primivite_create (ms): ",
            non_zero_round(total_time_create, 3),
        )

    if t_time != "" and float(t_time) > 0:
        print("Average Total time(ms) for each iteration: ", float(t_time) * 1000)
        print(
            "Percent of Primitive Execute in Total Time: ",
            non_zero_round(total_time / float(t_time) / 10, 3),
            "%",
        )
        if "create" in extras:
            print(
                "Percent of Primitive Create in Total Time: ",
                non_zero_round(total_time_create / float(t_time) / 10, 3),
                "%",
            )

    print("\nGrouped Execute Summary")
    if args.info:
        print(
            "% Primitive Exec Impact  --- Percentage contribution of each op within total op execution time in the run."
        )
        if t_time != "":
            print(
                "% E2E Impact  --- Percentage contribution of a particular op to the end to end execution time of the model."
            )
        print()
    print(group_table)

    if "create" in extras and total_time_create > 0:
        print("\nGrouped Create Summary")
        if args.info:
            print(
                "% Primitive Create Impact  --- Percentage contribution of each op within total op creation time in the run."
            )
            print()
        print(group_create_table)

    emb_dict, has_emb = embedding_summary_maker(op_info_dict, t_time)
    emb_df = None
    if has_emb:
        emb_table, emb_df = table_maker(emb_dict, "% Impact", False, hr_val=True)
        print("\nEmbedding Summary")
        print(emb_table)
        print_impact_footnote(t_time != "")

    print("\nNOTE: Matmul dimensions are in the format MxK:KxN:MxN")
    print("\nDetailed Op Summary")
    print(detailed_table)
    print_impact_footnote(t_time != "")
    if fl == 1:
        print("\nAlgo used for each matmul dimensions:\n")
        t = PrettyTable(["M", "N", "K", "ALGO_TYPE", "ALGO USED"])
        for ud in algo_t:
            t.add_row(ud)
        print(t)
    flops_dict, flop_extras = flops_dict_maker(op_info_dict, t_time, args, log_index)
    flops_table, flops_df = table_maker(
        flops_dict,
        "% Impact",
        False,
        {"Data Format": "max"},
        hr_val=True,
    )
    print("\nFLOPs and Efficiency for matmul and batch_matmul ops:")
    if not flops:
        print(
            "Efficiency has not been calculated. Use --roofline_path and -mc to enable kernel efficiency.\n"
        )

    if args.info:
        print(
            "GOPs (Giga Operations) - the total number of arithmetic operations that occurs for the given dimensionz, measured in Giga"
        )
        print(
            "GFlops - measures the number of floating-point operations that a system or processor can perform in one second, measured in billions."
        )
        print(
            "Efficiency - Comparing the achieved GFLOPS performance of a system against its peak theoretical GFLOPS performance."
        )
        print("Formulae :")
        print("GFlops = Gops * 1000 / Actual time in ms")
        print("GOPs = 2 * M * N * K * 0.000000001")
        print("Efficiency = Achieved GFLOPS/Theoretical GFLOPS * 100")
        print(
            "Theoretical GFlops = SIMD_Width / Data_Type_Size * Num_Cores * FMA_Per_Core *OPS_Per_FMA * DoublePumpMultiplier * Freq_Of_Core"
        )
        print()
    if flops:
        roofline_dir = getattr(args, "roofline_path", "") or "../../Roofline/"
        print(
            f"Machine config has been taken from '{roofline_dir}/machines_config.json'. Update the JSON for custom configuration."
        )
        print(
            "Machine frequency=",
            flop_extras["Frequency"],
            ", Threads=",
            flop_extras["Threads"],
            ", DoublePumpMultiplier=",
            flop_extras["DoublePumpMultiplier"],
        )
        print(
            "Theoretical GFlops for ",
            ", ".join(
                [
                    a + "=" + str(non_zero_round(flop_extras["Dtype"][a], 3))
                    for a in flop_extras["Dtype"]
                ]
            ),
        )
        print()
    print(flops_table)
    print_impact_footnote(t_time != "")
    if "faulty" in extras:
        faulty_detailed_dict, mis = detailed_dict_maker(
            extras["faulty"]["kwargs"]["op_info_dict"],
            op_info_dict1,
            "",
            args,
            {},
            1,
            1,
            0,
            0,
        )
        faulty_detailed_table, faulty_detailed_df = table_maker(
            faulty_detailed_dict, "Total Time (ms)", False, {"Data Format": "max"}, hr_val=True
        )
        print("\nFaulty Ops Detailed Summary:")
        print(
            "Since the tool is running in Safe Mode, the ops that were not considered for analysis are summarized below."
        )
        print(
            "Note : The following table contains the data from the entire log rather than a specific iteration.\n"
        )
        print(faulty_detailed_table)
    if export_csv:
        csv_file = input("\nEnter the file name for the csv (Without the extension) : ")
        csv_file += ".csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = cs.writer(csvfile)
            table_list = [group_table, detailed_table]
            csvwriter.writerow(["File name", args.input])
            csvwriter.writerow(["Total Iteration", args.iter])
            csvwriter.writerow(["Analysis Iteration", args.number])
            timestamp = time1.time()
            date_time = datetime.fromtimestamp(timestamp)
            formatted_date_time = date_time.strftime("%d-%m-%Y %H:%M:%S")
            csvwriter.writerow(["Time stamp", formatted_date_time])
            if flops:
                table_list.append(flops_table)
            for tab in table_list:
                csvwriter.writerow(tab.field_names)
                # PrettyTable has no public API to iterate raw row data;
                # _rows is the only way to access the underlying list.
                for row in tab._rows:
                    csvwriter.writerow(row)
                csvwriter.writerow([])
                csvwriter.writerow([])
    table_df_list = [group_df, detailed_df, flops_df, emb_df]
    for pr in extras["print_after_statements"]:
        print(pr)
    print("\n" + "=" * 80 + "\n")
    return table_df_list
