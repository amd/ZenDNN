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

"""
ZenML Profiler -- unified entry point.

Usage:
    python profiler.py -in1 <log1> -bk1 ZenDNN -it1 10 [options]

This replaces the old ZenML_Profiler.py with a modular structure.
"""

import argparse
import sys
import os
import json
import logging
import warnings
import subprocess

from zenml.utils.logging_fmt import ColoredFormatter
from zenml.utils.helpers import block_print, enable_print
from zenml.config.profiler_config import log_info, SAFE_MODE_OPS
from zenml.parsers.dispatcher import parse_log
from zenml.analysis.pattern_analyzer import custom_log_generator, pattern_analyzer
from zenml.analysis.iteration_splitter import iteration_splitter
from zenml.analysis.log_slicer import log_slicer
from zenml.reporting.orchestrator import result_generation
from zenml.reporting.comparison import compare_logs
from zenml.reporting.excel import excel_generator
from zenml.integrations.benchdnn import algo_extractor, benchdnn_support
from zenml.integrations.pytorch_profiler import (
    create_function_events,
    populate_cpu_children,
    calc_self_cpu,
    PyTorch_Profiler_view,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
log_format = "%(asctime)s | %(levelname)8s | %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = ColoredFormatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Suppress DeprecationWarning / FutureWarning noise from third-party libs
# (pandas, openpyxl) so profiler output stays clean.
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def args_parser():
    """
    Create an argument parser to handle command-line arguments
    and return the parsed arguments.

    Returns:
    - args (Namespace): Parsed arguments.
    - pt_prof_only (bool): True if only PyTorch profiler logs are provided.
    """
    try:
        parser = argparse.ArgumentParser(description="ZenProfiler")

        parser.add_argument(
            "-lc", "--log_count", default=1, type=int,
            help="Total number of logs passed",
        )
        parser.add_argument("-in1", "--input_1", default=False, help="Path to log file 1")
        parser.add_argument(
            "-ppj1", "--pytorch_profiler_json_1", default=False,
            help="Path to PyTorch profiler JSON of log file 1",
        )
        parser.add_argument(
            "-ppl1", "--pytorch_profiler_log_1", default=False,
            help="Path to PyTorch profiler log file 1",
        )

        args, unknown = parser.parse_known_args()
        pt_prof_only = False

        if args.input_1 is False:
            if args.pytorch_profiler_json_1 is not False or args.pytorch_profiler_log_1 is not False:
                pt_prof_only = True
            else:
                logging.critical(
                    "Please provide input log file or PyTorch profiler log file or PyTorch Profiler JSON!"
                )
                sys.exit(1)

        for i in range(1, args.log_count + 1):
            if i != 1:
                parser.add_argument(f"-in{i}", f"--input_{i}", help=f"Path to log file {i}")
                parser.add_argument(
                    f"-ppj{i}", f"--pytorch_profiler_json_{i}", default=False,
                    help=f"Path to PyTorch profiler JSON of log file {i}",
                )
                parser.add_argument(
                    f"-ppl{i}", f"--pytorch_profiler_log_{i}", default=False,
                    help=f"Path to PyTorch profiler log file {i}",
                )
            parser.add_argument(
                f"-it{i}", f"--iter_{i}", default=False,
                help=f"Total number of iterations for log {i}",
            )
            parser.add_argument(
                f"-n{i}", f"--number_{i}", default=False,
                help=f"Nth iteration to be considered for log {i}",
            )
            parser.add_argument(f"-ln{i}", f"--log_name_{i}", default=False, help=f"Log {i} title")
            parser.add_argument(
                f"-t{i}", f"--time_{i}", default=False,
                help=f"Total time in seconds for Log {i}",
            )
            parser.add_argument(
                f"-bk{i}", f"--backend_{i}",
                choices=["ZenDNN_5.1", "OneDNN", "ZenDNN_4.2", "ZenDNN_5.2"],
                default="ZenDNN_5.2",
                help=f"Backend used for benchmarking while generating Log {i}",
            )

        parser.add_argument("-e", "--export", default=False, help="Export the sliced logs")
        parser.add_argument("--benchdnn", action="store_true", help="Enable BenchDNN support")
        parser.add_argument("--benchdnn_disp", action="store_true", help="BenchDNN display support")
        parser.add_argument(
            "-mc", "--machine", default="",
            choices=["genoa", "turin-dense", "turin-classic", "venice-pprox", "venice-whatif"],
            help="Machine used for benchmarking",
        )
        parser.add_argument(
            "--roofline_path", default="",
            help="Path to Roofline directory (containing mmRoofline.py and machines_config.json). "
                 "Required for kernel efficiency calculation. If not set, efficiency is skipped.",
        )
        parser.add_argument("--info", action="store_true", help="Enable information about the tables")
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable high verbosity")
        parser.add_argument("-dp", "--disable_pattern", action="store_true", help="Disable pattern search")
        parser.add_argument("-dx", "--disable_excel", action="store_true", help="Disable Excel generation")
        parser.add_argument(
            "--compare_mm", action="store_true",
            help="Generate additional matmul/batch_matmul comparison table with unified B, M, N, K columns",
        )
        parser.add_argument(
            "-is", "--inference_script", choices=["custom", "hugging_face"],
            default="custom", help="Inference script used for benchmarking model",
        )
        parser.add_argument("--csv", action="store_true", help="Enable exporting CSV file")
        parser.add_argument("-f", "--file_name", default="sliced_output.txt", help="Name of sliced logs file to be exported")
        parser.add_argument("-rf", "--result_file", default="ZenMLProfiler_report.txt", help="File to save the analysis report")
        parser.add_argument("--threshold", default=2, type=int, help="Threshold to filter the PyTorch Profiler result")
        parser.add_argument("--no_flags", action="store_true")

        args = parser.parse_args(unknown, namespace=args)
        return args, pt_prof_only
    except Exception as e:
        logging.critical(e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    """Main function to execute the ZenML Profiler workflow."""
    try:
        inps = []
        backends = []
        total_iter_list = []
        num_list = []
        log_times = []
        op_dicts = []
        t_time_list = []
        df_collection = {"group": [], "detail": [], "flops": [], "compare": []}

        try:
            args, pt_prof_only = args_parser()
            print("Command: python " + " ".join(sys.argv))
            print()
            compare = False

            if not pt_prof_only:
                if args.benchdnn_disp:
                    print("BENCHDNN COMPARISON RESULT")
                    block_print()

                for i in range(1, args.log_count + 1):
                    inps.append(getattr(args, f"input_{i}"))
                    total_iter_list.append(
                        int(getattr(args, f"iter_{i}"))
                        if getattr(args, f"iter_{i}") is not False
                        else None
                    )
                    if getattr(args, f"number_{i}") is False:
                        num = None
                    else:
                        num = int(getattr(args, f"number_{i}"))
                    num_list.append(num)

                    if getattr(args, f"backend_{i}"):
                        backends.append(getattr(args, f"backend_{i}"))
                    else:
                        logging.info(
                            f"Note: Backend for log file {i} was not specified separately. ZenDNN_5.2 is taken as the default backend."
                        )
                        backends.append("ZenDNN_5.2")

                    if isinstance(num, int) and num > total_iter_list[-1]:
                        logging.warning(
                            f"Analysis requested for iteration {num_list[-1]}, while total iterations is given as {total_iter_list[-1]} for log {i}."
                        )
                        logging.error("Provide a num value less than or equal to the total iterations.")
                        sys.exit(1)

                export = bool(args.export)
                file_name = args.file_name

                args.flops = args.roofline_path != "" and args.machine != ""
                if args.machine != "" and args.roofline_path == "":
                    logging.warning(
                        "Roofline path not set (--roofline_path). "
                        "Kernel efficiency calculation will be skipped. "
                        "FLOPs (GOPs/GFlops) will still be computed."
                    )

                if args.log_count > 1:
                    compare = True
                elif args.compare_mm:
                    logging.warning("--compare_mm requires at least 2 logs (log_count > 1). Flag will be ignored.")
            else:
                for i in range(1, args.log_count + 1):
                    inps.append("")

        except argparse.ArgumentError as e:
            print(str(e))
            args.print_help()
            sys.exit(1)

        timings_list = []
        log_exist = False

        for i in range(len(inps)):
            print_statements = []

            if not os.path.isfile(inps[i]) and not pt_prof_only:
                logging.warning("No file found at the given location, hence skipping file.")
                continue

            t_time = (
                "" if not isinstance(getattr(args, f"time_{i+1}"), str) else getattr(args, f"time_{i+1}")
            )

            extras = {}
            j_t_time = ""

            if t_time == "" and isinstance(getattr(args, f"pytorch_profiler_json_{i+1}"), str):
                try:
                    with open(getattr(args, f"pytorch_profiler_json_{i+1}"), "r", encoding="utf-8") as f:
                        json_info = json.load(f)
                    if "traceEvents" in json_info:
                        trace_pt_profiler = json_info["traceEvents"]
                    else:
                        trace_pt_profiler = json_info
                    function_events = create_function_events(trace_pt_profiler)
                    updated_events = populate_cpu_children(function_events)
                    self_cpu, total_self_cpu = calc_self_cpu(updated_events)
                    j_t_time = total_self_cpu / 1000000
                    extras["pt_profiler"] = trace_pt_profiler
                except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                    logging.warning(f"Error with pytorch_profiler_json for log {i+1}: {e}")
                    j_t_time = ""

            elif t_time == "" and isinstance(getattr(args, f"pytorch_profiler_log_{i+1}"), str):
                try:
                    with open(getattr(args, f"pytorch_profiler_log_{i+1}"), "r", encoding="utf-8") as f:
                        pt_prof_info = f.read()
                        pt_lines = pt_prof_info.split("\n")
                        start = -1
                        end = len(pt_lines)
                        total_time = ""
                        for i_p, j in enumerate(pt_lines):
                            if start == -1:
                                if (
                                    "Name    Self CPU %" in j
                                    and i_p > 0
                                    and "------------" in pt_lines[i_p - 1]
                                ):
                                    start = i_p + 1
                            else:
                                if (
                                    "------------" in j
                                    and i_p + 1 < len(pt_lines)
                                    and "Self CPU time total:" in pt_lines[i_p + 1]
                                ):
                                    end = i_p + 2
                                    total_time = (
                                        pt_lines[i_p + 1]
                                        .split("Self CPU time total: ")[1]
                                        .strip()
                                    )
                                    break
                        extras["pt_profiler_log"] = pt_lines[start:end] if start != -1 else []
                        extras["pt_profiler_log time"] = total_time
                        if "ms" == total_time[-2:]:
                            j_t_time = str(float(total_time[:-2]) * 0.001)
                        elif "us" == total_time[-2:]:
                            j_t_time = str(float(total_time[:-2]) * 0.000001)
                        elif total_time[-2].isnumeric() and total_time[-1] == "s":
                            j_t_time = str(float(total_time[:-1]))
                except (FileNotFoundError, IndexError, ValueError) as e:
                    logging.warning(
                        f"Error with pytorch_profiler_log for log {i+1}: {e}"
                    )
                    j_t_time = ""

            if j_t_time != "":
                t_time = j_t_time
            if not pt_prof_only:
                safe_mode = False

                with open(inps[i], "r", encoding="utf-8") as file:
                    log = file.read()

                print_statements.append("Log being processed: " + str(inps[i]))
                log_exist = True
                th = ""
                backend = backends[i]

                if t_time == "":
                    ini, th = parse_log(log, backend, safe_mode, "execute", args.inference_script, True)
                else:
                    th = t_time
                    ini = parse_log(log, backend, safe_mode, "execute", args.inference_script, False)

                if total_iter_list[i] == 1:
                    if getattr(args, f"time_{i+1}") is False:
                        th = ""
                    else:
                        th = t_time

                if total_iter_list[i] is None:
                    if args.disable_pattern:
                        temp_total = 1
                        print_statements.append("Pattern analyzer is disabled, hence concatenated analysis will be done.")
                    else:
                        temp_total = pattern_analyzer(custom_log_generator(log, backend))
                        print_statements.append(f"Pattern analyzer calculated the total iterations to be {temp_total}")
                    total_iter_list[i] = temp_total
                    setattr(args, f"iter_{i+1}", temp_total)
                else:
                    print_statements.append("User input for the total iterations: " + str(getattr(args, f"iter_{i+1}")))

                if num_list[i] is None:
                    num = int(getattr(args, f"iter_{i+1}")) - 1 if int(getattr(args, f"iter_{i+1}")) != 1 else 1
                    num_list[i] = num
                    setattr(args, f"number_{i+1}", num)

                if len(ini) == 0:
                    logging.warning("No logs were parsed")
                    if log.strip() == "":
                        logging.warning(f"The input file {inps[i]} is empty")
                        print_statements.append(f"The input file {inps[i]} is empty")
                    else:
                        logging.warning("LOGS MIGHT BE CORRUPTED!")
                        logging.warning(f"The backend chosen is {backend} for the log {inps[i]}. Please check whether it is correct.")
                        print_statements.append(f"The backend chosen is {backend} for the log {inps[i]}. Please check whether it is correct.")
                    sys.exit(0)

                lis, anomaly, create_anomaly = iteration_splitter(ini, backend, total_iter_list[i], log, safe_mode)

                if len(lis) < total_iter_list[i]:
                    safe_mode = True
                    ini = parse_log(log, backend, safe_mode, "execute", args.inference_script, False)
                    lis, anomaly, create_anomaly = iteration_splitter(ini, backend, total_iter_list[i], log, safe_mode)

                    if len(lis) < total_iter_list[i]:
                        logging.warning("The tool was not able to make sense of the log, please verify the logs.")
                        logging.info("Performing entire log analysis.")
                        print_statements.append("The tool was not able to make sense of the log, please verify the logs.")
                        print_statements.append("Performing entire log analysis.")
                        num = 1
                        total_iter_list[i] = 1
                    else:
                        logging.warning("Note : The tool is running in safe mode!!")
                        print_statements.append("Note : The tool is running in safe mode!!")

                        faulty = parse_log(log, backend, safe_mode, "execute", args.inference_script, False, True)
                        extras["faulty"] = {"type": "detailed", "kwargs": {"op_info_dict": faulty, "t_time": ""}}

                        safe_string = ",".join(SAFE_MODE_OPS)
                        logging.info(f"Only {safe_string} are being considered for analysis.")
                        print_statements.append(f"Only {safe_string} are being considered for analysis.")
                        logging.info("Inconsistent behaviour was observed with some other ops, which triggered the Safe mode.")

                sl, sl1 = log_slicer(num_list[i], total_iter_list[i], log, lis, export, file_name)

                analysis_dict = parse_log(sl, backend, safe_mode, "execute", args.inference_script)
                op_dicts.append(analysis_dict)

                if sl1 is not None:
                    one_before_dict = parse_log(sl1, backend, safe_mode, "execute", args.inference_script)
                else:
                    one_before_dict = {}

                if "create" in log_info[backend]:
                    create_dict = parse_log(sl, backend, safe_mode, "create", args.inference_script)
                    extras["create"] = {"type": "group", "kwargs": {"op_info_dict": create_dict, "t_time": th}}

                if "prof" in log_info[backend]:
                    prof_dict = parse_log(sl, backend, safe_mode, "prof", args.inference_script)
                    extras["prof"] = {"type": "detail+", "kwargs": {"op_info_dict": prof_dict}}

                fl, algo_det, algo_print = algo_extractor(log, log_info, backend)
                if algo_print != "":
                    print_statements.append(algo_print)
                if fl == 0:
                    logging.warning("Couldn't parse algo information from the log.")

                print_after_statements = []

                if len(anomaly) > 0:
                    logging.warning("Some anomaly has been detected in the logs.")
                    print_after_statements.append("Some anomaly has been detected in the logs.")
                    print_after_statements.append("Please check the following line numbers of the log, and verify if they are correct")
                    print_after_statements.append("")
                    for i_an, j_an in enumerate(anomaly):
                        if i_an != len(anomaly) - 1:
                            print_after_statements[-1] = print_after_statements[-1] + str(j_an + 1) + " , "
                            if i_an > 15:
                                logging.warning("etc......\nToo many anomalies detected, please check the log and the settings used.")
                                break
                        else:
                            print_after_statements[-1] = print_after_statements[-1] + str(j_an + 1)
                            print_after_statements.append("")

                extras["print_statements"] = print_statements
                extras["print_after_statements"] = print_after_statements

                temp_catch = result_generation(
                    th, analysis_dict, one_before_dict, args, extras,
                    total_iter_list[i], num_list[i], i, fl, algo_det,
                )
                df_collection["group"].append(temp_catch[0])
                df_collection["detail"].append(temp_catch[1])
                if len(temp_catch[2]) > 0:
                    df_collection["flops"].append(temp_catch[2])

                if len(create_anomaly) > 0:
                    logging.info("Some anomaly has been detected in the CREATE logs.")

                timings_list.append(th)
            else:
                PyTorch_Profiler_view(extras, args)

        if compare and log_exist and not pt_prof_only:
            if args.benchdnn_disp:
                enable_print()
            df_collection["compare"].append(compare_logs(op_dicts, backends, args, timings_list))

        if not args.disable_excel and log_exist and not pt_prof_only:
            excel_generator(
                args, backends, timings_list,
                df_collection["group"], df_collection["detail"],
                df_collection["flops"], df_collection["compare"],
            )

        if args.benchdnn and log_exist and not pt_prof_only:
            if backend == "ZenDNN_5.1":
                logging.info("BENCHDNN ANALYSIS IN PROGRESS!")
                json_file = os.path.join(os.path.dirname(__file__), "zenml", "config", "args.json")
                with open(json_file, "r", encoding="utf-8") as openfile:
                    args_benchdnn = json.load(openfile)

                data_type = args_benchdnn["benchDNN"]["data_type"]
                gemm_algo_type = args_benchdnn["benchDNN"]["gemm_algo_type"]
                input_threads = args_benchdnn["benchDNN"]["input_threads"]
                iterations = args_benchdnn["benchDNN"]["iterations"]
                input_file = "input.txt"

                try:
                    output_file = args_benchdnn["benchDNN"]["output_file"]
                    if output_file == "":
                        output_file = "BenchDNN_Analysis.txt"
                except (KeyError, TypeError):
                    output_file = "BenchDNN_Analysis.txt"

                try:
                    result = args_benchdnn["benchDNN"]["result"]
                    if result == "":
                        result = "BenchDNN_Comparison.txt"
                except (KeyError, TypeError):
                    result = "BenchDNN_Comparison.txt"

                zenml_file = "ZenML_Profiler_ip.log"
                backend_b = args_benchdnn["benchDNN"]["backend"]
                count_file = "count.txt"
                ip_only = args_benchdnn["benchDNN"]["ip_only"]

                benchdnn_support(input_file, df_collection["detail"], count_file)
                logging.info(
                    "Input Dimensions to BenchDNN are saved as input.txt, and the number of occurrences of each dimension is saved in count.txt!"
                )

                if ip_only == "" or ip_only.lower() == "false":
                    benchdnn_file = args_benchdnn["benchDNN"]["benchdnn_file"]
                    logging.info(f"Running BENCHDNN Infra for input dimensions in {input_file}.")

                    with open(output_file, "w", encoding="utf-8") as benchdnn_analysis:
                        subprocess.run(
                            ["bash", benchdnn_file]
                            + [data_type, gemm_algo_type, input_threads, iterations, input_file, count_file],
                            stdout=benchdnn_analysis,
                        )
                    logging.info("Completed!!!")
                    logging.info(f"Output log saved as ZenML_Profiler_ip.log and BenchDNN analysis can be found in {output_file}!")

                    logging.info("Running Comparison!!!")
                    old_args = vars(args)
                    new_args_2 = [
                        "--input_2", zenml_file, "--backend_2", backend_b,
                        "--log_name_2", "BenchDNN", "--iter_2", "1",
                        "--number_2", "1", "--log_name_1", "ZenML Profiler", "--benchdnn_disp",
                    ]
                    new_args = []
                    for o in old_args:
                        if o == "log_count":
                            if old_args[o] > 1:
                                cnt_cr = old_args[o] + 1
                                old_args[o] = cnt_cr
                                new_args_2 = [
                                    "--input_" + str(cnt_cr), zenml_file,
                                    "--backend_" + str(cnt_cr), backend_b,
                                    "--log_name_" + str(cnt_cr), "BenchDNN",
                                    "--iter_" + str(cnt_cr), "1",
                                    "--number_" + str(cnt_cr), "1", "--benchdnn_disp",
                                ]
                            else:
                                old_args[o] = 2
                        if (
                            o == "file_name" or old_args[o] == False or old_args[o] == True
                            or o == "benchdnn" or old_args[o] == None or "-ppj" in o
                        ):
                            pass
                        else:
                            new_args.append("--" + str(o))
                            new_args.append(str(old_args[o]))
                    new_args = new_args + new_args_2
                    with open(result, "w", encoding="utf-8") as comparison:
                        subprocess.run(
                            [sys.executable, __file__] + new_args, stdout=comparison,
                        )
                    logging.info("ANALYSIS COMPLETED!")
                    logging.info(f"Comparison output saved in {result}!")
            else:
                logging.warning("Currently only ZenDNN is supported for BenchDNN analysis")

    except Exception as e:
        logging.critical("Note!!!! Error occurred during execution.")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        logging.debug("%s at line %s", exc_type, exc_tb.tb_lineno)
        logging.error(e)


if __name__ == "__main__":
    main()
