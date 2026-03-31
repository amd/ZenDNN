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

# Wrapper script to check python version

import sys
import os
import subprocess
import json
import logging
try:
    from pytimedinput import timedInput
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytimedinput"])
    from pytimedinput import timedInput
from zenml.utils.logging_fmt import ColoredFormatter

log_format = '%(asctime)s | %(levelname)8s | %(message)s'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = ColoredFormatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.info("ZenML Profiler tool invoked!")

# Verify Python 3 is in use
try:
    result = subprocess.run(
        [sys.executable, "--version"],
        capture_output=True, text=True, check=False,
    )
    py_ver = result.stdout.strip() or result.stderr.strip()
    if py_ver.split()[0] == "Python":
        if py_ver.split()[1].split(".")[0] == "2":
            logging.warning("Python 2 is being used, please use python 3")
            sys.exit(1)
    else:
        logging.error("Python not found")
        sys.exit(1)
except Exception:
    logging.critical("Please check the python version being used.")
    sys.exit(1)

# json file storing input args
json_file = os.path.join(os.path.dirname(__file__), "zenml", "config", "args.json")
check_flag = 0
# command line or json input
userChoice = "C"
timedOut1 = False

# check if input json is provided
try:
    check_file = sys.argv[1]
    if check_file.endswith(".json"):
        json_file = check_file
        check_flag = 1
except (IndexError, ValueError):
    pass

if check_flag == 1:
    logging.info("Input json file name {} is given with the command".format(check_file))
else:
    # user guide
    """
    The program is designed such a way that user has the flexibility to use it in interactive way as well as idle mode.
    The user can pass input parameters to the tool either through command line or as JSON input.
    If no choice is selected,
        If no JSON file is provided, the mode will be defaulted to read from command line
        If JSON file is given with the command, args will be parsed from it.
    If JSON (J) is selected,
        User can either pass the JSON file as input or the tool will automatically check the default file.
    """
    print(
        "\nDo you want to read the input arguments from command line or through JSON? \n"
    )
    print("If JSON, enter J else C.")
    try:
        userChoice, timedOut1 = timedInput("C/J? --->", timeout=5)
    except RuntimeError:
        userChoice = "C"
        timedOut1 = True

if userChoice == "J" or userChoice == "j" or check_flag == 1:
    logging.info(
        "Please update the required flags/arguments. Refer README_ZenML_Profiler.md to know more about the arguments!"
    )
    logging.info("Reading from JSON!")

    # Collecting the arguments from json file
    with open(json_file, "r", encoding="utf-8") as openfile:
        # Reading from json file
        args_ = json.load(openfile)

    # List to store arguments/flags for Profiler run
    arguments = []
    for ele in args_["ZenML"]:
        # ignore helper message
        if ele == "_helper":
            pass
        # check if a particular flag is empty or FALSE
        elif (
            len(str(args_["ZenML"][ele])) == 0
            or str(args_["ZenML"][ele]).upper() == "FALSE"
        ):
            pass
        # if a flag is set to TRUE, append the flag name to the list
        elif str(args_["ZenML"][ele]).upper() == "TRUE":
            arguments.append(str(ele))
        # append flag name and value to the list
        else:
            arguments.append(str(ele))
            arguments.append(str(args_["ZenML"][ele]))

elif timedOut1:
    logging.info("TIME OUT!!!! Reading from command line!")
    if check_flag == 1:
        arguments = sys.argv[2:]
    else:
        arguments = sys.argv[1:]

else:
    logging.info("Reading from command line!")
    if check_flag == 1:
        arguments = sys.argv[2:]
    else:
        arguments = sys.argv[1:]
# Passing the arguments to the actual ZenML Profiler
if "--result_file" in arguments:
    index_ = arguments.index("--result_file")
    if index_ + 1 < len(arguments):
        file_n = str(arguments[index_ + 1])
    else:
        logging.warning("--result_file flag provided without a value. Using default.")
        file_n = "ZenMLProfiler_report.txt"
elif "-rf" in arguments:
    index_ = arguments.index("-rf")
    if index_ + 1 < len(arguments):
        file_n = str(arguments[index_ + 1])
    else:
        logging.warning("-rf flag provided without a value. Using default.")
        file_n = "ZenMLProfiler_report.txt"
else:
    file_n = "ZenMLProfiler_report.txt"

logging.info("ANALYSIS IN PROGRESS!")

# check for required arguments
arg_check = 1
for ar in arguments:
    if ar.startswith("-in") or ar.startswith("--in") or ar.startswith("-pp") or ar.startswith("--pytorch"):
        arg_check = 1
        break
    else:
        arg_check = 0

if arg_check == 0:
    logging.error("Please provide input log file or PyTorch profiler log file or PyTorch Profiler JSON!")
else:
    try:
        profiler_script = os.path.join(os.path.dirname(__file__), "profiler.py")
        with open(file_n, 'w', encoding="utf-8") as result_file:
            ret = subprocess.run([sys.executable, profiler_script] + arguments, stdout=result_file)
        if ret.returncode != 0:
            logging.error("Profiler exited with return code %d.", ret.returncode)
        else:
            logging.info("COMPLETED! Profiler analysis saved in {}!".format(file_n))
    except (subprocess.SubprocessError, OSError) as e:
        logging.error(f"Failed to run profiler: {e}")
