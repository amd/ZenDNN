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

import sys
import os

_devnull = None
_prev_stdout = None


def block_print():
    """Redirect stdout to devnull to suppress print output."""
    global _devnull, _prev_stdout
    if _devnull is not None:
        return
    _prev_stdout = sys.stdout
    _devnull = open(os.devnull, "w")
    sys.stdout = _devnull


def enable_print():
    """Restore stdout to whatever it was before block_print() and close
    the devnull handle."""
    global _devnull, _prev_stdout
    if _prev_stdout is not None:
        sys.stdout = _prev_stdout
        _prev_stdout = None
    else:
        sys.stdout = sys.__stdout__
    if _devnull is not None:
        _devnull.close()
        _devnull = None
