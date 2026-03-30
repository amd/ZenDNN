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

"""Shared helpers for the data collection pipeline."""

from __future__ import annotations

MIN_SELF_CPU_RATIO = 0.9

# Conversion factors: divide µs by this to get ms, multiply s by this to get ms.
_US_TO_MS = 1000.0
_S_TO_MS = 1000.0

# Default decimal precision when rounding converted time values.
_TIME_PRECISION_MS = 4


def non_zero_round(num: float | None, precision: int = 0, max_precision: int = 16) -> float | int:
    """Round to the first non-zero decimal place.

    Increases decimal precision until the rounded value is non-zero,
    or until ``max_precision`` is reached, or returns 0 if the input
    is exactly zero.
    """
    if num is None:
        return 0
    if precision == 0:
        return round(num)
    while round(num, precision) == 0 and num != 0 and precision < max_precision:
        precision += 1
    return round(num, precision)


def time_converter(time_strings: list[str] | None) -> list[float | str]:
    """Convert profiler time strings (e.g., '1.23ms', '456us', '0.5s') to ms.

    Returns:
        List of numeric values in milliseconds, with NaN strings preserved.
    """
    if time_strings is None:
        return []
    result = []
    for raw in time_strings:
        if str(raw) == "nan":
            result.append(raw)
            continue

        # Use 0.0 as placeholder for bad entries to preserve index alignment
        # with other parallel columns (Name, Shape, Count, etc.).
        try:
            if raw.endswith("ms"):
                ms_val = float(raw[:-2])
            elif raw.endswith("us"):
                ms_val = float(raw[:-2]) / _US_TO_MS
            elif raw.endswith("s") and not (raw.endswith("ms") or raw.endswith("us")):
                ms_val = float(raw[:-1]) * _S_TO_MS
            else:
                print(f"WARNING: Unrecognized time unit: '{raw[-2:]}'")
                result.append(0.0)
                continue
        except (ValueError, AttributeError):
            print(f"WARNING: Malformed time value: {repr(raw)}")
            result.append(0.0)
            continue

        result.append(non_zero_round(ms_val, _TIME_PRECISION_MS))
    return result
