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

import pandas as pd
from prettytable import PrettyTable, ALL


def non_zero_round(num, precision=0):
    """Round ``num`` to ``precision`` decimal places, increasing precision
    if the result would be zero (and the input is non-zero)."""
    if precision == 0:
        return round(num)
    else:
        while round(num, precision) == 0 and num != 0:
            precision += 1
        return round(num, precision)


def print_impact_footnote(use_e2e):
    """Print a footnote explaining the % Impact denominator."""
    if use_e2e:
        print("* % Impact is w.r.t user-provided end-to-end time")
    else:
        print("* % Impact is w.r.t total library execution time (sum of all primitive ops)")


def table_maker(data_dict, sort_value, asc, width=None, hr_val=False):
    """
    Create a PrettyTable from a dictionary and return both the table and
    a sorted DataFrame.

    Parameters:
    data_dict (dict)    : Column name -> list of values.
    sort_value (str)    : Column name to sort by.
    asc (bool)          : True for ascending sort.
    width (dict)        : Column name -> max width override.
    hr_val (bool)       : Enable horizontal rules on every row.

    Returns:
    (PrettyTable, DataFrame)
    """
    if width is None:
        width = {}
    df = pd.DataFrame(data_dict)
    table = PrettyTable()
    if hr_val:
        table.hrules = ALL
    table.field_names = df.columns
    for field in table.field_names:
        table.align[field] = "c"
    df = df.sort_values(by=sort_value, ascending=asc)
    df_temp = pd.DataFrame(data_dict)
    df_temp = df_temp.sort_values(by=sort_value, ascending=asc)
    object_columns = df.select_dtypes(include=["object"]).columns
    df[object_columns] = df[object_columns].fillna("-")
    df_temp[object_columns] = df_temp[object_columns].fillna("-")
    for i in df.columns:
        if str(df.dtypes[i]) == "object" and i not in width:
            try:
                df_temp[i] = [": ".join([y for y in x.split(":")]) for x in df_temp[i]]
                df_temp[i] = df_temp[i].str.replace(": :", "::")
                max_len = max(
                    [
                        max(
                            [len(y) for y in x.strip().split()]
                            if x.strip() != ""
                            else [0]
                        )
                        for x in df_temp[i]
                    ]
                )
                if max_len < 55:
                    table.max_width[i] = max_len
                else:
                    table.max_width[i] = 55
            except (KeyError, TypeError):
                pass
    for i in df.columns:
        df_temp[i] = df_temp[i].replace(-1, "-")
    for row in df_temp.itertuples(index=False):
        table.add_row(row)
    if len(df) != 0:
        for i in width:
            if i in table.field_names:
                if isinstance(width[i], str):
                    max_len = []
                    for x in df[i]:
                        temp_len = [0]
                        for y in str(x).strip().split():
                            temp_len.append(len(y))
                        max_len.append(max(temp_len))
                    max_len = max(max_len)
                    if max_len < 55:
                        table.max_width[i] = max_len
                    else:
                        table.max_width[i] = 55
                else:
                    table.max_width[i] = width[i]
    return table, df
