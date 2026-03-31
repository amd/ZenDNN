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
import re
from zenml.config.profiler_config import log_info, SAFE_MODE_OPS

# Any ANSI escape sequence marks the start of injected process output.
# PROF lines never use ANSI codes, so everything from the first \x1b[
# onward is foreign content (vLLM, Ray, or any other concurrent writer).
_ANSI_INJECTION_RE = re.compile(r"\x1b\[.*")

_PROF_TIME_END_RE = re.compile(r"time[=:][\d.]+ms\s*$")

# Regexes for embedding sub-types
_EMB_CTX_CREATE_RE = re.compile(
    r"Embedding context create - table\[(\d+),(\d+)\]:(\w+):\w+:,time:([\d.]+)ms"
)
_EMB_OP_CREATE_RE = re.compile(
    r"Embedding operator create - ([^,]+),table\[(\d+),(\d+)\]:(\w+):\w+:,"
    r":?time:([\d.]+)ms"
)
_EMB_OP_EXECUTE_RE = re.compile(
    r"Embedding operator execute - ([^,]+),kernel:(\w+),"
    r"indices\[(\d+),(\d+)\]:\w+:\w+:,"
    r"output\[(\d+),(\d+)\]:(\w+):\w+:,"
    r"time:([\d.]+)ms"
)


def parse_embedding_line(line, backend):
    """Parse an embedding log line and return (ind_key, op_time) or (None, None).

    Three sub-types are supported:
      - Embedding context create  -> op = embedding_context_create
      - Embedding operator create -> op = embedding_op_create
      - Embedding operator execute -> op = embedding

    Returns the same (ind, op_time) tuple format as kv_build_ind so the
    result slots directly into op_info_dict.
    """
    if "Embedding" not in line:
        return None, None

    m = _EMB_OP_EXECUTE_RE.search(line)
    if m:
        plugin_op = m.group(1)
        kernel = m.group(2)
        seq_len = m.group(4)
        out_seq = m.group(5)
        hidden = m.group(6)
        dtype = m.group(7)
        time_ms = float(m.group(8))
        dim = f"{out_seq}x{hidden}"
        data_format = f"src_{dtype}::dst_{dtype}"
        ind = f"embedding,{kernel},{dim},NA,{backend},{data_format},{plugin_op}"
        return ind, time_ms

    m = _EMB_CTX_CREATE_RE.search(line)
    if m:
        vocab = m.group(1)
        hidden = m.group(2)
        dtype = m.group(3)
        time_ms = float(m.group(4))
        dim = f"{vocab}x{hidden}"
        data_format = f"src_{dtype}"
        ind = f"embedding_context_create,NA,{dim},NA,{backend},{data_format}"
        return ind, time_ms

    m = _EMB_OP_CREATE_RE.search(line)
    if m:
        plugin_op = m.group(1)
        vocab = m.group(2)
        hidden = m.group(3)
        dtype = m.group(4)
        time_ms = float(m.group(5))
        dim = f"{vocab}x{hidden}"
        data_format = f"src_{dtype}"
        ind = (
            f"embedding_op_create,NA,{dim},NA,{backend},{data_format},{plugin_op}"
        )
        return ind, time_ms

    return None, None


def fix_interleaved_lines(log_text):
    """Detect and repair PROF log lines broken by interleaved process output.

    Any concurrent writer (vLLM APIServer, Ray, custom loggers, etc.) that
    uses ANSI color codes can inject content mid-line.  Since PROF lines
    never contain ANSI escapes, everything from the first ``\\x1b[`` onward
    is foreign and gets stripped.  If the stripped line is incomplete
    (missing ``time=...ms``), the next non-PROF line is appended as the
    continuation fragment.

    Example breakage::

        [PROF ...]:LOWOHA matmul_direct: M=2048, K=40\\x1b[0;36m(proc)...
        96, alpha=1, ... time=117.726ms

    After fix:  ``[PROF ...]:LOWOHA matmul_direct: M=2048, K=4096, ... time=117.726ms``

    The original log file on disk is **never modified**; this operates
    purely on the in-memory string.

    Returns:
        (cleaned_text, fix_count) -- the repaired text and how many lines
        were fixed.
    """
    lines = log_text.split("\n")
    fixed = []
    fix_count = 0
    i = 0
    while i < len(lines):
        line = lines[i]

        if "[PROF" in line and "\x1b[" in line:
            cleaned = _ANSI_INJECTION_RE.sub("", line).rstrip()

            if not _PROF_TIME_END_RE.search(cleaned):
                if i + 1 < len(lines) and "[PROF" not in lines[i + 1]:
                    cleaned = cleaned + lines[i + 1]
                    i += 1
            fixed.append(cleaned)
            fix_count += 1
        else:
            fixed.append(line)
        i += 1

    if fix_count:
        logging.warning(
            "Fixed %d interleaved log lines (foreign process output "
            "was injected into profiling lines)", fix_count
        )

    return "\n".join(fixed), fix_count


def parse_kv_line(line, typ_cfg):
    """
    Parse a single KV-format log line (e.g., ZenDNN_5.2), matching only lines
    that contain the configured op_identifier.

    The log line format is:
        [prefix]:LOWOHA <op_name>: key1=val1, key2=val2, ...

    Parameters:
    - line (string): The log line to parse.
    - typ_cfg (dict): Per-op-type configuration from log_info, must contain
      'start' and 'op_identifier'.

    Returns:
    - kv_dict (dict or None): Parsed key-value pairs with '_op' set to the
      operation name, or None if the line doesn't match.
    """
    start = typ_cfg["start"]
    op_identifier = typ_cfg["op_identifier"]

    if start not in line:
        return None

    relevant = line[line.find(start) + len(start) :]

    # Extract op name (text before the first colon)
    colon_idx = relevant.find(":")
    if colon_idx == -1:
        return None
    op_name = relevant[:colon_idx].strip()

    # Only match lines whose op name matches this config entry
    if op_name != op_identifier:
        return None

    # Parse key=value pairs, respecting bracket-enclosed values like [gelu_erf]
    kv_part = relevant[colon_idx + 1 :].strip()
    kv_dict = {}
    bracket_depth = 0
    current = ""
    for char in kv_part:
        if char == "[":
            bracket_depth += 1
            current += char
        elif char == "]":
            bracket_depth -= 1
            current += char
        elif char == "," and bracket_depth == 0:
            if "=" in current:
                key, _, value = current.strip().partition("=")
                kv_dict[key.strip()] = value.strip()
            current = ""
        else:
            current += char
    if current.strip() and "=" in current:
        key, _, value = current.strip().partition("=")
        kv_dict[key.strip()] = value.strip()

    kv_dict["_op"] = op_name
    return kv_dict


def kv_build_ind(kv, backend, typ_cfg):
    """
    Build the standard op_info_dict key from a parsed KV log line, using
    the per-op-type config from log_info.json to map fields.

    The output key format matches the existing convention:
        op,ker,dim,post,backend,data_format[,plugin_op]

    Config fields used:
    - mapped_op      : Canonical op name (e.g., "matmul", "softmax")
    - dim_fields     : KV field names to extract dimensions from
    - dim_type       : How to build dim ("matmul" -> MxK:KxN:MxN, "softmax" -> batchxaxis_dim)
    - batch_fields   : (optional) KV field names for batch info
    - ker            : KV field name for kernel (null -> "NA")
    - post           : KV field name for post-ops (null -> "NA")
    - time           : KV field name for time value
    - time_suffix    : Suffix to strip from time (default "ms")
    - format_fields  : KV field names for dtype info
    - format_prefixes: Corresponding prefixes (e.g., ["src","dst","wei"])
    - plugin_op      : (optional) KV field name for plugin_op

    Parameters:
    - kv (dict): Parsed key-value dict from parse_kv_line.
    - backend (string): Backend name (e.g., 'ZenDNN_5.2').
    - typ_cfg (dict): Per-op-type configuration from log_info.

    Returns:
    - ind (string or None): The composed key, or None if essential fields are missing.
    - op_time (float or None): The execution time in ms.
    """
    op = typ_cfg["mapped_op"]

    # --- Dimensions ---
    dim_type = typ_cfg.get("dim_type", "generic")
    dim_values = {}
    for field in typ_cfg.get("dim_fields", []):
        val = kv.get(field, "")
        if val == "":
            return None, None
        dim_values[field] = val

    if dim_type == "matmul":
        M, N, K = dim_values["M"], dim_values["N"], dim_values["K"]
        batch = 1
        for bf in typ_cfg.get("batch_fields", []):
            b = int(kv.get(bf, "1") or "1")
            batch = max(batch, b)
        if batch > 1:
            dim = f"{batch}x{M}x{K}:{batch}x{K}x{N}:{batch}x{M}x{N}"
            if op == "matmul":
                op = "batch_matmul"
        else:
            dim = f"{M}x{K}:{K}x{N}:{M}x{N}"
    elif dim_type == "softmax":
        dim = "x".join(dim_values[f] for f in typ_cfg["dim_fields"])
    elif dim_type == "convolution":
        batch = dim_values.get("batch", "1")
        in_c = dim_values["in_c"]
        out_c = dim_values["out_c"]
        in_h = dim_values["in_h"]
        in_w = dim_values["in_w"]
        fh = dim_values["filter_h"]
        fw = dim_values["filter_w"]
        dim = f"mb{batch}ic{in_c}oc{out_c}_{in_h}x{in_w}_{fh}x{fw}"
    else:
        dim = "x".join(dim_values[f] for f in typ_cfg["dim_fields"])

    # --- Kernel ---
    ker_field = typ_cfg.get("ker")
    ker = kv.get(ker_field, "NA") if ker_field else "NA"

    # --- Post-ops ---
    post_field = typ_cfg.get("post")
    if post_field:
        post_raw = kv.get(post_field, "[none]")
        if post_raw in ("[none]", "", "none"):
            post = "NA"
        else:
            post = post_raw.strip("[]")
    else:
        post = "NA"

    # --- Data format (constructed from dtype fields) ---
    format_fields = typ_cfg.get("format_fields", [])
    format_prefixes = typ_cfg.get("format_prefixes", [])
    if format_fields and format_prefixes:
        fmt_parts = []
        for prefix, field in zip(format_prefixes, format_fields):
            val = kv.get(field, "NA")
            fmt_parts.append(f"{prefix}_{val}")
        data_format = "::".join(fmt_parts)
    else:
        data_format = "NA"

    # --- Build the ind key ---
    ind = f"{op},{ker},{dim},{post},{backend},{data_format}"

    # --- Plugin op (optional) ---
    plugin_field = typ_cfg.get("plugin_op")
    if plugin_field:
        plugin_val = kv.get(plugin_field, "")
        if plugin_val:
            ind = ind + "," + plugin_val

    # --- Time ---
    time_field = typ_cfg.get("time", "time")
    time_str = kv.get(time_field, "")
    time_suffix = typ_cfg.get("time_suffix", "ms")
    if time_str.endswith(time_suffix):
        time_str = time_str[: -len(time_suffix)]
    try:
        op_time = float(time_str)
    except ValueError:
        return None, None

    return ind, op_time


def kv_log_parser(log, backend, safe_mode, log_type, in_sc, t_time=False, special=False):
    """
    Parse key-value format logs (e.g., ZenDNN_5.2) and return the same
    op_info_dict structure as the positional log_parser.

    Iterates over each log line and tries each per-op-type config entry
    (e.g., matmul_direct, softmax_direct) to find matching lines.

    Parameters:
    - log (string): The log content.
    - backend (string): Backend name.
    - safe_mode (bool): Flag for safe mode.
    - log_type (string): Type of log (execute/create/prof).
    - in_sc (string): Inference script.
    - t_time (bool): Whether to return total time.
    - special (bool): Analyse only faulty ops.

    Returns:
    - Same return format as log_parser.
    """
    op_info_dict = {}
    t_t = ""

    if log_type not in log_info[backend]:
        if t_time:
            return op_info_dict, ""
        return op_info_dict

    log, _ = fix_interleaved_lines(log)

    for i, j in enumerate(log.strip().split("\n")):
        # Hugging Face total time extraction
        if in_sc == "hugging_face":
            if (
                "Model Name             Batch Size     Seq Length     Time in s"
                in j
            ):
                t_t = log.strip().split("\n")[i + 2].split()[-1]
            if "Latency for step " in j:
                t_t = str(float(j[16:].strip().split(" ")[1]) / 1000)

        # Try each per-op-type config against this line
        matched = False
        for typ_name in log_info[backend][log_type]:
            typ_cfg = log_info[backend][log_type][typ_name]

            kv = parse_kv_line(j, typ_cfg)
            if kv is None:
                continue

            ind, op_time = kv_build_ind(kv, backend, typ_cfg)
            if ind is None or op_time is None:
                continue

            op = ind.split(",")[0]

            if safe_mode and op not in SAFE_MODE_OPS:
                continue
            if special and op in SAFE_MODE_OPS:
                continue

            if ind not in op_info_dict:
                op_info_dict[ind] = [op_time]
            else:
                op_info_dict[ind].append(op_time)
            matched = True

        # Fallback: try embedding parser for lines not matched by KV config
        if not matched and "Embedding" in j:
            ind, op_time = parse_embedding_line(j, backend)
            if ind is not None and op_time is not None:
                op = ind.split(",")[0]
                if safe_mode and op not in SAFE_MODE_OPS:
                    continue
                if special and op in SAFE_MODE_OPS:
                    continue
                if ind not in op_info_dict:
                    op_info_dict[ind] = [op_time]
                else:
                    op_info_dict[ind].append(op_time)

    if t_time:
        try:
            float(t_t.strip())
            return op_info_dict, t_t
        except Exception:
            return op_info_dict, ""

    return op_info_dict
