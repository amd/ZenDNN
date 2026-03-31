#!/bin/bash
# *******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/
#
# Run mini-benchdnn: run benchdnn matmul and bmm for input files under benchdnn/input/matmul.
# Shapes run in input-file order. Matmul and BMM outputs are merged
# separately (BMM has an extra BS column). Outputs: <output_base>_matmul.csv and <output_base>_bmm.csv.
# BMM (batched matmul) inputs have 5 columns (Name,bs,M,K,N); use --ndims=3 for those.
#
# Defaults:
#   recsys:           M=256,2048; iters=1000, warmup=5000; OMP=8
#   llm_generative:   M=1,32; iters=2000, warmup=5000; OMP=cores per socket
#   pytorch*:         iters=1000, warmup=2000; OMP=cores per socket
#
# Kernel: matmul uses aocl_dlp_blocked, BMM uses aocl_dlp (--kernel overrides both).
# Datatype: --dtype src:wei:dst (e.g. bf16:bf16:bf16).
# Binary post-ops (from input models): --post-op-dt (default f32); see benchdnn doc/matmul.md.
# Weight quant scales (optional): --weight-scale-granularity, --weight-group-size, --weight-scale-dt
# (benchdnn --scale_granularity, --group_size, --scale_dt; omit to use benchdnn defaults).
# Scaling: --alpha and --beta (defaults 1.0 and 0.0).
# Per-input iters/warmup/OMP via --recsys-iters, --llm-omp, etc.
# Uses numactl --physcpubind=0-(numthreads-1) --interleave=0 when available.
#
# Usage:
#   ./run_mini_benchdnn.sh [options]
#
#   [--op matmul|bmm|all] [--recsys-m M[,M...]] [--llm-m M[,M...]]
#   [--dtype src:wei:dst] [--bias-dtype DT] [--post-op-dt DT]
#   [--weight-scale-granularity G] [--weight-group-size N] [--weight-scale-dt DT]
#   [--alpha A] [--beta B]
#   [--recsys-omp N] [--llm-omp N] [--pytorch-omp N]
#   [--recsys-iters N] [--recsys-warmup N] [--llm-iters N] [--llm-warmup N]
#   [--pytorch-iters N] [--pytorch-warmup N] [--kernel NAME]
#   [--lowoha true|false|1|0] [--no-numactl] [-o|--output BASE.csv] [-h|--help]
#   (-o writes BASE_matmul.csv and BASE_bmm.csv; omit --lowoha for benchdnn default, enabled)
#
# Examples:
#   ./run_mini_benchdnn.sh --op matmul
#   ./run_mini_benchdnn.sh --op bmm
#   ./run_mini_benchdnn.sh --dtype bf16:bf16:bf16 --recsys-m 256,2048 --llm-m 1,32
#   ./run_mini_benchdnn.sh --recsys-iters 1000 --recsys-warmup 5000 --llm-iters 2000 --llm-warmup 5000 --pytorch-iters 1000 --pytorch-warmup 2000
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default M values (comma-separated; single M has no comma)
RECSYS_M_DEFAULT="256,2048"
LLM_M_DEFAULT="1,8,32"
# Per-input iters and warmup (recsys, llm_generative, pytorch)
RECSYS_ITERS_DEFAULT=1000
RECSYS_WARMUP_DEFAULT=5000
LLM_ITERS_DEFAULT=2000
LLM_WARMUP_DEFAULT=5000
PYTORCH_ITERS_DEFAULT=1000
PYTORCH_WARMUP_DEFAULT=2000
RECSYS_OMP_DEFAULT=8
LLM_OMP_DEFAULT=""      # empty = cores per socket
PYTORCH_OMP_DEFAULT=""  # empty = cores per socket
DTYPE_DEFAULT="bf16:bf16:bf16"
BIAS_DT_DEFAULT="bf16"
POST_OP_DT_DEFAULT="f32"
# Empty = do not pass to benchdnn (use its defaults for scale_granularity / group_size / scale_dt).
WEIGHT_SCALE_GRANULARITY_DEFAULT=""
WEIGHT_GROUP_SIZE_DEFAULT=""
WEIGHT_SCALE_DT_DEFAULT=""
ALPHA_DEFAULT="1.0"
BETA_DEFAULT="0.0"
# Matmul (ndims=2) vs BMM (ndims=3); --kernel overrides both.
KERNEL_MATMUL_DEFAULT="aocl_dlp_blocked"
KERNEL_BMM_DEFAULT="aocl_dlp"
OUTPUT_DEFAULT="mini_benchdnn.csv"
# Input files: benchdnn/input/matmul/mini_benchdnn_inputs (see INPUT_FILES_DEFAULT).
INPUT_FILES_DEFAULT="recsys.txt llm_generative.txt pytorch_matmul.txt pytorch_bmm.txt"
OP_FILTER_DEFAULT="all"   # matmul | bmm | all
# Empty = omit --lowoha (benchdnn default: lowoha enabled). Set to true/false to pass explicitly.
LOWOHA_DEFAULT=""

# Parsed options (set by parse_args)
RECSYS_M="$RECSYS_M_DEFAULT"
LLM_M="$LLM_M_DEFAULT"
DTYPE="$DTYPE_DEFAULT"
BIAS_DT="$BIAS_DT_DEFAULT"
POST_OP_DT="$POST_OP_DT_DEFAULT"
WEIGHT_SCALE_GRANULARITY="$WEIGHT_SCALE_GRANULARITY_DEFAULT"
WEIGHT_GROUP_SIZE="$WEIGHT_GROUP_SIZE_DEFAULT"
WEIGHT_SCALE_DT="$WEIGHT_SCALE_DT_DEFAULT"
ALPHA="$ALPHA_DEFAULT"
BETA="$BETA_DEFAULT"
KERNEL=""                 # set only when --kernel is passed
KERNEL_USER_SPECIFIED=0
KERNEL_MATMUL="$KERNEL_MATMUL_DEFAULT"
KERNEL_BMM="$KERNEL_BMM_DEFAULT"
RECSYS_OMP="$RECSYS_OMP_DEFAULT"
LLM_OMP="$LLM_OMP_DEFAULT"
PYTORCH_OMP="$PYTORCH_OMP_DEFAULT"
USE_NUMACTL=1
OUTPUT_PATH="$OUTPUT_DEFAULT"
OP_FILTER="$OP_FILTER_DEFAULT"
LOWOHA="$LOWOHA_DEFAULT"
RECSYS_ITERS="$RECSYS_ITERS_DEFAULT"
RECSYS_WARMUP="$RECSYS_WARMUP_DEFAULT"
LLM_ITERS="$LLM_ITERS_DEFAULT"
LLM_WARMUP="$LLM_WARMUP_DEFAULT"
PYTORCH_ITERS="$PYTORCH_ITERS_DEFAULT"
PYTORCH_WARMUP="$PYTORCH_WARMUP_DEFAULT"

# Physical cores per socket (no hyperthreading)
get_cores_per_socket() {
  if command -v lscpu &>/dev/null; then
    lscpu | awk '/^Core\(s\) per socket:/ { print $4 }' | head -1
  else
    local online_cpus=""
    if command -v getconf &>/dev/null; then
      online_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    fi
    if [[ -z "$online_cpus" ]] && command -v nproc &>/dev/null; then
      online_cpus="$(nproc 2>/dev/null || true)"
    fi
    if [[ -z "$online_cpus" ]]; then
      online_cpus="64"
    fi
    echo "$online_cpus"
  fi
}

# Comma-separated M list -> space-separated for "for m in ...". Positive integers only.
normalize_m_values() {
  local label="$1" s="$2"
  local comma_sep='^[1-9][0-9]*(,[1-9][0-9]*)*$'
  s="${s//, /,}"
  s="${s// ,/,}"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  if [[ -z "$s" ]]; then
    echo "Error: $label: empty M list" >&2
    return 1
  fi
  if [[ ! $s =~ $comma_sep ]]; then
    echo "Error: $label: expected comma-separated positive integers (e.g. 256,2048)" >&2
    return 1
  fi
  echo "${s//,/ }"
}

# Positive integer (for OMP thread counts; matches numactl binding guard).
require_positive_int() {
  local opt="$1" val="$2"
  if [[ -z "$val" || ! "$val" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: $opt must be a positive integer (got: ${val:-empty})" >&2
    return 1
  fi
}

# Non-negative integer (same pattern as --weight-group-size).
require_uint() {
  local opt="$1" val="$2"
  if [[ -z "$val" || ! "$val" =~ ^[0-9]+$ ]]; then
    echo "Error: $opt must be a non-negative integer (got: ${val:-empty})" >&2
    return 1
  fi
}

# --dtype must be exactly src:wei:dst (three non-empty fields; rejects bf16, bf16:f32, a:b:c:d).
require_dtype_src_wei_dst() {
  local val="$1"
  if [[ -z "$val" || ! "$val" =~ ^[^:]+:[^:]+:[^:]+$ ]]; then
    echo "Error: --dtype must be src:wei:dst with three non-empty fields (e.g. bf16:bf16:bf16); got: ${val:-empty}" >&2
    return 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --recsys-m)
        [[ $# -lt 2 ]] && { echo "Error: --recsys-m requires a value" >&2; exit 1; }
        RECSYS_M="$2"
        shift 2
        ;;
      --llm-m)
        [[ $# -lt 2 ]] && { echo "Error: --llm-m requires a value" >&2; exit 1; }
        LLM_M="$2"
        shift 2
        ;;
      --dtype)
        [[ $# -lt 2 ]] && { echo "Error: --dtype requires src:wei:dst (e.g. bf16:bf16:bf16)" >&2; exit 1; }
        DTYPE="$2"
        shift 2
        ;;
      --bias-dtype)
        [[ $# -lt 2 ]] && { echo "Error: --bias-dtype requires a datatype" >&2; exit 1; }
        BIAS_DT="$2"
        shift 2
        ;;
      --post-op-dt)
        [[ $# -lt 2 ]] && { echo "Error: --post-op-dt requires a datatype" >&2; exit 1; }
        POST_OP_DT="$2"
        shift 2
        ;;
      --weight-scale-granularity)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --weight-scale-granularity requires a value (e.g. per-channel, per-group, per-tensor)" >&2; exit 1; }
        WEIGHT_SCALE_GRANULARITY="$2"
        shift 2
        ;;
      --weight-group-size)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --weight-group-size requires a non-negative integer" >&2; exit 1; }
        [[ ! "$2" =~ ^[0-9]+$ ]] && { echo "Error: --weight-group-size must be a non-negative integer (got: $2)" >&2; exit 1; }
        WEIGHT_GROUP_SIZE="$2"
        shift 2
        ;;
      --weight-scale-dt)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --weight-scale-dt requires a datatype (e.g. f32, bf16)" >&2; exit 1; }
        WEIGHT_SCALE_DT="$2"
        shift 2
        ;;
      --alpha)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --alpha requires a numeric value" >&2; exit 1; }
        ALPHA="$2"; shift 2
        ;;
      --beta)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --beta requires a numeric value" >&2; exit 1; }
        BETA="$2"; shift 2
        ;;
      --kernel)
        [[ $# -lt 2 ]] && { echo "Error: --kernel requires a kernel name" >&2; exit 1; }
        KERNEL="$2"
        KERNEL_USER_SPECIFIED=1
        shift 2
        ;;
      --recsys-omp)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --recsys-omp requires a value" >&2; exit 1; }
        require_positive_int "--recsys-omp" "$2" || exit 1
        RECSYS_OMP="$2"
        shift 2
        ;;
      --llm-omp)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --llm-omp requires a value" >&2; exit 1; }
        require_positive_int "--llm-omp" "$2" || exit 1
        LLM_OMP="$2"
        shift 2
        ;;
      --pytorch-omp)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --pytorch-omp requires a value" >&2; exit 1; }
        require_positive_int "--pytorch-omp" "$2" || exit 1
        PYTORCH_OMP="$2"
        shift 2
        ;;
      --recsys-iters)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --recsys-iters requires a value" >&2; exit 1; }
        require_uint "--recsys-iters" "$2" || exit 1
        RECSYS_ITERS="$2"
        shift 2
        ;;
      --recsys-warmup)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --recsys-warmup requires a value" >&2; exit 1; }
        require_uint "--recsys-warmup" "$2" || exit 1
        RECSYS_WARMUP="$2"
        shift 2
        ;;
      --llm-iters)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --llm-iters requires a value" >&2; exit 1; }
        require_uint "--llm-iters" "$2" || exit 1
        LLM_ITERS="$2"
        shift 2
        ;;
      --llm-warmup)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --llm-warmup requires a value" >&2; exit 1; }
        require_uint "--llm-warmup" "$2" || exit 1
        LLM_WARMUP="$2"
        shift 2
        ;;
      --pytorch-iters)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --pytorch-iters requires a value" >&2; exit 1; }
        require_uint "--pytorch-iters" "$2" || exit 1
        PYTORCH_ITERS="$2"
        shift 2
        ;;
      --pytorch-warmup)
        [[ $# -lt 2 || -z "$2" ]] && { echo "Error: --pytorch-warmup requires a value" >&2; exit 1; }
        require_uint "--pytorch-warmup" "$2" || exit 1
        PYTORCH_WARMUP="$2"
        shift 2
        ;;
      --no-numactl)      USE_NUMACTL=0; shift ;;
      -o|--output)
        [[ $# -lt 2 ]] && { echo "Error: -o/--output requires a base path" >&2; exit 1; }
        OUTPUT_PATH="$2"
        shift 2
        ;;
      --op)
        [[ $# -lt 2 ]] && { echo "Error: --op requires matmul, bmm, or all" >&2; exit 1; }
        OP_FILTER="$2"
        shift 2
        ;;
      --lowoha)
        [[ $# -lt 2 ]] && { echo "Error: --lowoha requires true, false, 1, or 0" >&2; exit 1; }
        _lowoha_val="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
        case "$_lowoha_val" in
          true|1) LOWOHA="true" ;;
          false|0) LOWOHA="false" ;;
          *) echo "Error: --lowoha must be true|false|1|0 (got: $2)" >&2; exit 1 ;;
        esac
        shift 2
        ;;
      -h|--help)
        echo "Usage: $0 [--op matmul|bmm|all] [--recsys-m M[,M...]] [--llm-m M[,M...]]"
        echo "       [--dtype src:wei:dst] [--bias-dtype DT] [--post-op-dt DT]"
        echo "       [--weight-scale-granularity G] [--weight-group-size N] [--weight-scale-dt DT]"
        echo "       [--alpha A] [--beta B] [--recsys-omp N] [--llm-omp N] [--pytorch-omp N]"
        echo "       [--recsys-iters N] [--recsys-warmup N] [--llm-iters N] [--llm-warmup N]"
        echo "       [--pytorch-iters N] [--pytorch-warmup N] [--kernel NAME]"
        echo "       [--lowoha true|false|1|0] [--no-numactl] [-o|--output mini_benchdnn.csv] [-h|--help]"
        echo "  --op: run only matmul, only bmm, or both (default: all). BMM = input files with 5 columns (Name,bs,M,K,N)."
        echo "  -o: output base path; writes <base>_matmul.csv and <base>_bmm.csv (matmul and BMM not combined)."
        echo "  --kernel: benchdnn --kernel_name (default: aocl_dlp_blocked for matmul, aocl_dlp for BMM; this flag sets both)."
        echo "  --post-op-dt: benchdnn --post_op_dt for binary_add/binary_mul (default: f32)."
        echo "  --weight-scale-granularity: benchdnn --scale_granularity (per-channel|channel, per-group|group, per-tensor|tensor). Omit for benchdnn default."
        echo "  --weight-group-size: benchdnn --group_size for per-group scaling (non-negative int). Omit for benchdnn default."
        echo "  --weight-scale-dt: benchdnn --scale_dt for weight scale tensor (e.g. f32, bf16). Omit for benchdnn default."
        echo "  --alpha / --beta: benchdnn GEMM scaling factor (defaults: 1.0, 0.0)."
        echo "  --lowoha: pass benchdnn --lowoha=true|false (omit for benchdnn default: enabled)."
        echo "  --recsys-m / --llm-m: comma-separated batch sizes (e.g. 256,2048 or single 256)."
        exit 0
        ;;
      *) echo "Unknown option: $1"; exit 1 ;;
    esac
  done
  RECSYS_M="$(normalize_m_values "--recsys-m" "$RECSYS_M")" || exit 1
  LLM_M="$(normalize_m_values "--llm-m" "$LLM_M")" || exit 1
  [[ -z "$LLM_OMP" ]] && LLM_OMP="$(get_cores_per_socket)"
  [[ -z "$PYTORCH_OMP" ]] && PYTORCH_OMP="$(get_cores_per_socket)"
  if [[ "$KERNEL_USER_SPECIFIED" -eq 1 ]]; then
    KERNEL_MATMUL="$KERNEL"
    KERNEL_BMM="$KERNEL"
  fi
  case "$OP_FILTER" in
    matmul|bmm|all) ;;
    *) echo "Error: --op must be matmul, bmm, or all (got: $OP_FILTER)" >&2; exit 1 ;;
  esac
  require_dtype_src_wei_dst "$DTYPE" || exit 1
}

# Resolve paths (ROOT_DIR is canonical absolute path from script location)
BENCHDNN_BIN=""
BENCHDNN_BUILD=""

find_paths() {
  local base="$ROOT_DIR"
  if [[ -d "$base" ]]; then
    if [[ -x "$base/build/benchdnn/benchdnn" ]]; then
      BENCHDNN_BIN="$base/build/benchdnn/benchdnn"
    elif [[ -x "$base/build/install/benchdnn/bin/benchdnn" ]]; then
      BENCHDNN_BIN="$base/build/install/benchdnn/bin/benchdnn"
    fi
    if [[ -d "$base/build" ]]; then
      BENCHDNN_BUILD="$base/build"
    fi
    if [[ -d "$base/benchdnn/input/matmul/mini_benchdnn_inputs" ]]; then
      INPUT_BASE="$base/benchdnn/input/matmul/mini_benchdnn_inputs"
    fi
  fi
  if [[ -z "$BENCHDNN_BIN" || ! -x "$BENCHDNN_BIN" ]]; then
    echo "Error: benchdnn binary not found under $ROOT_DIR (expected build/benchdnn/benchdnn or build/install/benchdnn/bin/benchdnn)" >&2
    exit 1
  fi
  if [[ -z "$BENCHDNN_BUILD" || ! -d "$BENCHDNN_BUILD" ]]; then
    echo "Error: benchdnn build dir not found" >&2
    exit 1
  fi
  if [[ -z "$INPUT_BASE" || ! -d "$INPUT_BASE" ]]; then
    echo "Error: benchdnn/input/matmul/mini_benchdnn_inputs not found" >&2
    exit 1
  fi
}

# Run one benchdnn invocation (from build dir so timings_*.csv are written there).
# ndims: 2 for matmul, 3 for bmm (batched matmul).
run_benchdnn_one() {
  local input_model_file="$1"
  local m_val="$2"  # empty or number (used only for matmul, not bmm)
  local iters="$3"
  local warmup="$4"
  local omp="$5"
  local ndims="${6:-2}"  # 2=matmul, 3=bmm
  local numactl_cmd=()
  if [[ $USE_NUMACTL -eq 1 ]] && [[ "$omp" =~ ^[1-9][0-9]*$ ]] && command -v numactl &>/dev/null; then
    numactl_cmd=(numactl --physcpubind=0-$((omp - 1)) --interleave=0 --)
  fi
  local sdt wdt ddt
  sdt="${DTYPE%%:*}"
  ddt="${DTYPE##*:}"
  wdt="${DTYPE#*:}"; wdt="${wdt%:*}"

  local kernel_name="$KERNEL_MATMUL"
  [[ "$ndims" -eq 3 ]] && kernel_name="$KERNEL_BMM"

  local args=(
    --op=matmul
    --ndims="$ndims"
    --input_model_file="$input_model_file"
    --iters="$iters"
    --warmup_iters="$warmup"
    --sdt="$sdt" --wdt="$wdt" --ddt="$ddt"
    --bias_dt="$BIAS_DT"
    --kernel_name="$kernel_name"
    --isTransA=false --isTransB=false
    --alpha="$ALPHA" --beta="$BETA"
    --post_op_dt="$POST_OP_DT"
  )
  [[ -n "$WEIGHT_SCALE_GRANULARITY" ]] && args+=(--scale_granularity="$WEIGHT_SCALE_GRANULARITY")
  [[ -n "$WEIGHT_GROUP_SIZE" ]] && args+=(--group_size="$WEIGHT_GROUP_SIZE")
  [[ -n "$WEIGHT_SCALE_DT" ]] && args+=(--scale_dt="$WEIGHT_SCALE_DT")
  [[ -n "$LOWOHA" ]] && args+=(--lowoha="$LOWOHA")
  # --m override only for matmul (ndims=2)
  [[ -n "$m_val" && "$ndims" -eq 2 ]] && args+=(--m="$m_val")

  (cd "$BENCHDNN_BUILD" && env OMP_NUM_THREADS="$omp" "${numactl_cmd[@]}" "$BENCHDNN_BIN" "${args[@]}")
}

# Detect if input file is BMM (5 columns: Name,bs,M,K,N). Use ndims=3 for BMM.
is_bmm_input() {
  local f="$1"
  awk -F',' '
    /^#/ || /^[[:space:]]*$/ { next }
    NF >= 5 { print "1"; exit }
    { print "0"; exit }
  ' "$f"
}

# Run benchdnn for one input file; optional M override (multiple runs if multiple M values).
# Copies each generated timings_*.csv into SESSION_DIR/matmul/ or SESSION_DIR/bmm/, then removes that timings file.
run_benchdnn_category() {
  local input_file="$1"
  local category="$2"
  local iters="$3"
  local warmup="$4"
  local omp="$5"
  local m_values="$6"
  local ndims=2
  [[ "$(is_bmm_input "$input_file")" == "1" ]] && ndims=3
  local out_subdir="$SESSION_DIR/matmul"
  [[ "$ndims" -eq 3 ]] && out_subdir="$SESSION_DIR/bmm"

  if [[ -n "$m_values" && "$ndims" -eq 2 ]]; then
    for m in $m_values; do
      echo "  [${category}] M=$m ..."
      if run_benchdnn_one "$input_file" "$m" "$iters" "$warmup" "$omp" "$ndims"; then
        latest="$(ls -t "$BENCHDNN_BUILD"/timings_*.csv 2>/dev/null | head -1)"
        if [[ -n "$latest" ]] && cp "$latest" "$out_subdir/${category}_m${m}.csv"; then
          rm -f "$latest"
        fi
      fi
    done
  else
    echo "  [${category}] $([[ "$ndims" -eq 3 ]] && echo "bmm ")(single run) ..."
    if run_benchdnn_one "$input_file" "" "$iters" "$warmup" "$omp" "$ndims"; then
      latest="$(ls -t "$BENCHDNN_BUILD"/timings_*.csv 2>/dev/null | head -1)"
      if [[ -n "$latest" ]] && cp "$latest" "$out_subdir/${category}.csv"; then
        rm -f "$latest"
      fi
    fi
  fi
}

# Merge CSVs: first file's header, then data rows from all
merge_csvs() {
  local out="$1"
  shift
  local first=1
  for f in "$@"; do
    if [[ ! -f "$f" ]]; then continue; fi
    if [[ $first -eq 1 ]]; then
      head -1 "$f" > "$out"
      first=0
    fi
    tail -n +2 "$f" >> "$out"
  done
}

main() {
  parse_args "$@"
  find_paths
  [[ -n "$LOWOHA" ]] && echo "benchdnn --lowoha=$LOWOHA"

  SESSION_DIR="$(mktemp -d "${BENCHDNN_BUILD}/mini_benchdnn_session.XXXXXX")" || {
    echo "Error: could not create session directory under $BENCHDNN_BUILD" >&2
    exit 1
  }
  mkdir -p "$SESSION_DIR/matmul" "$SESSION_DIR/bmm"
  trap '[[ -n "${SESSION_DIR:-}" && -d "$SESSION_DIR" ]] && rm -rf -- "$SESSION_DIR"' EXIT

  for name in $INPUT_FILES_DEFAULT; do
    inp="$INPUT_BASE/$name"
    if [[ ! -f "$inp" ]]; then
      echo "Warning: skipping missing input $inp" >&2
      continue
    fi
    # Skip if --op filter doesn't match: matmul inputs (non-BMM) vs bmm inputs (5-col BMM)
    is_bmm="$(is_bmm_input "$inp")"
    if [[ "$OP_FILTER" == "matmul" && "$is_bmm" == "1" ]]; then
      echo "Skipping $name (bmm input, --op=matmul)" >&2
      continue
    fi
    if [[ "$OP_FILTER" == "bmm" && "$is_bmm" != "1" ]]; then
      echo "Skipping $name (matmul input, --op=bmm)" >&2
      continue
    fi
    base="${name%.txt}"
    if [[ "$base" == *recsys* ]]; then
      echo "[$base] OMP=$RECSYS_OMP iters=$RECSYS_ITERS warmup=$RECSYS_WARMUP M=($RECSYS_M)"
      run_benchdnn_category "$inp" "$base" "$RECSYS_ITERS" "$RECSYS_WARMUP" "$RECSYS_OMP" "$RECSYS_M"
    elif [[ "$base" == *llm_generative* ]]; then
      echo "[$base] OMP=$LLM_OMP iters=$LLM_ITERS warmup=$LLM_WARMUP M=($LLM_M)"
      run_benchdnn_category "$inp" "$base" "$LLM_ITERS" "$LLM_WARMUP" "$LLM_OMP" "$LLM_M"
    else
      echo "[$base] OMP=$PYTORCH_OMP iters=$PYTORCH_ITERS warmup=$PYTORCH_WARMUP"
      run_benchdnn_category "$inp" "$base" "$PYTORCH_ITERS" "$PYTORCH_WARMUP" "$PYTORCH_OMP" ""
    fi
  done

  shopt -s nullglob
  matmul_csvs=("$SESSION_DIR"/matmul/*.csv)
  bmm_csvs=("$SESSION_DIR"/bmm/*.csv)
  shopt -u nullglob
  if [[ ${#matmul_csvs[@]} -eq 0 && ${#bmm_csvs[@]} -eq 0 ]]; then
    echo "Error: no CSV outputs produced." >&2
    exit 1
  fi
  output_base="${OUTPUT_PATH%.csv}"
  output_matmul="${output_base}_matmul.csv"
  output_bmm="${output_base}_bmm.csv"
  if [[ ${#matmul_csvs[@]} -gt 0 ]]; then
    merge_csvs "$output_matmul" "${matmul_csvs[@]}"
    echo "Matmul CSV: $output_matmul"
  fi
  if [[ ${#bmm_csvs[@]} -gt 0 ]]; then
    merge_csvs "$output_bmm" "${bmm_csvs[@]}"
    echo "BMM CSV: $output_bmm"
  fi
}

main "$@"
