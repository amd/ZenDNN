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
# Pod side: runs replay binaries one after another with the OpenMP setup of a
# model server's CPU worker: the libraries of REPLAY_PRELOAD preloaded (the
# worker's tcmalloc and libiomp5), KMP_AFFINITY with an explicit proclist of
# the first REPLAY_THREADS CPUs of this pod's cpuset (on a pod with more CPUs,
# the threads of its first L3 domains), KMP_BLOCKTIME=5, KMP_TPAUSE=0,
# OMP_NUM_THREADS=REPLAY_THREADS, and no OMP_PROC_BIND / OMP_PLACES /
# OMP_WAIT_POLICY.  The caller's ZENDNNL_* and
# KMP_* variables are cleared first.  Use it for A/B checks, e.g. the built-in
# plan against a tuning file:
#   run_replay.sh out/check base:./replay:10,tuned:./replay:10:ZENDNNL_MATMUL_NATIVE_TUNING_FILE=/path/t.json \
#       -- --wei s8 --shapes qkv:6144:4096 --ms 1,8,32 --check 1
# Usage: run_replay.sh <out_prefix> <item>[,<item>...] -- <replay args>
#   item: <label>:<binary>:<algo>[:<NAME=VALUE>[+<NAME=VALUE>...]]  (empty algo: ZENDNNL_MATMUL_ALGO unset)
#   env:  REPLAY_PRELOAD (colon-separated libraries), REPLAY_THREADS (32)
set -uo pipefail
[ "$#" -ge 2 ] || { sed -n '29,31p' "$0" >&2; exit 2; }
OUT=$1
ITEMS=$2
shift 2
[ "${1:-}" = -- ] && shift
PRE=${REPLAY_PRELOAD:-}
CPUS=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status)
LIST=$(echo "$CPUS" | tr ',' '\n' |
    awk -F- '{ if (NF == 2) for (i = $1; i <= $2; i++) printf "%d\n", i; else printf "%d\n", $1 }' |
    head -n "${REPLAY_THREADS:-32}" | paste -sd, -)
unset OMP_PROC_BIND OMP_PLACES OMP_WAIT_POLICY
for v in $(compgen -e | grep -E '^(ZENDNNL_|KMP_)'); do unset "$v"; done
export OMP_NUM_THREADS=${REPLAY_THREADS:-32} KMP_AFFINITY="granularity=fine,explicit,proclist=[$LIST]"
export KMP_BLOCKTIME=5 KMP_TPAUSE=0
echo "host=$(hostname) cpus=$CPUS preload=${PRE:-none} items=$ITEMS args=$* date=$(date -u '+%F %T UTC')"
i=0
IFS=',' read -r -a LIST_ITEMS <<< "$ITEMS"
for it in "${LIST_ITEMS[@]}"; do
    i=$((i + 1))
    IFS=':' read -r label bin alg extra <<< "$it"
    f=$OUT.$i.$label.txt
    EX=()
    [ -n "$extra" ] && IFS='+' read -r -a EX <<< "$extra"
    [ -n "$alg" ] && EX+=("ZENDNNL_MATMUL_ALGO=$alg")
    [ -n "$PRE" ] && EX+=("LD_PRELOAD=$PRE")
    echo "== run $i $label $bin env=${EX[*]:-none} -> $f"
    env "${EX[@]}" "$bin" "$@" > "$f" 2>&1
    echo "rc=$? $(grep -c '^RESULT' "$f") results"
    grep -E '^(CONFIG|TOTAL)' "$f"
done
