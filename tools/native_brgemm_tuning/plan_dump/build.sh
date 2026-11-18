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
# Builds the plan dump (plan_dump.cpp) against a ZenDNN worktree and its build:
# the planner's headers come from the source tree (they are not installed), the
# code from the static zendnnl archive and the dependencies under
# <build>/install.
# Usage: build.sh <worktree> <out-binary> [build dir (default <worktree>/build)]
# env: CXX (c++)
set -euo pipefail
[ "$#" -ge 2 ] && [ "$#" -le 3 ] || { echo "usage: $0 <worktree> <out-binary> [build dir]" >&2; exit 2; }
WT=$(readlink -f "$1")
OUT=$2
B=$(readlink -f "${3:-$WT/build}")
I=$B/install
D=$I/deps
LIB=$B/zendnnl/src/libzendnnl_archive.a
SRC=$(dirname "$(readlink -f "$0")")/plan_dump.cpp
[ -f "$LIB" ] || { echo "no $LIB: build zendnnl first" >&2; exit 1; }
extra=()
for a in "$D/aoclutils/lib/libaoclutils.a" "$D/aocldlp/lib/libaocl-dlp.a" "$D/onednn/lib/libdnnl.a" \
         "$D/libxsmm/lib/libxsmm.a" "$D/parlooper/lib/libparlooper.a"; do
    [ -f "$a" ] && extra+=("$a")
done
inc=()
for d in "$I/zendnnl/include" "$D/aoclutils/include" "$D/json/include" "$D/aocldlp/include" \
         "$D/onednn/include" "$D/libxsmm/include"; do
    [ -d "$d" ] && inc+=("-I$d")
done
"${CXX:-c++}" -std=c++17 -O2 -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1 -I"$WT/zendnnl/src" "${inc[@]}" \
    "$SRC" -o "$OUT" "$LIB" -Wl,--push-state,--whole-archive "${extra[@]}" -Wl,--pop-state -ldl -lpthread
echo "built $OUT from $WT ($(git -C "$WT" rev-parse --short=8 HEAD 2> /dev/null || echo "no git"), \
$(git -C "$WT" status --porcelain --untracked-files=no 2> /dev/null | wc -l) dirty)"
