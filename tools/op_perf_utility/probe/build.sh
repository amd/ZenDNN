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
# Builds the probe (probe.cpp) against a ZenDNN build, on the host: the static
# zendnnl archive and the dependencies installed under <build>/install, with the
# library's hardware detection from this checkout's zendnnl/src.  The AVX-512
# kernels carry their ISA as function attributes and run only where the CPU has
# it, so one binary serves every machine.
# deps/aoclutils is required; the dependencies a build can leave out
# (deps/aocldlp, deps/onednn, deps/libxsmm, deps/parlooper) are linked when
# installed.
# Usage: build.sh <worktree or build dir> <out-binary>     env: CXX (c++)
set -euo pipefail
[ "$#" -eq 2 ] || { echo "usage: $0 <worktree or build dir> <out-binary>" >&2; exit 2; }
ARG=$(readlink -f "$1")
OUT=$2
if [ -d "$ARG/build/install" ]; then
    B=$ARG/build
else
    B=$ARG
fi
I=$B/install
D=$I/deps
HERE=$(dirname "$(readlink -f "$0")")
REPO=$(readlink -f "$HERE/../../..")
SRC=$HERE/probe.cpp
LIB=$B/zendnnl/src/libzendnnl_archive.a
UTILS=$D/aoclutils/lib/libaoclutils.a
for f in "$LIB" "$UTILS" "$REPO/zendnnl/src/lowoha_operators/matmul/matmul_native/common/native_hw.hpp"; do
    [ -f "$f" ] || { echo "no $f: build zendnnl first (and run this script from a ZenDNN checkout)" >&2; exit 1; }
done
extra=()
for a in "$D/aocldlp/lib/libaocl-dlp.a" "$D/onednn/lib/libdnnl.a" "$D/libxsmm/lib/libxsmm.a" \
         "$D/parlooper/lib/libparlooper.a"; do
    [ -f "$a" ] && extra+=("$a")
done
inc=("-I$REPO/zendnnl/src")
for d in "$I/zendnnl/include" "$D/aoclutils/include" "$D/json/include"; do
    [ -d "$d" ] && inc+=("-I$d")
done
"${CXX:-c++}" -std=c++17 -O2 -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1 "${inc[@]}" \
    "$SRC" -o "$OUT" "$LIB" \
    -Wl,--push-state,--whole-archive "$UTILS" "${extra[@]}" -Wl,--pop-state \
    -ldl -lpthread
echo "built $OUT from $B ($(git -C "$REPO" rev-parse --short=8 HEAD 2> /dev/null || echo "no git"))"
