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
# Rewrites the tuner's fixtures from a ZenDNN build: fixtures/schema_dump.json
# from the log of its NativeConfigSchema.SchemaDump gtest, and
# fixtures/plan_check.json from its planner's plans for the cases of
# tests/plan_cases.py. The plan fixture is written only when plan.py gives the
# library's plan in every case; otherwise the differences are listed and
# plan.py must follow the library first. The kernel registry follows the ISA
# (an f16 call takes the F16_PH kernels only with AVX512-FP16), so the plans
# come from a machine of the schema dump's ISA: run the script there, or give
# DUMP, the plan dump's output for the cases made there (plan_dump/plan_dump <
# the output of tests/plan_cases.py cases). Take the dump from a machine with
# every kernel kind the planner has, so that the planner's test of the
# fixture (NativeConfigSchema.TheTunerFollowsThePlanner) passes on every
# machine.
# Usage: refresh_fixtures.sh <worktree> <SchemaDump gtest log> [build dir]
# env: PYTHON (python3), CXX (c++), DUMP (a plan dump made elsewhere)
set -euo pipefail
[ "$#" -ge 2 ] && [ "$#" -le 3 ] || {
    echo "usage: $0 <worktree> <SchemaDump gtest log> [build dir]" >&2
    exit 2
}
HERE=$(dirname "$(readlink -f "$0")")
PY=${PYTHON:-python3}
WT=$(readlink -f "$1")
LOG=$(readlink -f "$2")
B=${3:-$WT/build}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
"$PY" "$HERE/tune.py" schema --extract "$LOG" -o "$T/schema.json"
"$PY" "$HERE/tune.py" schema --schema "$T/schema.json" --compare "$HERE/fixtures/schema_dump.json" || true
if [ -n "${DUMP:-}" ]; then
    cp "$DUMP" "$T/dump.txt"
else
    bash "$HERE/plan_dump/build.sh" "$WT" "$T/plan_dump" "$B"
    "$PY" "$HERE/tests/plan_cases.py" cases > "$T/cases.txt"
    "$T/plan_dump" < "$T/cases.txt" > "$T/dump.txt"
fi
if ! "$PY" "$HERE/tests/plan_cases.py" compare "$T/dump.txt" --schema "$T/schema.json"; then
    echo "plan.py differs from the library: the fixtures are not rewritten" >&2
    exit 1
fi
cp "$T/schema.json" "$HERE/fixtures/schema_dump.json"
"$PY" "$HERE/tests/plan_cases.py" fixture "$T/dump.txt" --schema "$HERE/fixtures/schema_dump.json" \
    > "$HERE/fixtures/plan_check.json"
echo "fixtures refreshed from $WT ($(git -C "$WT" rev-parse --short=8 HEAD 2> /dev/null || echo "no git"))"
