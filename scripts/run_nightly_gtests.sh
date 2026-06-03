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
# ===== USAGE_BEGIN =====
# Run the full ZenDNN GTest suite for nightly correctness sign-off.
#
# Each operator's tests run sequentially under their own --gtest_filter so that a
# single suite hang/crash cannot poison the others.
#
# Output artifacts (all under <output-dir>/gtest_nightly_<YYYYMMDD>/):
#   * <op>_<YYYYMMDD>.txt    — per-op gtest log, prefixed with a self-describing
#                              metadata header (op, cmd, host, start_utc, ...).
#   * SUMMARY_<YYYYMMDD>.txt — human-readable run log; mirrors the live console
#                              stream (timestamped INFO/ERROR events).
#   * summary.tsv            — machine-readable, one row per op:
#                              op  status  passed  failed  skipped  duration_s
#                              rc  start_utc  end_utc  log_file
#
# Log format is deliberately vendor-neutral: UTC ISO-8601 timestamps + INFO /
# WARN / ERROR levels + key=value fields. No CI-specific markers (no
# GitHub-Actions ::error::, no TeamCity service messages, no JUnit XML); any
# CI dashboard can grep for the single final `RESULT status=PASS|FAIL` line.
#
# Exit code: 0 if every op suite passed, 1 if any op suite failed, 2 if the
# invocation itself was malformed (bad flag, missing binary, etc.).
#
# Usage:
#   ./scripts/run_nightly_gtests.sh [options]
#
# Options:
#   --binary PATH       Path to the gtests executable. If omitted, the script
#                       probes the following locations in order and uses the
#                       first executable hit:
#                         1. $ROOT_DIR/install/gtests/gtests
#                         2. $ROOT_DIR/build/install/gtests/gtests
#                         3. $ROOT_DIR/build/zendnnl/gtests/gtests
#                       (1)/(2) are the install-tree locations; (3) is the
#                       CMake build-tree location used before `make install`.
#   --output-dir DIR    Parent directory for the run folder
#                       (default: $PWD; the run folder is gtest_nightly_<date>)
#   --date YYYYMMDD     Override the date stamp (default: today's UTC date,
#                       chosen to match the UTC ISO-8601 stamps used in logs)
#   --ops LIST          Comma-separated subset of ops to run; default =
#                       full nightly sweep (everything except batchmatmul).
#                       Valid names: matmul, group_matmul, reorder, softmax,
#                                    sdpa, normalization, embedding,
#                                    embedding_bag,
#                                    lru_cache, postop_cache, omp_api,
#                                    matmul_ai_primitive, matmul_ai_lowoha,
#                                    embag_ai_primitive, embag_ai_lowoha,
#                                    batchmatmul (opt-in only; ~38h)
#                       (*_ai_* ops pin --ai_test_mode postsub plus
#                        --lowoha false (primitive) or --lowoha true (lowoha).)
#   --seed N            Fixed seed for reproducibility (default: time-based)
#   --num-threads N     Forwarded to gtests --num_threads
#   --lowoha VAL        Forwarded to gtests --lowoha (true|false|1|0)
#   --extra ARG         Extra arg forwarded verbatim to gtests. Repeatable:
#                       pass --extra once per arg to preserve shell quoting,
#                       e.g. --extra --gtest_repeat=2 --extra --gtest_color=no.
#                       (A single --extra value containing spaces is also
#                       accepted for convenience and will be word-split.)
#   --timeout SEC       Per-op wall-clock timeout in seconds (default 14400,
#                       i.e. 4 h). Overrides every op's built-in budget. Use 0
#                       to disable the timeout entirely. A timed-out op is
#                       reported with status=TIMEOUT and counts as a failure.
#   --dry-run           Print the command for each op without executing
#   -h|--help           Show this help and exit
#
# Examples:
#   ./scripts/run_nightly_gtests.sh
#   ./scripts/run_nightly_gtests.sh --seed 1776662967 --ops matmul,group_matmul
#   ./scripts/run_nightly_gtests.sh --binary ./build/install/gtests/gtests \
#                                   --output-dir /scratch/nightly
#   ./scripts/run_nightly_gtests.sh \
#       --ops matmul_ai_primitive,matmul_ai_lowoha,embag_ai_primitive,embag_ai_lowoha
#   ./scripts/run_nightly_gtests.sh --ops batchmatmul   # opt-in; ~38h
# ===== USAGE_END =====
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ------------------------------------------------------------------------------
# Logging helpers
#
# Format: <UTC ISO-8601 timestamp> <LEVEL> <message>
#   e.g.  2026-05-12T05:08:59Z INFO  [matmul] STARTED filter=Matmul/*
#
# This format is intentionally generic: no CI-vendor-specific markers
# (GitHub Actions ::error::, TeamCity service messages, JUnit XML, etc.) so
# the log stream is portable. Per-event key=value fields make grep/awk/jq
# ingestion trivial; INFO/WARN/ERROR levels match the de-facto convention
# every CI dashboard understands.
# ------------------------------------------------------------------------------
_now_utc()   { date -u +%Y-%m-%dT%H:%M:%SZ; }
_log()       { printf '%s %-5s %s\n' "$(_now_utc)" "$1" "${*:2}"; }
# Plain log_* helpers always write to stdout; callers attach `>&2` themselves
# when stderr semantics are required. Keeping the helpers redirection-free
# means a pipe like `log_error "..." | tee -a summary >&2` can read their
# output (a function body's `>&2` would consume stdout before the pipe).
log_info()   { _log INFO  "$@"; }
log_warn()   { _log WARN  "$@"; }
log_error()  { _log ERROR "$@"; }

# ------------------------------------------------------------------------------
# Op -> gtest filter + per-op extra args table.
#
# Two kinds of ops live in OP_FILTER:
#   * Randomised ops (matmul, group_matmul, reorder, ..., batchmatmul) sweep
#     random parameter tuples and inherit the global --seed / --num-threads /
#     --lowoha if supplied; their OP_EXTRA entry is empty.
#   * AI ops (matmul_ai_*, embag_ai_*) run the deterministic curated parameter
#     set under a specific test mode. They pin --ai_test_mode and --lowoha at
#     the per-op level via OP_EXTRA so the policy is recorded with the op,
#     not implicit on the command line.
#
# group_matmul uses a wildcard (GroupMatmul*/*) so all twelve group_matmul
# instantiations land in one log file.
#
# OP_NAMES is the *default* nightly sweep (order = run order). Everything
# listed here finishes within ~3-4 hours total. batchmatmul takes ~38h on
# default test_num=400 and is therefore excluded from the default; it stays
# in OP_FILTER so users can opt in explicitly via `--ops batchmatmul`.
# ------------------------------------------------------------------------------
OP_NAMES=(
  matmul
  group_matmul
  reorder
  softmax
  sdpa
  normalization
  embedding
  embedding_bag
  lru_cache
  postop_cache
  omp_api
  matmul_ai_primitive
  matmul_ai_lowoha
  embag_ai_primitive
  embag_ai_lowoha
)
declare -A OP_FILTER=(
  [matmul]='Matmul/*'
  [batchmatmul]='BatchMatmul/*'
  [group_matmul]='GroupMatmul*/*'
  [reorder]='Reorder/*'
  [softmax]='Softmax/*'
  [sdpa]='Sdpa/*'
  [normalization]='Normalization/*'
  [embedding]='Embedding/*'
  [embedding_bag]='EmbeddingBag/*'
  # Fast, deterministic unit/utility suites (no randomized data). Included in
  # the default sweep; each finishes in well under a second.
  [lru_cache]='LruCacheTryGet.*'
  [postop_cache]='*PostopCache*'
  [omp_api]='OmpApiTest.*'
  [matmul_ai_primitive]='AITests/TestMatmul*'
  [matmul_ai_lowoha]='AITests/TestMatmul*'
  [embag_ai_primitive]='AITests/TestEmbagAI*'
  [embag_ai_lowoha]='AITests/TestEmbagAI*'
)
# Per-op flag policy. Appended AFTER the global --seed/--num-threads/--lowoha
# /--extra so they win on conflict (the gtests parser uses last-write-wins via
# its umap, see gtest_utils.cpp::Parser::operator()).
#   *_primitive : --lowoha false  -> primitive operator path
#   *_lowoha    : --lowoha true   -> LOWOHA operator path
# Both AI variants pin --ai_test_mode postsub so the curated parameter set is
# deterministic and identical across the two kernel paths.
declare -A OP_EXTRA=(
  [matmul_ai_primitive]='--ai_test_mode postsub --lowoha false'
  [matmul_ai_lowoha]='--ai_test_mode postsub --lowoha true'
  [embag_ai_primitive]='--ai_test_mode postsub --lowoha false'
  [embag_ai_lowoha]='--ai_test_mode postsub --lowoha true'
)

# Per-op wall-clock timeout in seconds. A hung op is killed (SIGTERM, then
# SIGKILL after a 30s grace) so it cannot block the rest of the sweep.
# Defaults are observed-runtime * generous safety margin; --timeout SEC on the
# command line overrides for all ops. --timeout 0 disables the timeout.
TIMEOUT_DEFAULT=14400   # 4 h — covers every default-sweep op with margin.
declare -A OP_TIMEOUT=(
  [batchmatmul]=172800  # 48 h — observed ~38 h on test_num=400.
)

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------
BINARY=""
OUTPUT_PARENT="$PWD"
# UTC, not local: matches the inner _now_utc() timestamps so a run started
# near local midnight cannot land in a folder whose date disagrees with the
# events recorded inside it.
DATE_STAMP="$(date -u +%Y%m%d)"
OPS_REQUESTED=""
SEED=""
NUM_THREADS=""
LOWOHA=""
EXTRA_ARGS=()
TIMEOUT_OVERRIDE=""
DRY_RUN=0

# usage(): extract the help block bounded by the explicit USAGE_BEGIN /
# USAGE_END markers in this file's comment header, strip the leading '# '
# from each line, and drop the marker lines themselves. The explicit
# markers keep the help text from accidentally absorbing the copyright /
# license block above it (or anything we add below it later).
usage() {
  sed -n '/^# ===== USAGE_BEGIN =====/,/^# ===== USAGE_END =====/p' "$0" \
    | sed -e '/^# ===== USAGE_\(BEGIN\|END\) =====/d' -e 's/^# \?//'
  exit 0
}

# ------------------------------------------------------------------------------
# Arg parsing
#
# need_value: assert that a flag taking a value actually has one. Without it,
# `--binary` (no argument) under `set -u` would terminate with the unfriendly
# "$2: unbound variable" instead of a user-readable error. The strict variant
# also rejects values that look like another option (start with '-'), so a
# typo like `--binary --output-dir /tmp` is caught at the point of the bug
# instead of silently assigning '--output-dir' to BINARY and failing later.
# need_value_raw is the permissive variant for flags like --extra that
# legitimately forward '-'-prefixed args verbatim to the gtests binary.
# ------------------------------------------------------------------------------
_missing_value() {
  log_error "Option '$1' requires a value (got: ${2:-<empty>})" >&2
  log_error "Run with -h for help." >&2
  exit 2
}
need_value() {
  local val="${2-}"
  if [[ -z "$val" ]]; then
    _missing_value "$1" "$val"
  fi
  # A value starting with '-' followed by a non-digit (e.g. '--output-dir',
  # '-h') is treated as a missing-value typo: the user almost certainly
  # forgot the value and the next flag got consumed. Negative numbers
  # ('-1', '-1.5') are accepted so genuinely numeric flags still work.
  if [[ "$val" == -* && ! "$val" =~ ^-[0-9]+(\.[0-9]+)?$ ]]; then
    _missing_value "$1" "$val"
  fi
}
need_value_raw() {
  if [[ -z "${2-}" ]]; then
    _missing_value "$1" "${2-}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)       need_value     "$1" "${2-}"; BINARY="$2";           shift 2 ;;
    --output-dir)   need_value     "$1" "${2-}"; OUTPUT_PARENT="$2";    shift 2 ;;
    --date)         need_value     "$1" "${2-}"; DATE_STAMP="$2";       shift 2 ;;
    --ops)          need_value     "$1" "${2-}"; OPS_REQUESTED="$2";    shift 2 ;;
    --seed)         need_value     "$1" "${2-}"; SEED="$2";             shift 2 ;;
    --num-threads)  need_value     "$1" "${2-}"; NUM_THREADS="$2";      shift 2 ;;
    --lowoha)       need_value     "$1" "${2-}"; LOWOHA="$2";           shift 2 ;;
    # --extra is repeatable so users get true shell-quoting preservation:
    # `--extra "--gtest_filter=Foo Bar"` reaches the binary as one arg, not
    # two. For convenience a single --extra value containing spaces is also
    # accepted and word-split below (matches the prior single-string API).
    --extra)        need_value_raw "$1" "${2-}"; EXTRA_ARGS+=("$2");    shift 2 ;;
    --timeout)      need_value     "$1" "${2-}"; TIMEOUT_OVERRIDE="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1;          shift   ;;
    -h|--help)      usage ;;
    *) log_error "Unknown option: $1" >&2; log_error "Run with -h for help." >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------------------
# Resolve binary path
# ------------------------------------------------------------------------------
if [[ -z "$BINARY" ]]; then
  for cand in \
      "$ROOT_DIR/install/gtests/gtests" \
      "$ROOT_DIR/build/install/gtests/gtests" \
      "$ROOT_DIR/build/zendnnl/gtests/gtests"; do
    if [[ -x "$cand" ]]; then BINARY="$cand"; break; fi
  done
fi
if [[ ! -x "$BINARY" ]]; then
  log_error "gtests binary not found or not executable: ${BINARY:-<unset>}" >&2
  log_error "Pass --binary <path> or build the project first." >&2
  exit 2
fi

# Validate --timeout if supplied. (The presence check for timeout(1) is
# deferred until after op selection so dry-run and --timeout 0 don't require
# coreutils to be installed.)
if [[ -n "$TIMEOUT_OVERRIDE" && ! "$TIMEOUT_OVERRIDE" =~ ^[0-9]+$ ]]; then
  log_error "--timeout must be a non-negative integer in seconds, got '$TIMEOUT_OVERRIDE'" >&2
  exit 2
fi

# Validate DATE_STAMP: it's concatenated into RUN_DIR and every per-op
# filename, so a value containing '/', '..', or other path-active characters
# would let `--date` write outside OUTPUT_PARENT or clobber unrelated paths.
# The default YYYYMMDD form fits ^[0-9]{8}$; we relax slightly to also accept
# common rerun-suffix conventions like 20260512-rerun2 or nightly_2026_05_12.
if [[ ! "$DATE_STAMP" =~ ^[A-Za-z0-9_-]+$ ]]; then
  log_error "--date must match [A-Za-z0-9_-]+ (no path separators, no '.'), got '$DATE_STAMP'" >&2
  exit 2
fi

# ------------------------------------------------------------------------------
# Resolve op selection
# ------------------------------------------------------------------------------
if [[ -n "$OPS_REQUESTED" ]]; then
  IFS=',' read -r -a SELECTED_RAW <<< "$OPS_REQUESTED"
  SELECTED=()
  # Track first-seen ops so a duplicate (likely a typo) fails fast. We refuse
  # rather than silently dedup because the second run would overwrite the
  # first's ${op}_${DATE_STAMP}.txt log and produce two summary.tsv rows
  # pointing at the same file -- a quietly broken state worse than a typo.
  declare -A _seen=()
  for op in "${SELECTED_RAW[@]}"; do
    # Trim leading/trailing whitespace so `--ops "matmul, group_matmul"`
    # (with spaces after commas, a natural CLI habit) works correctly. Also
    # drop empty tokens that result from stray/double commas or --ops "".
    op="${op#"${op%%[![:space:]]*}"}"
    op="${op%"${op##*[![:space:]]}"}"
    [[ -z "$op" ]] && continue
    if [[ -z "${OP_FILTER[$op]:-}" ]]; then
      # List every accepted op (OP_FILTER keys), not just the default sweep,
      # so users can discover opt-in entries like batchmatmul.
      valid_ops=$(printf '%s\n' "${!OP_FILTER[@]}" | sort | paste -sd, -)
      log_error "unknown op '$op'. Valid: $valid_ops" >&2
      exit 2
    fi
    if [[ -n "${_seen[$op]:-}" ]]; then
      log_error "--ops contains duplicate op '$op'. Each op may appear at most once." >&2
      exit 2
    fi
    _seen[$op]=1
    SELECTED+=("$op")
  done
  unset _seen
  if [[ ${#SELECTED[@]} -eq 0 ]]; then
    log_error "--ops produced no op names after trimming whitespace/empty tokens" >&2
    exit 2
  fi
else
  SELECTED=("${OP_NAMES[@]}")
fi

# Check for timeout(1) only when at least one selected op will actually use
# it: dry-run never executes the binary, and --timeout 0 (or every per-op
# OP_TIMEOUT == 0) explicitly disables the wrapper. This lets users on
# systems without coreutils still drive `--dry-run` and `--timeout 0`.
if [[ "$DRY_RUN" -eq 0 ]]; then
  needs_timeout_cmd=0
  for op in "${SELECTED[@]}"; do
    if [[ -n "$TIMEOUT_OVERRIDE" ]]; then
      [[ "$TIMEOUT_OVERRIDE" -gt 0 ]] && needs_timeout_cmd=1
    else
      effective=${OP_TIMEOUT[$op]:-$TIMEOUT_DEFAULT}
      [[ "$effective" -gt 0 ]] && needs_timeout_cmd=1
    fi
    [[ "$needs_timeout_cmd" -eq 1 ]] && break
  done
  if [[ "$needs_timeout_cmd" -eq 1 ]] && ! command -v timeout >/dev/null 2>&1; then
    log_error "'timeout' command not found (expected from coreutils); pass --timeout 0 to disable or install coreutils" >&2
    exit 2
  fi
fi

# ------------------------------------------------------------------------------
# Prepare output directory + summary artifacts
# ------------------------------------------------------------------------------
RUN_DIR="$OUTPUT_PARENT/gtest_nightly_$DATE_STAMP"
SUMMARY_FILE="$RUN_DIR/SUMMARY_$DATE_STAMP.txt"   # human-readable, log-style
SUMMARY_TSV="$RUN_DIR/summary.tsv"                # machine-readable, one row per op

# Create RUN_DIR and reset both summary files at run start (the script may
# be invoked multiple times in a day, e.g. dev triage; each invocation owns
# a fresh summary). These are explicit `||` checks rather than relying on
# `set -e` so the user sees the documented exit-2 + structured ERROR log
# instead of a bare `mkdir: cannot create directory ...` shell message.
if ! mkdir -p "$RUN_DIR" 2>/dev/null; then
  log_error "failed to create run directory '$RUN_DIR' (check that '$OUTPUT_PARENT' exists and is writable)" >&2
  exit 2
fi
if ! : > "$SUMMARY_FILE" 2>/dev/null; then
  log_error "failed to write summary file '$SUMMARY_FILE' (check disk space and permissions on '$RUN_DIR')" >&2
  exit 2
fi
if ! : > "$SUMMARY_TSV" 2>/dev/null; then
  log_error "failed to write summary TSV '$SUMMARY_TSV' (check disk space and permissions on '$RUN_DIR')" >&2
  exit 2
fi
if ! printf 'op\tstatus\tpassed\tfailed\tskipped\tduration_s\trc\tstart_utc\tend_utc\tlog_file\n' \
        >> "$SUMMARY_TSV" 2>/dev/null; then
  log_error "failed to seed TSV header in '$SUMMARY_TSV'" >&2
  exit 2
fi

# tee_log: write a log line to stdout AND to the human summary file. Used so
# `tail -f SUMMARY_<date>.txt` mirrors the live console stream exactly.
tee_log_info()  { log_info  "$@" | tee -a "$SUMMARY_FILE"; }
tee_log_warn()  { log_warn  "$@" | tee -a "$SUMMARY_FILE" >&2; }
tee_log_error() { log_error "$@" | tee -a "$SUMMARY_FILE" >&2; }

# Common forwarded args (built once, applied to every op invocation)
COMMON_ARGS=()
[[ -n "$SEED"        ]] && COMMON_ARGS+=(--seed "$SEED")
[[ -n "$NUM_THREADS" ]] && COMMON_ARGS+=(--num_threads "$NUM_THREADS")
[[ -n "$LOWOHA"      ]] && COMMON_ARGS+=(--lowoha "$LOWOHA")
if (( ${#EXTRA_ARGS[@]} == 1 )) && [[ "${EXTRA_ARGS[0]}" == *[[:space:]]* ]]; then
  # Legacy convenience: a single --extra value containing spaces is word-split,
  # matching the prior single-string API. Pass --extra once per arg to disable
  # this and preserve quoting verbatim. We use `read -r -a` rather than an
  # unquoted expansion so the split tokens are NOT pathname-expanded against
  # the cwd -- crucial for filter patterns like `--gtest_filter=Matmul/*`,
  # which `+=( ${var} )` would otherwise glob against local files and corrupt.
  read -r -a _extra_split <<< "${EXTRA_ARGS[0]}"
  COMMON_ARGS+=("${_extra_split[@]}")
  unset _extra_split
else
  COMMON_ARGS+=("${EXTRA_ARGS[@]}")
fi

# ------------------------------------------------------------------------------
# Banner — structured INFO events. RUN_START is grep-friendly; the metadata
# fields below it document what was run.
# ------------------------------------------------------------------------------
RUN_START_UTC="$(_now_utc)"
RUN_START_EPOCH="$(date +%s)"
SCRIPT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

tee_log_info "RUN_START date=$DATE_STAMP host=$(hostname) script_sha=$SCRIPT_SHA"
tee_log_info "binary=$BINARY"
tee_log_info "run_dir=$RUN_DIR"
tee_log_info "ops_selected=$(IFS=,; echo "${SELECTED[*]}") ops_count=${#SELECTED[@]}"
tee_log_info "seed=${SEED:-time-based} num_threads=${NUM_THREADS:-gtest-default} lowoha=${LOWOHA:-unset} extra=${EXTRA_ARGS[*]:-none}"

# ------------------------------------------------------------------------------
# Run loop
#
# For each op:
#   * Emit STARTED event with cmd/filter/log_file fields.
#   * Prepend a self-describing metadata header into the per-op .txt log so
#     each archived log can be triaged without referring back to SUMMARY.
#   * Run the binary, appending its stdout+stderr to the per-op log.
#   * Parse gtest's [  PASSED  ] / [  FAILED  ] / [  SKIPPED ] lines.
#   * Emit FINISHED event with status + counts + duration.
#   * Append a TSV row to summary.tsv for machine ingestion.
# ------------------------------------------------------------------------------
overall_failed=0
overall_timed_out=0
ops_attempted=0  # ops that left dry-run mode and entered the run path,
                 # whether or not the binary actually executed. A per-op
                 # setup failure (e.g. cannot_write_log) still counts as
                 # attempted so the RESULT arithmetic balances:
                 #   ops_run = ops_passed + ops_failed + ops_timed_out
                 # Differs from ${#SELECTED[@]} in --dry-run (selected but
                 # not attempted) and in `continue`-on-setup-fail paths.
total_passed=0
total_failed=0
total_skipped=0

for op in "${SELECTED[@]}"; do
  filter="${OP_FILTER[$op]}"
  log="$RUN_DIR/${op}_${DATE_STAMP}.txt"
  log_basename="$(basename "$log")"

  # Word-split per-op extra flags (e.g. `--ai_test_mode postsub --lowoha false`)
  # via `read -r -a` rather than an unquoted array expansion: the unquoted
  # form would also perform pathname expansion against cwd, which would
  # corrupt any future OP_EXTRA value containing glob chars like '*' or '?'
  # (e.g. a `--gtest_filter=Foo/*` override). No OP_EXTRA value uses globs
  # today, but the safe pattern costs nothing.
  read -r -a op_extra <<< "${OP_EXTRA[$op]:-}"
  cmd=("$BINARY" "--gtest_filter=$filter" "${COMMON_ARGS[@]}" "${op_extra[@]}")
  cmd_str=$(printf '%q ' "${cmd[@]}")
  cmd_str=${cmd_str% }   # strip trailing space

  # Resolve the effective per-op timeout (seconds). Precedence:
  #   1. --timeout SEC on the command line wins for every op.
  #   2. OP_TIMEOUT[<op>] is the per-op override hardcoded above.
  #   3. TIMEOUT_DEFAULT covers everything else.
  # A value of 0 means "no timeout" — the binary runs unbounded.
  if [[ -n "$TIMEOUT_OVERRIDE" ]]; then
    op_timeout="$TIMEOUT_OVERRIDE"
  else
    op_timeout="${OP_TIMEOUT[$op]:-$TIMEOUT_DEFAULT}"
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    tee_log_info "[$op] DRY_RUN timeout_s=$op_timeout cmd=$cmd_str"
    continue
  fi

  op_start_utc="$(_now_utc)"
  op_start_epoch=$(date +%s)
  tee_log_info "[$op] STARTED filter=$filter timeout_s=$op_timeout log=$log_basename"

  # Record a per-op IO failure (truncation or header write) without
  # aborting the sweep. The "an op's failure can't poison the others"
  # invariant in the file header is more valuable than failing fast on a
  # mid-run disk-full / quota / permission change. Caller is expected to
  # `continue` after invoking this; we update both summary artifacts here
  # so the caller's `continue` site stays a single line.
  fail_op_setup() {
    local reason="$1"
    tee_log_error "[$op] FINISHED status=FAIL passed=0 failed=0 skipped=0 duration_s=0 rc=N/A reason=$reason log=$log"
    overall_failed=$(( overall_failed + 1 ))
    ops_attempted=$(( ops_attempted + 1 ))
    printf '%s\t%s\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s\n' \
        "$op" "FAIL" 0 0 0 0 "N/A" "$op_start_utc" "$(_now_utc)" "$log_basename" >> "$SUMMARY_TSV"
  }

  # Truncate the per-op log so a same-day re-run starts fresh. Without this,
  # `>>` would append a second header + gtest output on top of the previous
  # run's data and break the "header is at the top of the file" guarantee.
  # SUMMARY_<date>.txt and summary.tsv are truncated at run start above; this
  # gives every artifact the same overwrite-on-rerun semantics.
  if ! : > "$log" 2>/dev/null; then
    fail_op_setup cannot_write_log
    continue
  fi

  # Self-describing header at the top of the per-op log so the archived file
  # is independently triageable. Uses '#' so it won't confuse the gtest parser.
  # Same fail-soft policy as the truncation above: a mid-run IO failure
  # marks just this op as FAIL and the sweep moves on.
  if ! {
    echo "# ===== ZenDNN nightly gtest log ====="
    echo "# op           : $op"
    echo "# filter       : $filter"
    echo "# binary       : $BINARY"
    echo "# cmd          : $cmd_str"
    echo "# timeout_s    : $op_timeout"
    echo "# host         : $(hostname)"
    echo "# started_utc  : $op_start_utc"
    echo "# script_sha   : $SCRIPT_SHA"
    echo "# ===================================="
  } >> "$log" 2>/dev/null; then
    fail_op_setup cannot_write_log_header
    continue
  fi

  # Wrap the binary in `timeout` unless explicitly disabled (op_timeout=0).
  # --kill-after escalates to SIGKILL 30 s after the initial SIGTERM, in case
  # the process is stuck in an uninterruptible state. Exit codes of interest:
  #   124 -> SIGTERM fired (op exceeded its budget).
  #   137 -> SIGKILL fired (op also ignored SIGTERM, fell back to KILL).
  # Both map to status=TIMEOUT below.
  set +e
  if [[ "$op_timeout" -eq 0 ]]; then
    "${cmd[@]}" &>> "$log"
  else
    timeout --kill-after=30s "${op_timeout}s" "${cmd[@]}" &>> "$log"
  fi
  rc=$?
  set -e
  op_end_utc="$(_now_utc)"
  elapsed=$(( $(date +%s) - op_start_epoch ))

  # Extract pass/fail/skip from gtest summary lines. Use sed to pull the
  # integer directly (bracket-agnostic; awk field indexing breaks because gtest
  # prints '[  PASSED  ] N tests.' with brackets as separate whitespace tokens).
  # The `tests?` and `(\.|,)` lookahead accept both gtest variants:
  #   '[  PASSED  ] 1 test.'           (singular, count == 1)
  #   '[  PASSED  ] 100 tests.'        (plural)
  #   '[  FAILED  ] 1 test, listed below:'
  #   '[  FAILED  ] 9 tests, listed below:'
  # Each pipeline is guarded so a binary that crashes before printing a summary
  # doesn't abort the script under set -euo pipefail.
  passed=$( { grep -E "^\[  PASSED  \] [0-9]+ tests?(\.|,)" "$log" || true; } \
           | tail -1 | sed -E 's/.*\] ([0-9]+) tests?.*/\1/')
  failed=$( { grep -E "^\[  FAILED  \] [0-9]+ tests?(\.|,)" "$log" || true; } \
           | tail -1 | sed -E 's/.*\] ([0-9]+) tests?.*/\1/')
  skipped=$( { grep -cE "^\[  SKIPPED \]" "$log" || true; } )
  skipped=$(( ${skipped:-0} / 2 ))   # gtest logs each skip twice (inline + summary)
  passed=${passed:-0}; failed=${failed:-0}
  # Treat any non-numeric (e.g. empty / unparsed) value as 0.
  [[ "$passed"  =~ ^[0-9]+$ ]] || passed=0
  [[ "$failed"  =~ ^[0-9]+$ ]] || failed=0
  [[ "$skipped" =~ ^[0-9]+$ ]] || skipped=0

  if [[ "$rc" -eq 124 || "$rc" -eq 137 ]]; then
    status="TIMEOUT"
    overall_timed_out=$(( overall_timed_out + 1 ))
    tee_log_error "[$op] FINISHED status=$status passed=$passed failed=$failed skipped=$skipped duration_s=$elapsed rc=$rc timeout_s=$op_timeout"
  elif [[ "$rc" -ne 0 || "$failed" -gt 0 ]]; then
    status="FAIL"
    overall_failed=$(( overall_failed + 1 ))
    tee_log_error "[$op] FINISHED status=$status passed=$passed failed=$failed skipped=$skipped duration_s=$elapsed rc=$rc"
  else
    status="PASS"
    tee_log_info "[$op] FINISHED status=$status passed=$passed failed=$failed skipped=$skipped duration_s=$elapsed rc=$rc"
  fi

  ops_attempted=$(( ops_attempted + 1 ))
  total_passed=$((  total_passed  + passed  ))
  total_failed=$((  total_failed  + failed  ))
  total_skipped=$(( total_skipped + skipped ))

  # Machine-readable row for CI ingestion. Tab-separated to avoid quoting.
  printf '%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\n' \
      "$op" "$status" "$passed" "$failed" "$skipped" "$elapsed" "$rc" \
      "$op_start_utc" "$op_end_utc" "$log_basename" >> "$SUMMARY_TSV"
done

# ------------------------------------------------------------------------------
# Footer — final aggregate RESULT event. Single grep-friendly line:
#   ... INFO  RESULT status=PASS ops_passed=12 ops_failed=0 total_passed=... duration_s=...
# CI dashboards can scrape this one line; humans get the same info in context.
# ------------------------------------------------------------------------------
overall_duration=$(( $(date +%s) - RUN_START_EPOCH ))
overall_bad=$(( overall_failed + overall_timed_out ))
# ops_passed is derived from ops_attempted (not ${#SELECTED[@]}) so --dry-run
# honestly reports ops_run=0 / ops_passed=0 instead of pretending the
# selected ops all passed when they were never actually launched. The
# invariant `ops_run = ops_passed + ops_failed + ops_timed_out` holds even
# when a per-op setup failure (cannot_write_log) is counted as both
# attempted and failed.
ops_passed=$(( ops_attempted - overall_bad ))

if [[ "$DRY_RUN" -eq 1 ]]; then
  result_status="DRY_RUN"
  result_logger="tee_log_info"
elif [[ "$overall_bad" -eq 0 ]]; then
  result_status="PASS"
  result_logger="tee_log_info"
else
  result_status="FAIL"
  result_logger="tee_log_error"
fi

# ops_planned is exposed alongside ops_run so dry-run consumers can still
# see how many ops *would* have executed; in non-dry-run mode the two are
# always equal and downstream tooling can ignore one of them.
$result_logger \
  "RESULT status=$result_status ops_run=$ops_attempted ops_planned=${#SELECTED[@]} ops_passed=$ops_passed ops_failed=$overall_failed ops_timed_out=$overall_timed_out total_passed=$total_passed total_failed=$total_failed total_skipped=$total_skipped duration_s=$overall_duration"
tee_log_info "RUN_END date=$DATE_STAMP run_dir=$RUN_DIR"

[[ "$overall_bad" -eq 0 ]] || exit 1
exit 0
