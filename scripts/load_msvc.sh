#!/usr/bin/env bash
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
# Load the MSVC (x64) build environment into the CURRENT Git Bash session so
# that cl / rc / link, and the INCLUDE, LIB and PATH variables, are all set for
# a native Windows ZenDNN build.
#
# A batch file can only set variables in the process that runs it, so it cannot
# modify a Git Bash session directly. This script therefore runs the batch
# environment in a child `cmd`, dumps the resulting variables with `set`, and
# imports them into the current shell (converting PATH to POSIX form).
#
# USAGE (must be sourced, not executed, so the variables persist):
#
#     source scripts/load_msvc.sh
#     command -v cl                 # -> .../VC/Tools/MSVC/.../cl
#     echo "SDK=[$WindowsSDKVersion]"
#
# CONFIGURATION (all optional; override by exporting before sourcing):
#
#     VS_PATH       Visual Studio (or Build Tools) install root. If unset, the
#                   script auto-discovers it via vswhere.exe (shipped by every
#                   VS 2017+ / Build Tools installer at a fixed location), then
#                   falls back to the standard Program Files install paths for
#                   VS 2022/2019. Set this only to force a specific install.
#     VSDEV_ARCH    Architecture passed to vcvarsall.bat (default: x64).
#     MSVCENV_BAT   Optional wrapper .bat that sets up MSVC and injects a Windows
#                   SDK that vcvars cannot auto-discover (for hosts with a
#                   damaged/undiscoverable SDK). Unset by default; when it points
#                   at an existing file it is used instead of vcvarsall.
# *******************************************************************************

# --- must be sourced ---------------------------------------------------------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "error: this script must be sourced so the environment persists:" >&2
  echo "       source scripts/load_msvc.sh" >&2
  exit 1
fi

# Locate a Visual Studio / Build Tools install that provides the C++ x64
# toolset. Precedence: explicit VS_PATH -> vswhere.exe -> well-known roots.
# Prints a POSIX path on success; returns non-zero if nothing is found.
zl_discover_vs_path() {
  if [ -n "${VS_PATH:-}" ]; then
    printf '%s\n' "${VS_PATH}"
    return 0
  fi

  # vswhere is authoritative and independent of the install location; the VS /
  # Build Tools installer always drops it here, even on 64-bit Windows.
  local vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
  if [ -x "${vswhere}" ]; then
    local found
    found="$("${vswhere}" -latest -products '*' \
              -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
              -property installationPath 2>/dev/null | tr -d '\r' | head -n1)"
    if [ -n "${found}" ]; then
      # vswhere prints a Windows path; hand back POSIX for the caller's tests.
      cygpath -u "${found}" 2>/dev/null || printf '%s\n' "${found}"
      return 0
    fi
  fi

  # Fallback: probe the standard install roots (edition- and year-agnostic).
  local base ed
  for base in "/c/Program Files/Microsoft Visual Studio/2022" \
              "/c/Program Files (x86)/Microsoft Visual Studio/2022" \
              "/c/Program Files/Microsoft Visual Studio/2019" \
              "/c/Program Files (x86)/Microsoft Visual Studio/2019"; do
    for ed in Enterprise Professional Community BuildTools Preview; do
      if [ -d "${base}/${ed}" ]; then
        printf '%s\n' "${base}/${ed}"
        return 0
      fi
    done
  done

  return 1
}

zl_load_msvc() {
  local vsdev_arch="${VSDEV_ARCH:-x64}"
  local msvcenv_bat="${MSVCENV_BAT:-}"
  local vs_path setup_cmd bat_win line name

  if [ -n "${msvcenv_bat}" ] && [ -f "${msvcenv_bat}" ]; then
    # Host with a damaged/undiscoverable SDK: use the user-provided wrapper
    # that injects the SDK paths vcvars cannot find on its own.
    # Short 8.3 path + no quotes: MSYS mangles embedded quotes when launching
    # native cmd, and a short path carries no spaces so none are needed.
    bat_win="$(cygpath -w -s "${msvcenv_bat}")"
    setup_cmd="call ${bat_win}"
    echo "load_msvc: using wrapper ${msvcenv_bat}"
  else
    # Healthy host: auto-discover VS and let stock vcvarsall set up the SDK.
    vs_path="$(zl_discover_vs_path)"
    if [ -z "${vs_path}" ]; then
      echo "load_msvc: ERROR - no Visual Studio C++ toolset found." >&2
      echo "           Install VS 2022 or Build Tools with 'Desktop development" >&2
      echo "           with C++', or export VS_PATH=/c/path/to/VS before sourcing." >&2
      return 1
    fi
    local vcvars="${vs_path}/VC/Auxiliary/Build/vcvarsall.bat"
    if [ ! -f "${vcvars}" ]; then
      echo "load_msvc: ERROR - vcvarsall.bat not found under ${vs_path}" >&2
      echo "           (expected ${vcvars}). Set VS_PATH to a valid VS install." >&2
      return 1
    fi
    # Short 8.3 path + no quotes (see wrapper branch): survives MSYS->cmd and
    # handles standard VS paths under "Program Files" (spaces) alike.
    bat_win="$(cygpath -w -s "${vcvars}")"
    setup_cmd="call ${bat_win} ${vsdev_arch}"
    echo "load_msvc: using ${vcvars} ${vsdev_arch}"
  fi

  # Resolve Windows cmd.exe by absolute path rather than by PATH lookup. Under a
  # stripped or reordered PATH (e.g. a fresh tmux session) a bare `cmd` can
  # resolve to a non-Windows `cmd` (such as /usr/bin/cmd) and the MSVC setup then
  # silently fails ("cl not found after setup"). COMSPEC is the authoritative
  # cmd.exe location; fall back to the standard System32 path, then to `cmd`.
  local win_cmd="cmd" comspec_u=""
  [ -n "${COMSPEC:-}" ] && comspec_u="$(cygpath -u "${COMSPEC}" 2>/dev/null)"
  if [ -n "${comspec_u}" ] && [ -x "${comspec_u}" ]; then
    win_cmd="${comspec_u}"
  elif [ -x "/c/Windows/System32/cmd.exe" ]; then
    win_cmd="/c/Windows/System32/cmd.exe"
  fi

  while IFS= read -r line; do
    # `cmd` emits CRLF line endings; strip the trailing CR so it does not leak
    # into every imported value (which corrupts PATH/INCLUDE/LIB and the shell).
    line="${line%$'\r'}"
    case "${line}" in
      PATH=*)
        # Windows ;-separated PATH -> POSIX :-separated so bash finds the tools.
        export PATH="$(cygpath -u -p "${line#PATH=}")" ;;
      *=*)
        name="${line%%=*}"
        # Skip names that are not valid shell identifiers (e.g. ProgramFiles(x86)).
        case "${name}" in
          *[!A-Za-z0-9_]*) ;;
          *) export "${name}=${line#*=}" ;;
        esac ;;
    esac
  done < <("${win_cmd}" //c "${setup_cmd} >nul 2>&1 && set")

  if ! command -v cl >/dev/null 2>&1; then
    echo "load_msvc: ERROR - the MSVC C++ compiler 'cl' is not available after loading" >&2
    echo "  the build environment. Visual Studio's C++ toolset or a Windows SDK is likely" >&2
    echo "  missing or not discoverable on this machine." >&2
    echo "  Fix: install Visual Studio 2022 or the Build Tools with the 'Desktop" >&2
    echo "  development with C++' workload, then re-run:  source scripts/load_msvc.sh" >&2
    echo "  Advanced: set VS_PATH=<VS root> or MSVCENV_BAT=<SDK setup .bat> to override." >&2
    return 1
  fi
  echo "load_msvc: cl  = $(command -v cl)"
  echo "load_msvc: SDK = [${WindowsSDKVersion}]"
  case "${INCLUDE}" in
    *ucrt*) : ;;
    *) echo "load_msvc: WARNING - INCLUDE has no UCRT path; SDK may be missing" >&2 ;;
  esac
}

zl_load_msvc
