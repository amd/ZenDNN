#!/bin/bash
# Build the vLLM (CPU) wheel with the oneDNN + ZenDNN (zen64) backend.
#
# Layout assumption: this script lives in REPO_ROOT, the PARENT of the vLLM
# checkout (i.e. the vLLM repo is at ${REPO_ROOT}/vllm).
#
# ZenDNN is enabled ONLY when BOTH ZENDNN_DIR and FETCHCONTENT_SOURCE_DIR_ONEDNN
# are set to real paths. Override them via env, or edit the two lines below.
set -euo pipefail

export VLLM_TARGET_DEVICE=cpu
export MAX_JOBS="${MAX_JOBS:-32}"

# --- Set these (via env or by editing here) -------------------------------
# ZENDNN_DIR: either the ZenDNN repo dir (whose install tree is at
#   build/install/zendnnl) or the install tree itself (ZENDBUILDPATH). The
#   bundled patch detects which one you passed; if neither resolves to a real
#   zendnnl install tree, cmake fails with a FATAL_ERROR rather than quietly
#   building without ZenDNN. Build ZenDNN before running this.
export ZENDNN_DIR="${ZENDNN_DIR:-<SET_ZENDNN_DIR>}"
# FETCHCONTENT_SOURCE_DIR_ONEDNN: the upstream oneDNN source dir (uxlfoundation/
#   oneDNN) on a ref that has the zen64 backend (#5511) — main, or a tag known to
#   contain it. vLLM compiles it from source.
export FETCHCONTENT_SOURCE_DIR_ONEDNN="${FETCHCONTENT_SOURCE_DIR_ONEDNN:-<SET_ONEDNN_DIR>}"
# ---------------------------------------------------------------------------

# Guard: refuse to build with unset placeholders (would silently drop ZenDNN).
for v in ZENDNN_DIR FETCHCONTENT_SOURCE_DIR_ONEDNN; do
    val="${!v}"
    case "$val" in
        "<SET_"*|"")
            echo "ERROR: $v is not set (still '$val')." >&2
            echo "       ZenDNN is enabled ONLY when BOTH ZENDNN_DIR and" >&2
            echo "       FETCHCONTENT_SOURCE_DIR_ONEDNN point at real paths." >&2
            echo "       Export them or edit build_wheel.sh, then re-run." >&2
            exit 1
            ;;
    esac
    [ -d "$val" ] || { echo "ERROR: $v='$val' is not a directory." >&2; exit 1; }
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}/vllm"

# Keep rust off the home quota (must match create_env.sh).
export RUSTUP_HOME="${RUSTUP_HOME:-${REPO_ROOT}/.rust/rustup}"
export CARGO_HOME="${CARGO_HOME:-${REPO_ROOT}/.rust/cargo}"
if [ -f "${CARGO_HOME}/env" ]; then source "${CARGO_HOME}/env"; fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: cargo not found. Run create_env.sh first." >&2
    exit 1
fi

# gcc/g++ >= 12.3: fall back to the conda compilers if CC/CXX are unset.
if [ -z "${CC:-}" ] && [ -n "${CONDA_PREFIX:-}" ] \
   && [ -x "${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc" ]; then
    export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
    export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
fi

# rm -rf build/ .deps/   # uncomment for a clean rebuild against custom oneDNN+ZenDNN

pip wheel . --no-deps --no-build-isolation -w dist/
echo "Wheel written to: $(ls -t dist/*.whl | head -1)"
