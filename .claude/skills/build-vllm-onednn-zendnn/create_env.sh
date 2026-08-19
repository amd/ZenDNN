#!/usr/bin/env bash
# Create/reuse the conda build env for the vLLM + oneDNN + ZenDNN (zen64) build.
#
# Layout assumption: this script lives in REPO_ROOT, the PARENT of the vLLM
# checkout (i.e. the vLLM repo is at ${REPO_ROOT}/vllm). Rust state is placed
# under ${REPO_ROOT}/.rust to stay off the home quota.
#
# Usage:
#   bash create_env.sh              # build-only deps
#   bash create_env.sh --runtime    # + runtime deps + CPU torch/vision/audio
set -euo pipefail

ENV_NAME="${ENV_NAME:-vllm-build-env-312}"
PY_VERSION="3.12"
RUST_CHANNEL="1.95"

WITH_RUNTIME=0
[ "${1:-}" = "--runtime" ] && WITH_RUNTIME=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="${REPO_ROOT}/vllm"
BUILD_REQS="${VLLM_DIR}/requirements/build/cpu.txt"
RUNTIME_REQS="${VLLM_DIR}/requirements/cpu.txt"

# Only require the runtime requirements file when --runtime was asked for, so a
# build-only run does not break on upstream vLLM moving requirements/cpu.txt.
REQS_TO_CHECK=("${BUILD_REQS}")
[ "${WITH_RUNTIME}" -eq 1 ] && REQS_TO_CHECK+=("${RUNTIME_REQS}")

for f in "${REQS_TO_CHECK[@]}"; do
    [ -f "$f" ] || { echo "ERROR: requirements file not found: $f" >&2; exit 1; }
done

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "    env '${ENV_NAME}' already exists, reusing it"
else
    conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
fi

conda install -n "${ENV_NAME}" -c conda-forge \
    "gxx_linux-64>=12.3,<13" "gcc_linux-64>=12.3,<13" libnuma numactl -y

set +u; conda activate "${ENV_NAME}"; set -u

# Keep rust off the home quota.
export RUSTUP_HOME="${RUSTUP_HOME:-${REPO_ROOT}/.rust/rustup}"
export CARGO_HOME="${CARGO_HOME:-${REPO_ROOT}/.rust/cargo}"
mkdir -p "${RUSTUP_HOME}" "${CARGO_HOME}"

if command -v cargo >/dev/null 2>&1; then
    echo "    found existing cargo: $(cargo --version)"
else
    if ! command -v rustup >/dev/null 2>&1; then
        # Download the installer to a file first so it can be inspected/audited
        # rather than piping remote content straight into a shell.
        RUSTUP_INIT="$(mktemp -t rustup-init-XXXXXX.sh)"
        trap 'rm -f "${RUSTUP_INIT}"' EXIT
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o "${RUSTUP_INIT}"
        sh "${RUSTUP_INIT}" -y --profile minimal --no-modify-path
        rm -f "${RUSTUP_INIT}"
        trap - EXIT
    fi
    source "${CARGO_HOME}/env"
    rustup toolchain install "${RUST_CHANNEL}" --profile minimal
fi

pip install -r "${BUILD_REQS}"

if [ "${WITH_RUNTIME}" -eq 1 ]; then
    pip install -r "${RUNTIME_REQS}"
    # CPU index only, or CUDA builds slip in.
    pip install --force-reinstall --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.11.0 torchvision torchaudio
fi

echo "Done. conda activate ${ENV_NAME} ; bash build_wheel.sh"
