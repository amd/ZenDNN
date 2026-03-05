#!/bin/bash
set -euo pipefail

# ===========================================================================
# AI Matmul Benchmark
#
# Usage:
#   ./run_ai_matmul_benchmark.sh <input> [algo1 algo2 ...]
#
# Input shortcuts:
#   bf16             -> benchmark_sweep/bf16_generative_models_eval.txt
#   fp32             -> benchmark_sweep/fp32_generative_models_eval.txt
#   bf16_pytorch     -> benchmark_sweep/bf16_pytorch_models_eval.txt
#   fp32_pytorch     -> benchmark_sweep/fp32_pytorch_models_eval.txt
#   /path/to/file    -> custom input file
#
# Algo values: 1 (aocl_dlp_blocked), 10 (ai_gemm), 11 (ai_brgemm)
# If no algo specified, runs all three (1, 10, 11).
#
# Examples:
#   ./run_ai_matmul_benchmark.sh bf16                  # all 3 algos
#   ./run_ai_matmul_benchmark.sh bf16 1 10             # ALGO=1 and ALGO=10
#   ./run_ai_matmul_benchmark.sh fp32 10               # only ALGO=10
#   ./run_ai_matmul_benchmark.sh bf16_pytorch 1 11     # ALGO=1 and ALGO=11
#   ./run_ai_matmul_benchmark.sh /tmp/my_shapes.txt 10 # custom file, ALGO=10
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SWEEP_DIR="$REPO_ROOT/benchdnn/input/matmul/benchmark_sweep"

BENCHDNN_BIN="$REPO_ROOT/build/install/benchdnn/bin/benchdnn"
if [ ! -f "$BENCHDNN_BIN" ]; then
    echo "ERROR: benchdnn binary not found at $BENCHDNN_BIN"; exit 1
fi

# --- Resolve input file ---
ARG="${1:-bf16}"
shift || true

case "$ARG" in
    bf16|BF16)             INPUT_FILE="$SWEEP_DIR/bf16_generative_models_eval.txt"; TAG="bf16" ;;
    fp32|FP32)             INPUT_FILE="$SWEEP_DIR/fp32_generative_models_eval.txt"; TAG="fp32" ;;
    bf16_pytorch|BF16_PYTORCH) INPUT_FILE="$SWEEP_DIR/bf16_pytorch_models_eval.txt"; TAG="bf16_pytorch" ;;
    fp32_pytorch|FP32_PYTORCH) INPUT_FILE="$SWEEP_DIR/fp32_pytorch_models_eval.txt"; TAG="fp32_pytorch" ;;
    *)                     INPUT_FILE="$ARG"; TAG="custom" ;;
esac

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input file not found at $INPUT_FILE"; exit 1
fi

# --- Resolve algos ---
ALGOS=("$@")
if [ ${#ALGOS[@]} -eq 0 ]; then
    ALGOS=(1 10 11)
fi

algo_label() {
    case "$1" in
        1)  echo "ALGO=1 (aocl_dlp_blocked)" ;;
        10) echo "ALGO=10 (ai_gemm)" ;;
        11) echo "ALGO=11 (ai_brgemm)" ;;
        *)  echo "ALGO=$1" ;;
    esac
}

# --- Environment ---
: "${JEMALLOC_LIB:=/usr/local/lib/libjemalloc.so}"
: "${IOMP_LIB:=/opt/intel/oneapi/compiler/2025.0/lib/libiomp5.so}"
PRELOAD_VALUE="${LD_PRELOAD-}"
if [ -f "$IOMP_LIB" ]; then
    PRELOAD_VALUE="${IOMP_LIB}${PRELOAD_VALUE:+:${PRELOAD_VALUE}}"
fi
if [ -f "$JEMALLOC_LIB" ]; then
    PRELOAD_VALUE="${JEMALLOC_LIB}${PRELOAD_VALUE:+:${PRELOAD_VALUE}}"
fi
if [ -n "$PRELOAD_VALUE" ]; then
    export LD_PRELOAD="$PRELOAD_VALUE"
fi
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
: "${KMP_AFFINITY:=granularity=fine,compact,1,0}"
export KMP_AFFINITY
: "${KMP_BLOCKTIME:=1}"
export KMP_BLOCKTIME
: "${OMP_NUM_THREADS:=$(nproc)}"
export OMP_NUM_THREADS

OUTDIR="$REPO_ROOT/build"

echo "================================================================"
echo "  AI Matmul Benchmark"
echo "  Input : $INPUT_FILE"
echo "  Algos : ${ALGOS[*]}"
echo "  Threads: $OMP_NUM_THREADS"
echo "================================================================"
echo ""

for algo in "${ALGOS[@]}"; do
    LABEL=$(algo_label "$algo")
    OUTFILE="$OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c.txt"

    echo "================================================================"
    echo "  Running $LABEL"
    echo "  Output : $OUTFILE"
    echo "================================================================"
    echo ""

    export ZENDNNL_MATMUL_ALGO=$algo
    CPULIST="0-$((OMP_NUM_THREADS - 1))"
    numactl --physcpubind="$CPULIST" \
        "$BENCHDNN_BIN" --op=matmul --lowoha=true --input_file="$INPUT_FILE" \
        2>&1 | tee "$OUTFILE"

    echo ""
    echo "=== $LABEL DONE ==="
    echo ""
done

echo "================================================================"
echo "Results:"
for algo in "${ALGOS[@]}"; do
    echo "  $(algo_label "$algo") : $OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c.txt"
done
echo "================================================================"
