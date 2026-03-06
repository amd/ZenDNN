#!/bin/bash
set -euo pipefail

# ===========================================================================
# AI Matmul Benchmark Runner
#
# Usage:
#   ./run_matmul_benchmark_sweep.sh [options] algo [algo ...]
#
# Options:
#   -i, --input <file|shortcut>   Input file or shortcut (default: bf16)
#   -t, --threads <N>             Number of OMP threads (default: all cores)
#   -o, --outdir <dir>            Output directory (default: build/)
#   -p, --perf                    Run each shape under perf stat to collect
#                                 per-shape L2/L3 HW counters. Requires sudo.
#                                 Outputs a separate _perf_raw.txt per algo.
#   -h, --help                    Show this help
#
# Input shortcuts:
#   bf16             -> benchmark_sweep/bf16_generative_models_eval.txt
#   fp32             -> benchmark_sweep/fp32_generative_models_eval.txt
#   bf16_pytorch     -> benchmark_sweep/bf16_pytorch_models_eval.txt
#   fp32_pytorch     -> benchmark_sweep/fp32_pytorch_models_eval.txt
#   <path>           -> custom input file
#
# Algo values (positional, after options):
#   Pass one or more algo numbers (e.g. 1, 3, 10, 11).
#   At least one algo must be specified.
#
# Examples:
#   ./run_matmul_benchmark_sweep.sh -t 64 -i bf16 1 11       # 64 threads, ALGO 1 & 11
#   ./run_matmul_benchmark_sweep.sh -i fp32 -t 1 10          # fp32, single thread, ALGO 10
#   sudo ./run_matmul_benchmark_sweep.sh -t 1 -p -i bf16 1   # timing + HW counters
#   ./run_matmul_benchmark_sweep.sh -i /tmp/shapes.txt 11    # custom file, ALGO 11
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SWEEP_DIR="$REPO_ROOT/benchdnn/input/matmul/benchmark_sweep"

INPUT_ARG="bf16"
NUM_THREADS=""
OUTDIR="$REPO_ROOT/build"
PERF_MODE=0

show_help() {
    cat <<'HELP'
AI Matmul Benchmark Runner

Usage:
  ./run_matmul_benchmark_sweep.sh [options] algo [algo ...]

Options:
  -i, --input <file|shortcut>   Input file or shortcut (default: bf16)
  -t, --threads <N>             Number of OMP threads (default: all cores)
  -o, --outdir <dir>            Output directory (default: build/)
  -p, --perf                    Run each shape under perf stat to collect
                                per-shape L2/L3 HW counters. Requires sudo.
                                Outputs a separate _perf_raw.txt per algo.
  -h, --help                    Show this help

Input shortcuts:
  bf16             -> benchmark_sweep/bf16_generative_models_eval.txt
  fp32             -> benchmark_sweep/fp32_generative_models_eval.txt
  bf16_pytorch     -> benchmark_sweep/bf16_pytorch_models_eval.txt
  fp32_pytorch     -> benchmark_sweep/fp32_pytorch_models_eval.txt
  <path>           -> custom input file

Algo values (positional, after options):
  Pass one or more algo numbers (e.g. 1, 3, 10, 11).
  At least one algo must be specified.

Examples:
  ./run_matmul_benchmark_sweep.sh -t 64 -i bf16 1 11       # 64 threads, ALGO 1 & 11
  ./run_matmul_benchmark_sweep.sh -i fp32 -t 1 10          # fp32, single thread, ALGO 10
  sudo ./run_matmul_benchmark_sweep.sh -t 1 -p -i bf16 1   # timing + HW counters
  ./run_matmul_benchmark_sweep.sh -i /tmp/shapes.txt 11    # custom file, ALGO 11
HELP
    exit 0
}

# --- Parse options ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)   INPUT_ARG="$2"; shift 2 ;;
        -t|--threads) NUM_THREADS="$2"; shift 2 ;;
        -o|--outdir)  OUTDIR="$2"; shift 2 ;;
        -p|--perf)    PERF_MODE=1; shift ;;
        -h|--help)    show_help ;;
        -*)           echo "Unknown option: $1"; show_help ;;
        *)            break ;;
    esac
done

# --- Resolve input file ---
case "$INPUT_ARG" in
    bf16|BF16)                     INPUT_FILE="$SWEEP_DIR/bf16_generative_models_eval.txt"; TAG="bf16" ;;
    fp32|FP32)                     INPUT_FILE="$SWEEP_DIR/fp32_generative_models_eval.txt"; TAG="fp32" ;;
    bf16_pytorch|BF16_PYTORCH)     INPUT_FILE="$SWEEP_DIR/bf16_pytorch_models_eval.txt";    TAG="bf16_pytorch" ;;
    fp32_pytorch|FP32_PYTORCH)     INPUT_FILE="$SWEEP_DIR/fp32_pytorch_models_eval.txt";    TAG="fp32_pytorch" ;;
    *)                             INPUT_FILE="$INPUT_ARG"; TAG="$(basename "${INPUT_FILE%.*}")" ;;
esac

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input file not found: $INPUT_FILE"; exit 1
fi

# --- Resolve threads ---
export OMP_NUM_THREADS="${NUM_THREADS:-$(nproc)}"

# --- Resolve algos (remaining positional args) ---
ALGOS=("$@")
if [ ${#ALGOS[@]} -eq 0 ]; then
    echo "ERROR: at least one algo number is required (e.g. 1, 3, 10, 11)"; show_help
fi

# --- Locate benchdnn ---
BENCHDNN_BIN="$REPO_ROOT/build/benchdnn/benchdnn"
if [ ! -f "$BENCHDNN_BIN" ]; then
    BENCHDNN_BIN="$REPO_ROOT/build/install/benchdnn/bin/benchdnn"
fi
if [ ! -f "$BENCHDNN_BIN" ]; then
    echo "ERROR: benchdnn not found in build/ or build/install/"; exit 1
fi

# --- Optional library preloads ---
: "${JEMALLOC_LIB:=/usr/local/lib/libjemalloc.so}"
: "${IOMP_LIB:=/opt/intel/oneapi/compiler/2025.0/lib/libiomp5.so}"
PRELOAD_VALUE="${LD_PRELOAD-}"
[ -f "$IOMP_LIB" ]    && PRELOAD_VALUE="${IOMP_LIB}${PRELOAD_VALUE:+:${PRELOAD_VALUE}}"
[ -f "$JEMALLOC_LIB" ] && PRELOAD_VALUE="${JEMALLOC_LIB}${PRELOAD_VALUE:+:${PRELOAD_VALUE}}"
[ -n "$PRELOAD_VALUE" ] && export LD_PRELOAD="$PRELOAD_VALUE"

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

# --- CPU binding ---
CPU_BIND="0-$((OMP_NUM_THREADS - 1))"

# --- AMD Zen 5 PMU events for L2/L3 cache analysis (via perf stat counting mode) ---
# perf stat gives exact aggregate counters per run (not statistical samples).
# AMD raw event format: rUUEE (UU=unit_mask, EE=event_select)
#
# PMCx064 L2CacheReqStat — "Core to L2 Cacheable Request Access Status"
#   NOTE: Does NOT include L2 Prefetcher requests (those are PMCx070-072).
#   Bit 0: LsRdBlkCS    DC Shared Read HIT     ┐
#   Bit 1: LsRdBlkLHitX DC Read HIT Modifiable │ umask 0x0F = all DC hits
#   Bit 2: LsRdBlkLHitS DC Read HIT Non-Mod    │
#   Bit 3: LsRdBlkX     DC Store HIT           ┘
#   Bit 4: LsRdBlkC     DC Req MISS            ← umask 0x10 = DC miss
#   Bit 5: IcFillHitX   IC HIT Modifiable      ┐
#   Bit 6: IcFillHitS   IC HIT Non-Modifiable  │ (IC events, not tracked)
#   Bit 7: IcFillMiss   IC Req MISS            ┘
#
# PMCx070-072 — L2 Prefetcher events (separate from demand requests)
#   rFF70 = L2PfHitL2       PF accepted by L2, data found in L2
#   rFF71 = L2PfMissL2HitL3 PF accepted by L2, missed L2, hit L3
#   rFF72 = L2PfMissL2L3    PF accepted by L2, missed both L2 & L3 → DRAM
#
PERF_EVENTS="L1-dcache-loads,L1-dcache-load-misses,rFF70,rFF71,rFF72,r0F64,r1064"

mkdir -p "$OUTDIR"

echo "================================================================"
echo "  AI Matmul Benchmark"
echo "  Input   : $INPUT_FILE"
echo "  Algos   : ${ALGOS[*]}"
echo "  Threads : $OMP_NUM_THREADS"
echo "  CPU bind: $CPU_BIND"
echo "  HW Perf : $([ $PERF_MODE -eq 1 ] && echo 'ON (-p)' || echo 'OFF')"
echo "  Output  : $OUTDIR/"
echo "================================================================"
echo ""

for algo in "${ALGOS[@]}"; do
    OUTFILE="$OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c.txt"

    if [[ $PERF_MODE -eq 1 ]]; then
        # --- Per-shape perf stat (counting mode) ---
        # Runs each shape individually under "perf stat" to get exact
        # L2/L3 hardware counter totals. Requires sudo.
        PERF_RAW="$OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c_perf_raw.txt"
        echo "--- ALGO=$algo (per-shape perf stat counting mode) ---"
        > "$PERF_RAW"

        total=$(grep -c '[^[:space:]]' "$INPUT_FILE" || echo 0)
        idx=0

        while IFS= read -r line || [[ -n "$line" ]]; do
            [[ -z "${line// /}" ]] && continue
            idx=$((idx + 1))

            echo "$line" > /tmp/_benchdnn_single.txt

            echo "=== SHAPE $idx/$total ===" >> "$PERF_RAW"
            echo "INPUT: $line" >> "$PERF_RAW"

            perf stat -e "$PERF_EVENTS" -- \
                env OMP_NUM_THREADS="$OMP_NUM_THREADS" ZENDNNL_MATMUL_ALGO="$algo" \
                numactl --physcpubind="$CPU_BIND" \
                "$BENCHDNN_BIN" --op=matmul --lowoha=true \
                --input_file=/tmp/_benchdnn_single.txt \
                >> "$PERF_RAW" 2>&1

            echo "" >> "$PERF_RAW"

            if (( idx % 10 == 0 )) || (( idx == 1 )); then
                echo "  [$idx/$total] done"
            fi
        done < "$INPUT_FILE"

        echo "--- ALGO=$algo perf counters → $PERF_RAW ---"
    else
        # --- Normal mode: all shapes in one benchdnn run ---
        echo "--- ALGO=$algo ---"

        ZENDNNL_MATMUL_ALGO=$algo \
        numactl --physcpubind="$CPU_BIND" \
            "$BENCHDNN_BIN" --op=matmul --lowoha=true --input_file="$INPUT_FILE" \
            2>&1 | tee "$OUTFILE"

        echo "--- ALGO=$algo done → $OUTFILE ---"
    fi

    echo ""
done

echo "================================================================"
echo "Results:"
for algo in "${ALGOS[@]}"; do
    if [[ $PERF_MODE -eq 1 ]]; then
        echo "  ALGO=$algo perf raw: $OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c_perf_raw.txt"
    else
        echo "  ALGO=$algo timing:   $OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c.txt"
    fi
done
echo ""
echo "To analyze:"
if [[ $PERF_MODE -eq 1 ]]; then
    echo "  python3 scripts/analyze_benchmark.py --perf -t $OMP_NUM_THREADS <perf_raw_file>"
    echo "  python3 scripts/analyze_benchmark.py --perf -t $OMP_NUM_THREADS -v -b <perf_raw_file>"
else
    echo "  python3 scripts/analyze_benchmark.py <timing_file>"
fi
echo "================================================================"
