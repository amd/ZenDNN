#!/bin/bash
set -euo pipefail

# ===========================================================================
# Matmul / BMM / Group Matmul Benchmark Runner
#
# Usage:
#   ./run_matmul_benchmark_sweep.sh [options]
#
# Options:
#   --op <matmul|bmm|grp_matmul>  Operator (default: matmul)
#   -a, --algo <N>[,N,...]        Algo number(s) to benchmark (required for matmul/bmm)
#                                 Comma-separated or repeated: -a 1,11 or -a 1 -a 11
#                                   1  = AOCL DLP Blocked
#                                   3  = OneDNN BRGEMM
#                                   4  = AOCL DLP
#                                   5  = OneDNN
#                                   6  = LibxSMM
#                                   10 = Native GEMM
#                                   11 = Native BRGEMM
#   -v, --ver <N>[,N,...]         Group matmul strategy version(s) (for grp_matmul)
#                                   0  = Auto (selects V1, V2, or V3 based on shape)
#                                   1  = Sequential (experts serial, all threads per GEMM)
#                                   2  = Flat CCD adaptive tile (hybrid M/N, framework-safe)
#                                   3  = Flat CCD N-tile (no nested OMP, framework-safe)
#                                   4  = Multilevel CCD-aware (nested OMP)
#                                   5  = Per-expert (1 thread per expert, parallel-for)
#   -i, --input <file|shortcut>   Input file or shortcut (default: bf16)
#   -t, --threads <N>             Number of OMP threads (default: all cores)
#   -o, --outdir <dir>            Output directory (default: build/)
#   -p, --perf [profile]          External perf stat (matmul/bmm only)
#   -P, --perf-internal [profile] Internal perf counters (matmul/bmm only)
#   -h, --help                    Show this help
#
# Input shortcuts (matmul):
#   bf16             -> benchmark_sweep/bf16_generative_models_eval.txt
#   fp32             -> benchmark_sweep/fp32_generative_models_eval.txt
#   bf16_pytorch     -> benchmark_sweep/bf16_pytorch_models_eval.txt
#   fp32_pytorch     -> benchmark_sweep/fp32_pytorch_models_eval.txt
#
# Input shortcuts (bmm):
#   sdpa             -> input/bmm/sdpa_bmm_inputs.txt
#   pytorch          -> input/bmm/pytorch_bmm_inputs.txt
#
# Input shortcuts (grp_matmul):
#   uniform          -> input/grp_matmul/moe_uniform.txt
#   prompt           -> input/grp_matmul/grp_matmul_prompt.txt
#   decode           -> input/grp_matmul/grp_matmul_decode.txt
#
# Examples:
#   ./run_matmul_benchmark_sweep.sh -a 1,11 -i bf16 -t 128
#   ./run_matmul_benchmark_sweep.sh --op grp_matmul -v 0,1,2,3 -i uniform -t 128
#   ./run_matmul_benchmark_sweep.sh --op grp_matmul -v 1,2,3 -i prompt -t 128
#   ./run_matmul_benchmark_sweep.sh --op bmm -a 4,5,6 -i sdpa -t 128
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SWEEP_DIR="$REPO_ROOT/benchdnn/input/matmul/benchmark_sweep"
BMM_DIR="$REPO_ROOT/benchdnn/input/bmm"
GRP_DIR="$REPO_ROOT/benchdnn/input/grp_matmul"

OP="matmul"
INPUT_ARG="bf16"
NUM_THREADS=""
OUTDIR="$REPO_ROOT/build"
PERF_MODE=0
PERF_PROFILE="cache"
ALGOS=()
VERS=()

show_help() {
    sed -n '3,/^# ====/p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# --- Parse options ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --op)         OP="$2"; shift 2 ;;
        -a|--algo)
            IFS=',' read -ra _vals <<< "$2"
            ALGOS+=("${_vals[@]}")
            shift 2 ;;
        -v|--ver)
            IFS=',' read -ra _vals <<< "$2"
            VERS+=("${_vals[@]}")
            shift 2 ;;
        -i|--input)   INPUT_ARG="$2"; shift 2 ;;
        -t|--threads) NUM_THREADS="$2"; shift 2 ;;
        -o|--outdir)  OUTDIR="$2"; shift 2 ;;
        -p|--perf)
            PERF_MODE=1
            if [[ -n "${2:-}" && "$2" =~ ^(cache|tlb|stalls)$ ]]; then
                PERF_PROFILE="$2"; shift 2
            else
                PERF_PROFILE="cache"; shift
            fi ;;
        -P|--perf-internal)
            PERF_MODE=2
            if [[ -n "${2:-}" && "$2" =~ ^(cache|tlb|stalls)$ ]]; then
                PERF_PROFILE="$2"; shift 2
            else
                PERF_PROFILE="cache"; shift
            fi ;;
        -h|--help)    show_help ;;
        -*)           echo "Unknown option: $1"; show_help ;;
        *)            ALGOS+=("$1"); shift ;;
    esac
done

# --- Resolve input file ---
if [[ "$OP" == "grp_matmul" ]]; then
    case "$INPUT_ARG" in
        uniform)     INPUT_FILE="$GRP_DIR/moe_uniform.txt"; TAG="uniform" ;;
        prompt)      INPUT_FILE="$GRP_DIR/grp_matmul_prompt.txt"; TAG="prompt" ;;
        decode)      INPUT_FILE="$GRP_DIR/grp_matmul_decode.txt"; TAG="decode" ;;
        bf16)        INPUT_FILE="$GRP_DIR/moe_uniform.txt"; TAG="uniform" ;;
        *)           INPUT_FILE="$INPUT_ARG"; TAG="$(basename "${INPUT_FILE%.*}")" ;;
    esac
elif [[ "$OP" == "bmm" ]]; then
    case "$INPUT_ARG" in
        sdpa|SDPA)       INPUT_FILE="$BMM_DIR/sdpa_bmm_inputs.txt";    TAG="sdpa" ;;
        pytorch|PYTORCH) INPUT_FILE="$BMM_DIR/pytorch_bmm_inputs.txt";  TAG="pytorch" ;;
        bf16)            INPUT_FILE="$BMM_DIR/sdpa_bmm_inputs.txt";     TAG="sdpa" ;;
        *)               INPUT_FILE="$INPUT_ARG"; TAG="$(basename "${INPUT_FILE%.*}")" ;;
    esac
else
    case "$INPUT_ARG" in
        bf16|BF16)                     INPUT_FILE="$SWEEP_DIR/bf16_generative_models_eval.txt"; TAG="bf16" ;;
        fp32|FP32)                     INPUT_FILE="$SWEEP_DIR/fp32_generative_models_eval.txt"; TAG="fp32" ;;
        bf16_pytorch|BF16_PYTORCH)     INPUT_FILE="$SWEEP_DIR/bf16_pytorch_models_eval.txt";    TAG="bf16_pytorch" ;;
        fp32_pytorch|FP32_PYTORCH)     INPUT_FILE="$SWEEP_DIR/fp32_pytorch_models_eval.txt";    TAG="fp32_pytorch" ;;
        *)                             INPUT_FILE="$INPUT_ARG"; TAG="$(basename "${INPUT_FILE%.*}")" ;;
    esac
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input file not found: $INPUT_FILE"; exit 1
fi

# --- Resolve threads ---
export OMP_NUM_THREADS="${NUM_THREADS:-$(nproc)}"

# --- Validate args ---
if [[ "$OP" == "grp_matmul" ]]; then
    if [ ${#VERS[@]} -eq 0 ]; then VERS=(1); fi
    if [ ${#ALGOS[@]} -eq 0 ]; then ALGOS=(11); fi
elif [[ "$OP" == "bmm" ]]; then
    if [ ${#ALGOS[@]} -eq 0 ]; then
        echo "ERROR: -a/--algo is required for bmm (e.g. -a 6 for libxsmm, -a 5,6)"; show_help
    fi
else
    if [ ${#ALGOS[@]} -eq 0 ]; then
        echo "ERROR: -a/--algo is required (e.g. -a 1 or -a 1,11)"; show_help
    fi
fi

# --- Locate benchdnn ---
BENCHDNN_BIN="$REPO_ROOT/build/benchdnn/benchdnn"
if [ ! -f "$BENCHDNN_BIN" ]; then
    BENCHDNN_BIN="$REPO_ROOT/build/install/benchdnn/bin/benchdnn"
fi
if [ ! -f "$BENCHDNN_BIN" ]; then
    echo "ERROR: benchdnn not found in build/ or build/install/"; exit 1
fi

# --- Standard env ---
: "${JEMALLOC_LIB:=/usr/local/lib/libjemalloc.so}"
: "${IOMP_LIB:=/opt/intel/oneapi/compiler/2025.0/lib/libiomp5.so}"
PRELOAD_VALUE="${LD_PRELOAD-}"
[ -f "$IOMP_LIB" ]    && PRELOAD_VALUE="${IOMP_LIB}${PRELOAD_VALUE:+:${PRELOAD_VALUE}}"
[ -f "$JEMALLOC_LIB" ] && PRELOAD_VALUE="${JEMALLOC_LIB}${PRELOAD_VALUE:+:${PRELOAD_VALUE}}"
[ -n "$PRELOAD_VALUE" ] && export LD_PRELOAD="$PRELOAD_VALUE"

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

CPU_BIND="0-$((OMP_NUM_THREADS - 1))"

# --- Perf events ---
case "$PERF_PROFILE" in
    cache)  PERF_EVENTS="L1-dcache-loads,L1-dcache-load-misses,rFF70,rFF71,rFF72,rF064,r0864" ;;
    tlb)    PERF_EVENTS="L1-dcache-loads,L1-dcache-load-misses,r0F45,rF045,r00C0,r0076" ;;
    stalls) PERF_EVENTS="r00C0,r0076,r20AE,r40AE,r02AE,r20AF" ;;
esac

mkdir -p "$OUTDIR"

echo "================================================================"
echo "  Benchmark: $OP"
echo "  Input   : $INPUT_FILE"
if [[ "$OP" == "grp_matmul" ]]; then
echo "  Versions: ${VERS[*]}"
echo "  Algo    : ${ALGOS[*]}"
else
echo "  Algos   : ${ALGOS[*]}"
fi
if [[ "$OP" == "bmm" ]]; then
echo "  ndims   : 3 (batched)"
fi
echo "  Threads : $OMP_NUM_THREADS"
echo "  CPU bind: $CPU_BIND"
if [[ $PERF_MODE -eq 1 ]]; then echo "  HW Perf : External perf stat ($PERF_PROFILE)"
elif [[ $PERF_MODE -eq 2 ]]; then echo "  HW Perf : Internal perf_event_open ($PERF_PROFILE)"
else echo "  HW Perf : OFF"; fi
echo "  Output  : $OUTDIR/"
echo "================================================================"
echo ""

# ── grp_matmul mode: loop over versions × algos ─────────────────────────
if [[ "$OP" == "grp_matmul" ]]; then
    for algo in "${ALGOS[@]}"; do
        for ver in "${VERS[@]}"; do
            OUTFILE="$OUTDIR/grp_matmul_${TAG}_v${ver}_algo${algo}_${OMP_NUM_THREADS}t.csv"
            echo "--- grp_matmul V${ver} ALGO=${algo} ---"

            ZENDNNL_GRP_MATMUL_ALGO=$ver \
            ZENDNNL_MATMUL_ALGO=$algo \
            numactl --physcpubind="$CPU_BIND" \
                "$BENCHDNN_BIN" --op=grp_matmul --input_file="$INPUT_FILE" \
                2>&1 | tee "$OUTFILE"

            echo "--- V${ver} ALGO=${algo} done → $OUTFILE ---"
            echo ""
        done
    done

# ── bmm mode: loop over algos with --ndims=3 ─────────────────────────────
elif [[ "$OP" == "bmm" ]]; then
    for algo in "${ALGOS[@]}"; do
        OUTFILE="$OUTDIR/bmm_${TAG}_algo${algo}_${OMP_NUM_THREADS}c.txt"

        if [[ $PERF_MODE -eq 1 ]]; then
            PERF_RAW="$OUTDIR/bmm_${TAG}_algo${algo}_${OMP_NUM_THREADS}c_perf_raw.txt"
            echo "--- BMM ALGO=$algo (per-shape perf stat) ---"
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
                    env OMP_NUM_THREADS="$OMP_NUM_THREADS" ZENDNNL_BMM_ALGO="$algo" \
                    numactl --physcpubind="$CPU_BIND" \
                    "$BENCHDNN_BIN" --op=matmul --ndims=3 \
                    --input_file=/tmp/_benchdnn_single.txt \
                    >> "$PERF_RAW" 2>&1
                echo "" >> "$PERF_RAW"
                if (( idx % 10 == 0 )) || (( idx == 1 )); then
                    echo "  [$idx/$total] done"
                fi
            done < "$INPUT_FILE"
            echo "--- BMM ALGO=$algo perf → $PERF_RAW ---"
        elif [[ $PERF_MODE -eq 2 ]]; then
            echo "--- BMM ALGO=$algo (internal perf) ---"
            ZENDNNL_BMM_ALGO=$algo \
            numactl --physcpubind="$CPU_BIND" \
                "$BENCHDNN_BIN" --op=matmul --ndims=3 \
                "--perf-counters=$PERF_PROFILE" \
                --input_file="$INPUT_FILE" \
                2>&1 | tee "$OUTFILE"
            echo "--- BMM ALGO=$algo done → $OUTFILE ---"
        else
            echo "--- BMM ALGO=$algo ---"
            ZENDNNL_BMM_ALGO=$algo \
            numactl --physcpubind="$CPU_BIND" \
                "$BENCHDNN_BIN" --op=matmul --ndims=3 --input_file="$INPUT_FILE" \
                2>&1 | tee "$OUTFILE"
            echo "--- BMM ALGO=$algo done → $OUTFILE ---"
        fi
        echo ""
    done

# ── matmul mode: loop over algos (existing behavior) ────────────────────
else
    for algo in "${ALGOS[@]}"; do
        OUTFILE="$OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c.txt"

        if [[ $PERF_MODE -eq 1 ]]; then
            PERF_RAW="$OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c_perf_raw.txt"
            echo "--- ALGO=$algo (per-shape perf stat) ---"
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
            echo "--- ALGO=$algo perf → $PERF_RAW ---"
        elif [[ $PERF_MODE -eq 2 ]]; then
            echo "--- ALGO=$algo (internal perf) ---"
            ZENDNNL_MATMUL_ALGO=$algo \
            numactl --physcpubind="$CPU_BIND" \
                "$BENCHDNN_BIN" --op=matmul --lowoha=true \
                "--perf-counters=$PERF_PROFILE" \
                --input_file="$INPUT_FILE" \
                2>&1 | tee "$OUTFILE"
            echo "--- ALGO=$algo done → $OUTFILE ---"
        else
            echo "--- ALGO=$algo ---"
            ZENDNNL_MATMUL_ALGO=$algo \
            numactl --physcpubind="$CPU_BIND" \
                "$BENCHDNN_BIN" --op=matmul --lowoha=true --input_file="$INPUT_FILE" \
                2>&1 | tee "$OUTFILE"
            echo "--- ALGO=$algo done → $OUTFILE ---"
        fi
        echo ""
    done
fi

echo "================================================================"
echo "Results in $OUTDIR/"
echo "================================================================"
