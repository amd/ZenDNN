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
#   -v, --ver <N>[,N,...]         Group matmul selector(s) (for grp_matmul)
#                                   0  = Auto (selects V1, V2, or V3 based on shape)
#                                   1  = Sequential (experts serial, all threads per GEMM)
#                                   2  = Flat CCD adaptive tile (hybrid M/N, framework-safe)
#                                   3  = Flat CCD N-tile (no nested OMP, framework-safe)
#                                   4  = W8A8 fused-MoE fast path (AUTO fallback)
#                                   5  = Per-expert (1 thread per expert, parallel-for)
#                                   6  = Multilevel CCD-aware (nested OMP)
#   -i, --input <file|shortcut>   Input file or shortcut (default: bf16)
#   -t, --threads <N[,N,...]>     OMP thread/core count(s). Comma-separated
#                                 values sweep cores, e.g. -t 32,64,128
#                                 (default: all cores)
#   -o, --outdir <dir>            Output directory (default: build/)
#   -C, --cache-mode <m[,m,...]>  Cache mode(s): hot, cold, warm. Comma-separated
#                                 values sweep modes, e.g. -C hot,cold
#                                 (default: benchdnn default = hot)
#   -m, --m-sweep <M[:M:...]>     In-binary M sweep (colon-separated), e.g. -m 1:128:512.
#                                 Forwarded as --sweep=true --m_sweep=... (matmul/bmm only)
#   -d, --dtype-sweep <list|all>  In-binary dtype sweep (comma-separated names or 'all'),
#                                 e.g. -d all. Forwarded as --sweep=true --dtype_sweep=...
#                                 (matmul/bmm only)
#   -p, --perf [profile]          External perf stat (matmul/bmm only)
#   -P, --perf-internal [profile] Internal perf counters (matmul/bmm only)
#   -h, --help                    Show this help
#
# Input shortcuts (matmul):
#   bf16             -> benchmark_sweep/bf16_generative_models_eval.txt
#   fp32             -> benchmark_sweep/fp32_generative_models_eval.txt
#   bf16_pytorch     -> benchmark_sweep/bf16_pytorch_models_eval.txt
#   fp32_pytorch     -> benchmark_sweep/fp32_pytorch_models_eval.txt
#   s8s8_per_token   -> benchmark_sweep/s8s8_per_token_generative_models_eval.txt
#   s8s8_per_group   -> benchmark_sweep/s8s8_per_group_generative_models_eval.txt
#
# Input shortcuts (bmm):
#   sdpa             -> input/bmm/sdpa_bmm_inputs.txt
#   pytorch          -> input/bmm/pytorch_bmm_inputs.txt
#
# Input shortcuts (grp_matmul):
#   prompt           -> input/grp_matmul/grp_matmul_prompt.txt
#   decode           -> input/grp_matmul/grp_matmul_decode.txt
#   mixtral_fused    -> input/grp_matmul/moe_fused/mixtral_moe_fused.txt
#   qwen3_fused      -> input/grp_matmul/moe_fused/qwen3_30b_moe_fused.txt
#   gptoss_fused     -> input/grp_matmul/moe_fused/gpt_oss_moe_fused.txt
#   mixtral_full     -> input/grp_matmul/moe_fused_gate_up_down/mixtral_moe_full_block.txt
#   qwen3_full       -> input/grp_matmul/moe_fused_gate_up_down/qwen3_30b_moe_full_block.txt
#   gptoss_full      -> input/grp_matmul/moe_fused_gate_up_down/gpt_oss_moe_full_block.txt
#
# Examples:
#   ./run_matmul_benchmark_sweep.sh -a 1,11 -i bf16 -t 128
#   ./run_matmul_benchmark_sweep.sh -a 1,11 -i bf16 -t 32,64,128   # core sweep
#   ./run_matmul_benchmark_sweep.sh -a 1,11 -i bf16 -C hot,cold     # cache sweep
#   ./run_matmul_benchmark_sweep.sh -a 1,11 -i bf16 -m 1:128:512    # M sweep (in-binary)
#   ./run_matmul_benchmark_sweep.sh -a 1,11 -i bf16 -d all          # dtype sweep (in-binary)
#   ./run_matmul_benchmark_sweep.sh --op grp_matmul -v 1,2,3 -i mixtral_full -t 128
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
THREADS_ARG=""
CACHE_MODES_ARG=""
M_SWEEP_ARG=""
DTYPE_SWEEP_ARG=""
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
        -t|--threads) THREADS_ARG="$2"; shift 2 ;;
        -C|--cache-mode) CACHE_MODES_ARG="$2"; shift 2 ;;
        -m|--m-sweep)     M_SWEEP_ARG="$2"; shift 2 ;;
        -d|--dtype-sweep) DTYPE_SWEEP_ARG="$2"; shift 2 ;;
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
        prompt)         INPUT_FILE="$GRP_DIR/grp_matmul_prompt.txt"; TAG="prompt" ;;
        decode)         INPUT_FILE="$GRP_DIR/grp_matmul_decode.txt"; TAG="decode" ;;
        bf16)           INPUT_FILE="$GRP_DIR/grp_matmul_decode.txt"; TAG="decode" ;;
        mixtral_fused)  INPUT_FILE="$GRP_DIR/moe_fused/mixtral_moe_fused.txt"; TAG="mixtral_fused" ;;
        qwen3_fused)    INPUT_FILE="$GRP_DIR/moe_fused/qwen3_30b_moe_fused.txt"; TAG="qwen3_fused" ;;
        gptoss_fused)   INPUT_FILE="$GRP_DIR/moe_fused/gpt_oss_moe_fused.txt"; TAG="gptoss_fused" ;;
        mixtral_full)   INPUT_FILE="$GRP_DIR/moe_fused_gate_up_down/mixtral_moe_full_block.txt"; TAG="mixtral_full" ;;
        qwen3_full)     INPUT_FILE="$GRP_DIR/moe_fused_gate_up_down/qwen3_30b_moe_full_block.txt"; TAG="qwen3_full" ;;
        gptoss_full)    INPUT_FILE="$GRP_DIR/moe_fused_gate_up_down/gpt_oss_moe_full_block.txt"; TAG="gptoss_full" ;;
        *)              INPUT_FILE="$INPUT_ARG"; TAG="$(basename "${INPUT_FILE%.*}")" ;;
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
        s8s8_per_token|S8S8_PER_TOKEN) INPUT_FILE="$SWEEP_DIR/s8s8_per_token_generative_models_eval.txt"; TAG="s8s8_per_token" ;;
        s8s8_per_group|S8S8_PER_GROUP) INPUT_FILE="$SWEEP_DIR/s8s8_per_group_generative_models_eval.txt"; TAG="s8s8_per_group" ;;
        *)                             INPUT_FILE="$INPUT_ARG"; TAG="$(basename "${INPUT_FILE%.*}")" ;;
    esac
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input file not found: $INPUT_FILE"; exit 1
fi

# --- Resolve core-count sweep (comma-separated -t sweeps cores) ---
if [[ -n "$THREADS_ARG" ]]; then
    IFS=',' read -ra CORE_COUNTS <<< "$THREADS_ARG"
    for i in "${!CORE_COUNTS[@]}"; do
        CORE_COUNTS[$i]="${CORE_COUNTS[$i]//[[:space:]]/}"
        if [[ ! "${CORE_COUNTS[$i]}" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: invalid core count '${CORE_COUNTS[$i]}' (use positive integers)"; exit 1
        fi
    done
else
    CORE_COUNTS=("$(nproc)")
fi

# --- Resolve cache-mode sweep (comma-separated -C sweeps modes) ---
# An empty entry means "do not pass --cache_mode" (benchdnn default = hot).
if [[ -n "$CACHE_MODES_ARG" ]]; then
    IFS=',' read -ra CACHE_MODES <<< "$CACHE_MODES_ARG"
    for i in "${!CACHE_MODES[@]}"; do
        CACHE_MODES[$i]="${CACHE_MODES[$i]//[[:space:]]/}"
        CACHE_MODES[$i]="${CACHE_MODES[$i],,}"
        [[ -z "${CACHE_MODES[$i]}" ]] && continue
        if [[ ! "${CACHE_MODES[$i]}" =~ ^(hot|cold|warm)$ ]]; then
            echo "ERROR: invalid cache mode '${CACHE_MODES[$i]}' (use hot, cold, or warm)"; exit 1
        fi
    done
else
    CACHE_MODES=("")
fi

# --- Resolve in-binary M / dtype sweep (-m / -d) ---
# Unlike cores and cache mode (process-level bash loops), M and dtype are swept
# inside a single benchdnn process via --sweep. We just forward the flags; the
# binary cross-products M x dtype per input row.
SWEEP_ARGS=""
if [[ -n "$M_SWEEP_ARG" || -n "$DTYPE_SWEEP_ARG" ]]; then
    if [[ "$OP" == "grp_matmul" ]]; then
        echo "ERROR: -m/--m-sweep and -d/--dtype-sweep are not supported for --op grp_matmul"; exit 1
    fi
    SWEEP_ARGS="--sweep=true"
    [[ -n "$M_SWEEP_ARG" ]]     && SWEEP_ARGS="$SWEEP_ARGS --m_sweep=$M_SWEEP_ARG"
    [[ -n "$DTYPE_SWEEP_ARG" ]] && SWEEP_ARGS="$SWEEP_ARGS --dtype_sweep=$DTYPE_SWEEP_ARG"
    if [[ $PERF_MODE -eq 1 ]]; then
        echo "WARNING: -p/--perf runs one perf stat per input line; with -m/-d each line"
        echo "         expands to many configs, so counters aggregate across them."
    fi
fi

# --- Validate args ---
if [[ "$OP" == "grp_matmul" ]]; then
    if [ ${#VERS[@]} -eq 0 ]; then VERS=(1); fi
    if [ ${#ALGOS[@]} -eq 0 ]; then ALGOS=(1); fi
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
echo "  GRP_ALGO : ${VERS[*]}"
echo "  MATMUL_ALGO: ${ALGOS[*]}"
else
echo "  Algos   : ${ALGOS[*]}"
fi
if [[ "$OP" == "bmm" ]]; then
echo "  ndims   : 3 (batched)"
fi
echo "  Cores   : ${CORE_COUNTS[*]}"
if [[ -n "$CACHE_MODES_ARG" ]]; then
echo "  Cache   : ${CACHE_MODES[*]}"
else
echo "  Cache   : hot (default)"
fi
if [[ -n "$SWEEP_ARGS" ]]; then
echo "  M sweep : ${M_SWEEP_ARG:-(binary default)}"
echo "  Dtype   : ${DTYPE_SWEEP_ARG:-all}"
fi
if [[ $PERF_MODE -eq 1 ]]; then echo "  HW Perf : External perf stat ($PERF_PROFILE)"
elif [[ $PERF_MODE -eq 2 ]]; then echo "  HW Perf : Internal perf_event_open ($PERF_PROFILE)"
else echo "  HW Perf : OFF"; fi
echo "  Output  : $OUTDIR/"
echo "================================================================"
echo ""

# ── Core-count sweep: run the full operator dispatch once per core count ─
for NUM_THREADS in "${CORE_COUNTS[@]}"; do
export OMP_NUM_THREADS="$NUM_THREADS"
CPU_BIND="0-$((OMP_NUM_THREADS - 1))"
if (( OMP_NUM_THREADS > $(nproc) )); then
    echo "WARNING: requested $OMP_NUM_THREADS cores > $(nproc) available; numactl bind may fail."
fi
echo "################################################################"
echo "  CORES = $OMP_NUM_THREADS   (CPU bind $CPU_BIND)"
echo "################################################################"

# ── Cache-mode sweep: run the operator dispatch once per cache mode ──────
for CACHE_MODE in "${CACHE_MODES[@]}"; do
if [[ -n "$CACHE_MODE" ]]; then
    CACHE_ARG="--cache_mode=$CACHE_MODE"
    CTAG="_${CACHE_MODE}"
    echo ">>>>>>>>>>>>>>  CACHE MODE = $CACHE_MODE  <<<<<<<<<<<<<<"
else
    CACHE_ARG=""
    CTAG=""
fi

# ── grp_matmul mode: loop over versions × algos ─────────────────────────
if [[ "$OP" == "grp_matmul" ]]; then
    for algo in "${ALGOS[@]}"; do
        for ver in "${VERS[@]}"; do
            OUTFILE="$OUTDIR/grp_matmul_${TAG}_v${ver}_algo${algo}_${OMP_NUM_THREADS}t${CTAG}.csv"
            echo "--- grp_matmul GRP_ALGO=${ver} MATMUL_ALGO=${algo} ---"

            ZENDNNL_GRP_MATMUL_ALGO=$ver \
            ZENDNNL_MATMUL_ALGO=$algo \
            numactl --physcpubind="$CPU_BIND" \
                "$BENCHDNN_BIN" --op=grp_matmul --input_file="$INPUT_FILE" \
                $CACHE_ARG \
                2>&1 | tee "$OUTFILE"

            echo "--- GRP_ALGO=${ver} MATMUL_ALGO=${algo} done → $OUTFILE ---"
            echo ""
        done
    done

# ── bmm mode: loop over algos with --ndims=3 ─────────────────────────────
elif [[ "$OP" == "bmm" ]]; then
    for algo in "${ALGOS[@]}"; do
        OUTFILE="$OUTDIR/bmm_${TAG}_algo${algo}_${OMP_NUM_THREADS}c${CTAG}.txt"

        if [[ $PERF_MODE -eq 1 ]]; then
            PERF_RAW="$OUTDIR/bmm_${TAG}_algo${algo}_${OMP_NUM_THREADS}c${CTAG}_perf_raw.txt"
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
                    $CACHE_ARG $SWEEP_ARGS \
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
                $CACHE_ARG $SWEEP_ARGS \
                --input_file="$INPUT_FILE" \
                2>&1 | tee "$OUTFILE"
            echo "--- BMM ALGO=$algo done → $OUTFILE ---"
        else
            echo "--- BMM ALGO=$algo ---"
            ZENDNNL_BMM_ALGO=$algo \
            numactl --physcpubind="$CPU_BIND" \
                "$BENCHDNN_BIN" --op=matmul --ndims=3 $CACHE_ARG $SWEEP_ARGS --input_file="$INPUT_FILE" \
                2>&1 | tee "$OUTFILE"
            echo "--- BMM ALGO=$algo done → $OUTFILE ---"
        fi
        echo ""
    done

# ── matmul mode: loop over algos (existing behavior) ────────────────────
else
    for algo in "${ALGOS[@]}"; do
        OUTFILE="$OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c${CTAG}.txt"

        if [[ $PERF_MODE -eq 1 ]]; then
            PERF_RAW="$OUTDIR/benchmark_${TAG}_algo${algo}_${OMP_NUM_THREADS}c${CTAG}_perf_raw.txt"
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
                    $CACHE_ARG $SWEEP_ARGS \
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
                $CACHE_ARG $SWEEP_ARGS \
                --input_file="$INPUT_FILE" \
                2>&1 | tee "$OUTFILE"
            echo "--- ALGO=$algo done → $OUTFILE ---"
        else
            echo "--- ALGO=$algo ---"
            ZENDNNL_MATMUL_ALGO=$algo \
            numactl --physcpubind="$CPU_BIND" \
                "$BENCHDNN_BIN" --op=matmul --lowoha=true $CACHE_ARG $SWEEP_ARGS --input_file="$INPUT_FILE" \
                2>&1 | tee "$OUTFILE"
            echo "--- ALGO=$algo done → $OUTFILE ---"
        fi
        echo ""
    done
fi

done  # end cache-mode sweep

done  # end core-count sweep

echo "================================================================"
echo "Results in $OUTDIR/"
echo "================================================================"
