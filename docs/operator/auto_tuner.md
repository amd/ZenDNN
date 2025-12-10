(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# ZenDNN* AutoTuner

The **ZenDNN* AutoTuner** is an intelligent performance optimization component that automatically selects the best-performing algorithm for operations at runtime. It dynamically evaluates multiple backend implementations and caches the optimal choice for specific workload characteristics, eliminating the need for manual algorithm selection and tuning.

## Overview

Modern deep learning inference and training workloads present a complex optimization challenge due to their inherent diversity and dynamic nature. The computational patterns, tensor dimensions, data types, memory layouts, and hardware characteristics can vary significantly across different models, layers, and deployment scenarios. What performs optimally for one configuration may be suboptimal for another, making static algorithm selection impractical.

### The AutoTuner Solution

The ZenDNN* AutoTuner addresses these challenges through an intelligent, adaptive approach:

- **Automatic Algorithm Selection**: Evaluates multiple backend algorithms using time-based profiling and selects the fastest for each unique workload without requiring manual intervention or domain expertise.

- **Intelligent Caching**: Stores optimal algorithm choices in a workload-aware cache, avoiding redundant benchmarking and amortizing tuning overhead across multiple invocations.

- **Multi-Phase Execution**: Employs a scientifically rigorous three-phase methodology (warmup, evaluation, execution) to ensure robust performance measurements that account for cache effects and system stabilization.

- **Workload-Specific Optimization**: Maintains separate cache entries for different tensor shapes, transpositions, data types, and other operation parameters, ensuring fine-grained optimization.

- **Zero Manual Intervention**: Operates transparently without requiring users to understand algorithmic trade-offs, library capabilities, or hardware characteristics.

ZenDNN AutoTuner will exhibit non-deterministic behavior during its initial tuning phase, but deterministic behavior after caching. Here’s why:

### Initial Evaluation Phase (Non-Deterministic)
The AutoTuner benchmarks multiple algorithms using time-based profiling.
Runtime conditions such as CPU frequency scaling, NUMA effects, background processes, and cache state can influence timing measurements.
Therefore, the algorithm selected during the first run for a given workload may vary slightly across executions on the same hardware, especially if system load changes.

### Caching Phase (Deterministic)
Once the best algorithm is selected and stored in the cache for a specific workload key (e.g., M, K, N, transpose flags, data type), subsequent executions for that workload will always use the cached algorithm.
This makes repeated runs for the same workload deterministic after tuning is complete.


## AutoTuner Flow Diagram

### High-Level AutoTuner Workflow

```text
                      Operation Request
                           (params)
                              |
                              v
                    +---------+----------+
                    | Check Cache        |
                    | (Key: M,K,N,Trans) |
                    +---------+----------+
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
          Cache Hit                    Cache Miss
          (Algorithm Found)            (New Workload)
                 |                          |
                 |                          v
                 |                  +-------+--------+
                 |                  | Initialize     |
                 |                  | Tuning State   |
                 |                  +-------+--------+
                 |                          |
                 |                          v
                 |            +-------------+--------------+
                 |            |   Phase 1: Skip/Warmup     |
                 |            | (SKIP_ITER iterations)     |
                 |            | - Warm up CPU caches       |
                 |            | - Stabilize system         |
                 |            +-------------+--------------+
                 |                          |
                 |                          v
                 |            +-------------+--------------+
                 |            |   Phase 2: Evaluation      |
                 |            | (EVAL_ITER iterations)     |
                 |            | - Test Algorithm           |
                 |            | - Measure Time             |
                 |            | - Cycle through all algos  |
                 |            |   each iteration           |
                 |            +-------------+--------------+
                 |                          |
                 |                          v
                 |                  +-------+--------+
                 |                  | Select Best    |
                 |                  | Algorithm      |
                 |                  | (Min Time)     |
                 |                  +-------+--------+
                 |                          |
                 |                          v
                 |                  +-------+--------+
                 |                  | Cache Result   |
                 |                  | Store in Map   |
                 |                  +-------+--------+
                 |                          |
                 +------------+-------------+
                              |
                              v
                    +---------+----------+
                    |   Phase 3: Execute |
                    | Use Cached Algo    |
                    +---------+----------+
                              |
                              v
                    Execute MatMul with
                    Selected Algorithm
                              |
                              v
                        Return Result
```
---

## Supported Algorithms

The AutoTuner currently evaluates between the following high-performance backends:

1. **oneDNN Blocked Algorithm** (`matmul_algo_t::onednn_blocked`)
   - Optimized for general-purpose matrix multiplication
   - Efficient cache utilization through blocking strategies
   - Strong performance across diverse matrix dimensions

2. **AOCL DLP/BLIS Blocked Algorithm** (`matmul_algo_t::aocl_blis_blocked`)
   - AMD-optimized implementation leveraging AOCL libraries
   - Tailored for AMD CPU architectures
   - Excellent performance for specific problem sizes and data layouts

---

## AutoTuner Execution Flow

The AutoTuner operates in three distinct phases:

### Phase 1: Skip (Warmup) Phase
- **Purpose**: Warm up CPU caches and stabilize system performance
- **Duration**: Configurable via `ZENDNNL_MATMUL_SKIP_ITER` (default: 2 iterations)
- **Behavior**: Executes operations using the different algorithms
- **No Measurements**: Performance data is not collected during this phase

### Phase 2: Evaluation Phase
- **Purpose**: Benchmark all available algorithms and collect timing data
- **Duration**: Configurable via `ZENDNNL_MATMUL_EVAL_ITER` (default: 3 iterations)
- **Behavior**: Cycles through all algorithms (using modulo arithmetic)
- **Timing Collection**: Measures execution time for each algorithm
- **Selection Logic**: Determines the best algorithm based on collected metrics

### Phase 3: Execution Phase
- **Purpose**: Use the cached optimal algorithm for all subsequent operations
- **Duration**: All remaining iterations after evaluation completes
- **Behavior**: Executes operations using the selected best-performing algorithm
- **No Overhead**: No additional timing or decision-making overhead

### Phase Transition Diagram
    SKIP_ITER = 2, EVAL_ITER = 3
    
    Iteration:  0    1   │  2    3    4  │   5     6    7 ...
                ─────────┼───────────────┼─────────────────
    Phase:      SKIP     │  EVALUATION   │   EXECUTION
                         │               │
    Algo Used:  default  │  1    2    1  |  best  best  best
                         │               │
    Timing:     ✗   ✗   │  ✓    ✓   ✓  │  ✗     ✗    ✗
                         │               │
    Cache:      N/A      │  building...  │  locked (best)
                         │  comparing    │
                         │  updating     │

---

## Cache Key Structure

    Key_matmul Components:
    ┌──────────────────────────────────┐
    │  transA     : 'n' or 't'         │
    │  transB     : 'n' or 't'         │
    │  M          : Rows of output     │
    │  K          : Inner dimension    │
    │  N          : Cols of output     │
    │  lda        : Leading dim A      │
    │  ldb        : Leading dim B      │
    │  weights*   : Weight pointer     │  ← For constant weights
    │  algo       : Algorithm ID       │
    └──────────────────────────────────┘
                |
                v
    Unique workload identifier
    Used for cache lookup

## AutoTuner Versions

ZenDNN* provides two implementations of the AutoTuner with different caching strategies and performance characteristics.

### Version 1: Layer-based Algorithm Caching (`auto_compute_matmul_v1`)

#### Design Philosophy
Version 1 uses a **greedy selection strategy** that immediately updates the cached algorithm whenever a faster implementation is discovered during the evaluation phase of each layer (based on key).

#### Internal Data Structures
```cpp
// Primary cache: Stores the best algorithm for each MatMul
static std::unordered_map<Key_matmul, matmul_algo_t> matmul_kernel_map;

// Auxiliary data: Tracks iteration count, best time, and current best algorithm
static std::unordered_map<Key_matmul, std::tuple<unsigned int, float, matmul_algo_t>> 
    matmul_kernel_map1_helper;
```

#### Selection Flow

    For each MatMul with key (M, N, K, Trans,...):
    
    Iteration 0-1: Skip Phase
    ┌─────────────────────────────────┐
    │  Execute with different algo    │
    │  No timing measurement          │
    └─────────────────────────────────┘
    
    Iteration 2-4: Evaluation Phase
    ┌─────────────────────────────────┐
    │  iter % NUM_ALGO → Select Algo  │
    │  ┌───────────────────────────┐  │
    │  │ Algo 1: Measure time_1    │  │
    │  │ if time_1 < best_time:    │  │
    │  │   Update cache → Algo 1   │◄─┼─── Immediate Update
    │  └───────────────────────────┘  │
    │  ┌───────────────────────────┐  │
    │  │ Algo 2: Measure time_2    │  │
    │  │ if time_2 < best_time:    │  │
    │  │   Update cache → Algo 2   │◄─┼─── Immediate Update
    │  └───────────────────────────┘  │
    └─────────────────────────────────┘
    
    Iteration 5+: Execution Phase
    ┌─────────────────────────────────┐
    │  Use cached best algorithm      │
    │  No timing overhead             │
    └─────────────────────────────────┘

### Version 2: Graph-based Algorithm Caching (`auto_compute_matmul_v2`)

#### Design Philosophy
Version 2 uses a statistical approach that accumulates timing data across all evaluation iterations and selects the algorithm with the minimum total execution time after evaluation completes.

#### Internal Data Structures
```cpp
// Iteration counter: Tracks execution count for each MatMul
static std::unordered_map<Key_matmul, unsigned int> matmul_kernel_map1_helper;

// Timing vector: Accumulates execution times for each iteration
static std::vector<double> algo_time_vec(skip_iter + evaluate_iter, 0.0);
```

#### Selection flow

    For each MatMul with key (M, N, K, Trans,...):
    
    Iteration 0-1: Skip Phase
    ┌─────────────────────────────────┐
    │  Execute with different algo    │
    │  algo_time_vec[0,1] = INT_MAX   │
    └─────────────────────────────────┘
    
    Iteration 2-4: Evaluation Phase
    ┌─────────────────────────────────┐
    │  iter % NUM_ALGO → Select Algo  │
    │  ┌───────────────────────────┐  │
    │  │ Algo 1: Measure time_1    │  │
    │  │ algo_time_vec[idx1] += t_1│◄─┼─── Accumulate
    │  └───────────────────────────┘  │
    │  ┌───────────────────────────┐  │
    │  │ Algo 2: Measure time_2    │  │
    │  │ algo_time_vec[idx2] += t_2│◄─┼─── Accumulate
    │  └───────────────────────────┘  │
    │  ┌───────────────────────────┐  │
    │  │ Repeat for all iterations │  │
    │  └───────────────────────────┘  │
    └─────────────────────────────────┘
                    |
                    v
    ┌─────────────────────────────────┐
    │  After last eval iteration:     │
    │  1. Find min(algo_time_vec)     │
    │  2. Map index → Algorithm       │
    │  3. Cache selected algorithm    │
    └─────────────────────────────────┘
    
    Iteration 5+: Execution Phase
    ┌─────────────────────────────────┐
    │  Use cached best algorithm      │
    │  No timing overhead             │
    └─────────────────────────────────┘

---

## Configuration and Environment Variables

The AutoTuner behavior can be controlled through various environment variables, allowing fine-grained control over its operation without code modifications.

### AutoTuner Control Variables

```
# Enable the AutoTuner
export ZENDNNL_MATMUL_ALGO=auto

# Select AutoTuner version (optional, default: 2)
export ZENDNNL_AUTO_TUNER_TYPE=1/2

# Set Skip Iterations (optional, default: 2)
export ZENDNNL_MATMUL_SKIP_ITER=<number>

# Set Evaluate Iterations (optional, default: 3)
export ZENDNNL_MATMUL_EVAL_ITER=<number>
```
---
**Note**: The AutoTuner for MatMul operation is currently in development.

>ZenDNN* : ZenDNN is currently undergoing a strategic re-architecture and refactoring to enhance performance, maintainability, and scalability.