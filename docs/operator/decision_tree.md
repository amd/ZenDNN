(Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.)

# ZENDNN* Decision Tree

The **Decision Tree Algorithm** is a kernel selection mechanism in the ZenDNN* framework derived using machine learning techniques. During development, an ML pipeline analyzes performance data to automatically generate optimal decision logic, which is then converted into native C++ conditional statements. At runtime, these precomputed if-else conditions select the best-performing kernel for matrix multiplication operations based on input dimensions, ensuring maximum performance with zero ML overhead.

---

## 1. Decision Tree in ZENDNN*

### Overview

In ZENDNN*, multiple kernels are available for executing matrix multiplication (matmul) operations. Each kernel exhibits different performance characteristics depending on the metadata of the matmul operation.

### Decision Tree solution

#### What is a Decision Tree?

A decision tree is a hierarchical structure of conditional rules that evaluates input parameters (such as matrix dimensions) through a series of comparisons to arrive at an optimal decision.

**Usage in ZENDNN***: The decision tree is used to internally select the optimal kernel for execution based on the input dimensions. The decision logic is precomputed during the training phase, and the generated conditional statements are directly integrated into the library.

### Runtime Execution Flowchart

```text
              ┌──────────────────────────────────────────────┐
              │         Runtime Execution (ZenDNN*)          │
              │         Input: M, N, K, BS                   │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │      Decision Tree Conditional Checks and    │
              │            Optimal Kernel selection          │
              │        (Based on decision tree output)       │
              │  ┌────────────────────────────────────────┐  │
              │  │ if (M <= threshold_1)                  │  │
              │  │   if (K <= threshold_2)                │  │
              │  │     return ALGO_A                      │  │
              │  │   else                                 │  │
              │  │     return ALGO_B                      │  │
              │  │ else                                   │  │
              │  │   return ALGO_A                        │  │
              │  └────────────────────────────────────────┘  │
              └──────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │   Execute MatMul with Selected Algorithm     │
              └──────────────────────────────────────────────┘
```

### Enable the Decision Tree

```bash
export ZENDNNL_MATMUL_ALGO=0
```

### Supported Algorithms

The Decision Tree currently picks one of the following high-performance backends based on the input features:

1. **oneDNN Blocked Algorithm** (`matmul_algo_t::onednn_blocked`)
   - Optimized for general-purpose matrix multiplication
   - Efficient cache utilization through blocking strategies
   - Strong performance across diverse matrix dimensions

2. **AOCL DLP Blocked Algorithm** (`matmul_algo_t::aocl_dlp_blocked`)
   - AMD-optimized implementation leveraging AOCL libraries
   - Tailored for AMD CPU architectures
   - Excellent performance for specific problem sizes and data layouts

---

## 2. Limitations of Manually-Crafted Decision Trees

Manually-Crafted decision trees suffer from several critical limitations:

**Stale Performance Models**: Hardware improvements, compiler optimizations, and library updates can change performance characteristics over time, requiring manual updates.

**Manual Maintenance Overhead**: Every significant code change requires manual re-analysis and tree updates, consuming valuable engineering time.

**Pattern Recognition Difficulty**: Humans struggle to identify optimal decision boundaries when analyzing thousands of performance measurements across multiple dimensions.

**Time-Intensive**: Manual analysis and rule crafting can take days or weeks for complex datasets.

**The ML-based approach solves these issues by enabling periodic retraining on fresh performance data.**

---

## 3. Why Use Machine Learning for Decision Trees?

The machine learning approach offers significant advantages over traditional manual rule creation:

### Automated Rule Generation
The model automatically learns decision logic during training, removing the need for hand-crafted rules. Rules are derived from actual performance measurements through statistical optimization rather than human intuition, ensuring objective and data-driven algorithm selection.

### Pattern Discovery and Optimization
ML algorithms can identify non-obvious patterns across matmul metadata (such as matrix dimensions M, N, K, BS, and other operational parameters) that are difficult for humans to detect. Manual analysis becomes impractical with thousands of data points and multiple features. The model discovers subtle relationships between matmul characteristics and optimal algorithms that might be missed in manual inspection, automatically determining optimal decision boundaries rather than relying on trial-and-error.

### Scalable Design
Additional input features such as preceding operation, thread count can be easily integrated into the model. The framework adapts to new optimization criteria without redesigning the entire selection logic, providing a future-proof architecture.

### Efficiency and Deployment
The model can be fine-tuned using grid search and cross-validation to identify optimal hyperparameters. ML pipelines process thousands of configurations in minutes, producing reproducible results with consistent hyperparameters. As new backends are added, the model can be retrained seamlessly. Although trained using scikit-learn, the final decision logic is translated into pure C++ conditional statements, requiring no Python runtime or ML libraries at deployment, executing as fast as hand-written conditionals.

---

## 4. Why Decision Tree Model Specifically?

Decision trees are uniquely suited for kernel selection in performance-critical systems, as they rely solely on comparison operators at inference time and produce exactly one deterministic output for each input perfectly aligned with the requirement to select a single best-performing kernel.

### Computational Efficiency
Decision trees require no floating-point arithmetic, exponentials, or matrix multiplications. Modern CPUs handle conditional branches efficiently with branch prediction, making decision trees ideal for performance-sensitive tasks with minimal overhead.

### Advantages Over Alternative ML Models
Unlike other ML models, decision trees avoid computational complexity at runtime. Logistic regression requires dot products and sigmoid functions, neural networks involve matrix multiplications and activation functions, random forests aggregate predictions from multiple trees (adding latency and complexity), support vector machines need distance calculations, and k-nearest neighbors requires computation across all training points. All these alternatives introduce unnecessary overhead that is unsuitable for deterministic, performance-critical kernel selection.

---

## 5. Machine Learning Pipeline

The ML pipeline transforms raw performance data into optimized C++ decision logic through several key stages:

### Data Preparation

**Source**: All training and testing data is extracted from benchmarking various models listed in the **PyTorch Dashboard** across all available backends. The dataset contains real-world workload configurations. This ensures the decision tree is trained on realistic, production-relevant data.

**Note**: All training data has been collected using AMD EPYC Gen 4 and Gen 5 architectures.

**Train/Test Split**: A 70/30 stratified split maintains class distribution across both performance ratio groups and target algorithms.

**Feature Engineering**: Matrix dimensions (M, N, K, BS) serve as input features, with the target variable being the optimal algorithm choice based on performance ratios.

**Hyperparameter Optimization**: Grid search performs exhaustive exploration over hyperparameter combinations.

**Instance Weighting**: Records with significant performance differences receive higher weights during training, prioritizing correct predictions for high-impact configurations.

**Cross-Validation**: StratifiedKFold ensures robust performance estimates and guards against overfitting.

**Custom Error Metric**: The evaluation calculates the sum of performance ratios for misclassified instances. Lower values indicate better model performance, prioritizing the avoidance of costly misclassifications where performance differences are large. Unlike accuracy, this metric accounts for the real-world impact of incorrect predictions.

### Model Selection and Pruning

**Pruning**: Insignificant nodes are removed to reduce tree complexity and improve generalization.

**Manual Selection**: Models are evaluated based on error metric (sum of performance ratio differences), tree depth (complexity), number of conditions (nodes), and cross-validation standard deviation (stability). All candidate models are scored and sorted to select the optimal balance between accuracy and simplicity.

### Code Generation

The trained decision tree is automatically converted to C++ conditional statements and integrated directly into ZENDNN* source files.

### Key Characteristics

- **Precomputed Logic**: The decision tree is trained offline using ML techniques and converted into native C++ conditional statements.
- **Zero Runtime Overhead**: No model loading, inference libraries, or mathematical computations required only simple comparison-based if-else conditions execute at runtime.
- **Deterministic Selection**: Produces exactly one kernel choice for each input configuration.
- **Configuration-Agnostic Training**: The ML pipeline can generate optimized decision trees for any hardware architecture, thread configuration, or operational context by simply providing the corresponding performance data.

### Pipeline Flowchart

```text
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         Data_processing.sh                              │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                      Data Collection                              │  │
    │  │  ┌─────────────────────────────────────────────────────────────┐  │  │
    │  │  │ - Benchmark PT Dashboard models                             │  │  │
    │  │  │ - Execute across all backends                               │  │  │
    │  │  │ - Collect performance metrics                               │  │  │
    │  │  └─────────────────────────────────────────────────────────────┘  │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    │                                 │                                       │
    │                                 ▼                                       │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                      Process the Data                             │  │
    │  │  ┌─────────────────────────────────────────────────────────────┐  │  │
    │  │  │ - Extract performance ratios                                │  │  │
    │  │  │ - Format dataset (M, N, K, BS → optimal algo)               │  │  │
    │  │  │ - Prepare ML training dataset                               │  │  │
    │  │  └─────────────────────────────────────────────────────────────┘  │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                       DT_Automation_pipeline.py                         │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                      Preprocess the Data                          │  │
    │  │  ┌─────────────────────────────────────────────────────────────┐  │  │
    │  │  │ - 70/30 stratified train/test split                         │  │  │
    │  │  │ - Feature engineering (M, N, K, BS)                         │  │  │
    │  │  │ - Target: optimal algorithm selection                       │  │  │
    │  │  └─────────────────────────────────────────────────────────────┘  │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    │                                 │                                       │
    │                                 ▼                                       │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                      Train the ML Model                           │  │
    │  │  ┌─────────────────────────────────────────────────────────────┐  │  │
    │  │  │ Grid Search: Hyperparameter optimization                    │  │  │
    │  │  ├─────────────────────────────────────────────────────────────┤  │  │
    │  │  │ Instance Weighting                                          │  │  │
    │  │  ├─────────────────────────────────────────────────────────────┤  │  │
    │  │  │ Cross-Validation (StratifiedKFold)                          │  │  │
    │  │  ├─────────────────────────────────────────────────────────────┤  │  │
    │  │  │ Pruning                                                     │  │  │
    │  │  ├─────────────────────────────────────────────────────────────┤  │  │
    │  │  │ Custom Error Metric                                         │  │  │
    │  │  └─────────────────────────────────────────────────────────────┘  │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    │                                 │                                       │
    │                                 ▼                                       │
    │  ┌───────────────────────────────────────────────────────────────────┐  │
    │  │                 Convert the Model into a Function                 │  │
    │  │  ┌─────────────────────────────────────────────────────────────┐  │  │
    │  │  │ Model Selection                                             │  │  │
    │  │  ├─────────────────────────────────────────────────────────────┤  │  │
    │  │  │ Code Generation                                             │  │  │
    │  │  │  - Convert decision tree to C++ if-else statements          │  │  │
    │  │  │  - Pure conditional logic (no ML dependencies)              │  │  │
    │  │  └─────────────────────────────────────────────────────────────┘  │  │
    │  └───────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │    Plug the Function to ZenDNN*              │
              │  ┌────────────────────────────────────────┐  │
              │  │ - Integrate C++ code into source       │  │
              │  │ - Zero runtime ML overhead             │  │
              │  └────────────────────────────────────────┘  │
              └──────────────────────────────────────────────┘
```
**Note**: This entire process is performed offline during development. The final C++ code is integrated into ZenDNN* for runtime use.

---

## 6. Current Status and Future Work

The pipeline successfully generates and evaluates multiple decision tree candidates using the ML framework. The user manually selects the best-performing model, which is then converted to C++ and integrated into ZENDNN* for runtime algorithm selection with zero ML overhead.

**Current Achievements**: Automated decision tree generation from performance data, stratified train/test split with instance weighting, and custom error metric accounting for performance impact.

**Current Limitations**: Manual model selection required.

**Planned Enhancements**: This is the initial version of the ML-based decision tree pipeline. Future work focuses on further fine-tuning the pipeline, automating the model selection criteria, and expanding feature-based analysis to include thread scaling, pre-op, post-op and other operational context parameters.

---

**Note**: The ML pipeline used for decision tree generation is currently in development.

>ZENDNN* : ZenDNN is currently undergoing a strategic re-architecture and refactoring to enhance performance, maintainability, and scalability.