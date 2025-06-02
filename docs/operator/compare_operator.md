
# Compare Operator Support and Examples

## Overview

The Compare Operator is designed to perform element-wise comparison between two tensors. It calculates and returns the following statistics:
- `match_percent`: The percentage of elements that match between the two tensors.
- `max_deviation`: The maximum deviation found between corresponding elements of the tensors.
- `mean_deviation`: The average deviation across all elements.
- `min_deviation`: The minimum deviation found between corresponding elements of the tensors.

This document provides an overview of the Compare Operator, its functionality, and examples demonstrating its usage.

## General Compare Operation

Let:

- \f$A \in \mathbb{R}^{M \times N}\f$: First tensor
- \f$B \in \mathbb{R}^{M \times N}\f$: Second tensor

The comparison operation calculates the following metrics:

1. **Match Percentage**:
   $$
   \text{match\_percent} = \frac{\text{Number of Matching Elements}}{\text{Total Elements}} \times 100
   $$

2. **Maximum Deviation**:
   $$
   \text{max\_deviation} = \max(|A_{ij} - B_{ij}|)
   $$

3. **Mean Deviation**:
   $$
   \text{mean\_deviation} = \frac{\sum_{i,j} |A_{ij} - B_{ij}|}{\text{Total Elements}}
   $$

4. **Minimum Deviation**:
   $$
   \text{min\_deviation} = \min(|A_{ij} - B_{ij}|)
   $$

## Step-by-Step Operation

1. **Input Tensors**:
   - Two tensors, $A$ and $B$, with the same dimensions are provided as input.

2. **Element-Wise Comparison**:
   - Each element of $A$ is compared with the corresponding element of $B$.

3. **Metric Calculation**:
   - Compute `match_percent`, `max_deviation`, `mean_deviation`, and `min_deviation`.

4. **Return Results**:
   - The calculated metrics are returned as output.

## Compare Support Table

This table outlines the support for Compare operations with various data types.

| Data Type | Supported Metrics |
|-----------|-------------------|
| FP32      | Match Percent, Max Deviation, Mean Deviation, Min Deviation |

## Examples

### 1. Compare two FP32 Tensors

This example demonstrates how to use the Compare Operator to compare two tensors and retrieve the metrics.

```cpp
int compare_operator_example() {
  try {
    // Status variable to track success or failure of operations
    status_t status;

    // Factory object to create tensors with specified properties
    tensor_factory_t tensor_factory;

    // Create a expected tensor with dimensions [M, N], data type float32,
    // initialized with uniform values of 1.0, and named "expected_tensor"
    auto expected = tensor_factory.uniform_tensor({M, N},
                                                 data_type_t::f32,
                                                 1.0, "expected_tensor");

    // Create a test tensor with dimensions [M, N], data type float32,
    // initialized with uniform values of 1.1, and named "test_rensor"
    auto test = tensor_factory.uniform_tensor({M, N},
                                              data_type_t::f32,
                                              1.1, "test_tensor");

    // Create a diff tensor with dimensions [M, N], data type float32,
    // it will store the difference of elements, and named "diff_tensor"
    auto diff = tensor_factory.zero_tensor({M, N},
                                           data_type_t::f32,
                                           "diff_tensor");

    // Create a compare context by setting tolerance.
    auto compare_context = compare_context_t()
                           .set_tolerance(1e-07f)
                           .create();

    // Create a compare operator using the defined context
    // This operator will execute the compare operation
    auto compare_operator = compare_operator_t()
                            .set_name("compare_operator")
                            .set_context(compare_context)
                            .create();

    // Check if the operator was created successfully
    if (! compare_operator.check()) {
      testlog_error("operator ", compare_operator.get_name(), " creation failed");
      return NOT_OK;
    }
    // Set the expected , test and diff tensors and execute the compare operator
    status = compare_operator
             .set_input("expected_tensor", expected)
             .set_input("test_tensor", test)
             .set_output("diff_tensor", diff)
             .execute();

    // Get the tensor comparision statistics
    auto stats = compare_operator.get_compare_stats();

    // Log the results
    if (status == status_t::success) {
      testlog_info("operator ", compare_operator.get_name(), " execution successful.");
      testlog_verbose("Match Percent:",stats.match_percent, "%, ",
                      "Mean Deviation:",stats.mean_deviation,", ",
                      "Max Deviation:",stats.max_deviation,", ",
                      "Min Deviation:",stats.min_deviation,", ",
                      "output[", MATMUL_M/2, ",", MATMUL_N/2,"] = "
                      ,diff_tensor.at({MATMUL_M/2, MATMUL_N/2}));
    }
    else {
      testlog_info("operator ", compare_operator.get_name(), " execution failed");
      return NOT_OK;
    }

  }
  catch (const exception_t &ex) {
    std::cout << ex.what() << std::endl;
    return NOT_OK;
  }

  return OK;
}
```

## Error Handling

The Compare Operator includes error checking to ensure that:
- The input tensors have the same dimensions.
- The operator creation and execution status are validated.
- Relevant logging is provided for any errors encountered.

## Logger

Utility functions such as `testlog_info` and `testlog_error` are used for logging information and errors during the operation flow.

These examples demonstrate the functionality of the Compare Operator, showcasing its ability to calculate metrics for element-wise comparison between tensors in the `zendnnl` library.
