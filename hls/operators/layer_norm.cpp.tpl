#include <cmath>

// Operator: LayerNorm
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${output_0_size}],
    ${cpp_dtype} out[${output_0_size}]
) {
layer_norm_outer_loop:
  for (int outer = 0; outer < ${outer_size}; ++outer) {
    ${cpp_dtype} mean = (${cpp_dtype})0;
layer_norm_mean_loop:
    for (int idx = 0; idx < ${normalized_size}; ++idx) {
#pragma HLS PIPELINE II=1
      mean += input[outer * ${normalized_size} + idx];
    }
    mean /= (${cpp_dtype})${normalized_size};

    ${cpp_dtype} variance = (${cpp_dtype})0;
layer_norm_var_loop:
    for (int idx = 0; idx < ${normalized_size}; ++idx) {
#pragma HLS PIPELINE II=1
      const ${cpp_dtype} centered = input[outer * ${normalized_size} + idx] - mean;
      variance += centered * centered;
    }
    variance /= (${cpp_dtype})${normalized_size};
    const ${cpp_dtype} inv_std = (${cpp_dtype})1 / std::sqrt(variance + (${cpp_dtype})${epsilon});

layer_norm_write_loop:
    for (int idx = 0; idx < ${normalized_size}; ++idx) {
#pragma HLS PIPELINE II=1
      out[outer * ${normalized_size} + idx] =
          (input[outer * ${normalized_size} + idx] - mean) * inv_std;
    }
  }
}
