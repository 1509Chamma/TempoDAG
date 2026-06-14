#include <cmath>

// Operator: Softmax
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${output_0_size}],
    ${cpp_dtype} out[${output_0_size}]
) {
softmax_outer_loop:
  for (int outer = 0; outer < ${outer_size}; ++outer) {
softmax_inner_loop:
    for (int inner = 0; inner < ${inner_size}; ++inner) {
      ${cpp_dtype} max_value = input[outer * ${axis_size} * ${inner_size} + inner];
softmax_max_loop:
      for (int axis = 1; axis < ${axis_size}; ++axis) {
#pragma HLS PIPELINE II=1
        const int idx = outer * ${axis_size} * ${inner_size} + axis * ${inner_size} + inner;
        max_value = input[idx] > max_value ? input[idx] : max_value;
      }

      ${cpp_dtype} denom = (${cpp_dtype})0;
softmax_exp_loop:
      for (int axis = 0; axis < ${axis_size}; ++axis) {
#pragma HLS PIPELINE II=1
        const int idx = outer * ${axis_size} * ${inner_size} + axis * ${inner_size} + inner;
        const ${cpp_dtype} value = std::exp(input[idx] - max_value);
        out[idx] = value;
        denom += value;
      }

softmax_norm_loop:
      for (int axis = 0; axis < ${axis_size}; ++axis) {
#pragma HLS PIPELINE II=1
        const int idx = outer * ${axis_size} * ${inner_size} + axis * ${inner_size} + inner;
        out[idx] = out[idx] / denom;
      }
    }
  }
}
