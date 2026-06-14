#include <cmath>

// Operator: Sigmoid
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${input_0_size}],
    ${cpp_dtype} out[${output_0_size}]
) {
sigmoid_loop:
  for (int idx = 0; idx < ${output_0_size}; ++idx) {
#pragma HLS PIPELINE II=1
    out[idx] = (${cpp_dtype})1 / ((${cpp_dtype})1 + std::exp(-input[idx]));
  }
}
