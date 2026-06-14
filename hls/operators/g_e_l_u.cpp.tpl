#include <cmath>

// Operator: GELU
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${input_0_size}],
    ${cpp_dtype} out[${output_0_size}]
) {
  static constexpr ${cpp_dtype} kAlpha = (${cpp_dtype})0.7978845608028654;
  static constexpr ${cpp_dtype} kBeta = (${cpp_dtype})0.044715;

gelu_loop:
  for (int idx = 0; idx < ${output_0_size}; ++idx) {
#pragma HLS PIPELINE II=1
    const ${cpp_dtype} x = input[idx];
    const ${cpp_dtype} inner = kAlpha * (x + kBeta * x * x * x);
    out[idx] = (${cpp_dtype})0.5 * x * ((${cpp_dtype})1 + std::tanh(inner));
  }
}
