// Operator: Mul
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} lhs[${input_0_size}],
    const ${cpp_dtype} rhs[${input_1_size}],
    ${cpp_dtype} out[${output_0_size}]
) {
  static constexpr int kOutputSize = ${output_0_size};
  static constexpr bool kScalarLhs = ${has_scalar_lhs};
  static constexpr bool kScalarRhs = ${has_scalar_rhs};

mul_loop:
  for (int idx = 0; idx < kOutputSize; ++idx) {
#pragma HLS PIPELINE II=1
    const ${cpp_dtype} lhs_value = kScalarLhs ? lhs[0] : lhs[idx];
    const ${cpp_dtype} rhs_value = kScalarRhs ? rhs[0] : rhs[idx];
    out[idx] = lhs_value * rhs_value;
  }
}
