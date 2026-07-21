// Operator: FusedScaleAdd
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${output_0_size}],
    const ${cpp_dtype} scale[${output_0_size}],
    const ${cpp_dtype} bias[${output_0_size}],
    ${cpp_dtype} out[${output_0_size}]
) {
  static constexpr int kOutputSize = ${output_0_size};
  static constexpr bool kScalarScale = ${has_scalar_scale};
  static constexpr bool kScalarBias = ${has_scalar_bias};

fused_scale_add_loop:
  for (int idx = 0; idx < kOutputSize; ++idx) {
#pragma HLS PIPELINE II=1
    const ${cpp_dtype} scale_value = kScalarScale ? scale[0] : scale[idx];
    const ${cpp_dtype} bias_value = kScalarBias ? bias[0] : bias[idx];
    out[idx] = input[idx] * scale_value + bias_value;
  }
}
