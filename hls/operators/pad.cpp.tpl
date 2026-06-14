// Operator: Pad
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${input_size}],
    ${cpp_dtype} out[${output_size}]
) {
pad_loop:
  for (int idx = 0; idx < ${output_size}; ++idx) {
#pragma HLS PIPELINE II=1
    if (idx < ${pad_before} || idx >= ${pad_before} + ${input_size}) {
      out[idx] = (${cpp_dtype})0;
    } else {
      out[idx] = input[idx - ${pad_before}];
    }
  }
}
