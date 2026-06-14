// Operator: Reshape
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${num_elements}],
    ${cpp_dtype} out[${num_elements}]
) {
reshape_copy_loop:
  for (int idx = 0; idx < ${num_elements}; ++idx) {
#pragma HLS PIPELINE II=1
    out[idx] = input[idx];
  }
}
