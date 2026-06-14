// Operator: Shift
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${output_size}],
    ${cpp_dtype} out[${output_size}]
) {
shift_loop:
  for (int idx = 0; idx < ${output_size}; ++idx) {
#pragma HLS PIPELINE II=1
    const int source = idx - (${amount});
    out[idx] = (source >= 0 && source < ${output_size})
        ? input[source]
        : (${cpp_dtype})0;
  }
}
