// Operator: Mean
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${input_size}],
    ${cpp_dtype} out[${output_size}]
) {
mean_output_loop:
  for (int out_idx = 0; out_idx < ${output_size}; ++out_idx) {
    ${cpp_dtype} acc = (${cpp_dtype})0;
mean_reduce_loop:
    for (int red_idx = 0; red_idx < ${reduction_size}; ++red_idx) {
#pragma HLS PIPELINE II=1
      acc += input[out_idx * ${reduction_size} + red_idx];
    }
    out[out_idx] = acc / (${cpp_dtype})${reduction_size};
  }
}
