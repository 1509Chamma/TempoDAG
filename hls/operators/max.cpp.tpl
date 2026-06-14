// Operator: Max
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${input_size}],
    ${cpp_dtype} out[${output_size}]
) {
max_output_loop:
  for (int out_idx = 0; out_idx < ${output_size}; ++out_idx) {
    ${cpp_dtype} current = input[out_idx * ${reduction_size}];
max_reduce_loop:
    for (int red_idx = 1; red_idx < ${reduction_size}; ++red_idx) {
#pragma HLS PIPELINE II=1
      const ${cpp_dtype} value = input[out_idx * ${reduction_size} + red_idx];
      current = value > current ? value : current;
    }
    out[out_idx] = current;
  }
}
