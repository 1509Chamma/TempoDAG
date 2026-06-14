// Operator: Concat
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype}* const inputs[${num_inputs}],
    ${cpp_dtype} out[${output_size}]
) {
  static constexpr int kNumInputs = ${num_inputs};
  const int input_sizes[kNumInputs] = {${input_sizes_csv}};
  int out_offset = 0;

concat_input_loop:
  for (int input_idx = 0; input_idx < kNumInputs; ++input_idx) {
concat_copy_loop:
    for (int elem = 0; elem < input_sizes[input_idx]; ++elem) {
#pragma HLS PIPELINE II=1
      out[out_offset + elem] = inputs[input_idx][elem];
    }
    out_offset += input_sizes[input_idx];
  }
}
