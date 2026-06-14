// Operator: Transpose
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${rows}][${cols}],
    ${cpp_dtype} out[${cols}][${rows}]
) {
transpose_row_loop:
  for (int row = 0; row < ${rows}; ++row) {
transpose_col_loop:
    for (int col = 0; col < ${cols}; ++col) {
#pragma HLS PIPELINE II=1
      out[col][row] = input[row][col];
    }
  }
}
