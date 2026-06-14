// Operator: MatMul
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} lhs[${m_dim}][${k_dim}],
    const ${cpp_dtype} rhs[${k_dim}][${n_dim}],
    ${cpp_dtype} out[${m_dim}][${n_dim}]
) {
matmul_row_loop:
  for (int row = 0; row < ${m_dim}; ++row) {
matmul_col_loop:
    for (int col = 0; col < ${n_dim}; ++col) {
#pragma HLS PIPELINE II=1
      ${cpp_dtype} acc = (${cpp_dtype})0;
matmul_reduce_loop:
      for (int k = 0; k < ${k_dim}; ++k) {
#pragma HLS UNROLL factor=4
        acc += lhs[row][k] * rhs[k][col];
      }
      out[row][col] = acc;
    }
  }
}
