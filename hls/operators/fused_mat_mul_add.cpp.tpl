// Operator: FusedMatMulAdd
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
void ${op_id}_kernel(
    const ${cpp_dtype} lhs[${m_dim}][${k_dim}],
    const ${cpp_dtype} rhs[${k_dim}][${n_dim}],
    const ${cpp_dtype} bias[${output_0_size}],
    ${cpp_dtype} out[${m_dim}][${n_dim}]
) {
fused_matmul_row_loop:
  for (int row = 0; row < ${m_dim}; ++row) {
fused_matmul_col_loop:
    for (int col = 0; col < ${n_dim}; ++col) {
#pragma HLS PIPELINE II=1
      ${cpp_dtype} acc = bias[row * ${n_dim} + col];
fused_matmul_reduce_loop:
      for (int k = 0; k < ${k_dim}; ++k) {
#pragma HLS UNROLL factor=4
        acc += lhs[row][k] * rhs[k][col];
      }
      out[row][col] = acc;
    }
  }
}
