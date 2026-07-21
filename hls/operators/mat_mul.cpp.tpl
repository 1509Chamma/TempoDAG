// Operator: MatMul
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
// Reduction: parallel multiplies + balanced adder tree (float order is
// reassociated vs a sequential accumulate; verified under testbench
// tolerance -- oracle-relative gate, not bit-exact to numpy's dot order).
void ${op_id}_kernel(
    const ${cpp_dtype} lhs[${m_dim}][${k_dim}],
    const ${cpp_dtype} rhs[${k_dim}][${n_dim}],
    ${cpp_dtype} out[${m_dim}][${n_dim}]
) {
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=2
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1
matmul_row_loop:
  for (int row = 0; row < ${m_dim}; ++row) {
matmul_col_loop:
    for (int col = 0; col < ${n_dim}; ++col) {
#pragma HLS PIPELINE II=1
      ${cpp_dtype} tree[${k_dim_pow2}];
#pragma HLS ARRAY_PARTITION variable=tree complete
matmul_mul_loop:
      for (int k = 0; k < ${k_dim_pow2}; ++k) {
#pragma HLS UNROLL
        tree[k] = (k < ${k_dim})
            ? lhs[row][k] * rhs[k][col]
            : (${cpp_dtype})0;
      }
matmul_tree_loop:
      for (int stride = ${k_dim_pow2} / 2; stride > 0; stride >>= 1) {
#pragma HLS UNROLL
        for (int i = 0; i < stride; ++i) {
#pragma HLS UNROLL
          tree[i] = tree[i] + tree[i + stride];
        }
      }
      out[row][col] = tree[0];
    }
  }
}
