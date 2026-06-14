// Operator: $op_type
// Kernel: ${op_id}_kernel
// Delay attrs: ${attrs}
// Inputs: $inputs
// Outputs: $outputs
template <typename T, int Depth>
void ${op_id}_kernel(const T input, T& output, T state[Depth]) {
#pragma HLS INLINE
  output = state[0];
delay_shift_loop:
  for (int idx = 0; idx < Depth - 1; ++idx) {
#pragma HLS PIPELINE II=1
    state[idx] = state[idx + 1];
  }
  state[Depth - 1] = input;
}
