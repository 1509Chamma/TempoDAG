// Operator: $op_type
// Kernel: ${op_id}_kernel
// RollingWindow attrs: ${attrs}
// Inputs: $inputs
// Outputs: $outputs
template <typename T, int Window>
void ${op_id}_kernel(const T input, T window[Window]) {
#pragma HLS INLINE
rolling_window_shift_loop:
  for (int idx = 0; idx < Window - 1; ++idx) {
#pragma HLS PIPELINE II=1
    window[idx] = window[idx + 1];
  }
  window[Window - 1] = input;
}
