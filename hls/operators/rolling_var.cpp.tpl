// Operator: $op_type
// Kernel: ${op_id}_kernel
// RollingVar attrs: ${attrs}
// Inputs: $inputs
// Outputs: $outputs
template <typename T, int Window>
void ${op_id}_kernel(const T input, T window[Window], T& variance) {
#pragma HLS INLINE
rolling_var_shift_loop:
  for (int idx = 0; idx < Window - 1; ++idx) {
#pragma HLS PIPELINE II=1
    window[idx] = window[idx + 1];
  }
  window[Window - 1] = input;

  T mean = (T)0;
rolling_var_mean_loop:
  for (int idx = 0; idx < Window; ++idx) {
#pragma HLS PIPELINE II=1
    mean += window[idx];
  }
  mean /= (T)Window;

  T accum = (T)0;
rolling_var_accum_loop:
  for (int idx = 0; idx < Window; ++idx) {
#pragma HLS PIPELINE II=1
    const T centered = window[idx] - mean;
    accum += centered * centered;
  }
  variance = accum / (T)Window;
}
