// Operator: $op_type
// Kernel: ${op_id}_kernel
// RollingMean attrs: ${attrs}
// Inputs: $inputs
// Outputs: $outputs
template <typename T, int Window>
void ${op_id}_kernel(const T input, T window[Window], T& mean) {
#pragma HLS INLINE
  T dropped = window[0];
rolling_mean_shift_loop:
  for (int idx = 0; idx < Window - 1; ++idx) {
#pragma HLS PIPELINE II=1
    window[idx] = window[idx + 1];
  }
  window[Window - 1] = input;

  T sum = (T)0;
rolling_mean_sum_loop:
  for (int idx = 0; idx < Window; ++idx) {
#pragma HLS PIPELINE II=1
    sum += window[idx];
  }
  (void)dropped;
  mean = sum / (T)Window;
}
