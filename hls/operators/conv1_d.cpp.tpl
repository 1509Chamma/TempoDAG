// Operator: Conv1D
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} input[${batch}][${in_channels}][${input_length}],
    const ${cpp_dtype} weight[${out_channels}][${in_channels}][${kernel_width}]${bias_parameter},
    ${cpp_dtype} out[${batch}][${out_channels}][${output_length}]
) {
conv_batch_loop:
  for (int batch = 0; batch < ${batch}; ++batch) {
conv_out_channel_loop:
    for (int out_channel = 0; out_channel < ${out_channels}; ++out_channel) {
conv_time_loop:
      for (int out_t = 0; out_t < ${output_length}; ++out_t) {
#pragma HLS PIPELINE II=1
        ${cpp_dtype} acc = ${bias_init};
conv_in_channel_loop:
        for (int in_channel = 0; in_channel < ${in_channels}; ++in_channel) {
conv_kernel_loop:
          for (int kernel = 0; kernel < ${kernel_width}; ++kernel) {
#pragma HLS UNROLL factor=2
            const int in_t = out_t * ${stride} + kernel * ${dilation} - ${padding};
            if (in_t >= 0 && in_t < ${input_length}) {
              acc += input[batch][in_channel][in_t] *
                     weight[out_channel][in_channel][kernel];
            }
          }
        }
        out[batch][out_channel][out_t] = acc;
      }
    }
  }
}
