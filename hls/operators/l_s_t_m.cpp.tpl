#include <cmath>

// Operator: LSTM
// Kernel: ${op_id}_kernel
// Inputs: ${inputs}
// Outputs: ${outputs}
// Input shapes: ${input_shapes}
// Output shapes: ${output_shapes}
void ${op_id}_kernel(
    const ${cpp_dtype} x[${seq_len}][${batch}][${input_size}],
    const ${cpp_dtype} w[${num_directions}][${hidden_size} * 4][${input_size}],
    const ${cpp_dtype} r[${num_directions}][${hidden_size} * 4][${hidden_size}]${bias_parameter},
    ${cpp_dtype} y[${seq_len}][${num_directions}][${batch}][${hidden_size}]
) {
lstm_direction_loop:
  for (int direction = 0; direction < ${num_directions}; ++direction) {
lstm_batch_loop:
    for (int batch = 0; batch < ${batch}; ++batch) {
      ${cpp_dtype} hidden[${hidden_size}];
      ${cpp_dtype} cell[${hidden_size}];
#pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=cell cyclic factor=4

lstm_state_init_loop:
      for (int hidden_idx = 0; hidden_idx < ${hidden_size}; ++hidden_idx) {
#pragma HLS PIPELINE II=1
        hidden[hidden_idx] = (${cpp_dtype})0;
        cell[hidden_idx] = (${cpp_dtype})0;
      }

lstm_time_loop:
      for (int step = 0; step < ${seq_len}; ++step) {
        const bool reverse = ${reverse_direction} || (${num_directions} == 2 && direction == 1);
        const int time_idx = reverse ? (${seq_len} - 1 - step) : step;

lstm_hidden_loop:
        for (int hidden_idx = 0; hidden_idx < ${hidden_size}; ++hidden_idx) {
#pragma HLS PIPELINE II=1
          ${cpp_dtype} gate_i = ${gate_i_bias};
          ${cpp_dtype} gate_o = ${gate_o_bias};
          ${cpp_dtype} gate_f = ${gate_f_bias};
          ${cpp_dtype} gate_c = ${gate_c_bias};

lstm_input_loop:
          for (int input_idx = 0; input_idx < ${input_size}; ++input_idx) {
#pragma HLS UNROLL factor=2
            const ${cpp_dtype} input_value = x[time_idx][batch][input_idx];
            gate_i += input_value * w[direction][hidden_idx][input_idx];
            gate_o +=
                input_value * w[direction][${hidden_size} + hidden_idx][input_idx];
            gate_f += input_value *
                      w[direction][2 * ${hidden_size} + hidden_idx][input_idx];
            gate_c += input_value *
                      w[direction][3 * ${hidden_size} + hidden_idx][input_idx];
          }

lstm_recurrent_loop:
          for (int recurrent_idx = 0; recurrent_idx < ${hidden_size};
               ++recurrent_idx) {
#pragma HLS UNROLL factor=2
            const ${cpp_dtype} recurrent_value = hidden[recurrent_idx];
            gate_i += recurrent_value * r[direction][hidden_idx][recurrent_idx];
            gate_o += recurrent_value *
                      r[direction][${hidden_size} + hidden_idx][recurrent_idx];
            gate_f += recurrent_value *
                      r[direction][2 * ${hidden_size} + hidden_idx][recurrent_idx];
            gate_c += recurrent_value *
                      r[direction][3 * ${hidden_size} + hidden_idx][recurrent_idx];
          }

          const ${cpp_dtype} input_gate =
              (${cpp_dtype})1 / ((${cpp_dtype})1 + std::exp(-gate_i));
          const ${cpp_dtype} output_gate =
              (${cpp_dtype})1 / ((${cpp_dtype})1 + std::exp(-gate_o));
          const ${cpp_dtype} forget_gate =
              (${cpp_dtype})1 / ((${cpp_dtype})1 + std::exp(-gate_f));
          const ${cpp_dtype} candidate = std::tanh(gate_c);

          cell[hidden_idx] = forget_gate * cell[hidden_idx] + input_gate * candidate;
          hidden[hidden_idx] = output_gate * std::tanh(cell[hidden_idx]);
          y[time_idx][direction][batch][hidden_idx] = hidden[hidden_idx];
        }
      }
    }
  }
}
