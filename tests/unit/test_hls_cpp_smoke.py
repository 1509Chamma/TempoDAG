import shutil
import subprocess
from pathlib import Path

import pytest

from tempo_dag.codegen.hls.generator import render_operator_hls
from tempo_dag.codegen.hls.temporal_generator import (
    render_temporal_artifact_from_trace,
    render_temporal_process_hls,
)
from tempo_dag.ir.op import Operator
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ops.builtins import (
    GELU,
    LSTM,
    Add,
    Concat,
    Conv1D,
    Div,
    LayerNorm,
    MatMul,
    Max,
    Mean,
    Mul,
    Pad,
    ReLU,
    Reshape,
    Shift,
    Sigmoid,
    Slice,
    Softmax,
    Sub,
    Sum,
    Tanh,
    Transpose,
)
from tempo_dag.parsers.temporal_onnx import (
    TemporalONNXParser,
    build_demo_temporal_onnx_model,
)
from tempo_dag.verification.golden_trace import load_golden_trace


def make_tensor(
    value_id: str,
    shape: list[int],
    axes: list[str] | None = None,
) -> Value:
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=shape,
        axes=axes or [f"axis_{idx}" for idx in range(len(shape))],
    )


def make_scalar(value_id: str) -> Value:
    return Value(
        value_id=value_id,
        vtype=ValueType.SCALAR,
        dtype="float32",
        shape=[],
        axes=[],
    )


def test_representative_hls_operator_templates_compile_as_cpp(
    tmp_path: Path,
) -> None:
    compiler = _cpp_compiler()

    snippets = [
        _render_binary(Add(op_id="add_0", inputs=["lhs", "rhs"], outputs=["out"])),
        _render_binary(Sub(op_id="sub_0", inputs=["lhs", "rhs"], outputs=["out"])),
        _render_binary(Mul(op_id="mul_0", inputs=["lhs", "rhs"], outputs=["out"])),
        _render_binary(Div(op_id="div_0", inputs=["lhs", "rhs"], outputs=["out"])),
        _render_unary(Sigmoid(op_id="sigmoid_0", inputs=["x"], outputs=["y"])),
        _render_unary(Tanh(op_id="tanh_0", inputs=["x"], outputs=["y"])),
        _render_unary(ReLU(op_id="relu_0", inputs=["x"], outputs=["y"])),
        _render_unary(GELU(op_id="gelu_0", inputs=["x"], outputs=["y"])),
        _render_reduction(Sum(op_id="sum_0", inputs=["x"], outputs=["y"])),
        _render_reduction(Mean(op_id="mean_0", inputs=["x"], outputs=["y"])),
        _render_reduction(Max(op_id="max_0", inputs=["x"], outputs=["y"])),
        _render_softmax(),
        _render_matmul(),
        _render_transpose(),
        _render_reshape(),
        _render_concat(),
        _render_slice(),
        _render_layer_norm(),
        _render_conv1d(),
        _render_pad(),
        _render_shift(),
        _render_lstm(),
    ]
    translation_unit = "\n\n".join(
        [
            "#include <cmath>",
            "#include <cstdint>",
            *snippets,
            "int main() { return 0; }",
            "",
        ]
    )

    _compile_cpp(compiler, tmp_path / "operators.cpp", translation_unit)


def test_hls_operator_templates_execute_against_golden_data(tmp_path: Path) -> None:
    compiler = _cpp_compiler()
    snippets = [
        _render_binary(Add(op_id="add_0", inputs=["lhs", "rhs"], outputs=["out"])),
        _render_add_scalar_rhs(),
        _render_binary(Sub(op_id="sub_0", inputs=["lhs", "rhs"], outputs=["out"])),
        _render_binary(Mul(op_id="mul_0", inputs=["lhs", "rhs"], outputs=["out"])),
        _render_binary(Div(op_id="div_0", inputs=["lhs", "rhs"], outputs=["out"])),
        _render_unary(ReLU(op_id="relu_0", inputs=["x"], outputs=["y"])),
        _render_reduction(Sum(op_id="sum_0", inputs=["x"], outputs=["y"])),
        _render_reduction(Mean(op_id="mean_0", inputs=["x"], outputs=["y"])),
        _render_reduction(Max(op_id="max_0", inputs=["x"], outputs=["y"])),
        _render_softmax(),
        _render_matmul(),
        _render_transpose(),
        _render_reshape(),
        _render_concat(),
        _render_slice(),
        _render_layer_norm(),
        _render_pad(),
        _render_shift(),
        _render_shift_left(),
        _render_lstm(),
    ]
    translation_unit = "\n\n".join(
        [
            "#include <cmath>",
            "#include <cstdint>",
            "#include <iostream>",
            *snippets,
            _golden_runtime_main(),
            "",
        ]
    )

    _compile_and_run_cpp(compiler, tmp_path / "operator_golden.cpp", translation_unit)


def test_temporal_hls_artifact_compiles_with_testbench(tmp_path: Path) -> None:
    compiler = _cpp_compiler()
    process = (
        TemporalONNXParser()
        .parse_model(
            build_demo_temporal_onnx_model(),
            process_id="demo_process",
        )
        .process
    )
    trace = load_golden_trace("tests/verification/golden_traces/rolling_mean.json")
    artifact = render_temporal_artifact_from_trace(process, trace)

    process_path = tmp_path / "demo_process.cpp"
    testbench_path = tmp_path / "demo_process_tb.cpp"
    process_path.write_text(artifact.process_hls, encoding="utf-8")
    testbench_path.write_text(artifact.testbench_hls, encoding="utf-8")

    command = [
        compiler,
        "-std=c++17",
        "-Wno-unknown-pragmas",
        str(process_path),
        str(testbench_path),
        "-o",
        str(tmp_path / "demo_process_tb"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_temporal_process_hls_emits_operator_kernels_at_file_scope() -> None:
    process = (
        TemporalONNXParser()
        .parse_model(
            build_demo_temporal_onnx_model(),
            process_id="demo_process",
        )
        .process
    )

    rendered = render_temporal_process_hls(process)

    assert (
        "template <typename T, int Window>\nvoid rolling_mean_node_kernel" in rendered
    )
    assert "void demo_process_step() {" in rendered
    assert "#pragma HLS DATAFLOW" in rendered
    assert (
        "Operator invocation wiring is emitted by the next scheduler layer" in rendered
    )


def _render_binary(operator: Operator) -> str:
    values = {
        "lhs": make_tensor("lhs", [2, 3], ["batch", "feature"]),
        "rhs": make_tensor("rhs", [2, 3], ["batch", "feature"]),
        "out": make_tensor("out", [2, 3], ["batch", "feature"]),
    }
    return render_operator_hls(operator, values)


def _render_add_scalar_rhs() -> str:
    values = {
        "lhs": make_tensor("lhs", [3], ["feature"]),
        "rhs": make_scalar("rhs"),
        "out": make_tensor("out", [3], ["feature"]),
    }
    return render_operator_hls(
        Add(op_id="add_scalar_0", inputs=["lhs", "rhs"], outputs=["out"]),
        values,
    )


def _render_unary(operator: Operator) -> str:
    values = {
        "x": make_tensor("x", [2, 3], ["batch", "feature"]),
        "y": make_tensor("y", [2, 3], ["batch", "feature"]),
    }
    return render_operator_hls(operator, values)


def _render_reduction(operator: Operator) -> str:
    values = {
        "x": make_tensor("x", [2, 3], ["batch", "feature"]),
        "y": make_tensor("y", [2], ["batch"]),
    }
    operator.attrs["axis"] = 1
    return render_operator_hls(operator, values)


def _render_softmax() -> str:
    values = {
        "x": make_tensor("x", [2, 3], ["batch", "feature"]),
        "y": make_tensor("y", [2, 3], ["batch", "feature"]),
    }
    return render_operator_hls(
        Softmax(op_id="softmax_0", inputs=["x"], outputs=["y"], attrs={"axis": 1}),
        values,
    )


def _render_matmul() -> str:
    values = {
        "lhs": make_tensor("lhs", [2, 3], ["rows", "inner"]),
        "rhs": make_tensor("rhs", [3, 4], ["inner", "cols"]),
        "out": make_tensor("out", [2, 4], ["rows", "cols"]),
    }
    return render_operator_hls(
        MatMul(op_id="matmul_0", inputs=["lhs", "rhs"], outputs=["out"]),
        values,
    )


def _render_transpose() -> str:
    values = {
        "x": make_tensor("x", [2, 3], ["rows", "cols"]),
        "y": make_tensor("y", [3, 2], ["cols", "rows"]),
    }
    return render_operator_hls(
        Transpose(
            op_id="transpose_0",
            inputs=["x"],
            outputs=["y"],
            attrs={"perm": [1, 0]},
        ),
        values,
    )


def _render_reshape() -> str:
    values = {
        "x": make_tensor("x", [2, 3], ["rows", "cols"]),
        "y": make_tensor("y", [6], ["flat"]),
    }
    return render_operator_hls(
        Reshape(op_id="reshape_0", inputs=["x"], outputs=["y"], attrs={"shape": [6]}),
        values,
    )


def _render_concat() -> str:
    values = {
        "a": make_tensor("a", [2], ["feature"]),
        "b": make_tensor("b", [3], ["feature"]),
        "out": make_tensor("out", [5], ["feature"]),
    }
    return render_operator_hls(
        Concat(op_id="concat_0", inputs=["a", "b"], outputs=["out"], attrs={"axis": 0}),
        values,
    )


def _render_slice() -> str:
    values = {
        "x": make_tensor("x", [6], ["feature"]),
        "y": make_tensor("y", [2], ["feature"]),
    }
    return render_operator_hls(
        Slice(
            op_id="slice_0",
            inputs=["x"],
            outputs=["y"],
            attrs={"axis": 0, "start": 1, "end": 5, "step": 2},
        ),
        values,
    )


def _render_layer_norm() -> str:
    values = {
        "x": make_tensor("x", [2, 3], ["batch", "feature"]),
        "y": make_tensor("y", [2, 3], ["batch", "feature"]),
    }
    return render_operator_hls(
        LayerNorm(op_id="layer_norm_0", inputs=["x"], outputs=["y"], attrs={"axis": 1}),
        values,
    )


def _render_conv1d() -> str:
    values = {
        "x": make_tensor("x", [1, 2, 8], ["batch", "channel", "time"]),
        "w": make_tensor("w", [4, 2, 3], ["out_channel", "in_channel", "kernel"]),
        "y": make_tensor("y", [1, 4, 8], ["batch", "channel", "time"]),
    }
    return render_operator_hls(
        Conv1D(
            op_id="conv_0",
            inputs=["x", "w"],
            outputs=["y"],
            attrs={"stride": 1, "padding": 1, "dilation": 1},
        ),
        values,
    )


def _render_pad() -> str:
    values = {
        "x": make_tensor("x", [4], ["feature"]),
        "y": make_tensor("y", [7], ["feature"]),
    }
    return render_operator_hls(
        Pad(op_id="pad_0", inputs=["x"], outputs=["y"], attrs={"pads": [2, 1]}),
        values,
    )


def _render_shift() -> str:
    values = {
        "x": make_tensor("x", [4], ["feature"]),
        "y": make_tensor("y", [4], ["feature"]),
    }
    return render_operator_hls(
        Shift(
            op_id="shift_0",
            inputs=["x"],
            outputs=["y"],
            attrs={"axis": 0, "amount": 1},
        ),
        values,
    )


def _render_shift_left() -> str:
    values = {
        "x": make_tensor("x", [4], ["feature"]),
        "y": make_tensor("y", [4], ["feature"]),
    }
    return render_operator_hls(
        Shift(
            op_id="shift_left_0",
            inputs=["x"],
            outputs=["y"],
            attrs={"axis": 0, "amount": -1},
        ),
        values,
    )


def _render_lstm() -> str:
    values = {
        "x": make_tensor("x", [3, 1, 2], ["time", "batch", "feature"]),
        "w": make_tensor("w", [1, 16, 2], ["direction", "gate_hidden", "feature"]),
        "r": make_tensor("r", [1, 16, 4], ["direction", "gate_hidden", "hidden"]),
        "y": make_tensor("y", [3, 1, 1, 4], ["time", "direction", "batch", "hidden"]),
    }
    return render_operator_hls(
        LSTM(
            op_id="lstm_0",
            inputs=["x", "w", "r"],
            outputs=["y"],
            attrs={"hidden_size": 4},
        ),
        values,
    )


def _golden_runtime_main() -> str:
    return r"""
bool approx_equal(float lhs, float rhs, float tolerance = 1.0e-4f) {
  return std::fabs(lhs - rhs) <= tolerance;
}

bool check_array(const char* name, const float* actual, const float* expected,
                 int size, float tolerance = 1.0e-4f) {
  for (int idx = 0; idx < size; ++idx) {
    if (!approx_equal(actual[idx], expected[idx], tolerance)) {
      std::cerr << name << "[" << idx << "] expected " << expected[idx]
                << " got " << actual[idx] << "\n";
      return false;
    }
  }
  return true;
}

int main() {
  bool ok = true;

  float lhs[6] = {1.0f, -2.0f, 0.0f, 4.5f, -1.0f, 3.0f};
  float rhs[6] = {2.0f, 2.0f, -1.0f, 0.5f, 1.0f, -3.0f};
  float out6[6] = {};

  add_0_kernel(lhs, rhs, out6);
  const float add_expected[6] = {3.0f, 0.0f, -1.0f, 5.0f, 0.0f, 0.0f};
  ok = check_array("add", out6, add_expected, 6) && ok;

  float scalar_rhs[1] = {2.5f};
  float lhs3[3] = {-1.0f, 0.0f, 4.0f};
  float out3[3] = {};
  add_scalar_0_kernel(lhs3, scalar_rhs, out3);
  const float add_scalar_expected[3] = {1.5f, 2.5f, 6.5f};
  ok = check_array("add_scalar", out3, add_scalar_expected, 3) && ok;

  sub_0_kernel(lhs, rhs, out6);
  const float sub_expected[6] = {-1.0f, -4.0f, 1.0f, 4.0f, -2.0f, 6.0f};
  ok = check_array("sub", out6, sub_expected, 6) && ok;

  mul_0_kernel(lhs, rhs, out6);
  const float mul_expected[6] = {2.0f, -4.0f, -0.0f, 2.25f, -1.0f, -9.0f};
  ok = check_array("mul", out6, mul_expected, 6) && ok;

  div_0_kernel(lhs, rhs, out6);
  const float div_expected[6] = {0.5f, -1.0f, -0.0f, 9.0f, -1.0f, -1.0f};
  ok = check_array("div", out6, div_expected, 6) && ok;

  relu_0_kernel(lhs, out6);
  const float relu_expected[6] = {1.0f, 0.0f, 0.0f, 4.5f, 0.0f, 3.0f};
  ok = check_array("relu", out6, relu_expected, 6) && ok;

  float reduction_input[6] = {1.0f, 2.0f, 3.0f, -1.0f, -5.0f, 4.0f};
  float reduction_out[2] = {};
  sum_0_kernel(reduction_input, reduction_out);
  const float sum_expected[2] = {6.0f, -2.0f};
  ok = check_array("sum", reduction_out, sum_expected, 2) && ok;
  mean_0_kernel(reduction_input, reduction_out);
  const float mean_expected[2] = {2.0f, -0.6666667f};
  ok = check_array("mean", reduction_out, mean_expected, 2) && ok;
  max_0_kernel(reduction_input, reduction_out);
  const float max_expected[2] = {3.0f, 4.0f};
  ok = check_array("max", reduction_out, max_expected, 2) && ok;

  float softmax_input[6] = {0.0f, 0.0f, 0.0f, 2.0f, 1.0f, 0.0f};
  softmax_0_kernel(softmax_input, out6);
  const float softmax_expected[6] = {
      0.3333333f, 0.3333333f, 0.3333333f,
      0.6652409f, 0.2447285f, 0.0900306f};
  ok = check_array("softmax", out6, softmax_expected, 6, 1.0e-3f) && ok;

  float mat_lhs[2][3] = {{1.0f, 2.0f, 3.0f}, {-1.0f, 0.0f, 2.0f}};
  float mat_rhs[3][4] = {
      {1.0f, 0.0f, 2.0f, -1.0f},
      {0.0f, 1.0f, -1.0f, 2.0f},
      {3.0f, 1.0f, 0.0f, 1.0f}};
  float mat_out[2][4] = {};
  matmul_0_kernel(mat_lhs, mat_rhs, mat_out);
  const float mat_expected[8] = {10.0f, 5.0f, 0.0f, 6.0f,
                                 5.0f, 2.0f, -2.0f, 3.0f};
  ok = check_array("matmul", &mat_out[0][0], mat_expected, 8) && ok;

  float transpose_input[2][3] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
  float transpose_out[3][2] = {};
  transpose_0_kernel(transpose_input, transpose_out);
  const float transpose_expected[6] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
  ok = check_array("transpose", &transpose_out[0][0], transpose_expected, 6) && ok;

  reshape_0_kernel(transpose_input[0], out6);
  const float reshape_expected[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  ok = check_array("reshape", out6, reshape_expected, 6) && ok;

  float concat_a[2] = {7.0f, 8.0f};
  float concat_b[3] = {-1.0f, -2.0f, -3.0f};
  const float* concat_inputs[2] = {concat_a, concat_b};
  float concat_out[5] = {};
  concat_0_kernel(concat_inputs, concat_out);
  const float concat_expected[5] = {7.0f, 8.0f, -1.0f, -2.0f, -3.0f};
  ok = check_array("concat", concat_out, concat_expected, 5) && ok;

  float slice_input[6] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  float slice_out[2] = {};
  slice_0_kernel(slice_input, slice_out);
  const float slice_expected[2] = {20.0f, 40.0f};
  ok = check_array("slice", slice_out, slice_expected, 2) && ok;

  float layer_input[6] = {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f};
  layer_norm_0_kernel(layer_input, out6);
  const float layer_expected[6] = {-1.2247356f, 0.0f, 1.2247356f,
                                   0.0f, 0.0f, 0.0f};
  ok = check_array("layer_norm", out6, layer_expected, 6, 1.0e-3f) && ok;

  float pad_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float pad_out[7] = {};
  pad_0_kernel(pad_input, pad_out);
  const float pad_expected[7] = {0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 0.0f};
  ok = check_array("pad", pad_out, pad_expected, 7) && ok;

  float shift_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float shift_out[4] = {};
  shift_0_kernel(shift_input, shift_out);
  const float shift_expected[4] = {0.0f, 1.0f, 2.0f, 3.0f};
  ok = check_array("shift", shift_out, shift_expected, 4) && ok;
  shift_left_0_kernel(shift_input, shift_out);
  const float shift_left_expected[4] = {2.0f, 3.0f, 4.0f, 0.0f};
  ok = check_array("shift_left", shift_out, shift_left_expected, 4) && ok;

  float lstm_x[3][1][2] = {};
  float lstm_w[1][16][2] = {};
  float lstm_r[1][16][4] = {};
  float lstm_y[3][1][1][4] = {};
  lstm_0_kernel(lstm_x, lstm_w, lstm_r, lstm_y);
  const float lstm_expected[12] = {};
  ok = check_array("lstm_zero", &lstm_y[0][0][0][0], lstm_expected, 12) && ok;

  return ok ? 0 : 1;
}
"""


def _cpp_compiler() -> str:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("C++ compiler not available for HLS smoke tests")
    return compiler


def _compile_cpp(compiler: str, path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    command = [
        compiler,
        "-std=c++17",
        "-Wno-unknown-pragmas",
        "-fsyntax-only",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def _compile_and_run_cpp(compiler: str, path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    executable = path.with_suffix(".exe")
    command = [
        compiler,
        "-std=c++17",
        "-Wno-unknown-pragmas",
        str(path),
        "-o",
        str(executable),
    ]
    compile_result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr
