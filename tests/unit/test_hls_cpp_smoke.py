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
    assert "void demo_process_step() {\n  // Operator invocation wiring" in rendered


def _render_binary(operator: Operator) -> str:
    values = {
        "lhs": make_tensor("lhs", [2, 3], ["batch", "feature"]),
        "rhs": make_tensor("rhs", [2, 3], ["batch", "feature"]),
        "out": make_tensor("out", [2, 3], ["batch", "feature"]),
    }
    return render_operator_hls(operator, values)


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
