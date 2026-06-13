from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import FPGACost, Operator
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.numerical_parity import (
    NumericalParityConfig,
    ONNXRuntimeParityAdapter,
    TensorFlowKerasParityAdapter,
    TorchQuantizedModelSimulator,
    run_numerical_parity_test,
)
from tempo_dag.quantization_config import (
    FixedPointSpec,
    OverflowPolicy,
    QuantizationScheme,
    QuantizationSpec,
    StateQuantSpec,
)


class MockOp(Operator):
    OP_TYPE = "Mock"

    def validate(self, values):
        del values

    def estimate_fpga_cost(self, values):
        del values
        return FPGACost(1)

    def hls_template_path(self):
        return ""

    def hls_context(self, values):
        del values
        return {}


class TinyParityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)

        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor([[0.5, -0.25], [0.75, 0.5], [-0.5, 0.125]])
            )
            self.fc1.bias.copy_(torch.tensor([0.1, -0.2, 0.05]))
            self.fc2.weight.copy_(torch.tensor([[0.25, -0.75, 0.5]]))
            self.fc2.bias.copy_(torch.tensor([0.125]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class ConstantModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        return self.identity(torch.zeros(1, dtype=torch.float32))


class NonFiniteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        del x
        values = torch.tensor([float("nan")], dtype=torch.float32)
        return self.identity(values)


def test_numerical_parity_config_accepts_temporal_state_quantization() -> None:
    config = NumericalParityConfig.from_input(
        {
            "state_quantization": {
                "hidden": {
                    "dtype": "fixed16",
                    "scale": 2**-8,
                    "overflow_policy": "saturate",
                    "fixed_point": {"integer_bits": 8, "fractional_bits": 8},
                },
                "window": StateQuantSpec(
                    dtype="fixed24",
                    scale=2**-12,
                    overflow_policy=OverflowPolicy.SATURATE,
                    fixed_point=FixedPointSpec(
                        integer_bits=12,
                        fractional_bits=12,
                    ),
                ),
            }
        }
    )

    assert config.state_quantization["hidden"].fixed_point == FixedPointSpec(
        integer_bits=8,
        fractional_bits=8,
    )
    assert config.state_quantization["window"].dtype == "fixed24"


class _FakeONNXValueInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeONNXSession:
    def __init__(
        self,
        *,
        output_shift: float = 0.0,
        hidden_shift: float = 0.0,
    ) -> None:
        self.output_shift = output_shift
        self.hidden_shift = hidden_shift

    def get_inputs(self) -> list[_FakeONNXValueInfo]:
        return [_FakeONNXValueInfo("input")]

    def get_outputs(self) -> list[_FakeONNXValueInfo]:
        return [_FakeONNXValueInfo("output")]

    def run(
        self,
        output_names: list[str],
        feeds: dict[str, object],
    ) -> list[object]:
        input_array = torch.as_tensor(feeds["input"], dtype=torch.float32).numpy()
        hidden = input_array + 0.5 + self.hidden_shift
        output = hidden * 1.5 + self.output_shift
        tensors = {"output": output, "hidden": hidden}
        return [tensors[name] for name in output_names]


def _fixed_point_spec(integer_bits: int, fractional_bits: int) -> QuantizationSpec:
    return QuantizationSpec(
        bit_width=integer_bits + fractional_bits,
        scheme=QuantizationScheme.SYMMETRIC,
        fixed_point=FixedPointSpec(
            integer_bits=integer_bits,
            fractional_bits=fractional_bits,
        ),
    )


def _make_graph(output_shape: list[int]) -> Graph:
    input_value = Value("input", ValueType.TENSOR, "float32", [1, 2], ["N", "C"])
    output_value = Value(
        "output",
        ValueType.TENSOR,
        "float32",
        output_shape,
        ["N", "C"] if len(output_shape) == 2 else ["N"],
        producer_op_id="op0",
    )
    op = MockOp("op0", ["input"], ["output"])
    return Graph(
        {"input": input_value, "output": output_value},
        {"op0": op},
        ["input"],
        ["output"],
    )


def _make_tensorflow_model():
    tf = pytest.importorskip("tensorflow")
    inputs = tf.keras.Input(shape=(2,), name="input")
    hidden = tf.keras.layers.Dense(
        3,
        activation="relu",
        name="dense_hidden",
        kernel_initializer=tf.keras.initializers.Constant(
            [[0.5, -0.25, 0.125], [0.75, 0.5, -0.5]]
        ),
        bias_initializer=tf.keras.initializers.Constant([0.1, -0.2, 0.05]),
    )(inputs)
    outputs = tf.keras.layers.Dense(
        1,
        name="dense_output",
        kernel_initializer=tf.keras.initializers.Constant([[0.25], [-0.75], [0.5]]),
        bias_initializer=tf.keras.initializers.Constant([0.125]),
    )(hidden)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_numerical_parity_identical_models_have_zero_error() -> None:
    model = TinyParityModel().eval()

    def dataset():
        yield from (
            torch.tensor([0.2, -0.1], dtype=torch.float32),
            torch.tensor([1.0, 0.5], dtype=torch.float32),
            torch.tensor([-0.75, 0.25], dtype=torch.float32),
        )

    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=model,
        dataset=dataset(),
        config={
            "metrics": ["mae", "max_error", "relative_error"],
            "thresholds": {
                "mae": 0.0,
                "max_error": 0.0,
                "relative_error": 0.0,
            },
        },
    )

    assert result["pass"] is True
    assert result["violations"] == []
    assert result["metrics"]["global"]["mae"] == 0.0
    assert result["metrics"]["global"]["max_error"] == 0.0
    assert result["metrics"]["global"]["relative_error"] == 0.0
    assert "fc1" in result["metrics"]["layers"]
    assert result["diagnostics"]["top_k_worst_samples"][0]["score"] == 0.0


def test_numerical_parity_detects_quantization_noise() -> None:
    model = TinyParityModel().eval()
    simulator = TorchQuantizedModelSimulator(
        model,
        activation_spec=_fixed_point_spec(4, 4),
        weight_spec=_fixed_point_spec(4, 4),
        input_spec=_fixed_point_spec(4, 4),
        output_spec=_fixed_point_spec(4, 4),
    )
    dataset = [
        torch.tensor([0.2, -0.1], dtype=torch.float32),
        torch.tensor([0.7, 0.6], dtype=torch.float32),
        torch.tensor([-0.9, 0.4], dtype=torch.float32),
    ]

    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=simulator,
        dataset=dataset,
        config={
            "metrics": ["mae", "max_error", "relative_error", "sqnr"],
            "thresholds": {"max_error": 0.01},
        },
    )

    assert result["pass"] is False
    assert result["metrics"]["global"]["mae"] > 0.0
    assert result["metrics"]["global"]["max_error"] > 0.01
    assert result["metrics"]["global"]["max_error"] < 0.2
    assert result["diagnostics"]["highest_deviation_layer"] is not None
    assert result["diagnostics"]["quantization_reports"]


def test_numerical_parity_reports_clipping_diagnostics() -> None:
    model = TinyParityModel().eval()
    simulator = TorchQuantizedModelSimulator(
        model,
        activation_spec=_fixed_point_spec(2, 2),
        weight_spec=_fixed_point_spec(2, 2),
        input_spec=_fixed_point_spec(2, 2),
        output_spec=_fixed_point_spec(2, 2),
    )
    dataset = [torch.tensor([8.0, -8.0], dtype=torch.float32)]

    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=simulator,
        dataset=dataset,
        config={"metrics": ["max_error"], "thresholds": {"max_error": 0.4}},
    )

    assert result["pass"] is False
    assert result["diagnostics"]["quantization_reports"][0]["total_clipped_values"] > 0
    assert result["metrics"]["global"]["max_error"] > 0.4


def test_numerical_parity_handles_constant_and_nonfinite_outputs() -> None:
    constant_model = ConstantModel().eval()
    constant_result = run_numerical_parity_test(
        fp32_model=constant_model,
        quantized_model=constant_model,
        dataset=[torch.tensor([1.0], dtype=torch.float32)],
        config={"metrics": ["relative_error"], "thresholds": {"relative_error": 0.0}},
    )

    assert constant_result["pass"] is True
    assert constant_result["metrics"]["global"]["relative_error"] == 0.0

    nonfinite_result = run_numerical_parity_test(
        fp32_model=constant_model,
        quantized_model=NonFiniteModel().eval(),
        dataset=[torch.tensor([1.0], dtype=torch.float32)],
        config={"metrics": ["max_error"], "thresholds": {"max_error": 0.0}},
    )

    assert nonfinite_result["pass"] is False
    assert any(
        violation["metric"] == "nonfinite_count"
        for violation in nonfinite_result["violations"]
    )


def test_numerical_parity_surfaces_ir_mismatches() -> None:
    model = TinyParityModel().eval()
    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=model,
        dataset=[torch.tensor([0.1, 0.2], dtype=torch.float32)],
        config={
            "fp32_ir": _make_graph([1, 1]),
            "quantized_ir": _make_graph([1, 2]),
        },
    )

    assert result["pass"] is False
    assert result["diagnostics"]["ir"]["pass"] is False
    assert any(violation["scope"] == "ir" for violation in result["violations"])


def test_onnx_runtime_adapter_supports_identical_models_and_layer_capture() -> None:
    adapter = ONNXRuntimeParityAdapter(
        FakeONNXSession(),
        layer_output_names=("hidden",),
    )
    dataset = [
        torch.tensor([0.1, 0.2], dtype=torch.float32),
        torch.tensor([-0.5, 0.3], dtype=torch.float32),
    ]

    result = run_numerical_parity_test(
        fp32_model=adapter,
        quantized_model=adapter,
        dataset=dataset,
        config={
            "metrics": ["mae", "max_error"],
            "thresholds": {"mae": 0.0, "max_error": 0.0},
        },
    )

    assert result["pass"] is True
    assert result["metrics"]["global"]["mae"] == 0.0
    assert "hidden" in result["metrics"]["layers"]
    assert "output" in result["metrics"]["outputs"]


def test_onnx_runtime_adapter_detects_output_and_layer_drift() -> None:
    fp32_adapter = ONNXRuntimeParityAdapter(
        FakeONNXSession(),
        layer_output_names=("hidden",),
    )
    quantized_adapter = ONNXRuntimeParityAdapter(
        FakeONNXSession(output_shift=0.2, hidden_shift=0.1),
        layer_output_names=("hidden",),
    )

    result = run_numerical_parity_test(
        fp32_model=fp32_adapter,
        quantized_model=quantized_adapter,
        dataset=[{"input": torch.tensor([0.3, -0.4], dtype=torch.float32)}],
        config={
            "metrics": ["max_error"],
            "thresholds": {"max_error": 0.05},
        },
    )

    assert result["pass"] is False
    assert result["diagnostics"]["highest_deviation_layer"] == "hidden"
    assert any(violation["scope"] == "layer" for violation in result["violations"])


def test_onnx_runtime_adapter_validates_input_arity() -> None:
    adapter = ONNXRuntimeParityAdapter(FakeONNXSession())

    with pytest.raises(ValueError, match="expected 1 inputs"):
        adapter.parity_forward(
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
            capture_layers=False,
        )


def test_tensorflow_keras_adapter_supports_identical_models_and_layer_capture() -> None:
    tf = pytest.importorskip("tensorflow")
    model = _make_tensorflow_model()
    adapter = TensorFlowKerasParityAdapter(
        model,
        default_layer_names=("dense_hidden", "dense_output"),
    )
    dataset = [
        tf.constant([[0.2, -0.1]], dtype=tf.float32),
        tf.constant([[1.0, 0.5]], dtype=tf.float32),
    ]

    result = run_numerical_parity_test(
        fp32_model=adapter,
        quantized_model=adapter,
        dataset=dataset,
        config={
            "metrics": ["mae", "max_error"],
            "thresholds": {"mae": 0.0, "max_error": 0.0},
        },
    )

    assert result["pass"] is True
    assert result["metrics"]["global"]["max_error"] == 0.0
    assert "dense_hidden" in result["metrics"]["layers"]
    assert "dense_output" in result["metrics"]["layers"]


def test_tensorflow_keras_adapter_detects_weight_noise() -> None:
    tf = pytest.importorskip("tensorflow")
    fp32_model = _make_tensorflow_model()
    quantized_model = tf.keras.models.clone_model(fp32_model)
    quantized_model.set_weights(fp32_model.get_weights())

    kernel, bias = quantized_model.get_layer("dense_output").get_weights()
    quantized_model.get_layer("dense_output").set_weights([kernel + 0.15, bias + 0.05])

    fp32_adapter = TensorFlowKerasParityAdapter(fp32_model)
    quantized_adapter = TensorFlowKerasParityAdapter(
        quantized_model,
        default_layer_names=("dense_hidden", "dense_output"),
    )
    dataset = [tf.constant([[0.25, 0.75]], dtype=tf.float32)]

    result = run_numerical_parity_test(
        fp32_model=fp32_adapter,
        quantized_model=quantized_adapter,
        dataset=dataset,
        config={
            "metrics": ["max_error", "mae"],
            "thresholds": {"max_error": 0.01},
        },
    )

    assert result["pass"] is False
    assert result["metrics"]["global"]["max_error"] > 0.01
    assert result["diagnostics"]["highest_deviation_layer"] == "dense_output"


def test_tensorflow_keras_adapter_respects_explicit_layer_selection() -> None:
    tf = pytest.importorskip("tensorflow")
    adapter = TensorFlowKerasParityAdapter(_make_tensorflow_model())

    output_map, layer_map = adapter.parity_forward(
        tf.constant([[0.2, -0.1]], dtype=tf.float32),
        capture_layers=True,
        layer_names=("dense_hidden",),
    )

    assert "output" in output_map
    assert "dense_hidden" in layer_map
    assert "dense_output" not in layer_map
