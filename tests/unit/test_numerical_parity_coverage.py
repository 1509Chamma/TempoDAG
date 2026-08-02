"""Coverage-focused tests for tempo_dag.numerical_parity.

These tests deterministically exercise config coercion errors, adapter edge
cases (via stub tensorflow/onnxruntime modules), quantization branches, IR
comparison violations, and the private normalization/adaptation helpers.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
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
    _adapt_sample,
    _metric_float,
    _MetricAccumulator,
    _normalize_output_structure,
    _numpy_to_like,
    _quantize_value_like,
    compare_ir_graphs,
    quantize_array,
    run_numerical_parity_test,
)
from tempo_dag.quantization_config import (
    FixedPointSpec,
    QuantizationScheme,
    QuantizationSpec,
    QuantizationType,
    StateQuantSpec,
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _fp_spec(integer_bits: int, fractional_bits: int) -> QuantizationSpec:
    return QuantizationSpec(
        bit_width=integer_bits + fractional_bits,
        scheme=QuantizationScheme.SYMMETRIC,
        fixed_point=FixedPointSpec(
            integer_bits=integer_bits,
            fractional_bits=fractional_bits,
        ),
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


class MockOpAlt(MockOp):
    OP_TYPE = "MockAlt"


def _make_graph(
    *,
    graph_inputs: tuple[str, ...] = ("input",),
    graph_outputs: tuple[str, ...] = ("output",),
    output_dtype: str = "float32",
    output_shape: tuple[int, ...] = (1, 1),
    extra_value: bool = False,
    op_cls: type[MockOp] = MockOp,
) -> Graph:
    values = {
        "input": Value("input", ValueType.TENSOR, "float32", [1, 2], ["N", "C"]),
        "output": Value(
            "output",
            ValueType.TENSOR,
            output_dtype,
            list(output_shape),
            ["N", "C"],
            producer_op_id="op0",
        ),
    }
    if extra_value:
        values["extra"] = Value("extra", ValueType.TENSOR, "float32", [1], ["N"])
    op = op_cls("op0", ["input"], ["output"])
    return Graph(values, {"op0": op}, list(graph_inputs), list(graph_outputs))


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor([[0.5, -0.25], [0.75, 0.5], [-0.5, 0.125]])
            )
            self.fc1.bias.copy_(torch.tensor([0.1, -0.2, 0.05]))
            self.fc2.weight.copy_(torch.tensor([[0.25, -0.75, 0.5]]))
            self.fc2.bias.copy_(torch.tensor([0.125]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class NestedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        with torch.no_grad():
            self.block[0].weight.copy_(torch.tensor([[0.5, -0.5], [0.25, 0.75]]))
            self.block[0].bias.copy_(torch.tensor([0.0, 0.1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BufferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)
        self.register_buffer("float_buf", torch.tensor([0.25, -0.5]))
        self.register_buffer("int_buf", torch.tensor([1, 2]))
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[0.5, 0.25], [-0.25, 0.5]]))
            self.fc.bias.copy_(torch.tensor([0.0, 0.125]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.float_buf


class _FakeValueInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeONNXSession:
    def __init__(self, output_names: tuple[str, ...] = ("output",)) -> None:
        self._output_names = output_names

    def get_inputs(self) -> list[_FakeValueInfo]:
        return [_FakeValueInfo("input")]

    def get_outputs(self) -> list[_FakeValueInfo]:
        return [_FakeValueInfo(name) for name in self._output_names]

    def run(
        self,
        output_names: list[str],
        feeds: dict[str, object],
    ) -> list[object]:
        x = np.asarray(feeds["input"], dtype=np.float64)
        tensors = {"output": x * 1.5, "second": x - 1.0, "hidden": x + 0.5}
        return [tensors[name] for name in output_names]


class _FakeKerasModel:
    """Stands in for tf.keras.Model inside a stub tensorflow module."""

    def __init__(self, inputs=None, outputs=None) -> None:
        self.inputs = list(inputs or [])
        self.outputs = list(outputs or [])
        self.layers: list[object] = []

    def __call__(self, *args, training: bool = False, **kwargs):
        del training, kwargs
        x = np.asarray(args[0], dtype=np.float64)
        results = [fn(x) for fn in self.outputs]
        if len(results) == 1:
            return results[0]
        return results


class InputLayer:
    """Class name is significant: default layer resolution filters on it."""

    name = "input_stub"


class _FakeDenseLayer:
    def __init__(self, name: str, fn) -> None:
        self.name = name
        self.output = fn


def _double(x: np.ndarray) -> np.ndarray:
    return x * 2.0


def _shift(x: np.ndarray) -> np.ndarray:
    return x + 1.0


class _FakeFunctionalModel(_FakeKerasModel):
    def __init__(self) -> None:
        super().__init__(inputs=["x"])
        self._fns = {"double": _double, "shift": _shift}
        self.layers = [
            InputLayer(),
            _FakeDenseLayer("double", _double),
            _FakeDenseLayer("shift", _shift),
        ]

    def __call__(self, *args, training: bool = False, **kwargs):
        del training, kwargs
        return np.asarray(args[0], dtype=np.float64) * 2.0 + 1.0

    def get_layer(self, name: str) -> _FakeDenseLayer:
        try:
            fn = self._fns[name]
        except KeyError as exc:
            raise ValueError(f"No such layer: {name}") from exc
        return _FakeDenseLayer(name, fn)


class _FakeSubclassedModel(_FakeKerasModel):
    """No symbolic inputs, mimicking a subclassed (non-functional) model."""

    def __call__(self, *args, training: bool = False, **kwargs):
        del training, kwargs
        return np.asarray(args[0], dtype=np.float64) + 3.0


@pytest.fixture()
def fake_tf(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake = types.ModuleType("tensorflow")
    fake.keras = types.SimpleNamespace(Model=_FakeKerasModel)
    monkeypatch.setitem(sys.modules, "tensorflow", fake)
    return fake


class _LayerMapModel:
    """Minimal parity_forward adapter returning fixed output/layer maps."""

    def __init__(self, output_value, layer_map) -> None:
        self._output = np.asarray(output_value, dtype=np.float64)
        self._layers = {
            key: np.asarray(value, dtype=np.float64) for key, value in layer_map.items()
        }

    def parity_forward(self, *args, capture_layers=True, layer_names=None, **kwargs):
        del args, kwargs, capture_layers, layer_names
        return {"output": self._output}, dict(self._layers)


def _identity_model(x):
    return np.asarray(x, dtype=np.float64)


def _drift_model(x):
    return np.asarray(x, dtype=np.float64) * 1.1


# ---------------------------------------------------------------------------
# Config coercion
# ---------------------------------------------------------------------------


def test_config_from_input_none_and_instance_passthrough() -> None:
    default = NumericalParityConfig.from_input(None)
    assert isinstance(default, NumericalParityConfig)
    assert default.top_k_worst == 5

    instance = NumericalParityConfig()
    assert NumericalParityConfig.from_input(instance) is instance


def test_config_from_input_coerces_optional_fields() -> None:
    config = NumericalParityConfig.from_input(
        {
            "metrics": ["mae", "mse"],
            "layer_names": ["fc1", "fc2"],
            "top_k_worst": 3,
            "histogram_bins": [0.0, 0.5, 1.0],
            "state_quantization": {
                "hidden": StateQuantSpec(dtype="fixed16", scale=2**-8),
                "window": {"dtype": "fixed24", "scale": 2**-12},
            },
        }
    )

    assert config.metrics == ("mae", "mse")
    assert config.layer_names == ("fc1", "fc2")
    assert config.top_k_worst == 3
    assert config.histogram_bins == (0.0, 0.5, 1.0)
    assert config.state_quantization["hidden"].dtype == "fixed16"
    assert config.state_quantization["window"].dtype == "fixed24"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"metrics": 5}, "metrics"),
        ({"layer_names": 5}, "layer_names"),
        ({"state_quantization": "bad"}, "state_quantization"),
        ({"state_quantization": {1: {}}}, "keys must be strings"),
        ({"state_quantization": {"h": 42}}, "StateQuantSpec"),
        ({"thresholds": "bad"}, "thresholds"),
        ({"relative_error_epsilon": "bad"}, "numeric"),
        ({"top_k_worst": "bad"}, "integer"),
        ({"histogram_bins": "bad"}, "histogram_bins"),
    ],
)
def test_config_from_input_rejects_invalid_values(payload: dict, match: str) -> None:
    with pytest.raises(TypeError, match=match):
        NumericalParityConfig.from_input(payload)


# ---------------------------------------------------------------------------
# Metric accumulator and metric helpers
# ---------------------------------------------------------------------------


def test_metric_accumulator_rejects_shape_mismatch() -> None:
    accumulator = _MetricAccumulator(
        metrics=("mae",),
        relative_error_epsilon=1e-8,
        histogram_bins=None,
    )
    with pytest.raises(ValueError, match="identical shapes"):
        accumulator.update(np.zeros(2), np.zeros(3))


def test_metric_float_returns_zero_for_non_numeric() -> None:
    assert _metric_float({"metric": "not-a-number"}, "metric") == 0.0
    assert _metric_float({}, "missing") == 0.0


def test_histogram_metrics_accumulate_across_samples() -> None:
    dataset = [np.array([0.0, 1.0]), np.array([2.0, 3.0])]
    result = run_numerical_parity_test(
        fp32_model=_identity_model,
        quantized_model=lambda x: np.asarray(x, dtype=np.float64) + 0.05,
        dataset=dataset,
        config={"histogram_bins": 4},
    )

    histogram = result["metrics"]["global"]["abs_error_histogram"]
    assert len(histogram["bins"]) == 5
    assert sum(histogram["counts"]) == 4


def test_empty_dataset_produces_zeroed_metrics() -> None:
    result = run_numerical_parity_test(
        fp32_model=_identity_model,
        quantized_model=_identity_model,
        dataset=[],
        config=None,
    )

    assert result["pass"] is True
    assert result["diagnostics"]["sample_count"] == 0
    assert result["metrics"]["global"]["mae"] == 0.0
    assert result["metrics"]["global"]["sqnr"] == math.inf


def test_zero_signal_with_noise_yields_negative_infinite_sqnr() -> None:
    result = run_numerical_parity_test(
        fp32_model=lambda x: np.zeros(3),
        quantized_model=lambda x: np.ones(3),
        dataset=[np.array([1.0])],
        config={"fail_on_nonfinite": False},
    )

    assert result["metrics"]["global"]["sqnr"] == -math.inf


def test_nonfinite_candidate_output_is_flagged() -> None:
    result = run_numerical_parity_test(
        fp32_model=lambda x: np.zeros(1),
        quantized_model=lambda x: np.array([np.nan]),
        dataset=[np.array([1.0])],
        config=None,
    )

    assert result["pass"] is False
    assert any(v["metric"] == "nonfinite_count" for v in result["violations"])


# ---------------------------------------------------------------------------
# Torch simulator
# ---------------------------------------------------------------------------


def test_simulator_rejects_non_torch_module() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        TorchQuantizedModelSimulator("not-a-module", activation_spec=_fp_spec(4, 4))


def test_simulator_quantizes_floating_point_buffers() -> None:
    model = BufferModel().eval()
    simulator = TorchQuantizedModelSimulator(
        model,
        activation_spec=_fp_spec(4, 4),
        weight_spec=_fp_spec(4, 4),
    )

    quantized_buf = dict(simulator._module.named_buffers())["float_buf"]
    assert torch.allclose(quantized_buf, torch.tensor([0.25, -0.5]), atol=2**-4)
    int_buf = dict(simulator._module.named_buffers())["int_buf"]
    assert torch.equal(int_buf, torch.tensor([1, 2]))


def test_simulator_quantizes_keyword_arguments_and_reports() -> None:
    model = TinyModel().eval()
    simulator = TorchQuantizedModelSimulator(
        model,
        activation_spec=_fp_spec(3, 3),
        layer_specs={"fc1": _fp_spec(4, 4)},
    )

    output = simulator(x=torch.tensor([[0.5, -0.25]], dtype=torch.float32))
    assert output.shape == (1, 1)

    report = simulator.consume_last_quantization_report()
    assert "fc1" in report["layers"]
    follow_up = simulator.consume_last_quantization_report()
    assert follow_up["total_clipped_values"] == 0


def test_simulator_train_eval_roundtrip() -> None:
    simulator = TorchQuantizedModelSimulator(
        TinyModel(),
        activation_spec=_fp_spec(4, 4),
    )
    assert simulator.train() is simulator
    assert simulator.training is True
    assert simulator.eval() is simulator
    assert simulator.training is False


def test_simulator_parity_run_detects_quantization_noise() -> None:
    model = TinyModel().eval()
    simulator = TorchQuantizedModelSimulator(
        model,
        activation_spec=_fp_spec(3, 3),
        weight_spec=_fp_spec(3, 3),
    )
    dataset = [
        torch.tensor([0.2, -0.1], dtype=torch.float32),
        torch.tensor([0.7, 0.6], dtype=torch.float32),
    ]

    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=simulator,
        dataset=dataset,
        config={"thresholds": {"max_error": 1e-9}},
    )

    assert result["pass"] is False
    assert result["metrics"]["global"]["mae"] > 0.0
    assert result["diagnostics"]["quantization_reports"]
    assert result["diagnostics"]["highest_deviation_layer"] is not None


def test_layer_name_selection_limits_captured_layers() -> None:
    model = TinyModel().eval()
    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=model,
        dataset=[torch.tensor([0.1, 0.2], dtype=torch.float32)],
        config={"layer_names": ["fc1"]},
    )

    assert set(result["metrics"]["layers"]) == {"fc1"}


def test_nested_module_layer_capture_skips_containers() -> None:
    model = NestedModel().eval()
    result = run_numerical_parity_test(
        fp32_model=model,
        quantized_model=model,
        dataset=[torch.tensor([0.3, -0.2], dtype=torch.float32)],
        config=None,
    )

    layer_names = set(result["metrics"]["layers"])
    assert "block.0" in layer_names
    assert "block" not in layer_names


# ---------------------------------------------------------------------------
# ONNX Runtime adapter
# ---------------------------------------------------------------------------


def test_onnx_adapter_call_returns_single_output_array() -> None:
    adapter = ONNXRuntimeParityAdapter(FakeONNXSession())
    result = adapter(np.array([1.0, 2.0]))
    assert np.allclose(result, [1.5, 3.0])


def test_onnx_adapter_call_returns_map_for_multiple_outputs() -> None:
    adapter = ONNXRuntimeParityAdapter(FakeONNXSession(("output", "second")))
    result = adapter(np.array([1.0]))
    assert set(result) == {"output", "second"}
    assert np.allclose(result["second"], [0.0])


def test_onnx_adapter_accepts_mapping_and_keyword_feeds() -> None:
    adapter = ONNXRuntimeParityAdapter(FakeONNXSession())

    mapped, _ = adapter.parity_forward({"input": np.array([2.0])}, capture_layers=False)
    assert np.allclose(mapped["output"], [3.0])

    keyed, _ = adapter.parity_forward(input=np.array([2.0]), capture_layers=False)
    assert np.allclose(keyed["output"], [3.0])


def test_onnx_adapter_captures_requested_layer_outputs() -> None:
    adapter = ONNXRuntimeParityAdapter(
        FakeONNXSession(),
        layer_output_names=("hidden",),
    )
    outputs, layers = adapter.parity_forward(np.array([1.0]))
    assert np.allclose(outputs["output"], [1.5])
    assert np.allclose(layers["hidden"], [1.5])


def test_onnx_adapter_validates_positional_arity() -> None:
    adapter = ONNXRuntimeParityAdapter(FakeONNXSession())
    with pytest.raises(ValueError, match="expected 1 inputs"):
        adapter.parity_forward(np.array([1.0]), np.array([2.0]), capture_layers=False)


def test_onnx_adapter_mode_helpers_are_inert() -> None:
    adapter = ONNXRuntimeParityAdapter(FakeONNXSession())
    assert adapter.training is False
    assert adapter.eval() is adapter
    assert adapter.train() is adapter


def test_onnx_adapter_resolves_session_from_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda path: FakeONNXSession()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    adapter = ONNXRuntimeParityAdapter("model.onnx")
    assert np.allclose(adapter(np.array([2.0])), [3.0])


def test_onnx_adapter_rejects_invalid_session_object() -> None:
    with pytest.raises(TypeError, match="ONNX Runtime session"):
        ONNXRuntimeParityAdapter(12345)


# ---------------------------------------------------------------------------
# TensorFlow adapter (stubbed tensorflow module)
# ---------------------------------------------------------------------------


def test_tf_adapter_rejects_non_keras_model(fake_tf: types.ModuleType) -> None:
    with pytest.raises(TypeError, match="tf.keras.Model"):
        TensorFlowKerasParityAdapter(object())


def test_tf_adapter_call_train_eval_roundtrip(fake_tf: types.ModuleType) -> None:
    adapter = TensorFlowKerasParityAdapter(_FakeFunctionalModel())
    assert adapter.training is False
    assert adapter.train() is adapter
    assert adapter.training is True
    assert adapter.eval() is adapter
    assert adapter.training is False

    result = adapter(np.array([1.0, 2.0]))
    assert np.allclose(result, [3.0, 5.0])


def test_tf_adapter_parity_forward_capture_disabled(
    fake_tf: types.ModuleType,
) -> None:
    adapter = TensorFlowKerasParityAdapter(_FakeFunctionalModel())
    outputs, layers = adapter.parity_forward(np.array([1.0]), capture_layers=False)
    assert np.allclose(outputs["output"], [3.0])
    assert layers == {}


def test_tf_adapter_parity_forward_empty_layer_selection(
    fake_tf: types.ModuleType,
) -> None:
    adapter = TensorFlowKerasParityAdapter(_FakeFunctionalModel())
    outputs, layers = adapter.parity_forward(np.array([1.0]), layer_names=())
    assert np.allclose(outputs["output"], [3.0])
    assert layers == {}


def test_tf_adapter_captures_single_layer_and_caches_model(
    fake_tf: types.ModuleType,
) -> None:
    adapter = TensorFlowKerasParityAdapter(_FakeFunctionalModel())

    outputs, layers = adapter.parity_forward(np.array([2.0]), layer_names=("double",))
    assert np.allclose(outputs["output"], [5.0])
    assert np.allclose(layers["double"], [4.0])

    _, cached_layers = adapter.parity_forward(np.array([3.0]), layer_names=("double",))
    assert np.allclose(cached_layers["double"], [6.0])
    assert ("double",) in adapter._capture_model_cache


def test_tf_adapter_default_layer_resolution_captures_all(
    fake_tf: types.ModuleType,
) -> None:
    adapter = TensorFlowKerasParityAdapter(_FakeFunctionalModel())
    _, layers = adapter.parity_forward(np.array([1.0]))
    assert set(layers) == {"double", "shift"}


def test_tf_adapter_respects_default_layer_names(fake_tf: types.ModuleType) -> None:
    adapter = TensorFlowKerasParityAdapter(
        _FakeFunctionalModel(),
        default_layer_names=("shift",),
    )
    _, layers = adapter.parity_forward(np.array([1.0]))
    assert set(layers) == {"shift"}


def test_tf_adapter_subclassed_model_skips_layer_capture(
    fake_tf: types.ModuleType,
) -> None:
    adapter = TensorFlowKerasParityAdapter(_FakeSubclassedModel())
    outputs, layers = adapter.parity_forward(np.array([1.0]), layer_names=("double",))
    assert np.allclose(outputs["output"], [4.0])
    assert layers == {}
    assert adapter._capture_model_cache[("double",)] is None


def test_tf_adapter_unknown_layer_disables_capture(
    fake_tf: types.ModuleType,
) -> None:
    adapter = TensorFlowKerasParityAdapter(_FakeFunctionalModel())
    outputs, layers = adapter.parity_forward(np.array([1.0]), layer_names=("missing",))
    assert np.allclose(outputs["output"], [3.0])
    assert layers == {}
    assert adapter._capture_model_cache[("missing",)] is None


# ---------------------------------------------------------------------------
# quantize_array branches
# ---------------------------------------------------------------------------


def test_quantize_array_empty_input_short_circuits() -> None:
    result = quantize_array(np.array([]), _fp_spec(4, 4))
    assert result.dequantized.size == 0
    assert result.clipped_values == 0
    assert result.scale == 1.0
    assert result.zero_point == 0


def test_quantize_array_requires_fixed_point_spec() -> None:
    spec = QuantizationSpec(bit_width=8, scheme=QuantizationScheme.SYMMETRIC)
    with pytest.raises(ValueError, match="fixed_point"):
        quantize_array(np.array([1.0]), spec)


def test_quantize_array_fixed_point_with_explicit_scale() -> None:
    spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        fixed_point=FixedPointSpec(integer_bits=4, fractional_bits=4),
        scale=0.25,
        zero_point=0,
    )
    result = quantize_array(np.array([0.5, -0.75]), spec)
    assert np.allclose(result.dequantized, [0.5, -0.75])
    assert result.scale == 0.25


def test_quantize_array_integer_symmetric_roundtrip() -> None:
    spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.SYMMETRIC,
        qtype=QuantizationType.INTEGER,
    )
    values = np.array([-1.0, 0.5, 1.0])
    result = quantize_array(values, spec)
    assert result.zero_point == 0
    assert np.allclose(result.dequantized, values, atol=result.scale)


def test_quantize_array_integer_asymmetric_roundtrip() -> None:
    spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.ASYMMETRIC,
        qtype=QuantizationType.INTEGER,
    )
    values = np.array([0.0, 0.5, 1.0])
    result = quantize_array(values, spec)
    assert np.allclose(result.dequantized, values, atol=result.scale)


def test_quantize_array_integer_with_explicit_parameters() -> None:
    spec = QuantizationSpec(
        bit_width=8,
        scheme=QuantizationScheme.ASYMMETRIC,
        qtype=QuantizationType.INTEGER,
        scale=0.5,
        zero_point=10,
    )
    result = quantize_array(np.array([1.0]), spec)
    assert result.scale == 0.5
    assert result.zero_point == 10


# ---------------------------------------------------------------------------
# IR comparison
# ---------------------------------------------------------------------------


def test_compare_ir_graphs_skips_when_either_graph_missing() -> None:
    report = compare_ir_graphs(None, None)
    assert report["pass"] is True
    assert report["summary"] == "IR comparison skipped."


def test_compare_ir_graphs_reports_structural_violations() -> None:
    fp32 = _make_graph()
    quantized = _make_graph(
        graph_inputs=("other_input",),
        graph_outputs=("other_output",),
        output_dtype="float16",
        output_shape=(1, 2),
        extra_value=True,
        op_cls=MockOpAlt,
    )

    report = compare_ir_graphs(fp32, quantized)
    assert report["pass"] is False
    metrics = {violation["metric"] for violation in report["violations"]}
    assert metrics == {
        "graph_inputs",
        "graph_outputs",
        "operator_type",
        "value_presence",
        "shape",
        "dtype",
    }


def test_compare_ir_graphs_detects_missing_operator() -> None:
    fp32 = _make_graph()
    quantized = Graph(dict(fp32.values), {}, ["input"], ["output"])

    report = compare_ir_graphs(fp32, quantized)
    op_violations = [
        violation
        for violation in report["violations"]
        if violation["metric"] == "operator_type"
    ]
    assert op_violations and op_violations[0]["item"] == "op0"


# ---------------------------------------------------------------------------
# run_numerical_parity_test control flow
# ---------------------------------------------------------------------------


def test_top_k_heap_replaces_lower_scores() -> None:
    dataset = [np.array([1.0]), np.array([2.0]), np.array([0.5])]
    result = run_numerical_parity_test(
        fp32_model=_identity_model,
        quantized_model=_drift_model,
        dataset=dataset,
        config={"top_k_worst": 1},
    )

    worst = result["diagnostics"]["top_k_worst_samples"]
    assert len(worst) == 1
    assert worst[0]["sample_index"] == 1


def test_missing_and_unexpected_layer_names_are_violations() -> None:
    fp32 = _LayerMapModel([1.0], {"only_ref": [1.0]})
    quantized = _LayerMapModel([1.0], {"only_cand": [1.0]})

    result = run_numerical_parity_test(
        fp32_model=fp32,
        quantized_model=quantized,
        dataset=[np.array([1.0])],
        config=None,
    )

    assert result["pass"] is False
    metrics = {violation["metric"] for violation in result["violations"]}
    assert {"missing_candidate", "unexpected_candidate"} <= metrics
    assert set(result["diagnostics"]["failing_layers"]) == {"only_ref", "only_cand"}
    assert result["diagnostics"]["failing_samples"] == [0]


def test_layer_shape_mismatch_is_reported_not_raised() -> None:
    fp32 = _LayerMapModel([1.0], {"L": [1.0, 2.0]})
    quantized = _LayerMapModel([1.0], {"L": [1.0, 2.0, 3.0]})

    result = run_numerical_parity_test(
        fp32_model=fp32,
        quantized_model=quantized,
        dataset=[np.array([1.0])],
        config=None,
    )

    assert result["pass"] is False
    shape_violations = [
        violation
        for violation in result["violations"]
        if violation["metric"] == "shape_mismatch"
    ]
    assert shape_violations and shape_violations[0]["scope"] == "layer"
    assert result["diagnostics"]["failing_layers"] == ["L"]


def test_nonfinite_layer_output_marks_failing_layer() -> None:
    fp32 = _LayerMapModel([1.0], {"L": [1.0]})
    quantized = _LayerMapModel([1.0], {"L": [np.nan]})

    result = run_numerical_parity_test(
        fp32_model=fp32,
        quantized_model=quantized,
        dataset=[np.array([1.0])],
        config=None,
    )

    assert result["pass"] is False
    assert result["diagnostics"]["failing_layers"] == ["L"]
    assert any(
        violation["metric"] == "nonfinite_count" and violation["scope"] == "layer"
        for violation in result["violations"]
    )


def test_sample_adapter_unpacks_args_and_kwargs() -> None:
    def adapter(sample):
        return ((np.asarray(sample, dtype=np.float64) * 2.0,), {})

    result = run_numerical_parity_test(
        fp32_model=_identity_model,
        quantized_model=_identity_model,
        dataset=[np.array([0.5])],
        config={"sample_adapter": adapter},
    )
    assert result["pass"] is True


def test_sample_adapter_with_invalid_return_raises() -> None:
    def bad_adapter(sample):
        return np.asarray(sample)

    with pytest.raises(ValueError, match="sample_adapter"):
        run_numerical_parity_test(
            fp32_model=_identity_model,
            quantized_model=_identity_model,
            dataset=[np.array([0.5])],
            config={"sample_adapter": bad_adapter},
        )


def test_tuple_and_mapping_samples_are_unpacked() -> None:
    def two_arg_model(x, y):
        return np.asarray(x, dtype=np.float64) + np.asarray(y, dtype=np.float64)

    result = run_numerical_parity_test(
        fp32_model=two_arg_model,
        quantized_model=two_arg_model,
        dataset=[(np.array([1.0]), np.array([2.0]))],
        config=None,
    )
    assert result["pass"] is True

    def kwarg_model(x):
        return np.asarray(x, dtype=np.float64)

    result = run_numerical_parity_test(
        fp32_model=kwarg_model,
        quantized_model=kwarg_model,
        dataset=[{"x": np.array([1.0])}],
        config=None,
    )
    assert result["pass"] is True


def test_ir_graphs_flow_through_parity_run() -> None:
    result = run_numerical_parity_test(
        fp32_model=_identity_model,
        quantized_model=_identity_model,
        dataset=[np.array([1.0])],
        config={
            "fp32_ir": _make_graph(),
            "quantized_ir": _make_graph(output_dtype="float16"),
        },
    )

    assert result["pass"] is False
    assert result["diagnostics"]["ir"]["pass"] is False


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def test_normalize_output_structure_handles_nested_containers() -> None:
    arr = np.array([1.0])

    mapping = _normalize_output_structure({"a": arr, "b": {"c": arr}})
    assert set(mapping) == {"a", "b.c"}

    tupled = _normalize_output_structure((arr, {"k": arr}))
    assert set(tupled) == {"output.0", "output.1.k"}

    listed = _normalize_output_structure([arr, (arr,)])
    assert set(listed) == {"output.0", "output.1.output.0"}

    scalar = _normalize_output_structure(2.5)
    assert np.allclose(scalar["output"], 2.5)


def test_quantize_value_like_recurses_through_containers() -> None:
    spec = _fp_spec(4, 4)

    tupled, clipped = _quantize_value_like(
        (np.array([0.5]), [np.array([0.25])]),
        spec,
    )
    assert isinstance(tupled, tuple)
    assert isinstance(tupled[1], list)
    assert np.allclose(tupled[0], [0.5])
    assert np.allclose(tupled[1][0], [0.25])
    assert clipped == 0

    mapped, map_clipped = _quantize_value_like({"a": np.array([0.5])}, spec)
    assert np.allclose(mapped["a"], [0.5])
    assert map_clipped == 0

    tensor_out, _ = _quantize_value_like(torch.tensor([0.5], dtype=torch.float32), spec)
    assert isinstance(tensor_out, torch.Tensor)

    array_out, _ = _quantize_value_like(np.array([0.5]), spec)
    assert isinstance(array_out, np.ndarray)


def test_numpy_to_like_without_torch_reference_returns_array() -> None:
    result = _numpy_to_like(np.array([1.0, 2.0]), np.array([0.0]))
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, [1.0, 2.0])


def test_adapt_sample_variants() -> None:
    args, kwargs = _adapt_sample((1, 2), None)
    assert args == (1, 2)
    assert kwargs == {}

    args, kwargs = _adapt_sample({"x": 1}, None)
    assert args == ()
    assert kwargs == {"x": 1}

    args, kwargs = _adapt_sample(5, None)
    assert args == (5,)
    assert kwargs == {}

    def adapter(sample):
        return ((sample,), {"flag": True})

    args, kwargs = _adapt_sample(7, adapter)
    assert args == (7,)
    assert kwargs == {"flag": True}
