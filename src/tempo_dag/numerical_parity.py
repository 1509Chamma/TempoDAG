from __future__ import annotations

import copy
import heapq
import importlib
import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast

import numpy as np

from tempo_dag.quantization_config import (
    QuantizationScheme,
    QuantizationSpec,
    QuantizationType,
    compute_quant_params,
)

if TYPE_CHECKING:
    from tempo_dag.ir.graph import Graph


DEFAULT_METRICS = ("mae", "mse", "max_error", "relative_error", "sqnr")


class AbsErrorHistogram(TypedDict):
    bins: list[float]
    counts: list[int]


class MetricSummary(TypedDict, total=False):
    mae: float
    mse: float
    max_error: float
    relative_error: float
    sqnr: float
    nonfinite_count: int
    num_elements: int
    abs_error_histogram: AbsErrorHistogram


class Violation(TypedDict, total=False):
    scope: str
    scope_name: str
    sample_index: int
    metric: str
    actual: object
    expected: object
    threshold: float | int
    message: str
    item: str


class LayerQuantizationReport(TypedDict):
    clipped_values: int


class SimulationQuantizationReport(TypedDict):
    total_clipped_values: int
    layers: dict[str, LayerQuantizationReport]


class SampleQuantizationReport(SimulationQuantizationReport):
    sample_index: int


class WorstSample(TypedDict):
    sample_index: int
    score: float
    num_failures: int


class DiagnosticsReport(TypedDict):
    top_k_worst_samples: list[WorstSample]
    failing_samples: list[int]
    failing_layers: list[str]
    highest_deviation_layer: str | None
    sample_count: int
    quantization_reports: list[SampleQuantizationReport]
    ir: IRComparisonReport


IRComparisonReport = TypedDict(
    "IRComparisonReport",
    {
        "pass": bool,
        "violations": list[Violation],
        "summary": str,
    },
)

MetricsReport = TypedDict(
    "MetricsReport",
    {
        "global": MetricSummary,
        "outputs": dict[str, MetricSummary],
        "layers": dict[str, MetricSummary],
    },
)

NumericalParityResult = TypedDict(
    "NumericalParityResult",
    {
        "metrics": MetricsReport,
        "pass": bool,
        "violations": list[Violation],
        "summary": str,
        "diagnostics": DiagnosticsReport,
    },
)


class _ONNXValueInfoLike(Protocol):
    name: str


class _ONNXSessionLike(Protocol):
    def get_inputs(self) -> Sequence[_ONNXValueInfoLike]: ...

    def get_outputs(self) -> Sequence[_ONNXValueInfoLike]: ...

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, object],
    ) -> Sequence[object]: ...


@dataclass
class NumericalParityConfig:
    metrics: tuple[str, ...] = DEFAULT_METRICS
    thresholds: dict[str, float] = field(default_factory=dict)
    relative_error_epsilon: float = 1e-8
    top_k_worst: int = 5
    histogram_bins: int | Sequence[float] | None = None
    capture_layers: bool = True
    layer_names: tuple[str, ...] | None = None
    sample_adapter: Any | None = None
    fail_on_nonfinite: bool = True
    enforce_eval_mode: bool = True
    compare_ir: bool = True
    fp32_ir: Graph | None = None
    quantized_ir: Graph | None = None
    ranking_metric: str = "max_error"

    @classmethod
    def from_input(
        cls, config: NumericalParityConfig | Mapping[str, object] | None
    ) -> NumericalParityConfig:
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config

        metrics_value = _coerce_str_sequence(config.get("metrics"), DEFAULT_METRICS)
        layer_names_value = _coerce_optional_str_sequence(config.get("layer_names"))
        return cls(
            metrics=metrics_value,
            thresholds=_coerce_thresholds(config.get("thresholds")),
            relative_error_epsilon=_coerce_float(
                config.get("relative_error_epsilon"), 1e-8
            ),
            top_k_worst=_coerce_int(config.get("top_k_worst"), 5),
            histogram_bins=_coerce_histogram_bins(config.get("histogram_bins")),
            capture_layers=bool(config.get("capture_layers", True)),
            layer_names=layer_names_value,
            sample_adapter=config.get("sample_adapter"),
            fail_on_nonfinite=bool(config.get("fail_on_nonfinite", True)),
            enforce_eval_mode=bool(config.get("enforce_eval_mode", True)),
            compare_ir=bool(config.get("compare_ir", True)),
            fp32_ir=cast("Graph | None", config.get("fp32_ir")),
            quantized_ir=cast("Graph | None", config.get("quantized_ir")),
            ranking_metric=str(config.get("ranking_metric", "max_error")),
        )


@dataclass
class QuantizationSimulationResult:
    dequantized: np.ndarray
    quantized: np.ndarray
    clipped_values: int
    scale: float
    zero_point: int


def _coerce_str_sequence(
    value: object | None, default: tuple[str, ...]
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(str(item) for item in value)
    raise TypeError("config metrics must be a sequence of strings")


def _coerce_optional_str_sequence(value: object | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(str(item) for item in value)
    raise TypeError("config layer_names must be a sequence of strings")


def _coerce_thresholds(value: object | None) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("config thresholds must be a mapping")
    return {
        str(metric): _coerce_float(threshold) for metric, threshold in value.items()
    }


def _coerce_float(value: object | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    raise TypeError("config value must be numeric")


def _coerce_int(value: object | None, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    raise TypeError("config value must be an integer")


def _coerce_histogram_bins(
    value: object | None,
) -> int | Sequence[float] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return tuple(_coerce_float(item) for item in value)
    raise TypeError("config histogram_bins must be an integer or numeric sequence")


@dataclass
class _MetricAccumulator:
    metrics: tuple[str, ...]
    relative_error_epsilon: float
    histogram_bins: int | Sequence[float] | None
    abs_error_sum: float = 0.0
    sq_error_sum: float = 0.0
    rel_error_sum: float = 0.0
    signal_sq_sum: float = 0.0
    noise_sq_sum: float = 0.0
    element_count: int = 0
    max_error: float = 0.0
    nonfinite_count: int = 0
    histogram_counts: np.ndarray | None = None
    histogram_edges: np.ndarray | None = None

    def update(self, reference: np.ndarray, candidate: np.ndarray) -> MetricSummary:
        reference_arr = _as_float_array(reference)
        candidate_arr = _as_float_array(candidate)
        if reference_arr.shape != candidate_arr.shape:
            raise ValueError(
                "reference and candidate tensors must have identical shapes, got "
                f"{reference_arr.shape} and {candidate_arr.shape}"
            )

        nonfinite_mask = ~np.isfinite(reference_arr) | ~np.isfinite(candidate_arr)
        self.nonfinite_count += int(nonfinite_mask.sum())

        sanitized_reference = np.nan_to_num(
            reference_arr, nan=0.0, posinf=0.0, neginf=0.0
        )
        sanitized_candidate = np.nan_to_num(
            candidate_arr, nan=0.0, posinf=0.0, neginf=0.0
        )

        diff = sanitized_reference - sanitized_candidate
        abs_diff = np.abs(diff)
        sq_diff = diff * diff
        rel_denominator = np.maximum(
            np.abs(sanitized_reference), self.relative_error_epsilon
        )
        rel_diff = abs_diff / rel_denominator

        self.abs_error_sum += float(abs_diff.sum())
        self.sq_error_sum += float(sq_diff.sum())
        self.rel_error_sum += float(rel_diff.sum())
        self.signal_sq_sum += float(np.square(sanitized_reference).sum())
        self.noise_sq_sum += float(sq_diff.sum())
        self.element_count += int(abs_diff.size)

        sample_max_error = float(abs_diff.max()) if abs_diff.size else 0.0
        self.max_error = max(self.max_error, sample_max_error)

        if self.histogram_bins is not None:
            counts, edges = np.histogram(abs_diff, bins=self.histogram_bins)
            if self.histogram_counts is None:
                self.histogram_counts = counts.astype(np.int64)
                self.histogram_edges = edges
            else:
                self.histogram_counts += counts

        return _compute_metric_values(
            abs_error_sum=float(abs_diff.sum()),
            sq_error_sum=float(sq_diff.sum()),
            rel_error_sum=float(rel_diff.sum()),
            signal_sq_sum=float(np.square(sanitized_reference).sum()),
            noise_sq_sum=float(sq_diff.sum()),
            element_count=int(abs_diff.size),
            max_error=sample_max_error,
            nonfinite_count=int(nonfinite_mask.sum()),
        )

    def finalize(self) -> MetricSummary:
        metrics = _compute_metric_values(
            abs_error_sum=self.abs_error_sum,
            sq_error_sum=self.sq_error_sum,
            rel_error_sum=self.rel_error_sum,
            signal_sq_sum=self.signal_sq_sum,
            noise_sq_sum=self.noise_sq_sum,
            element_count=self.element_count,
            max_error=self.max_error,
            nonfinite_count=self.nonfinite_count,
        )
        metrics["num_elements"] = self.element_count
        metrics["nonfinite_count"] = self.nonfinite_count
        if self.histogram_counts is not None and self.histogram_edges is not None:
            metrics["abs_error_histogram"] = {
                "bins": self.histogram_edges.tolist(),
                "counts": self.histogram_counts.tolist(),
            }
        return metrics


class TorchQuantizedModelSimulator:
    """
    CPU-side quantization simulator for PyTorch modules.

    The simulator deep-copies the reference module, optionally quantizes weights,
    and quantizes inputs/activations/outputs back to dequantized floating-point
    tensors to approximate FPGA-facing numeric behavior during parity checks.
    """

    def __init__(
        self,
        module: Any,
        *,
        activation_spec: QuantizationSpec,
        weight_spec: QuantizationSpec | None = None,
        input_spec: QuantizationSpec | None = None,
        output_spec: QuantizationSpec | None = None,
        layer_specs: Mapping[str, QuantizationSpec] | None = None,
        quantize_inputs: bool = True,
        quantize_outputs: bool = True,
        quantize_weights: bool = True,
    ) -> None:
        try:
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TorchQuantizedModelSimulator requires PyTorch to be installed."
            ) from exc

        if not isinstance(module, nn.Module):
            raise TypeError("module must be a torch.nn.Module instance")

        self._module = copy.deepcopy(module)
        self._parity_layer_model = self._module
        self.activation_spec = activation_spec
        self.weight_spec = weight_spec or activation_spec
        self.input_spec = input_spec or activation_spec
        self.output_spec = output_spec or activation_spec
        self.layer_specs = dict(layer_specs or {})
        self.quantize_inputs = quantize_inputs
        self.quantize_outputs = quantize_outputs
        self.quantize_weights = quantize_weights
        self._last_quantization_report: SimulationQuantizationReport = {
            "total_clipped_values": 0,
            "layers": {},
        }

        if self.quantize_weights:
            self._quantize_module_parameters()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        import torch

        total_clipped_values = 0
        layer_reports: dict[str, LayerQuantizationReport] = {}
        hooks: list[Any] = []

        def make_activation_hook(name: str):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                nonlocal total_clipped_values
                spec = self.layer_specs.get(name, self.activation_spec)
                quantized_output, clipped = _quantize_value_like(output, spec)
                total_clipped_values += clipped
                layer_reports[name] = {"clipped_values": clipped}
                return quantized_output

            return hook

        for name, submodule in self._iter_capture_modules():
            hooks.append(submodule.register_forward_hook(make_activation_hook(name)))

        try:
            prepared_args = list(args)
            prepared_kwargs = dict(kwargs)
            if self.quantize_inputs:
                prepared_args, arg_clipped = _quantize_argument_sequence(
                    prepared_args, self.input_spec
                )
                prepared_kwargs, kwarg_clipped = _quantize_argument_mapping(
                    prepared_kwargs, self.input_spec
                )
                total_clipped_values += arg_clipped + kwarg_clipped

            with torch.no_grad():
                output = self._module(*prepared_args, **prepared_kwargs)

            if self.quantize_outputs:
                output, output_clipped = _quantize_value_like(output, self.output_spec)
                total_clipped_values += output_clipped

            self._last_quantization_report = {
                "total_clipped_values": total_clipped_values,
                "layers": layer_reports,
            }
            return output
        finally:
            for hook in hooks:
                hook.remove()

    def consume_last_quantization_report(self) -> SimulationQuantizationReport:
        report = self._last_quantization_report
        self._last_quantization_report = {
            "total_clipped_values": 0,
            "layers": {},
        }
        return report

    def eval(self) -> TorchQuantizedModelSimulator:
        self._module.eval()
        return self

    def train(self, mode: bool = True) -> TorchQuantizedModelSimulator:
        self._module.train(mode)
        return self

    @property
    def training(self) -> bool:
        return bool(getattr(self._module, "training", False))

    def _iter_capture_modules(self) -> list[tuple[str, Any]]:
        return [
            (name, module)
            for name, module in self._module.named_modules()
            if name and not any(True for _ in module.children())
        ]

    def _quantize_module_parameters(self) -> None:
        for parameter in self._module.parameters():
            quantized = quantize_array(
                parameter.detach().cpu().numpy(), self.weight_spec
            )
            parameter.data.copy_(_numpy_to_like(quantized.dequantized, parameter))

        for buffer in self._module.buffers():
            if bool(getattr(buffer, "is_floating_point", lambda: False)()):
                quantized = quantize_array(
                    buffer.detach().cpu().numpy(), self.weight_spec
                )
                buffer.data.copy_(_numpy_to_like(quantized.dequantized, buffer))


class ONNXRuntimeParityAdapter:
    """
    Adapter for ONNX Runtime inference sessions.

    Layer capture is supported when the requested layer output names are
    fetchable from the session, for example when an instrumented model exposes
    internal tensors as graph outputs.
    """

    def __init__(
        self,
        session_or_model: object,
        *,
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        layer_output_names: Sequence[str] | None = None,
    ) -> None:
        self._session = _resolve_onnx_session(session_or_model)
        self.input_names = (
            tuple(input_names)
            if input_names is not None
            else tuple(value_info.name for value_info in self._session.get_inputs())
        )
        self.output_names = (
            tuple(output_names)
            if output_names is not None
            else tuple(value_info.name for value_info in self._session.get_outputs())
        )
        self.layer_output_names = tuple(layer_output_names or ())
        self._parity_layer_model = self

    @property
    def training(self) -> bool:
        return False

    def eval(self) -> ONNXRuntimeParityAdapter:
        return self

    def train(self, mode: bool = True) -> ONNXRuntimeParityAdapter:
        del mode
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output_map, _layer_map = self.parity_forward(
            *args,
            capture_layers=False,
            layer_names=None,
            **kwargs,
        )
        if len(self.output_names) == 1:
            return output_map[self.output_names[0]]
        return output_map

    def parity_forward(
        self,
        *args: Any,
        capture_layers: bool = True,
        layer_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        feeds = self._build_feeds(args, kwargs)
        requested_layer_names = (
            tuple(layer_names) if layer_names is not None else self.layer_output_names
        )
        fetch_names = list(self.output_names)
        if capture_layers:
            for name in requested_layer_names:
                if name not in fetch_names:
                    fetch_names.append(name)

        session_outputs = self._session.run(fetch_names, feeds)
        result_map = {
            name: _as_float_array(value)
            for name, value in zip(fetch_names, session_outputs, strict=False)
        }
        output_map = {name: result_map[name] for name in self.output_names}
        layer_map = (
            {
                name: result_map[name]
                for name in requested_layer_names
                if name in result_map
            }
            if capture_layers
            else {}
        )
        return output_map, layer_map

    def _build_feeds(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        if kwargs:
            return {str(name): _as_float_array(value) for name, value in kwargs.items()}
        if len(args) == 1 and isinstance(args[0], Mapping):
            return {
                str(name): _as_float_array(value)
                for name, value in cast(Mapping[object, Any], args[0]).items()
            }
        if len(args) != len(self.input_names):
            raise ValueError(
                "ONNXRuntimeParityAdapter expected "
                f"{len(self.input_names)} inputs but received {len(args)}"
            )
        return {
            name: _as_float_array(value)
            for name, value in zip(self.input_names, args, strict=False)
        }


class TensorFlowKerasParityAdapter:
    """
    Adapter for TensorFlow/Keras models.

    Layer capture is available for built graph-style Keras models by constructing
    auxiliary submodels that expose selected intermediate layer outputs.
    """

    def __init__(
        self,
        model: object,
        *,
        default_layer_names: Sequence[str] | None = None,
    ) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TensorFlowKerasParityAdapter requires TensorFlow to be installed."
            ) from exc

        if not isinstance(model, tf.keras.Model):
            raise TypeError("model must be a tf.keras.Model instance")

        self._tf = tf
        self._model = model
        self._default_layer_names = tuple(default_layer_names or ())
        self._capture_model_cache: dict[tuple[str, ...], Any | None] = {}
        self._training = False
        self._parity_layer_model = self

    @property
    def training(self) -> bool:
        return self._training

    def eval(self) -> TensorFlowKerasParityAdapter:
        self._training = False
        return self

    def train(self, mode: bool = True) -> TensorFlowKerasParityAdapter:
        self._training = mode
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_model(*args, **kwargs)

    def parity_forward(
        self,
        *args: Any,
        capture_layers: bool = True,
        layer_names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        output = self._call_model(*args, **kwargs)
        output_map = _normalize_output_structure(output)
        if not capture_layers:
            return output_map, {}

        selected_layer_names = self._resolve_layer_names(layer_names)
        if not selected_layer_names:
            return output_map, {}

        capture_model = self._get_capture_model(selected_layer_names)
        if capture_model is None:
            return output_map, {}

        captured = capture_model(*args, training=False, **kwargs)
        if not isinstance(captured, list | tuple):
            captured = (captured,)

        layer_map: dict[str, np.ndarray] = {}
        for name, value in zip(selected_layer_names, captured, strict=False):
            normalized = _normalize_output_structure(value)
            layer_map.update(
                {
                    f"{name}.{key}" if key != "output" else name: tensor
                    for key, tensor in normalized.items()
                }
            )
        return output_map, layer_map

    def _call_model(self, *args: Any, **kwargs: Any) -> Any:
        call_kwargs = dict(kwargs)
        call_kwargs.setdefault("training", False)
        return self._model(*args, **call_kwargs)

    def _resolve_layer_names(
        self,
        layer_names: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if layer_names is not None:
            return tuple(layer_names)
        if self._default_layer_names:
            return self._default_layer_names
        return tuple(
            layer.name
            for layer in self._model.layers
            if layer.__class__.__name__ != "InputLayer" and hasattr(layer, "output")
        )

    def _get_capture_model(self, layer_names: tuple[str, ...]) -> Any | None:
        if layer_names in self._capture_model_cache:
            return self._capture_model_cache[layer_names]
        try:
            inputs = self._model.inputs
            if not inputs:
                self._capture_model_cache[layer_names] = None
                return None
            outputs = [self._model.get_layer(name).output for name in layer_names]
            capture_model = self._tf.keras.Model(inputs=inputs, outputs=outputs)
        except (AttributeError, ValueError):
            capture_model = None
        self._capture_model_cache[layer_names] = capture_model
        return capture_model


def quantize_array(
    values: Any,
    spec: QuantizationSpec,
) -> QuantizationSimulationResult:
    """
    Quantize values to an integer or fixed-point domain and dequantize back.
    """
    array = _as_float_array(values)
    if array.size == 0:
        return QuantizationSimulationResult(
            dequantized=array.copy(),
            quantized=array.astype(np.int64, copy=True),
            clipped_values=0,
            scale=1.0,
            zero_point=0,
        )

    if spec.qtype == QuantizationType.FIXED_POINT:
        if spec.fixed_point is None:
            raise ValueError("fixed-point quantization requires a fixed_point spec")
        scale = (
            float(spec.scale)
            if spec.scale is not None
            else 2.0 ** (-spec.fixed_point.fractional_bits)
        )
        zero_point = int(spec.zero_point or 0)
        qmin = -(2 ** (spec.bit_width - 1))
        qmax = (2 ** (spec.bit_width - 1)) - 1
    else:
        resolved_scale, resolved_zero_point = _resolve_integer_quant_params(array, spec)
        scale = resolved_scale
        zero_point = resolved_zero_point
        if spec.scheme == QuantizationScheme.SYMMETRIC:
            qmin = -(2 ** (spec.bit_width - 1))
            qmax = (2 ** (spec.bit_width - 1)) - 1
        else:
            qmin = 0
            qmax = (2**spec.bit_width) - 1

    scaled = array / scale + zero_point
    rounded = _round_half_away_from_zero(scaled)
    clipped = np.clip(rounded, qmin, qmax)
    clipped_values = int(np.count_nonzero(rounded != clipped))
    quantized = clipped.astype(np.int64, copy=False)
    dequantized = (quantized.astype(np.float64) - zero_point) * scale
    return QuantizationSimulationResult(
        dequantized=dequantized.astype(np.float64, copy=False),
        quantized=quantized,
        clipped_values=clipped_values,
        scale=scale,
        zero_point=zero_point,
    )


def compare_ir_graphs(
    fp32_ir: Graph | None,
    quantized_ir: Graph | None,
) -> IRComparisonReport:
    violations: list[Violation] = []
    if fp32_ir is None or quantized_ir is None:
        return {
            "pass": True,
            "violations": violations,
            "summary": "IR comparison skipped.",
        }

    if fp32_ir.graph_inputs != quantized_ir.graph_inputs:
        violations.append(
            {
                "scope": "ir",
                "metric": "graph_inputs",
                "actual": list(quantized_ir.graph_inputs),
                "expected": list(fp32_ir.graph_inputs),
                "message": "Quantized IR graph inputs differ from the FP32 reference.",
            }
        )

    if fp32_ir.graph_outputs != quantized_ir.graph_outputs:
        violations.append(
            {
                "scope": "ir",
                "metric": "graph_outputs",
                "actual": list(quantized_ir.graph_outputs),
                "expected": list(fp32_ir.graph_outputs),
                "message": "Quantized IR graph outputs differ from the FP32 reference.",
            }
        )

    fp32_op_types = {op_id: op.op_type for op_id, op in fp32_ir.ops.items()}
    quantized_op_types = {op_id: op.op_type for op_id, op in quantized_ir.ops.items()}
    for op_id in sorted(set(fp32_op_types) | set(quantized_op_types)):
        if fp32_op_types.get(op_id) != quantized_op_types.get(op_id):
            violations.append(
                {
                    "scope": "ir",
                    "metric": "operator_type",
                    "item": op_id,
                    "actual": quantized_op_types.get(op_id),
                    "expected": fp32_op_types.get(op_id),
                    "message": (
                        f"Operator '{op_id}' differs between FP32 and quantized IR."
                    ),
                }
            )

    for value_id in sorted(set(fp32_ir.values) | set(quantized_ir.values)):
        fp32_value = fp32_ir.values.get(value_id)
        quantized_value = quantized_ir.values.get(value_id)
        if fp32_value is None or quantized_value is None:
            violations.append(
                {
                    "scope": "ir",
                    "metric": "value_presence",
                    "item": value_id,
                    "actual": value_id in quantized_ir.values,
                    "expected": value_id in fp32_ir.values,
                    "message": f"Value '{value_id}' is missing in one IR graph.",
                }
            )
            continue
        if fp32_value.shape != quantized_value.shape:
            violations.append(
                {
                    "scope": "ir",
                    "metric": "shape",
                    "item": value_id,
                    "actual": list(quantized_value.shape),
                    "expected": list(fp32_value.shape),
                    "message": f"Value '{value_id}' shape differs between IR graphs.",
                }
            )
        if fp32_value.dtype != quantized_value.dtype:
            violations.append(
                {
                    "scope": "ir",
                    "metric": "dtype",
                    "item": value_id,
                    "actual": quantized_value.dtype,
                    "expected": fp32_value.dtype,
                    "message": f"Value '{value_id}' dtype differs between IR graphs.",
                }
            )

    summary = (
        "IR graphs match."
        if not violations
        else f"IR comparison found {len(violations)} violation(s)."
    )
    return {"pass": not violations, "violations": violations, "summary": summary}


def run_numerical_parity_test(
    fp32_model: Any,
    quantized_model: Any,
    dataset: Iterable[Any],
    config: NumericalParityConfig | Mapping[str, object] | None = None,
) -> NumericalParityResult:
    resolved_config = NumericalParityConfig.from_input(config)
    violations: list[Violation] = []
    output_accumulators: dict[str, _MetricAccumulator] = {}
    layer_accumulators: dict[str, _MetricAccumulator] = {}
    global_accumulator = _MetricAccumulator(
        metrics=resolved_config.metrics,
        relative_error_epsilon=resolved_config.relative_error_epsilon,
        histogram_bins=resolved_config.histogram_bins,
    )
    top_k_heap: list[tuple[float, int, WorstSample]] = []
    failing_samples: set[int] = set()
    failing_layers: set[str] = set()
    quantization_reports: list[SampleQuantizationReport] = []
    sample_count = 0

    ir_report = compare_ir_graphs(
        resolved_config.fp32_ir if resolved_config.compare_ir else None,
        resolved_config.quantized_ir if resolved_config.compare_ir else None,
    )
    violations.extend(ir_report["violations"])

    with ExitStack() as stack:
        _maybe_prepare_model(stack, fp32_model, resolved_config.enforce_eval_mode)
        _maybe_prepare_model(stack, quantized_model, resolved_config.enforce_eval_mode)

        for sample_index, sample in enumerate(dataset):
            sample_count += 1
            model_args, model_kwargs = _adapt_sample(
                sample, resolved_config.sample_adapter
            )

            fp32_outputs, fp32_layers = _run_model_with_optional_layer_capture(
                fp32_model, model_args, model_kwargs, resolved_config
            )
            (
                quantized_outputs,
                quantized_layers,
            ) = _run_model_with_optional_layer_capture(
                quantized_model, model_args, model_kwargs, resolved_config
            )

            sample_score = 0.0
            sample_failures = 0

            sample_score, sample_failures = _compare_scope_maps(
                reference_map=fp32_outputs,
                candidate_map=quantized_outputs,
                accumulators=output_accumulators,
                global_accumulator=global_accumulator,
                config=resolved_config,
                scope="output",
                sample_index=sample_index,
                violations=violations,
                failing_layers=failing_layers,
                current_score=sample_score,
                current_failures=sample_failures,
                update_global=True,
            )
            sample_score, sample_failures = _compare_scope_maps(
                reference_map=fp32_layers,
                candidate_map=quantized_layers,
                accumulators=layer_accumulators,
                global_accumulator=global_accumulator,
                config=resolved_config,
                scope="layer",
                sample_index=sample_index,
                violations=violations,
                failing_layers=failing_layers,
                current_score=sample_score,
                current_failures=sample_failures,
                update_global=False,
            )

            quantization_report = _consume_quantization_report(quantized_model)
            if quantization_report:
                quantization_reports.append(
                    {
                        "sample_index": sample_index,
                        "total_clipped_values": quantization_report[
                            "total_clipped_values"
                        ],
                        "layers": quantization_report["layers"],
                    }
                )

            if sample_failures:
                failing_samples.add(sample_index)

            sample_entry: WorstSample = {
                "sample_index": sample_index,
                "score": sample_score,
                "num_failures": sample_failures,
            }
            if len(top_k_heap) < resolved_config.top_k_worst:
                heapq.heappush(top_k_heap, (sample_score, sample_index, sample_entry))
            elif sample_score > top_k_heap[0][0]:
                heapq.heapreplace(
                    top_k_heap, (sample_score, sample_index, sample_entry)
                )

    output_metrics = {
        name: accumulator.finalize()
        for name, accumulator in sorted(output_accumulators.items())
    }
    layer_metrics = {
        name: accumulator.finalize()
        for name, accumulator in sorted(layer_accumulators.items())
    }
    global_metrics = global_accumulator.finalize()
    pass_flag = not violations

    worst_layer = None
    if layer_metrics:
        worst_layer = max(
            layer_metrics.items(),
            key=lambda item: _metric_float(item[1], resolved_config.ranking_metric),
        )[0]

    diagnostics: DiagnosticsReport = {
        "top_k_worst_samples": [
            item
            for _score, _sample_index, item in sorted(
                top_k_heap, key=lambda item: (item[0], item[1]), reverse=True
            )
        ],
        "failing_samples": sorted(failing_samples),
        "failing_layers": sorted(failing_layers),
        "highest_deviation_layer": worst_layer,
        "sample_count": sample_count,
        "quantization_reports": quantization_reports,
        "ir": ir_report,
    }

    summary = _build_summary(
        pass_flag=pass_flag,
        sample_count=sample_count,
        global_metrics=global_metrics,
        violations=violations,
        worst_layer=worst_layer,
        ranking_metric=resolved_config.ranking_metric,
    )

    metrics: MetricsReport = {
        "global": global_metrics,
        "outputs": output_metrics,
        "layers": layer_metrics,
    }

    return {
        "metrics": metrics,
        "pass": pass_flag,
        "violations": violations,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def _compare_scope_maps(
    *,
    reference_map: Mapping[str, np.ndarray],
    candidate_map: Mapping[str, np.ndarray],
    accumulators: dict[str, _MetricAccumulator],
    global_accumulator: _MetricAccumulator,
    config: NumericalParityConfig,
    scope: str,
    sample_index: int,
    violations: list[Violation],
    failing_layers: set[str],
    current_score: float,
    current_failures: int,
    update_global: bool,
) -> tuple[float, int]:
    for missing_name in sorted(set(reference_map) - set(candidate_map)):
        violations.append(
            {
                "scope": scope,
                "scope_name": missing_name,
                "sample_index": sample_index,
                "metric": "missing_candidate",
                "message": (
                    f"{scope.title()} '{missing_name}' is missing in quantized outputs."
                ),
            }
        )
        if scope == "layer":
            failing_layers.add(missing_name)
        current_failures += 1

    for extra_name in sorted(set(candidate_map) - set(reference_map)):
        violations.append(
            {
                "scope": scope,
                "scope_name": extra_name,
                "sample_index": sample_index,
                "metric": "unexpected_candidate",
                "message": (
                    f"{scope.title()} '{extra_name}' is missing in FP32 outputs."
                ),
            }
        )
        if scope == "layer":
            failing_layers.add(extra_name)
        current_failures += 1

    for name in sorted(set(reference_map) & set(candidate_map)):
        accumulator = accumulators.setdefault(
            name,
            _MetricAccumulator(
                metrics=config.metrics,
                relative_error_epsilon=config.relative_error_epsilon,
                histogram_bins=config.histogram_bins,
            ),
        )
        try:
            sample_metrics = accumulator.update(
                reference_map[name], candidate_map[name]
            )
            if update_global:
                global_accumulator.update(reference_map[name], candidate_map[name])
        except ValueError as exc:
            violations.append(
                {
                    "scope": scope,
                    "scope_name": name,
                    "sample_index": sample_index,
                    "metric": "shape_mismatch",
                    "message": str(exc),
                }
            )
            current_failures += 1
            if scope == "layer":
                failing_layers.add(name)
            continue
        current_score = max(
            current_score, _metric_float(sample_metrics, config.ranking_metric)
        )

        if config.fail_on_nonfinite and int(sample_metrics["nonfinite_count"]) > 0:
            violations.append(
                {
                    "scope": scope,
                    "scope_name": name,
                    "sample_index": sample_index,
                    "metric": "nonfinite_count",
                    "actual": int(sample_metrics["nonfinite_count"]),
                    "threshold": 0,
                    "message": f"{scope.title()} '{name}' produced NaN or Inf values.",
                }
            )
            current_failures += 1
            if scope == "layer":
                failing_layers.add(name)

        for metric_name, threshold in config.thresholds.items():
            actual_value = _metric_float(sample_metrics, metric_name)
            if actual_value > threshold:
                violations.append(
                    {
                        "scope": scope,
                        "scope_name": name,
                        "sample_index": sample_index,
                        "metric": metric_name,
                        "actual": actual_value,
                        "threshold": threshold,
                        "message": (
                            f"{scope.title()} '{name}' exceeded {metric_name}: "
                            f"{actual_value:.6g} > {threshold:.6g}"
                        ),
                    }
                )
                current_failures += 1
                if scope == "layer":
                    failing_layers.add(name)

    return current_score, current_failures


def _run_model_with_optional_layer_capture(
    model: Any,
    model_args: tuple[Any, ...],
    model_kwargs: dict[str, Any],
    config: NumericalParityConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    parity_forward = getattr(model, "parity_forward", None)
    if callable(parity_forward):
        return cast(
            tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
            parity_forward(
                *model_args,
                capture_layers=config.capture_layers,
                layer_names=config.layer_names,
                **model_kwargs,
            ),
        )

    capture_layers = config.capture_layers and _supports_torch_layer_capture(model)
    if not capture_layers:
        return _normalize_output_structure(model(*model_args, **model_kwargs)), {}

    layer_outputs: dict[str, np.ndarray] = {}
    hooks: list[Any] = []
    layer_model = _get_layer_capture_model(model)
    selected_names = set(config.layer_names or ())

    def make_hook(name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if selected_names and name not in selected_names:
                return
            layer_outputs.update(
                {
                    f"{name}.{key}" if key != "output" else name: value
                    for key, value in _normalize_output_structure(output).items()
                }
            )

        return hook

    for name, submodule in layer_model.named_modules():
        if not name:
            continue
        if selected_names:
            if name not in selected_names:
                continue
        elif any(True for _ in submodule.children()):
            continue
        hooks.append(submodule.register_forward_hook(make_hook(name)))

    try:
        output = model(*model_args, **model_kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    return _normalize_output_structure(output), layer_outputs


def _normalize_output_structure(output: Any) -> dict[str, np.ndarray]:
    if isinstance(output, Mapping):
        normalized: dict[str, np.ndarray] = {}
        for key, value in output.items():
            child_map = _normalize_output_structure(value)
            normalized.update(
                {
                    (
                        f"{key}.{child_key}" if child_key != "output" else str(key)
                    ): child_value
                    for child_key, child_value in child_map.items()
                }
            )
        return normalized
    if isinstance(output, tuple):
        normalized = {}
        for index, value in enumerate(output):
            child_map = _normalize_output_structure(value)
            normalized.update(
                {
                    (
                        f"output.{index}.{child_key}"
                        if child_key != "output"
                        else f"output.{index}"
                    ): child_value
                    for child_key, child_value in child_map.items()
                }
            )
        return normalized
    if isinstance(output, list):
        normalized = {}
        for index, value in enumerate(output):
            child_map = _normalize_output_structure(value)
            normalized.update(
                {
                    (
                        f"output.{index}.{child_key}"
                        if child_key != "output"
                        else f"output.{index}"
                    ): child_value
                    for child_key, child_value in child_map.items()
                }
            )
        return normalized
    return {"output": _as_float_array(output)}


def _build_summary(
    *,
    pass_flag: bool,
    sample_count: int,
    global_metrics: Mapping[str, object],
    violations: Sequence[Mapping[str, object]],
    worst_layer: str | None,
    ranking_metric: str,
) -> str:
    ranking_value = _metric_float(global_metrics, ranking_metric)
    if pass_flag:
        return (
            f"Numerical parity passed across {sample_count} sample(s); "
            f"global {ranking_metric}={ranking_value:.6g}."
        )

    return (
        f"Numerical parity failed with {len(violations)} violation(s) across "
        f"{sample_count} sample(s); global {ranking_metric}={ranking_value:.6g}, "
        f"highest deviation layer={worst_layer or 'n/a'}."
    )


def _compute_metric_values(
    *,
    abs_error_sum: float,
    sq_error_sum: float,
    rel_error_sum: float,
    signal_sq_sum: float,
    noise_sq_sum: float,
    element_count: int,
    max_error: float,
    nonfinite_count: int,
) -> MetricSummary:
    metrics: MetricSummary
    if element_count <= 0:
        metrics = {
            "mae": 0.0,
            "mse": 0.0,
            "max_error": 0.0,
            "relative_error": 0.0,
            "sqnr": math.inf,
            "nonfinite_count": nonfinite_count,
        }
        return metrics

    if noise_sq_sum == 0.0:
        sqnr = math.inf
    elif signal_sq_sum == 0.0:
        sqnr = -math.inf
    else:
        sqnr = 10.0 * math.log10(signal_sq_sum / noise_sq_sum)

    metrics = {
        "mae": abs_error_sum / element_count,
        "mse": sq_error_sum / element_count,
        "max_error": max_error,
        "relative_error": rel_error_sum / element_count,
        "sqnr": sqnr,
        "nonfinite_count": nonfinite_count,
    }
    return metrics


def _resolve_onnx_session(session_or_model: object) -> _ONNXSessionLike:
    if hasattr(session_or_model, "run") and hasattr(session_or_model, "get_inputs"):
        return cast(_ONNXSessionLike, session_or_model)
    if isinstance(session_or_model, str) or hasattr(session_or_model, "__fspath__"):
        try:
            ort = importlib.import_module("onnxruntime")
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ONNXRuntimeParityAdapter requires onnxruntime when given a model path."
            ) from exc
        return cast(_ONNXSessionLike, ort.InferenceSession(session_or_model))
    raise TypeError(
        "session_or_model must be an ONNX Runtime session or a path to an ONNX model"
    )


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))


def _metric_float(metrics: Mapping[str, object], key: str) -> float:
    value = metrics.get(key, 0.0)
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _resolve_integer_quant_params(
    array: np.ndarray,
    spec: QuantizationSpec,
) -> tuple[float, int]:
    scale = float(spec.scale) if spec.scale is not None else None
    zero_point = int(spec.zero_point) if spec.zero_point is not None else None
    if scale is None or scale <= 0.0 or zero_point is None:
        scale, zero_point = compute_quant_params(array, spec)
    if scale <= 0.0:
        scale = 1.0
    return scale, zero_point


def _quantize_argument_sequence(
    arguments: Sequence[Any],
    spec: QuantizationSpec,
) -> tuple[list[Any], int]:
    quantized_arguments: list[Any] = []
    total_clipped = 0
    for value in arguments:
        quantized_value, clipped = _quantize_value_like(value, spec)
        quantized_arguments.append(quantized_value)
        total_clipped += clipped
    return quantized_arguments, total_clipped


def _quantize_argument_mapping(
    arguments: Mapping[str, Any],
    spec: QuantizationSpec,
) -> tuple[dict[str, Any], int]:
    quantized_arguments: dict[str, Any] = {}
    total_clipped = 0
    for key, value in arguments.items():
        quantized_value, clipped = _quantize_value_like(value, spec)
        quantized_arguments[str(key)] = quantized_value
        total_clipped += clipped
    return quantized_arguments, total_clipped


def _quantize_value_like(value: Any, spec: QuantizationSpec) -> tuple[Any, int]:
    if isinstance(value, tuple):
        quantized_tuple_items: list[Any] = []
        total_clipped = 0
        for item in value:
            quantized_item, clipped = _quantize_value_like(item, spec)
            quantized_tuple_items.append(quantized_item)
            total_clipped += clipped
        return tuple(quantized_tuple_items), total_clipped
    if isinstance(value, list):
        quantized_list_items: list[Any] = []
        total_clipped = 0
        for item in value:
            quantized_item, clipped = _quantize_value_like(item, spec)
            quantized_list_items.append(quantized_item)
            total_clipped += clipped
        return quantized_list_items, total_clipped
    if isinstance(value, Mapping):
        quantized_items: dict[str, Any] = {}
        total_clipped = 0
        for key, item in value.items():
            quantized_item, clipped = _quantize_value_like(item, spec)
            quantized_items[str(key)] = quantized_item
            total_clipped += clipped
        return quantized_items, total_clipped

    quantized = quantize_array(value, spec)
    if _is_torch_tensor(value):
        return _numpy_to_like(quantized.dequantized, value), quantized.clipped_values
    return quantized.dequantized, quantized.clipped_values


def _numpy_to_like(array: np.ndarray, like: Any) -> Any:
    if _is_torch_tensor(like):
        import torch

        tensor = torch.from_numpy(np.asarray(array))
        return tensor.to(device=like.device, dtype=like.dtype)
    return np.asarray(array)


def _adapt_sample(
    sample: Any,
    adapter: Any | None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if adapter is not None:
        adapted = adapter(sample)
        if not isinstance(adapted, tuple) or len(adapted) != 2:
            raise ValueError("sample_adapter must return a tuple of (args, kwargs)")
        adapted_args, adapted_kwargs = adapted
        return tuple(adapted_args), dict(adapted_kwargs)
    if isinstance(sample, Mapping):
        return (), dict(sample)
    if isinstance(sample, tuple):
        return tuple(sample), {}
    return (sample,), {}


def _maybe_prepare_model(stack: ExitStack, model: Any, enforce_eval_mode: bool) -> None:
    if enforce_eval_mode and hasattr(model, "eval"):
        previous_training = getattr(model, "training", None)
        model.eval()
        if previous_training is not None and hasattr(model, "train"):
            stack.callback(model.train, previous_training)

    if _is_torch_module_or_wrapper(model):
        import torch

        stack.enter_context(torch.no_grad())


def _supports_torch_layer_capture(model: Any) -> bool:
    layer_model = _get_layer_capture_model(model)
    return hasattr(layer_model, "named_modules") and _is_torch_module_or_wrapper(
        layer_model
    )


def _get_layer_capture_model(model: Any) -> Any:
    return getattr(model, "_parity_layer_model", model)


def _consume_quantization_report(model: Any) -> SimulationQuantizationReport | None:
    reporter = getattr(model, "consume_last_quantization_report", None)
    if callable(reporter):
        return cast(SimulationQuantizationReport, reporter())
    return None


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "detach")


def _is_torch_module_or_wrapper(value: Any) -> bool:
    try:
        import torch.nn as nn
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, nn.Module) or isinstance(
        getattr(value, "_parity_layer_model", None), nn.Module
    )


def _as_float_array(value: Any) -> np.ndarray:
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(value, dtype=np.float64)


__all__ = [
    "NumericalParityConfig",
    "ONNXRuntimeParityAdapter",
    "TensorFlowKerasParityAdapter",
    "TorchQuantizedModelSimulator",
    "compare_ir_graphs",
    "quantize_array",
    "run_numerical_parity_test",
]
