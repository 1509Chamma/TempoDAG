from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import onnx

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.registry import OperatorRegistry
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import BufferSpec, Edge0, EdgeDelta, Kernel, Process
from tempo_dag.ops.builtins import register_builtin_operators
from tempo_dag.ops.temporal_builtins import TEMPORAL_BUILTIN_OPERATORS
from tempo_dag.parsers.onnx.parser import ONNXParser, ONNXTemporalPattern


@dataclass(frozen=True)
class TemporalLoweringReport:
    """Summary of how an ONNX model was lifted into temporal IR."""

    process_id: str
    detected_patterns: tuple[ONNXTemporalPattern, ...]
    lowered_ops: tuple[str, ...]
    states: tuple[str, ...]
    buffers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "process_id": self.process_id,
            "detected_patterns": [
                pattern.to_dict() for pattern in self.detected_patterns
            ],
            "lowered_ops": list(self.lowered_ops),
            "states": list(self.states),
            "buffers": list(self.buffers),
        }


@dataclass(frozen=True)
class TemporalLoweringResult:
    """Container returned by the temporal ONNX parser."""

    process: Process
    report: TemporalLoweringReport


class TemporalONNXParser:
    """Lower a supported ONNX model into TempoDAG temporal IR."""

    DEFAULT_EXTRA_OP_MAPPING = {
        "Delay": "Delay",
        "Loop": "ScanCell",
        "RollingMean": "RollingMean",
        "RollingVar": "RollingVar",
        "RollingWindow": "RollingWindow",
        "Scan": "ScanCell",
    }

    def __init__(
        self,
        *,
        registry: OperatorRegistry | None = None,
        extra_op_mapping: dict[str, str] | None = None,
    ) -> None:
        self.registry = registry or _build_temporal_registry()
        op_mapping = dict(self.DEFAULT_EXTRA_OP_MAPPING)
        if extra_op_mapping:
            op_mapping.update(extra_op_mapping)
        self.onnx_parser = ONNXParser(
            registry=self.registry,
            extra_op_mapping=op_mapping,
        )

    def parse(
        self,
        model_path: str,
        *,
        process_id: str = "temporal_process",
    ) -> TemporalLoweringResult:
        model = onnx.load(model_path)
        return self.parse_model(model, process_id=process_id)

    def parse_model(
        self,
        model: onnx.ModelProto,
        *,
        process_id: str = "temporal_process",
    ) -> TemporalLoweringResult:
        graph = self.onnx_parser.parse_model(model)
        _materialize_missing_values(graph)
        _promote_initializer_values_to_state(graph)
        patterns = tuple(self.onnx_parser.detect_temporal_patterns(model))

        buffers: dict[str, BufferSpec] = {}
        edge0: list[Edge0] = []
        edge_delta: list[EdgeDelta] = []
        states: set[str] = set()

        for operator in graph.ops.values():
            if hasattr(operator, "temporal_metadata"):
                metadata = operator.temporal_metadata(graph.values)
                states.update(metadata.state_reads)
                states.update(metadata.state_writes)
                for buffer_id in metadata.buffers:
                    if buffer_id not in buffers:
                        output_value = _primary_output_value(graph, operator)
                        buffers[buffer_id] = BufferSpec(
                            buffer_id=buffer_id,
                            dtype=output_value.dtype,
                            shape=tuple(output_value.shape),
                            depth=max(
                                1,
                                metadata.window_size or metadata.lag_cycles or 1,
                            ),
                            axes=tuple(output_value.axes),
                            metadata={"source_op": operator.op_id},
                        )
                for buffer_id in metadata.buffers:
                    edge0.append(
                        Edge0(
                            buffer_id,
                            "kernel_main",
                            value_id=operator.outputs[0],
                        )
                    )
                    edge_delta.append(
                        EdgeDelta(
                            "kernel_main",
                            buffer_id,
                            lag_cycles=max(1, metadata.lag_cycles or 1),
                            value_id=operator.outputs[0],
                        )
                    )

        for pattern in patterns:
            if pattern.stateful_inputs:
                for input_id in pattern.stateful_inputs:
                    buffer_id = f"{pattern.node_name}_{input_id}_buffer"
                    if buffer_id not in buffers:
                        source_value = graph.values.get(input_id)
                        if source_value is None:
                            continue
                        buffers[buffer_id] = BufferSpec(
                            buffer_id=buffer_id,
                            dtype=source_value.dtype,
                            shape=tuple(source_value.shape),
                            depth=max(1, pattern.body_node_count or 1),
                            axes=tuple(source_value.axes),
                            metadata={
                                "pattern": pattern.op_type,
                                "state_input": input_id,
                            },
                        )
                    states.add(input_id)
                    edge0.append(Edge0(buffer_id, "kernel_main", value_id=input_id))
                    edge_delta.append(
                        EdgeDelta(
                            "kernel_main",
                            buffer_id,
                            lag_cycles=1,
                            value_id=input_id,
                        )
                    )

        kernel = Kernel(kernel_id="kernel_main", graph=graph)
        process = Process(
            process_id=process_id,
            kernels={kernel.kernel_id: kernel},
            buffers=buffers,
            edge0=_dedupe_edge0(edge0),
            edge_delta=_dedupe_edge_delta(edge_delta),
            metadata={
                "source": "onnx",
                "detected_patterns": [pattern.to_dict() for pattern in patterns],
            },
        )
        process.validate()

        return TemporalLoweringResult(
            process=process,
            report=TemporalLoweringReport(
                process_id=process_id,
                detected_patterns=patterns,
                lowered_ops=tuple(graph.ops),
                states=tuple(sorted(states)),
                buffers=tuple(sorted(buffers)),
            ),
        )


def _build_temporal_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    register_builtin_operators(registry)
    builtin_op_types = set(registry.list_registered())
    for operator_cls in TEMPORAL_BUILTIN_OPERATORS:
        if operator_cls.operator_type() not in builtin_op_types:
            registry.register(operator_cls)
    return registry


def _primary_output_value(graph: Graph, operator: Any) -> Value:
    output_id = operator.outputs[0]
    return graph.values[output_id]


def _materialize_missing_values(graph: Graph) -> None:
    for operator in graph.ops.values():
        if all(output_id in graph.values for output_id in operator.outputs):
            continue
        if operator.op_type == "RollingMean":
            input_value = graph.values[operator.inputs[0]]
            output_shape = list(input_value.shape)
            output_axes = list(input_value.axes)
        elif operator.op_type == "Conv1D":
            input_value = graph.values[operator.inputs[0]]
            weight_value = graph.values[operator.inputs[1]]
            stride = _conv_attr_as_int(
                operator.attrs,
                singular="stride",
                plural="strides",
                default=1,
            )
            dilation = _coerce_attr_int(operator.attrs.get("dilation"), default=1)
            padding_value = operator.attrs.get(
                "padding",
                operator.attrs.get("pads", [0, 0]),
            )
            if isinstance(padding_value, list):
                padding = int(padding_value[0])
            else:
                padding = _coerce_attr_int(padding_value, default=0)
            batch, _, input_length = input_value.shape
            out_channels, _, kernel_width = weight_value.shape
            numerator = (
                input_length + (2 * padding) - (dilation * (kernel_width - 1)) - 1
            )
            output_length = (numerator // stride) + 1
            output_shape = [batch, out_channels, output_length]
            output_axes = list(input_value.axes)
        elif operator.op_type == "Add":
            lhs = graph.values[operator.inputs[0]]
            rhs = graph.values[operator.inputs[1]]
            if lhs.vtype is ValueType.SCALAR:
                output_shape = list(rhs.shape)
                output_axes = list(rhs.axes)
            else:
                output_shape = list(lhs.shape)
                output_axes = list(lhs.axes)
        else:
            input_value = graph.values[operator.inputs[0]]
            output_shape = list(input_value.shape)
            output_axes = list(input_value.axes)

        for output_id in operator.outputs:
            if output_id not in graph.values:
                graph.values[output_id] = Value(
                    value_id=output_id,
                    vtype=ValueType.TENSOR,
                    dtype=graph.values[operator.inputs[0]].dtype,
                    shape=output_shape,
                    axes=output_axes,
                    producer_op_id=operator.op_id,
                )


def _conv_attr_as_int(
    attrs: dict[str, object],
    *,
    singular: str,
    plural: str,
    default: int,
) -> int:
    if singular in attrs:
        return _coerce_attr_int(attrs[singular], default=default)
    plural_value = attrs.get(plural)
    if isinstance(plural_value, list) and plural_value:
        return int(plural_value[0])
    return default


def _coerce_attr_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"expected numeric attribute, got {type(value).__name__}")


def _dedupe_edge0(edges: list[Edge0]) -> list[Edge0]:
    seen = set()
    result = []
    for edge in edges:
        key = (edge.source, edge.target, edge.value_id)
        if key not in seen:
            seen.add(key)
            result.append(edge)
    return result


def _dedupe_edge_delta(edges: list[EdgeDelta]) -> list[EdgeDelta]:
    seen = set()
    result = []
    for edge in edges:
        key = (edge.source, edge.target, edge.lag_cycles, edge.value_id)
        if key not in seen:
            seen.add(key)
            result.append(edge)
    return result


def _promote_initializer_values_to_state(graph: Graph) -> None:
    produced_values = {
        output_id for operator in graph.ops.values() for output_id in operator.outputs
    }
    for value_id, value in graph.values.items():
        if value_id in graph.graph_inputs or value_id in produced_values:
            continue
        graph.states.setdefault(value_id, value)


def build_demo_temporal_onnx_model() -> onnx.ModelProto:
    """Create a tiny ONNX graph for the Week 4 temporal MVP demo."""

    helper = onnx.helper
    tensor_proto = onnx.TensorProto

    rolling = helper.make_node(
        "RollingMean",
        ["stream_in"],
        ["rolling_mean"],
        name="rolling_mean_node",
        window_size=4,
    )
    conv = helper.make_node(
        "Conv",
        ["rolling_mean", "conv_weight"],
        ["conv_out"],
        name="conv_node",
        strides=[1],
        pads=[1, 1],
    )
    head = helper.make_node(
        "Add",
        ["conv_out", "bias"],
        ["model_out"],
        name="head_add",
    )

    graph = helper.make_graph(
        [rolling, conv, head],
        "temporal_demo_graph",
        [helper.make_tensor_value_info("stream_in", tensor_proto.FLOAT, [1, 1, 1])],
        [helper.make_tensor_value_info("model_out", tensor_proto.FLOAT, [1, 1, 1])],
        initializer=[
            helper.make_tensor(
                "conv_weight",
                tensor_proto.FLOAT,
                [1, 1, 3],
                [0.25, 0.5, 0.25],
            ),
            helper.make_tensor("bias", tensor_proto.FLOAT, [1, 1, 1], [0.125]),
        ],
    )
    return helper.make_model(graph, producer_name="tempo_dag_demo")


__all__ = [
    "TemporalLoweringReport",
    "TemporalLoweringResult",
    "TemporalONNXParser",
    "build_demo_temporal_onnx_model",
]
