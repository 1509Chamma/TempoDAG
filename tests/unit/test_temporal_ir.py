from collections.abc import Mapping

import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.op import FPGACost, Operator
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import (
    BufferSpec,
    Edge0,
    EdgeDelta,
    Kernel,
    Process,
    StateKind,
    StateSpec,
    TemporalIRValidationError,
)


class PassOperator(Operator):
    OP_TYPE = "Pass"

    def validate(self, values: Mapping[str, Value]) -> None:
        return None

    def estimate_fpga_cost(self, values: Mapping[str, Value]) -> FPGACost:
        return FPGACost(latency_cycles=1)

    def hls_template_path(self) -> str:
        return "pass.cpp"

    def hls_context(self, values: Mapping[str, Value]) -> dict[str, object]:
        return {}


def _empty_kernel(kernel_id: str) -> Kernel:
    graph = Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[])
    return Kernel(kernel_id=kernel_id, graph=graph)


def test_temporal_process_serializes_core_components() -> None:
    kernel = _empty_kernel("feature_kernel")
    state = StateSpec(
        state_id="hidden",
        kind=StateKind.HIDDEN,
        dtype="float32",
        shape=(1, 8),
        axes=("batch", "channel"),
    )
    buffer = BufferSpec(
        buffer_id="rolling_window",
        dtype="float32",
        shape=(8,),
        depth=4,
        axes=("channel",),
    )
    process = Process(
        process_id="streaming_model",
        kernels={kernel.kernel_id: kernel},
        states={state.state_id: state},
        buffers={buffer.buffer_id: buffer},
        edge0=[Edge0("rolling_window", "feature_kernel", value_id="window")],
        edge_delta=[EdgeDelta("feature_kernel", "hidden", lag_cycles=1)],
    )

    process.validate()
    data = process.to_dict()

    assert data["process_id"] == "streaming_model"
    assert data["clocks"] == {
        "main": {"clock_id": "main", "period": 1, "unit": "cycle"}
    }
    assert data["states"]["hidden"]["kind"] == "hidden_state"
    assert data["buffers"]["rolling_window"]["depth"] == 4
    assert data["edge0"] == [
        {"source": "rolling_window", "target": "feature_kernel", "value_id": "window"}
    ]
    assert data["edge_delta"] == [
        {
            "source": "feature_kernel",
            "target": "hidden",
            "lag_cycles": 1,
            "value_id": None,
        }
    ]


def test_same_timestep_cycle_is_rejected() -> None:
    process = Process(
        process_id="bad_cycle",
        kernels={
            "a": _empty_kernel("a"),
            "b": _empty_kernel("b"),
        },
        edge0=[Edge0("a", "b"), Edge0("b", "a")],
    )

    with pytest.raises(TemporalIRValidationError, match="edge0 dependencies"):
        process.validate()


def test_temporal_cycle_with_positive_lag_is_allowed() -> None:
    process = Process(
        process_id="state_feedback",
        kernels={"cell": _empty_kernel("cell")},
        states={
            "hidden": StateSpec(
                state_id="hidden",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(8,),
            )
        },
        edge0=[Edge0("hidden", "cell")],
        edge_delta=[EdgeDelta("cell", "hidden", lag_cycles=1)],
    )

    process.validate()


def test_edge_delta_requires_positive_lag() -> None:
    process = Process(
        process_id="bad_lag",
        kernels={"cell": _empty_kernel("cell")},
        states={
            "hidden": StateSpec(
                state_id="hidden",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(8,),
            )
        },
        edge_delta=[EdgeDelta("cell", "hidden", lag_cycles=0)],
    )

    with pytest.raises(TemporalIRValidationError, match="positive integer"):
        process.validate()


def test_unknown_edge_endpoint_is_rejected() -> None:
    process = Process(
        process_id="unknown_endpoint",
        kernels={"cell": _empty_kernel("cell")},
        edge0=[Edge0("cell", "missing")],
    )

    with pytest.raises(TemporalIRValidationError, match="unknown 'missing'"):
        process.validate()


def test_component_clocks_must_be_declared() -> None:
    process = Process(
        process_id="missing_clock",
        kernels={
            "fast_kernel": Kernel(
                kernel_id="fast_kernel",
                graph=Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[]),
                clock_id="fast",
            )
        },
    )

    with pytest.raises(TemporalIRValidationError, match="unknown clock 'fast'"):
        process.validate()


def test_component_dictionary_keys_must_match_object_ids() -> None:
    process = Process(
        process_id="mismatched_key",
        kernels={"alias": _empty_kernel("actual")},
    )

    with pytest.raises(
        TemporalIRValidationError,
        match="kernel key 'alias' does not match kernel_id 'actual'",
    ):
        process.validate()


def test_kernel_graph_must_be_structurally_valid() -> None:
    graph = Graph(values={}, ops={}, graph_inputs=[], graph_outputs=["missing"])
    process = Process(
        process_id="bad_kernel_graph",
        kernels={"kernel": Kernel(kernel_id="kernel", graph=graph)},
    )

    with pytest.raises(
        TemporalIRValidationError,
        match="kernel 'kernel' graph is invalid",
    ):
        process.validate()


def test_kernel_graph_must_be_acyclic_within_timestep() -> None:
    a = Value(
        value_id="a",
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=[1],
        axes=["channel"],
        producer_op_id="op_a",
    )
    b = Value(
        value_id="b",
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=[1],
        axes=["channel"],
        producer_op_id="op_b",
    )
    graph = Graph(
        values={"a": a, "b": b},
        ops={
            "op_a": PassOperator(op_id="op_a", inputs=["b"], outputs=["a"]),
            "op_b": PassOperator(op_id="op_b", inputs=["a"], outputs=["b"]),
        },
        graph_inputs=[],
        graph_outputs=["a"],
    )
    process = Process(
        process_id="cyclic_kernel_graph",
        kernels={"kernel": Kernel(kernel_id="kernel", graph=graph)},
    )

    with pytest.raises(
        TemporalIRValidationError,
        match="kernel 'kernel' graph is invalid: Graph contains a cycle",
    ):
        process.validate()


def test_state_and_buffer_specs_validate_shapes() -> None:
    process = Process(
        process_id="bad_shape",
        states={
            "hidden": StateSpec(
                state_id="hidden",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(8,),
                axes=("batch", "channel"),
            )
        },
        buffers={
            "window": BufferSpec(
                buffer_id="window",
                dtype="float32",
                shape=(8,),
                depth=4,
            )
        },
    )

    with pytest.raises(TemporalIRValidationError, match="shape and axes"):
        process.validate()


def test_kernel_can_wrap_existing_same_timestep_graph() -> None:
    value = Value(
        value_id="x",
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=[1, 4],
        axes=["batch", "channel"],
    )
    graph = Graph(values={"x": value}, ops={}, graph_inputs=["x"], graph_outputs=["x"])
    process = Process(
        process_id="kernel_reuse",
        kernels={"identity": Kernel(kernel_id="identity", graph=graph)},
    )

    process.validate()

    kernel_dict = process.to_dict()["kernels"]["identity"]
    assert kernel_dict["graph"]["values"]["x"] == value.to_dict()
