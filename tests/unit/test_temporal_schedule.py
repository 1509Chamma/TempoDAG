from typing import cast

import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir.value import Value, ValueType
from tempo_dag.ir_temporal import (
    BufferSpec,
    Edge0,
    EdgeDelta,
    Kernel,
    Process,
    ResetPolicy,
    ScheduleEdgeKind,
    ScheduleNodeKind,
    StateKind,
    StateSpec,
    TemporalExecutionContract,
    TemporalStorageKind,
    derive_temporal_schedule,
)
from tempo_dag.ops.builtins import Add, MatMul, Mul


def tensor(
    value_id: str,
    shape: list[int],
    *,
    layout: str | None = None,
    quant: dict[str, float | int | str | None] | None = None,
) -> Value:
    return Value(
        value_id=value_id,
        vtype=ValueType.TENSOR,
        dtype="float32",
        shape=shape,
        axes=[f"axis_{idx}" for idx in range(len(shape))],
        layout=layout,
        quant=quant,
    )


def test_schedule_classifies_operator_streams_and_graph_ports() -> None:
    values = {
        "x": tensor("x", [2]),
        "y": tensor("y", [2]),
        "z": tensor("z", [2]),
        "out": tensor("out", [2]),
    }
    graph = Graph(
        values=values,
        ops={
            "add": Add("add", inputs=["x", "y"], outputs=["z"]),
            "mul": Mul("mul", inputs=["z", "y"], outputs=["out"]),
        },
        graph_inputs=["x", "y"],
        graph_outputs=["out"],
    )
    process = Process(
        process_id="stateless_chain",
        kernels={"main_kernel": Kernel("main_kernel", graph=graph)},
    )

    schedule = derive_temporal_schedule(process)
    edges = {edge.edge_id: edge for edge in schedule.edges}
    nodes = {node.node_id: node for node in schedule.nodes}

    assert nodes["main_kernel"].kind == ScheduleNodeKind.KERNEL
    assert nodes["main_kernel.add"].phase == 0
    assert nodes["main_kernel.mul"].phase == 1
    assert edges["main_kernel.input->add:x"].kind == ScheduleEdgeKind.GRAPH_INPUT
    assert edges["main_kernel.add->mul:z"].kind == ScheduleEdgeKind.STREAM
    assert edges["main_kernel.mul->output:out"].kind == ScheduleEdgeKind.GRAPH_OUTPUT
    assert schedule.estimated_initiation_interval == 1
    assert schedule.estimated_latency_cycles >= 1


def test_schedule_classifies_parameters_and_operator_cost_metadata() -> None:
    values = {
        "x": tensor("x", [2, 3]),
        "w": tensor("w", [3, 4], layout="parameter"),
        "out": tensor("out", [2, 4]),
    }
    graph = Graph(
        values=values,
        ops={"matmul": MatMul("matmul", inputs=["x", "w"], outputs=["out"])},
        graph_inputs=["x", "w"],
        graph_outputs=["out"],
    )
    process = Process(
        process_id="parameterized_kernel",
        kernels={"kernel": Kernel("kernel", graph=graph)},
    )

    schedule = derive_temporal_schedule(process)
    edges = {edge.edge_id: edge for edge in schedule.edges}
    matmul = next(node for node in schedule.nodes if node.node_id == "kernel.matmul")

    assert edges["kernel.param->matmul:w"].kind == ScheduleEdgeKind.PARAMETER_BLOCK
    assert edges["kernel.param->matmul:w"].source == "kernel.param"
    assert edges["kernel.param->matmul:w"].storage_kind == TemporalStorageKind.RAM
    assert matmul.metadata == {
        "kernel_id": "kernel",
        "op_type": "MatMul",
        "dsp": 3,
        "bram": 0,
        "lut": 8,
        "ff": 8,
    }


def test_schedule_classifies_state_buffer_and_temporal_delay_edges() -> None:
    graph = Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[])
    process = Process(
        process_id="rolling_state",
        kernels={"kernel": Kernel("kernel", graph=graph)},
        states={
            "hidden": StateSpec(
                state_id="hidden",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(4,),
            )
        },
        buffers={
            "window": BufferSpec(
                buffer_id="window",
                dtype="float32",
                shape=(1,),
                depth=8,
            )
        },
        edge0=[
            Edge0("hidden", "kernel", value_id="h_prev"),
            Edge0("window", "kernel", value_id="window_view"),
        ],
        edge_delta=[
            EdgeDelta("kernel", "hidden", lag_cycles=1, value_id="h_next"),
            EdgeDelta("kernel", "window", lag_cycles=8, value_id="x_t"),
        ],
    )

    schedule = derive_temporal_schedule(process)
    edge_kinds = {edge.edge_id: edge.kind for edge in schedule.edges}
    storage = {edge.edge_id: edge.storage_kind for edge in schedule.edges}

    assert edge_kinds["hidden->kernel:h_prev"] == ScheduleEdgeKind.STATE_READ
    assert edge_kinds["window->kernel:window_view"] == ScheduleEdgeKind.BUFFER_READ
    assert edge_kinds["kernel->hidden@1:h_next"] == ScheduleEdgeKind.TEMPORAL_DELAY
    assert storage["window->kernel:window_view"] == TemporalStorageKind.RING_BUFFER
    assert storage["kernel->window@8:x_t"] == TemporalStorageKind.SHIFT_REGISTER


def test_schedule_serializes_to_report_dict() -> None:
    process = Process(
        process_id="empty_kernel",
        kernels={
            "kernel": Kernel(
                "kernel",
                graph=Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[]),
            )
        },
    )

    payload = derive_temporal_schedule(process).to_dict()
    nodes = cast(list[dict[str, object]], payload["nodes"])

    assert payload["process_id"] == "empty_kernel"
    assert payload["estimated_latency_cycles"] == 1
    assert payload["estimated_initiation_interval"] == 1
    assert nodes[0]["node_id"] == "kernel"


def test_schedule_rejects_contract_for_different_process() -> None:
    process = Process(
        process_id="actual_process",
        kernels={
            "kernel": Kernel(
                "kernel",
                graph=Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[]),
            )
        },
    )
    contract = TemporalExecutionContract(
        process_id="other_process",
        reset_policy=ResetPolicy.ZERO,
        warmup_timesteps=0,
        flush_cycles=0,
        edge_delta_storage=(),
        buffer_storage=(),
    )

    with pytest.raises(ValueError, match="process_id"):
        derive_temporal_schedule(process, contract)
